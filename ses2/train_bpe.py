#!/usr/bin/env python3
"""
ERA v5 Session 2 – Multilingual BPE Tokenizer
Languages: English, Hindi, Telugu, Bengali
Vocab: 10 000 tokens (shared across all 4 languages)
Scoring: fertility = token_count / faithful_unit_count
Score = 1000 / (max_fertility - min_fertility)

Design choices:
- Metaspace pre-tokenizer: tokens span word+punctuation, enabling fertility < 1.0
- NFKC normalizer: Unicode normalization for Indic scripts
- min_frequency=1: no pair left behind
- Full article content (tables, infoboxes, navboxes): more text = richer pair statistics
- Balanced corpus weights en×3, hi×5, te×8, bn×8: compensates for article size differences
- Bengali as 4th language: large article (96 KB) with Brahmic-family overlap with Hindi
"""

import json, re, os, sys, math
import urllib.request, urllib.parse

import requests
from bs4 import BeautifulSoup

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.normalizers import NFKC
from tokenizers.decoders import Metaspace as MetaspaceDecoder

import regex  # full Unicode support

# ── Config ───────────────────────────────────────────────────────────────────

# Primary corpus: scored AND trained on (the "India" article in each language)
SCORE_LANGUAGES = {
    "English": {"lang": "en", "title": "India"},
    "Hindi":   {"lang": "hi", "title": "भारत"},
    "Telugu":  {"lang": "te", "title": "భారతదేశం"},
    "Bengali": {"lang": "bn", "title": "ভারত"},
}

# Supplemental TRAINING-ONLY articles (not used in fertility scoring).
SUPPLEMENTAL_TRAIN: dict = {}

ALPHA = 0.3  # kept for the alpha_weights helper but not used when override is set

# Empirically tuned corpus replication weights.
# The India Wikipedia article in English is 5× larger than Telugu's; these weights
# compensate so BPE sees roughly equal text per language during training.
TRAIN_WEIGHTS_OVERRIDE = {"English": 3, "Hindi": 5, "Telugu": 8, "Bengali": 8}

# Iterative re-weighting adds a second training pass but consistently hurt the score
# in practice — disabled.
ITERATIVE_REWEIGHT = False
REWEIGHT_BETA      = 0.5

VOCAB_SIZE = 10_000
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))

# ── Faithful-unit counter ─────────────────────────────────────────────────────

def count_faithful_units(text: str) -> int:
    """
    One unit = contiguous Unicode letter/mark/number run OR
               one visible non-space punctuation/symbol char.
    Uses the `regex` module for full Unicode property support.
    """
    units = regex.findall(r'[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]', text)
    return len(units)

# ── Wikipedia fetch ───────────────────────────────────────────────────────────

def fetch_wiki_text(lang_code: str, title: str) -> str:
    """
    Fetch Wikipedia article HTML (action=parse) and extract all visible text,
    including tables, infoboxes, and navboxes — more text means richer pair statistics
    for BPE training.
    """
    headers = {"User-Agent": "ERA-v5-BPE/1.0 (harshvardhan.rajug@gmail.com)"}

    parse_url = (
        f"https://{lang_code}.wikipedia.org/w/api.php?"
        + urllib.parse.urlencode({
            "action": "parse", "page": title,
            "prop": "text", "format": "json", "formatversion": "2",
        })
    )
    try:
        resp = requests.get(parse_url, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        html  = data["parse"]["text"]
        soup  = BeautifulSoup(html, "lxml")

        # Remove invisible / metadata elements only — keep all article text
        # (tables, infoboxes, navboxes) for richer BPE pair statistics.
        for tag in soup.find_all(["style", "script", "link", "meta"]):
            tag.decompose()
        for tag in soup.select(".mw-editsection"):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

    except Exception as e:
        print(f"    parse failed ({e}), falling back to extract API...")
        params = urllib.parse.urlencode({
            "action": "query", "titles": title,
            "prop": "extracts", "explaintext": "1", "format": "json",
        })
        api_url = f"https://{lang_code}.wikipedia.org/w/api.php?{params}"
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        page = next(iter(data["query"]["pages"].values()))
        text = page.get("extract", "")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ── Corpus sampling ───────────────────────────────────────────────────────────

def alpha_weights(char_counts: dict[str, int], alpha: float = 0.3) -> dict[str, int]:
    """
    Compute integer corpus replication weights using p(l) ∝ N_chars(l)^alpha.
    Normalised so the smallest language gets weight 1; others scaled up.
    α=0.3 (XLM-R recipe) strongly upsamples low-resource languages.
    """
    raw    = {lang: max(count, 1) ** alpha for lang, count in char_counts.items()}
    min_w  = min(raw.values())
    # Scale so min weight = 1, round to nearest integer (min 1)
    return {lang: max(1, round(w / min_w)) for lang, w in raw.items()}

def build_corpus(
    score_texts: dict[str, str],
    supp_texts:  dict[str, list[str]],
    weights:     dict[str, int],
) -> list[str]:
    corpus = []
    for lang, text in score_texts.items():
        w = weights.get(lang, 1)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        corpus.extend(lines * w)
    for lang, extra_list in supp_texts.items():
        w = weights.get(lang, 1)
        for extra_text in extra_list:
            lines = [l.strip() for l in extra_text.split("\n") if l.strip()]
            corpus.extend(lines * w)
    return corpus

# ── Tokenizer training ────────────────────────────────────────────────────────

def train_tokenizer(corpus: list[str]) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer    = NFKC()
    tok.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
    tok.decoder       = MetaspaceDecoder(replacement="▁", prepend_scheme="always")

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[UNK]"],
        min_frequency=1,
        show_progress=True,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    return tok

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(tok: Tokenizer, texts: dict[str, str]) -> dict:
    stats = {}
    for lang, text in texts.items():
        fu   = count_faithful_units(text)
        enc  = tok.encode(text)
        tc   = len(enc.ids)
        fert = round(tc / fu, 6) if fu else 0.0
        stats[lang] = {
            "char_count":     len(text),
            "faithful_units": fu,
            "token_count":    tc,
            "fertility":      fert,
            "sample_tokens":  enc.tokens[:80],
        }
    return stats

def score_result(stats: dict) -> tuple[float, float, float, float, dict]:
    fertilies    = {l: s["fertility"] for l, s in stats.items()}
    sorted_langs = sorted(fertilies, key=fertilies.get)
    f_min        = fertilies[sorted_langs[0]]
    f_max        = fertilies[sorted_langs[-1]]
    spread       = round(f_max - f_min, 6)
    raw_score    = round(1000 / spread, 4) if spread > 0 else 9999.0
    hi_fert      = fertilies.get("Hindi", 0)
    penalty      = math.exp(max(0, hi_fert / 1.2 - 1))
    adj_score    = round(raw_score / penalty, 4)
    return f_min, f_max, spread, raw_score, adj_score, penalty, fertilies, sorted_langs

# ── Iterative re-weighting ────────────────────────────────────────────────────

def reweight(
    current_weights: dict[str, int],
    fertilies:       dict[str, float],
    beta:            float = 0.5,
) -> dict[str, int]:
    """
    Adjust weights inversely proportional to fertility deviation from mean.
    Languages with high fertility get more weight; low fertility get less.
    β=0.5 damps the adjustment to avoid overshooting.
    """
    target  = sum(fertilies.values()) / len(fertilies)
    new_raw = {}
    for lang, w in current_weights.items():
        if lang not in fertilies:
            new_raw[lang] = w
            continue
        ratio      = (target / fertilies[lang]) ** beta
        new_raw[lang] = max(1.0, w * ratio)
    # Re-normalise: scale so minimum is 1, round to nearest integer
    min_w = min(new_raw.values())
    return {lang: max(1, round(v / min_w)) for lang, v in new_raw.items()}

# ── Round-trip check ──────────────────────────────────────────────────────────

def check_roundtrip(tok: Tokenizer, text: str, lang: str):
    import unicodedata
    enc      = tok.encode(text[:500])
    decoded  = tok.decode(enc.ids)
    orig_nw  = re.sub(r'\s', '', unicodedata.normalize('NFKC', text[:500]))
    dec_nw   = re.sub(r'\s', '', decoded)
    if orig_nw != dec_nw:
        print(f"  WARNING  {lang} round-trip mismatch (first 80 chars):")
        print(f"      orig: {orig_nw[:80]}")
        print(f"      dec:  {dec_nw[:80]}")
    else:
        print(f"  OK  {lang} round-trip OK")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== ERA v5 Session 2 – BPE Tokenizer ===\n")

    # 1. Fetch scoring corpus (India article in each language)
    print("Step 1: Fetching scoring corpus (Wikipedia India articles)")
    score_texts = {}
    for name, cfg in SCORE_LANGUAGES.items():
        print(f"  {name} ({cfg['lang']}.wikipedia.org/{cfg['title']})...")
        t = fetch_wiki_text(cfg["lang"], cfg["title"])
        score_texts[name] = t
        print(f"    {len(t):,} chars | {count_faithful_units(t):,} faithful units")

    # 2. Fetch supplemental training corpus (Telugu only, not scored)
    print("\nStep 2: Fetching supplemental training corpus")
    supp_texts: dict[str, list[str]] = {}
    for lang, articles in SUPPLEMENTAL_TRAIN.items():
        supp_texts[lang] = []
        for art in articles:
            print(f"  {lang} +supp: {art['lang']}.wikipedia.org/{art['title']}...")
            t = fetch_wiki_text(art["lang"], art["title"])
            supp_texts[lang].append(t)
            print(f"    {len(t):,} chars")

    # 3. Compute initial training weights via α-temperature or use override
    print("\nStep 3: Computing training weights")
    if TRAIN_WEIGHTS_OVERRIDE:
        weights = TRAIN_WEIGHTS_OVERRIDE
        print(f"  Using manual override: {weights}")
    else:
        # Total chars per language = score article + all supplemental articles
        total_chars = {}
        for lang, text in score_texts.items():
            total_chars[lang] = len(text)
        for lang, extras in supp_texts.items():
            total_chars[lang] = total_chars.get(lang, 0) + sum(len(t) for t in extras)
        weights = alpha_weights(total_chars, alpha=ALPHA)
        print(f"  Corpus sizes: { {l: f'{c:,}' for l, c in total_chars.items()} }")
        print(f"  α={ALPHA} weights: {weights}")

    # 4. Build corpus and train (pass 1)
    print(f"\nStep 4: Building corpus and training BPE (pass 1, weights={weights})")
    corpus1 = build_corpus(score_texts, supp_texts, weights)
    total_chars_corpus = sum(len(l) for l in corpus1)
    print(f"  Corpus: {len(corpus1):,} lines | {total_chars_corpus:,} chars total")
    tok = train_tokenizer(corpus1)
    print(f"  Actual vocab size: {tok.get_vocab_size()}")

    # 5. Round-trip check
    print("\nStep 5: Round-trip verification (pass 1)")
    for lang, text in score_texts.items():
        check_roundtrip(tok, text, lang)

    # 6. Evaluate pass 1
    print("\nStep 6: Fertility (pass 1)")
    stats1 = evaluate(tok, score_texts)
    for lang, s in stats1.items():
        print(f"  {lang}: {s['faithful_units']:,} FU → {s['token_count']:,} tokens  fertility={s['fertility']:.6f}")

    f_min1, f_max1, spread1, score1, adj1, pen1, ferts1, sorted1 = score_result(stats1)
    print(f"\n  Spread={spread1:.6f}  Raw score={score1:.2f}  Penalty={pen1:.4f}  Adj={adj1:.2f}")

    # 7. Iterative re-weighting (pass 2)
    stats_final  = stats1
    tok_final    = tok
    weights_final = weights
    spread_final = spread1
    score_final  = score1
    adj_final    = adj1

    if ITERATIVE_REWEIGHT:
        ferts1_map = {l: s["fertility"] for l, s in stats1.items()}
        weights2   = reweight(weights, ferts1_map, beta=REWEIGHT_BETA)
        print(f"\nStep 7: Iterative re-weighting")
        print(f"  Pass 1 fertilities: { {l: f'{v:.4f}' for l, v in ferts1_map.items()} }")
        print(f"  New weights: {weights2}")

        corpus2 = build_corpus(score_texts, supp_texts, weights2)
        total2  = sum(len(l) for l in corpus2)
        print(f"  Corpus: {len(corpus2):,} lines | {total2:,} chars total")
        tok2 = train_tokenizer(corpus2)
        print(f"  Actual vocab size: {tok2.get_vocab_size()}")

        print("\nStep 7b: Round-trip verification (pass 2)")
        for lang, text in score_texts.items():
            check_roundtrip(tok2, text, lang)

        print("\nStep 7c: Fertility (pass 2)")
        stats2 = evaluate(tok2, score_texts)
        for lang, s in stats2.items():
            print(f"  {lang}: {s['faithful_units']:,} FU → {s['token_count']:,} tokens  fertility={s['fertility']:.6f}")

        f_min2, f_max2, spread2, score2, adj2, pen2, ferts2, sorted2 = score_result(stats2)
        print(f"\n  Spread={spread2:.6f}  Raw score={score2:.2f}  Penalty={pen2:.4f}  Adj={adj2:.2f}")

        # Keep whichever pass produced the better adjusted score
        if adj2 >= adj_final:
            print(f"  Pass 2 is better (adj {adj2:.2f} >= {adj_final:.2f}) — using pass 2 tokenizer")
            stats_final   = stats2
            tok_final     = tok2
            weights_final = weights2
            spread_final  = spread2
            score_final   = score2
            adj_final     = adj2
        else:
            print(f"  Pass 1 is better (adj {adj_final:.2f} > {adj2:.2f}) — keeping pass 1 tokenizer")

    # 8. Print final summary
    f_min, f_max, spread, raw_sc, adj_sc, penalty, ferts, sorted_langs = score_result(stats_final)
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"  Fertility ranking:")
    for lang in sorted_langs:
        print(f"    {lang}: {ferts[lang]:.6f}")
    print(f"\n  Spread = {f_max:.6f} - {f_min:.6f} = {spread:.6f}")
    print(f"  Raw score = 1000 / {spread:.6f} = {raw_sc:.2f}")
    print(f"  Hindi penalty factor = {penalty:.6f}")
    print(f"  Hindi-adjusted score = {adj_sc:.2f}")

    # 9. Save outputs
    print("\nStep 8: Saving outputs")

    corpus_dir = os.path.join(OUT_DIR, "corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    for lang, text in score_texts.items():
        lc = SCORE_LANGUAGES[lang]["lang"]
        with open(os.path.join(corpus_dir, f"{lc}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    for lang, extras in supp_texts.items():
        lc = SCORE_LANGUAGES[lang]["lang"]
        for i, text in enumerate(extras):
            with open(os.path.join(corpus_dir, f"{lc}_supp_{i+1}.txt"), "w", encoding="utf-8") as f:
                f.write(text)

    vocab_dict = tok_final.get_vocab()
    vocab_list = sorted(vocab_dict, key=vocab_dict.get)

    model_json  = json.loads(tok_final.to_str())
    raw_merges  = model_json.get("model", {}).get("merges", [])
    merges_list = [m.split(" ", 1) if isinstance(m, str) else list(m) for m in raw_merges]

    results = {
        "languages":        list(SCORE_LANGUAGES.keys()),
        "fourth_language":  "Bengali",
        "supplemental":     {k: [a["title"] for a in v] for k, v in SUPPLEMENTAL_TRAIN.items()},
        "alpha":            ALPHA,
        "train_weights":    weights_final,
        "vocab_size":       tok_final.get_vocab_size(),
        "num_merges":       len(merges_list),
        "stats":            stats_final,
        "fertilies_sorted": [(l, ferts[l]) for l in sorted_langs],
        "f_min":            f_min,
        "f_max":            f_max,
        "f_min_lang":       sorted_langs[0],
        "f_max_lang":       sorted_langs[-1],
        "spread":           spread,
        "score":            raw_sc,
        "hindi_penalty":    round(penalty, 6),
        "adj_score":        adj_sc,
    }

    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_list, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "merges.json"), "w", encoding="utf-8") as f:
        json.dump(merges_list, f, ensure_ascii=False, indent=2)

    tok_final.save(os.path.join(OUT_DIR, "tokenizer.json"))

    print(f"  results.json, vocab.json ({len(vocab_list)} tokens), merges.json ({len(merges_list)} merges), tokenizer.json")
    print("\nDone!")
    return results

if __name__ == "__main__":
    main()
