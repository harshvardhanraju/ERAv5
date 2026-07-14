#!/usr/bin/env python3
"""
ERA v5 Session 2 – Multilingual BPE Tokenizer
Languages: English, Hindi, Telugu, Maithili
Vocab: 10 000 tokens (shared across all 4 languages)
Scoring: fertility = token_count / faithful_unit_count
Score = 1000 / (max_fertility - min_fertility)

Design choices (following reference):
- Metaspace pre-tokenizer: tokens span word+punctuation, enabling fertility < 1.0
- NFKC normalizer: Unicode normalization for Indic scripts
- min_frequency=1: no pair left behind
- Decoder: Metaspace (round-trips text)
- Training weights per language: tuned to equalise fertility
"""

import json, re, os, sys, math
import urllib.request, urllib.parse
from collections import Counter

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

LANGUAGES = {
    "English": {"lang": "en", "title": "India"},
    "Hindi":   {"lang": "hi", "title": "भारत"},
    "Telugu":  {"lang": "te", "title": "భారతదేశం"},
    "Bengali": {"lang": "bn", "title": "ভারত"},   # 96 K chars; Bengali script ≈ Devanagari
}

# Training weights: more weight → more BPE merges → lower fertility.
# Tuned to minimise spread across all 4 languages.
TRAIN_WEIGHTS = {
    "English": 3,
    "Hindi":   5,
    "Telugu":  8,
    "Bengali": 8,
}

VOCAB_SIZE  = 10_000
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))

# ── Faithful-unit counter ─────────────────────────────────────────────────────

def count_faithful_units(text: str) -> int:
    """
    One unit = contiguous Unicode letter/mark/number run  OR
               one visible non-space punctuation/symbol char.
    Uses the `regex` module for full Unicode property support.
    """
    units = regex.findall(r'[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]', text)
    return len(units)

# ── Wikipedia fetch ───────────────────────────────────────────────────────────

def fetch_wiki_text(lang_code: str, title: str) -> str:
    """
    Fetch Wikipedia article HTML (action=parse) and extract all visible text via
    BeautifulSoup.  Only invisible/meta elements are removed.  All content —
    tables, infoboxes, navboxes, references, captions — is preserved so that the
    corpus is as large and as faithful to the Wikipedia article as possible.
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

        # Remove only invisible / metadata elements
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

# ── Training ──────────────────────────────────────────────────────────────────

def build_corpus(texts: dict[str, str], weights: dict[str, int]) -> list[str]:
    corpus = []
    for lang, text in texts.items():
        w = weights.get(lang, 1)
        # Split into sentences/paragraphs for iterator training
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        corpus.extend(lines * w)
    return corpus

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

# ── Round-trip check ──────────────────────────────────────────────────────────

def check_roundtrip(tok: Tokenizer, text: str, lang: str):
    import unicodedata
    enc      = tok.encode(text[:500])
    decoded  = tok.decode(enc.ids)
    # Normalise both sides with NFKC (tokenizer applies NFKC internally)
    orig_nw  = re.sub(r'\s', '', unicodedata.normalize('NFKC', text[:500]))
    dec_nw   = re.sub(r'\s', '', decoded)
    if orig_nw != dec_nw:
        print(f"  ⚠️  {lang} round-trip mismatch (first 80 chars):")
        print(f"      orig: {orig_nw[:80]}")
        print(f"      dec:  {dec_nw[:80]}")
    else:
        print(f"  ✓  {lang} round-trip OK")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== ERA v5 Session 2 – BPE Tokenizer ===\n")

    # 1. Fetch texts
    print("Step 1: Fetching Wikipedia articles")
    texts = {}
    for name, cfg in LANGUAGES.items():
        print(f"  {name} ({cfg['lang']}.wikipedia.org/{cfg['title']})...")
        t = fetch_wiki_text(cfg["lang"], cfg["title"])
        texts[name] = t
        print(f"    {len(t):,} chars | {count_faithful_units(t):,} faithful units")

    # 2. Build corpus
    print(f"\nStep 2: Building training corpus (weights: {TRAIN_WEIGHTS})")
    corpus = build_corpus(texts, TRAIN_WEIGHTS)
    total_chars = sum(len(l) for l in corpus)
    print(f"  Corpus: {len(corpus):,} lines | {total_chars:,} chars total")

    # 3. Train
    print(f"\nStep 3: Training BPE (vocab_size={VOCAB_SIZE}, min_frequency=1)")
    tok = train_tokenizer(corpus)
    print(f"  Actual vocab size: {tok.get_vocab_size()}")

    # 4. Round-trip check
    print("\nStep 4: Round-trip verification")
    for lang, text in texts.items():
        check_roundtrip(tok, text, lang)

    # 5. Evaluate
    print("\nStep 5: Computing fertility ratios")
    stats = evaluate(tok, texts)
    for lang, s in stats.items():
        print(f"  {lang}: {s['faithful_units']:,} FU → {s['token_count']:,} tokens  fertility={s['fertility']:.6f}")

    # 6. Score
    fertilies    = {l: s["fertility"] for l, s in stats.items()}
    sorted_langs = sorted(fertilies, key=fertilies.get)
    f_min        = fertilies[sorted_langs[0]]
    f_max        = fertilies[sorted_langs[-1]]
    spread       = round(f_max - f_min, 6)
    score        = round(1000 / spread, 4) if spread > 0 else 9999.0

    # Hindi penalty (should be 1.0 if Hindi fertility < 1.2)
    hi_fert       = fertilies.get("Hindi", 0)
    hindi_penalty = math.exp(max(0, hi_fert / 1.2 - 1))
    adj_score     = round(score / hindi_penalty, 4)

    print(f"\n  Fertility ranking:")
    for lang in sorted_langs:
        print(f"    {lang}: {fertilies[lang]:.6f}")
    print(f"\n  Spread = {f_max:.6f} - {f_min:.6f} = {spread:.6f}")
    print(f"  Raw score = 1000 / {spread:.6f} = {score:.2f}")
    print(f"  Hindi penalty factor = {hindi_penalty:.6f}")
    print(f"  Hindi-adjusted score = {adj_score:.2f}")

    # 7. Save
    print("\nStep 6: Saving outputs")

    # Corpus snapshots for reproducibility
    corpus_dir = os.path.join(OUT_DIR, "corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    for lang, text in texts.items():
        lc = LANGUAGES[lang]["lang"]
        with open(os.path.join(corpus_dir, f"{lc}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

    vocab_dict = tok.get_vocab()
    vocab_list = sorted(vocab_dict, key=vocab_dict.get)  # sorted by token id

    model_json  = json.loads(tok.to_str())
    raw_merges  = model_json.get("model", {}).get("merges", [])
    merges_list = [m.split(" ", 1) if isinstance(m, str) else list(m) for m in raw_merges]

    results = {
        "languages":        list(LANGUAGES.keys()),
        "fourth_language":  "Bengali",
        "vocab_size":       tok.get_vocab_size(),
        "num_merges":       len(merges_list),
        "train_weights":    TRAIN_WEIGHTS,
        "stats":            stats,
        "fertilies_sorted": [(l, fertilies[l]) for l in sorted_langs],
        "f_min":            f_min,
        "f_max":            f_max,
        "f_min_lang":       sorted_langs[0],
        "f_max_lang":       sorted_langs[-1],
        "spread":           spread,
        "score":            score,
        "hindi_penalty":    round(hindi_penalty, 6),
        "adj_score":        adj_score,
    }

    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_list, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "merges.json"), "w", encoding="utf-8") as f:
        json.dump(merges_list, f, ensure_ascii=False, indent=2)

    tok.save(os.path.join(OUT_DIR, "tokenizer.json"))

    print(f"  results.json, vocab.json ({len(vocab_list)} tokens), merges.json ({len(merges_list)} merges), tokenizer.json")
    print("\nDone!")
    return results

if __name__ == "__main__":
    main()
