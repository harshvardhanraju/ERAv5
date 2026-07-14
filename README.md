# ERA v5 — School of AI

Coursework from the ERA v5 (Extensive & Reimagined AI) program.

## Sessions

| Session | Topic | Live Demo |
|---------|-------|-----------|
| [Session 1](./ses1/) | Neural Network Fundamentals — Activations, Depth, Embeddings, Generalization | [harsha-erav5-s1.netlify.app](https://harsha-erav5-s1.netlify.app) |
| [Session 2](./ses2/) | Multilingual BPE Tokenizer — English, Hindi, Telugu, Bengali | [harsha-erav5-s2.netlify.app](https://harsha-erav5-s2.netlify.app) |

---

## Session 1 — Neural Network Fundamentals

Four live proofs of core deep learning principles, implemented from scratch in vanilla JavaScript (no libraries):

- **S1-1 · Activations** — Linear vs ReLU on concentric rings; watch the boundary go from straight line to ring-shaped
- **S1-2 · Depth** — 1-layer linear = 5-layer linear = W₁×W₂×W₃×W₄×W₅ (all the same matrix)
- **S1-3 · Embeddings** — Next-token prediction spontaneously clusters animals/fruits/verbs in embedding space
- **S1-4 · Generalization** — Memorization gap at n=20 vs generalization at n=2000

All neural network code (forward pass, backpropagation, SGD) written from scratch in ~600 lines of vanilla JS.

---

## Session 2 — Multilingual BPE Tokenizer

A single 10,000-token BPE vocabulary trained on the Wikipedia "India" article in four languages. The goal is to minimise the spread of compression ratios across languages to maximise the assignment score.

**Score formula:** `1000 / (max_fertility − min_fertility)` where `fertility = BPE_tokens / faithful_units`

| Language | Script     | Faithful Units | BPE Tokens | Fertility |
|----------|------------|---------------|------------|-----------|
| Bengali  | Bengali    | 18,959        | 23,842     | 1.2576    |
| Hindi    | Devanagari | 20,223        | 25,554     | 1.2636    |
| English  | Latin      | 43,126        | 55,560     | 1.2883    |
| Telugu   | Telugu     | 8,183         | 10,565     | 1.2911    |

**Spread:** 0.0335 · **Raw score:** 29,820 · **Adjusted score:** 28,280

Key design choices:
- **Metaspace pre-tokeniser** — keeps "India's" as one unit; enables cross-punctuation merges
- **NFKC normalisation** — handles Indic script glyph variants consistently
- **min_frequency=1** — no character pair left unmerged
- **Balanced corpus weights** `en×3, hi×5, te×8, bn×8` — compensates for article size differences (English article is 5× larger than Telugu's)
- **Bengali as 4th language** — large article (96 KB) with Brahmic-family script overlap with Hindi

### Reproducing the results

```bash
cd ses2
pip install tokenizers requests beautifulsoup4 lxml regex
python3 train_bpe.py          # fetches Wikipedia, trains, saves all artefacts
```

Output files:
- `tokenizer.json` — full HuggingFace-compatible BPE model
- `vocab.json` — 10,000 tokens sorted by ID
- `merges.json` — 9,644 BPE merge rules
- `corpus/*.txt` — Wikipedia article snapshots used for scoring
