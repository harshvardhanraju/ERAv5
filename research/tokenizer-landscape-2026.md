# The State of Tokenizers in 2026 — Research Review

> Published as a blog post: https://harshvardhanraju.github.io/posts/tokenizer-landscape-2026.html

Extended reading connected to the [Session 2 multilingual BPE tokenizer](../ses2/).

---

## Algorithm Comparison

| Algorithm | Core Idea | Used By |
|-----------|-----------|---------|
| BPE | Merge most-frequent adjacent pair | GPT-4o, LLaMA 3, Qwen, DeepSeek |
| WordPiece | Merge pair with highest PMI score | BERT, older Google models |
| Unigram LM | Start big, prune lowest-loss tokens | Gemma, Gemini, T5, LLaMA 1/2 |
| SentencePiece | Framework wrapping BPE or Unigram | Same as above |

---

## Vocabulary Size Evolution

| Model | Vocab Size | Year |
|-------|-----------|------|
| GPT-2 | 50K | 2019 |
| LLaMA 1/2 | 32K | 2023 |
| GPT-4 (cl100k_base) | 100K | 2023 |
| LLaMA 3 (tiktoken) | 128K | 2024 |
| Qwen 2/3 | 152K | 2024 |
| GPT-4o (o200k_base) | 200K | 2024 |
| Gemma / Gemini | 256K | 2024 |

The LLaMA 2→3 jump (32K→128K) is ~1B extra parameters in embedding tables alone.
GPT-4o's o200k_base reduced CJK token counts by ~40% vs cl100k_base.

---

## The Token Tax

Cross-language fertility multipliers with cl100k_base (GPT-4):

| Language | Token multiplier | API cost multiplier |
|----------|-----------------|-------------------|
| English | 1.0× | baseline |
| Spanish | 1.55× | 1.55× |
| Japanese | 2.93× | 2.93× |
| Arabic | 3.30× | 3.30× |
| Bengali | ~4.5× | 4.5× |

Context window impact: a 128K-token window holds ~30K English words but only ~8K Arabic words.

### Connection to ses2 assignment
The fertility spread metric (1000 / (max_fertility − min_fertility)) measures the same thing
as the "token tax" across four languages. Reducing spread from 0.154 → 0.034 is a 4.3× improvement
in multilingual equity. Parity-Aware BPE (arXiv 2508.04796) tries to bake this fairness objective
into the training algorithm rather than doing it through corpus weight tuning.

---

## Byte-Level Models

The sequence-length problem: byte sequences are 4–6× longer than token sequences.
Standard attention is O(n²) — naive byte models are 16–36× more expensive.

### Solutions

**ByT5 (2022)** — baseline. Works, multilingual, robust. Slow.

**MrT5 — Stanford/USC (ICLR 2025, arXiv:2410.20771)**
- Learned "delete gate" inside the encoder
- Progressively removes low-information tokens after encoding layers
- Up to 80% sequence length reduction, matches ByT5 accuracy

**Byte Latent Transformer — Meta AI (ACL 2025, arXiv:2412.09871)**
- Entropy-based dynamic patching: boundaries placed where local byte-entropy is high
- Easy text → long patches (less compute). Hard text → short patches (more compute)
- 8B params / 4T bytes — **matches tokenized LLMs at fixed FLOP cost**
- Better robustness to typos, OCR errors, unusual encodings

**BoundlessBPE (COLM 2025, arXiv:2504.00178)**
- Relaxes BPE's constraint: merges can cross whitespace boundaries
- 15% improvement in bytes-per-token
- Training: 4.7 CPU-days on 1GB (vs 59 seconds for standard BPE) — research only

---

## Morphologically-Aware Tokenization

For agglutinative languages (Turkish, Finnish, Korean, Hungarian):

- **MYTE** (ACL 2024) — morpheme-based byte codes. 99 languages all shorter than UTF-8. Worst-case: 1.7× vs English, vs 3.5× for plain UTF-8.
- **Morpheus** (arXiv 2606.18717) — morphologically-aware tokenizer + word embedder for Turkish
- **Thunder-Tok / VerChol** (arXiv 2603.05883) — 10% fertility reduction for Korean via grammar-first pre-tokenization
- **Tokens with Meaning** (arXiv 2508.14292) — hybrid morphological + BPE approach for Turkish

---

## The NP-Completeness Result

**"Tokenisation is NP-Complete"** — Whittington, Bachmann, Pimentel (ACL 2025, arXiv:2412.15210)

Both optimal vocabulary selection AND optimal encoding given a fixed vocabulary are NP-complete (reduction from set cover).

Implication: every BPE tokenizer is a greedy approximation to an intractable problem. Optimal corpus weights
cannot be computed analytically — grid search is the right approach. The sharp ridges in the fertility landscape
are a property of NP-hard optimization, not measurement noise.

---

## Domain-Specific Tokenizers

**KL3M** (arXiv 2503.17247, March 2025)
- Trained on US legal, financial, and governmental text
- 35% smaller vocabulary than GPT-4o, but 9% fewer tokens for domain documents
- Up to **83% fewer tokens** for legal terminology vs LLaMA 3
- Open-source: `alea-institute/` on HuggingFace

**LiteToken** (arXiv 2602.04706, Feb 2026, Peking University)
- Found ~10% of tokens in o200k_base, Qwen, DeepSeek, LLaMA 3, Gemma 3, BLOOM are "intermediate merge residues"
- These tokens are almost never emitted at inference (always further merged)
- Removing them reduces vocabulary without performance loss; improves adversarial robustness

---

## Arithmetic Failures

Multi-digit numbers tokenize inconsistently: `12345` → `[123, 45]` or `[12, 345]` or `[1, 2345]`.
No consistent place-value structure. Models cannot "see" individual digits.

- **Paper:** "Tokenization Constraints in LLMs" (arXiv 2505.14178, 2025) — formally confirms arithmetic failure scales with operand size, directly attributable to tokenization
- **Google's fix:** Gemma split-digit tokenizer — numbers always become individual digit tokens
- **Production workaround:** route arithmetic to Python interpreter

---

## Key Papers

| Paper | Venue | arXiv |
|-------|-------|-------|
| Byte Latent Transformer (Meta AI) | ACL 2025 | 2412.09871 |
| Tokenisation is NP-Complete | ACL 2025 | 2412.15210 |
| MYTE: Morphology-Driven Byte Encoding | ACL 2024 | — |
| MrT5: Dynamic Token Merging | ICLR 2025 | 2410.20771 |
| BoundlessBPE | COLM 2025 | 2504.00178 |
| KL3M Domain Tokenizers | arXiv 2025 | 2503.17247 |
| LiteToken | arXiv 2026 | 2602.04706 |
| Parity-Aware BPE | arXiv 2025 | 2508.04796 |
| Tokens with Meaning (Turkish) | arXiv 2025 | 2508.14292 |
| Qtok Evaluation Framework | arXiv 2024 | 2410.12989 |
| Tokenization Constraints in LLMs | arXiv 2025 | 2505.14178 |
| Stop Taking Tokenizers for Granted | EACL 2026 | 2601.13260 |
