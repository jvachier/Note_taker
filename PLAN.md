# Plan: Two From-Scratch Models (Keras 3 + JAX, trained on Kaggle)

Scope: models only. No ASR, no app. Both models are written from scratch in Keras 3
(custom layers using `keras.ops`, backend-agnostic), trained with the JAX backend on
Kaggle GPUs. Local machine (M2, 16 GB) is for development, tests and CPU evaluation only.

All numbers below are starting values; tune them, but log every change in `configs/`.

---

## 0. Stack and repo

**Stack**
- Python 3.11, Keras 3 (`KERAS_BACKEND=jax`), JAX with CUDA on Kaggle, JAX CPU locally
- `tokenizers` (Hugging Face, Rust) to train our own WordPiece tokenizer
- `datasets` to download WikiText-103 and DialogSum
- `rouge-score`, `bert-score` (evaluation only), `scipy` (Spearman), `numpy`
- Teacher inference only (Kaggle): `vllm` with AWQ checkpoints; fallback `transformers` + `bitsandbytes` 4-bit
- `pytest`, `ruff`, `pre-commit`, `huggingface_hub`

**Layout**
```
configs/            # one YAML per run (tokenizer, model2, mlm, model1_A/B/C)
src/
  data/             # download, sentence/utterance splitting, noise, splits
  tokenizer/        # train + load WordPiece
  layers/           # embeddings, transformer block, heads (custom Keras layers)
  model2_cleanup/   # model, labels, training, eval
  model1_scorer/    # MLM pretraining, teacher labelling, students, eval
  eval/             # shared metrics
notebooks/kaggle/   # one self-contained notebook per model (own code, no repo imports)
tests/
```

**Data handling**
- Pre-tokenize everything once to `int32` NumPy arrays (`.npy`, memory-mapped). No `tf.data`, no TF dependency at training time.
- One Kaggle notebook per model runs its whole pipeline (data, tokenizer, arrays, training, evaluation) and writes to `/kaggle/working`; attaching its previous output resumes it. Trained models and results are pulled into the repo with `notebooks/kaggle/fetch_outputs.py` and used locally (evaluation, results tables, publishing).
- Checkpoint every epoch (sessions are capped at 12 h, ~30 GPU-h per week).

---

## 1. Data

| Dataset | Use | Notes |
|---|---|---|
| WikiText-103 (`wikitext-103-raw-v1`) | Tokenizer, Model 2 training, MLM pretraining | Use the **raw** version (keeps case and punctuation, no `<unk>`) |
| DialogSum | Tokenizer, Model 2 (in-domain mix + eval), Model 1 everything | ~12.5k train / 500 val / 500 test dialogues; test has 3 reference summaries. Non-commercial licence: state it in model cards |

**Splits**: use the official DialogSum splits. For WikiText use the official train/valid/test.
Test: no dialogue ID appears in more than one split.

**Deduplication (approved deviation)**: train items duplicated in validation/test are removed;
validation and test are never modified.
- DialogSum: 32 train dialogues whose utterance text equals a validation dialogue are dropped
  (12,460 → 12,428). 6 validation/test duplicates remain and are only reported.
- WikiText: 1 train article whose title is also a test article ("The Hustler (film)") is dropped.
- DialogSum test rows (`test_<n>_<k>`, 1,500) are merged into 500 dialogues with 3 references.
- 6 empty speaker turns (tag with no text) in DialogSum train are dropped.

**WikiText detokenisation (approved deviation)**: raw files keep the original tokenised text.
Every consumer (tokenizer corpus, Model 2 examples) reads it through
`src.data.wikitext.detokenize`, which undoes ` @-@ ` / ` @,@ ` / ` @.@ ` and the spaces around
punctuation, quotes, brackets and clitics (`don 't` → `don't`). Heading lines are skipped.

---

## 2. Tokenizer (shared by both models)

- **Type**: WordPiece (BERT-style), trained by us with `tokenizers.BertWordPieceTokenizer`.
- **Vocab size**: 16,000.
- **Normalisation**: lowercase, strip accents, NFD. Uncased is deliberate: Model 2 input is lowercase by construction, and Model 1 does not need case.
- **Pre-tokenisation**: whitespace + punctuation split (so `,` `.` `?` are their own tokens for Model 1).
- **Special tokens**: `[PAD]=0, [UNK], [CLS], [SEP], [MASK]`, plus speaker tokens `[S1]`…`[S8]` reserved.
- **Training corpus**: WikiText-103 train (detokenised, headings skipped) + DialogSum train utterances (speaker tags removed). No val/test text.
- **Output**: `tokenizer.json` + `vocab.txt`, versioned with the models and uploaded with them.
- **Report**: average subword tokens per word on WikiText and DialogSum test, `[UNK]` rate.

Tests: encode/decode round trip on lowercase text; special token IDs fixed; word-to-first-subword mapping is correct.

---

## 3. Shared building blocks (`src/layers/`)

Custom Keras layers, `keras.ops` only:
- `TokenAndPositionEmbedding`: token embedding + learned absolute position embedding (+ optional speaker embedding), dropout.
- `TransformerEncoderBlock`: **pre-LayerNorm**, multi-head self-attention (`keras.layers.MultiHeadAttention` with padding mask), residual, LayerNorm, FFN (Dense → GELU → Dense), residual, dropout 0.1.
- Final LayerNorm after the last block.
- Masked loss utilities: per-position `sample_weight` so padding and non-first subwords contribute zero.

Initialisation: truncated normal, std 0.02. Tie MLM output weights to the token embedding.

---

## 4. Model 2: Cleanup (punctuation + casing restoration)

**Task**: token classification. Input is lowercase text with no punctuation and synthetic word errors. For each word, predict:
- **Casing** (3 classes): `lower`, `Capitalised`, `UPPER`. Mixed-case words (e.g. "iPhone") get weight 0.
- **Trailing punctuation** (4 classes): `none`, `comma`, `period`, `question`. Map `!` `;` → `period`, `:` → `comma`, drop everything else.

Labels sit on the **first subword** of each word; other subwords get weight 0.

**Building examples**
1. Split text into sentences, then pack consecutive sentences into windows of ~64 words (no sentence cut across windows).
2. Record per-word labels from the original text.
3. Lowercase and strip punctuation.
4. Apply noise (per word, independently):
   - substitution p = 0.05 (random word from the top-20k frequency list)
   - deletion p = 0.03 (move its punctuation label to the previous word)
   - insertion p = 0.02 (inserted word gets `lower` / `none`)
5. Tokenize, truncate/pad to **128** tokens.

Training mix: WikiText-103 windows + DialogSum train dialogues oversampled ×5 (conversational domain).

**Architecture** (~7M params)
| | |
|---|---|
| Vocab | 16,000 |
| d_model | 256 |
| Layers | 4 |
| Heads | 4 |
| FFN dim | 1,024 |
| Max length | 128 |
| Heads on top | `Dense(3)` casing, `Dense(4)` punctuation |

**Training**
- Loss: sum of two masked sparse categorical cross-entropies. Class weights on punctuation (inverse sqrt frequency) because `none` dominates.
- AdamW, peak lr 5e-4, 2k warmup steps, cosine decay, weight decay 0.01, batch 128, mixed precision `mixed_float16`.
- Early stopping on validation punctuation macro-F1.

**Evaluation**
- Punctuation: per-class precision/recall/F1 and macro-F1 (excluding `none`).
- Casing: accuracy and F1 for `Capitalised` and `UPPER`.
- On two test sets: WikiText test (in-domain) and DialogSum test (conversational).
- Noise ablation: evaluate at 0%, 5%, 10%, 20% total noise.
- Rule baseline: capitalise first word and "i", period at end of window.

---

## 5. Model 1: Distilled utterance scorer (extractive notes)

**Unit**: one utterance (speaker turn) of a DialogSum dialogue. Speaker tags `#Person1#` etc. are removed from the text and kept as speaker IDs.

### 5.1 MLM pretraining of the encoder
- Corpus: WikiText-103 train + DialogSum train dialogues.
- Sequences of 512 tokens; mask 15% (80% `[MASK]`, 10% random, 10% unchanged).
- AdamW, peak lr 5e-4, 4k warmup steps, cosine decay, batch 64, mixed precision.
- Measure throughput first, then set the step budget to fit ~8-10 Kaggle GPU-hours.
- Report MLM loss/accuracy on held-out WikiText and DialogSum val.

### 5.2 Architecture (~17M params)
| | |
|---|---|
| Vocab | 16,000 (shared tokenizer) |
| d_model | 384 |
| Layers | 6 |
| Heads | 6 |
| FFN dim | 1,536 |
| Max length | 512 |
| Extra embedding | speaker embedding (8 speakers) |

**Input format (BertSum-style)**: `[CLS] utt_1 [SEP] [CLS] utt_2 [SEP] …`. The hidden state at each `[CLS]` represents its utterance. Dialogues over 512 tokens: truncate the longest utterances first; log how many are affected.

**Heads (on each `[CLS]` vector)**
- Importance: `Dense(1)` → sigmoid, score in [0, 1].
- Tag: `Dense(4)`: `none`, `key_point`, `decision`, `action_item`.

### 5.3 Teacher labelling (Kaggle GPU) — v2 (approved 2026-10-13)
- Teachers: **Qwen2.5-14B-Instruct-AWQ** (strong, reference teacher) and **Qwen2.5-1.5B-Instruct** (weak-teacher baseline), both Apache 2.0.
- Prompt: numbered utterances **and the dialogue's reference summary** (all 3 on test); the teacher rates how much each utterance contributes to it. JSON only, format shown **without example values**:
  `[{"id": <int>, "score": <int>, "tag": <string>}, …]`
- Score: anchored **0-3** scale (3 = main fact / problem / decision / plan, 2 = useful detail, 1 = context, 0 = greeting / filler / bare yes-no). Tags with stricter rules (`decision` only when explicitly agreed).
- **Self-consistency**: 3 samples per dialogue (T = 0.7, top-p 0.95); each validated (valid JSON, every ID once, integer score in range, tag in set); valid samples merged (mean score, majority tag, ties to the rarer tag). If no sample is valid: one greedy retry with the reason, then drop. Log failure rate and valid-sample rate per teacher.
- Normalise score to [0, 1].
- Label the full train/val/test splits if throughput allows, else a fixed subset (start with 5,000 train dialogues), identical across teachers. The teachers stop at 60% of a session and resume in the next one.
- Because the teacher sees the reference, its test row is a **reference-aware upper bound** (like the oracle), not a deployable system. Students never see references.

**Why v2.** v1 (7B + 1.5B, greedy, no reference, 0-10 scale with an example `"score": 3` in the format line) gave weak labels: the 1.5B teacher answered 3 for 98% of utterances (the example value), so its top-3 was Lead-3 through position tie-breaking; the 7B teacher's top-3 scored below Lead-3 on ROUGE-1 (0.203 vs 0.219), favoured late utterances (Spearman with position +0.35, the oracle favours early ones) and put only 40% of oracle utterances in its top-3. Student A reproduced the 7B teacher almost exactly (retention 0.997), so distillation worked and the labels were the bottleneck.

### 5.4 Oracle labels (no LLM)
Greedy selection per dialogue: repeatedly add the utterance that most increases ROUGE-1 F1 + ROUGE-2 F1 against the reference summary; stop when no gain or 3 utterances. Selected = 1, others = 0. No tags.

**Soft labels (v2)**: per utterance, ROUGE-1 F1 + ROUGE-2 F1 of that utterance alone against the references, divided by the dialogue's best single utterance → a continuous target in [0, 1]. v1's 0/1 labels (~2 positives per dialogue) gave the listwise loss little to rank.

### 5.5 Students (identical architecture, init from the MLM checkpoint)
| Student | Targets (v2) |
|---|---|
| A | 14B teacher scores + tags |
| B | 1.5B teacher scores + tags |
| C | **Soft** oracle labels (importance head only) |
| D (ablation) | Same as A but **no MLM pretraining** (random init) |
| E (added) | Importance = **0.5 × 14B teacher score + 0.5 × soft oracle label**; tags from the 14B teacher. |

v1 targets: A = 7B teacher, C = oracle 0/1, E = oracle 0/1 importance + 7B tags (E was added after A was found to copy the 7B teacher's bias towards late, long utterances).

**Loss**
- Importance: binary cross-entropy with soft targets + λ · listwise KL between softmax(student logits / T) and softmax(teacher scores / T) over the utterances of a dialogue. Start with λ = 1, T = 2.
- Tags (A, B, D, E): masked sparse categorical cross-entropy, weight 0.5, with **inverse-sqrt-frequency class weights** from the train tags (v2: `decision` is ~3% of utterances; v1 decision F1 was 0.04–0.12).
- AdamW, lr 1e-4 (encoder) / 1e-3 (heads), warmup 10%, batch 16 dialogues, max 10 epochs, early stopping and checkpoint selection on **validation loss**, patience 2 (v2; v1 used validation Spearman, patience 3, which kept training for 2–4 epochs after the validation loss started rising). Validation Spearman is still logged.
- 3 seeds per student; report mean ± std.

**Inference rule (same for every system)**: select the top-3 utterances by score, output them in dialogue order, grouped by predicted tag. In the evaluation, score ties are broken by seeded random keys per dialogue (v2; v1 broke them by position, which turned a teacher with many tied scores into Lead-3).

### 5.6 Evaluation (DialogSum test)
- Summary quality vs the 3 human references: ROUGE-1/2/L, BERTScore F1.
- **Retention** = student score / teacher score (target ≥ 0.90), where the teacher row is the teacher's own top-3 selection.
- Agreement with teacher: per-dialogue Spearman (averaged), top-3 Jaccard overlap.
- Tags: precision/recall/F1 for `decision` and `action_item` vs the reference (v2: 14B, v1: 7B) teacher's tags (state clearly this is agreement with the teacher, not ground truth).
- Efficiency: parameter count, file size, CPU latency per dialogue (M2), compared with the teachers.
- Reference rows: oracle selection (upper bound for extraction), lead-3 (first three utterances).

---

## 6. Results tables to produce

**Model 2**: rule baseline vs model, both test sets, all noise levels.

**Model 1 (v2, to fill)**:
| System | Params | ROUGE-1/2/L | BERTScore | Retention | Spearman vs 14B | Decision/Action F1 | CPU latency |
|---|---|---|---|---|---|---|---|
| Lead-3 | – | | | | | | |
| Oracle | – | | | | | | |
| Teacher 14B (sees reference) | 14B | | | 1.00 | 1.00 | | |
| Teacher 1.5B (sees reference) | 1.5B | | | | | | |
| Student A | ~17M | | | | | | |
| Student B | ~17M | | | | | | |
| Student C | ~17M | | | | | | |
| Student D | ~17M | | | | | | |
| Student E | ~17M | | | | | | |

**Model 1 v1 results** (Kaggle run of 2026-10-13: 7B / 1.5B teachers, v1 prompt; 498 test dialogues; students mean ± std over 3 seeds; `configs/model1_eval_v1.yaml`):
| System | Params | ROUGE-1 / 2 / L | BERTScore | Retention | Spearman vs 7B | Decision / Action F1 | CPU latency |
|---|---|---|---|---|---|---|---|
| Lead-3 | – | 0.219 / 0.069 / 0.170 | 0.862 | 1.107 | −0.359 | – | – |
| Oracle | – | 0.338 / 0.150 / 0.271 | 0.882 | 1.830 | −0.017 | – | – |
| Teacher 7B | 7B | 0.203 / 0.063 / 0.149 | 0.860 | 1.000 | 1.000 | 1.000 / 1.000 | 298 ms (GPU, batched) |
| Teacher 1.5B | 1.5B | 0.217 / 0.069 / 0.166 | 0.862 | 1.089 | 0.639 | 0.076 / 0.179 | 114 ms (GPU, batched) |
| Student A | 17.0M | 0.202 / 0.063 / 0.148 | 0.859 | 0.997 ± 0.007 | 0.670 ± 0.007 | 0.124 / 0.457 | 42.9 ms |
| Student B | 17.0M | 0.201 / 0.061 / 0.150 | 0.859 | 0.993 ± 0.021 | 0.415 ± 0.061 | 0.007 / 0.062 | 43.5 ms |
| Student C | 17.0M | 0.246 / 0.088 / 0.190 | 0.867 | 1.264 ± 0.016 | 0.007 ± 0.056 | – | 43.3 ms |
| Student D | 17.0M | 0.200 / 0.061 / 0.148 | 0.859 | 0.987 ± 0.014 | 0.569 ± 0.004 | 0.048 / 0.241 | 44.1 ms |
| Student E | 17.0M | 0.245 / 0.089 / 0.176 | 0.866 | 1.230 ± 0.008 | 0.045 ± 0.031 | 0.039 / 0.440 | 43.8 ms |

---

## 7. Tests (`tests/`)
- Tokenizer: round trip, fixed special IDs, first-subword alignment.
- Noise: labels stay aligned after substitution/deletion/insertion; deletion moves punctuation correctly.
- Layers: output shapes; padding mask changes nothing at padded positions.
- Losses: zero weight on padding and non-first subwords.
- Metrics: ROUGE wrapper and Spearman on hand-checked examples.
- Data: no ID overlap between splits; teacher JSON validator rejects malformed outputs.
- Smoke: one training step per model on CPU with a tiny config.

---

## 8. Order of work
1. Repo, configs, tests skeleton.
2. Data download + splits.
3. Train tokenizer.
4. Model 2: example builder → training on Kaggle → evaluation.
5. Model 1: MLM pretraining on Kaggle.
6. Teacher labelling (7B, 1.5B) on Kaggle + oracle labels.
7. Students A–D, 3 seeds each → evaluation → results tables. (Done: v1, plus Student E.)
7b. v2: teacher labelling (14B, 1.5B; reference, 0-3 scale, 3 samples) + soft oracle → students A–E, 3 seeds each → evaluation → results tables.
8. Publish both models to Hugging Face: `.keras` file, `tokenizer.json`, config, model card (data, licence, method, metrics, limitations, usage snippet).
