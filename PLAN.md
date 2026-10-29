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

## 5. Model 1: Distilled utterance scorer (extractive notes) — v1/v2, superseded by §10 for new work

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
9. Model 1 v3 (§10): follow §10.9. It replaces steps 6–7b for new work; Model 1 is published from v3.

---

## 9. Proposal: Model 1 v3 — abstractive notes (logged 2026-10-13; SUPERSEDED by §10, not implemented)

Not approved and not in the order of work. Nothing here is implemented.

**Why.** Extractive selection is capped on DialogSum because the references are third-person
rewrites. In the DialogSum paper (Tables 4–5), the extractive oracle reaches 37.9 ROUGE-1, a
**Transformer trained from scratch** writing summaries reaches 35.9, and pretrained BART-large
reaches 47.3. Our v1 students are near Lead on that scale. Raw numbers differ from ours because
the ROUGE implementation differs (their Lead-2 = 27.5, our Lead-3 = 21.9); compare ratios.

**Open decision:** (a) from scratch + a fine-tuned KerasHub BART-base as a reference row only
(recommended); (b) from scratch only; (c) switch to a pretrained model (requires rewriting
CLAUDE.md and this plan). Compute must stay small.

### 9.1 Architecture (~27M params)
- **Encoder**: the existing 6-layer, d384 encoder, initialised from the existing MLM checkpoint
  (no new MLM run). Same BertSum-style input (`[CLS] utt [SEP] …`, speaker embeddings) so the
  utterance heads keep working.
- **Decoder** (new, from scratch, `keras.ops`): 4 layers, d384, 6 heads, FFN 1,536, pre-LN,
  causal self-attention + cross-attention to the encoder. Token embedding and output projection
  tied to the encoder's embedding (shared 16k WordPiece vocab) → ~9.5M new parameters.
- **Heads**: the decoder's language-model head (summary) + the existing importance and tag
  heads on each `[CLS]` (multi-task, shared encoder).
- **Targets**: `#PersonN#` in summaries mapped to the reserved `[S1]`…`[S8]` tokens and back.
  Summaries are capped at 96 tokens (DialogSum summaries are ~30–40 tokens).
- **Casing**: the tokenizer is uncased, so generated summaries are lowercase (ROUGE is
  case-insensitive). For display, pass them through **Model 2** to restore casing.

### 9.2 Denoising pretraining (BART-style, fixed time budget)
- Input: 256-token chunks of DialogSum train dialogues (×5) + WikiText-103 train with **text
  infilling** (spans of Poisson(λ = 3) length replaced by one `[MASK]`, 30% of tokens).
  Target: the original chunk.
- Encoder from the MLM checkpoint, decoder random. Throughput probe, then a **2–3 GPU-hour**
  budget (same mechanism as the MLM stage).

### 9.3 Teacher summaries (same Kaggle pass as the v2 labels)
- Qwen2.5-14B writes **3 summaries per train/validation dialogue** (T = 0.7) in DialogSum style:
  third person, `#PersonN#` names, at most 3 sentences, with 2 human references from other train
  dialogues as style examples. The teacher does **not** see the dialogue's own reference.
- Filters: length 10–80 words, speaker tags valid for the dialogue, and drop the 10% with the
  lowest ROUGE-L against the human reference (computed in the labelling stage, not in
  training code).
- Cost: ~60 output tokens × 3 per dialogue ≈ 2.2M tokens, about +30–40% on top of the v2
  labelling, sharing the loaded model and prefix cache.
- No extra dialogue corpora by default: DailyDialog (a DialogSum source) could overlap
  validation/test and would need deduplication first.

### 9.4 Fine-tuning
- Per epoch, each train dialogue gives one target: the human summary with probability 0.5,
  else one of its teacher summaries (sequence-level distillation, Kim & Rush 2016; several
  teacher summaries per dialogue keep the student from collapsing to one style).
- Loss: token cross-entropy with label smoothing 0.1 + 0.5 × (importance + tag losses of §5.5,
  on the v2 targets).
- AdamW, lr 1e-4 encoder / 3e-4 decoder, warmup 10%, batch 32, up to 20 epochs, early stopping
  on validation loss (ROUGE stays evaluation-only).
- Decoding: beam search (4 beams, length penalty, no repeated 3-grams) written with `keras.ops`;
  no KV cache at first (summaries are short; check CPU latency).
- Budget: ~1 GPU-hour per seed, 3 seeds.

### 9.5 Output and evaluation
- Notes = generated summary (casing by Model 2) + **Decisions** and **Action items** bullets from
  the utterance heads.
- Summary rows: ROUGE-1/2/L and BERTScore vs the 3 references; Lead-3, extractive oracle, best
  extractive student (v2), Qwen-14B zero-shot summary (without the reference, so a fair upper
  baseline), and under option (a) fine-tuned BART-base as a reference row.
- Ablations, one seed each: without teacher summaries (human only); without denoising
  pretraining.
- Expected (guess): ~38–42 ROUGE-1 on the paper's scale — above the paper's from-scratch 35.9
  thanks to our pretraining and the extra targets, below BART-large's 47.3.

### 9.6 Optional: MiniLM-style encoder distillation (separate approval)
MiniLM-L6-H384 has the same shape as our encoder; it is strong because it is distilled from a
large encoder's self-attention relations (MiniLMv2: query–query, key–key, value–value
relations, KL loss, teacher's upper-middle layer) rather than trained with MLM from scratch.
Here: a RoBERTa teacher via KerasHub (JAX) during encoder pretraining only, relations
averaged from subwords to words because the tokenizers differ. This puts a pretrained teacher
in the training environment, which CLAUDE.md currently reserves for our own models. It helps the
encoder, not the extractive ceiling, so it would come after 9.1–9.5.

### 9.7 Compute (2× T4 sessions, 12 h cap)
| Session | Work | Time |
|---|---|---|
| 1 | v2 labels (14B, 3 samples) + teacher summaries, 1.5B labels | ~7–9 h (may need to resume) |
| 2 | denoising pretraining (2–3 h) + fine-tuning 3 seeds (~3 h) + extractive students if still wanted | ~7–9 h |
| optional | BART-base reference row (option a) | ~1 h |
Savings: reuse the MLM encoder; check teacher quality on the validation split (~15 min) before
the full run; drop extractive students B and D if compute is tight.

Sources: DialogSum (arXiv 2105.06762), MiniLMv2 (arXiv 2012.15828), sequence-level KD (Kim &
Rush 2016), factual dialogue summarization from LLMs (arXiv 2406.14709), DisCal (arXiv 2310.13760).

---

## 10. Model 1 v3: meeting-notes student distilled from Qwen's reasoning (APPROVED 2026-10-17)

This section **replaces §5.3–§5.6** (DialogSum teacher labelling and students) and §9
(abstractive v3) for all new work. The v1 results in §6 and the code in `src/model1_scorer/`
stay as history. New code goes in `src/model1_notes/`.

**Why.**
- The v1 student copied its Qwen teacher almost exactly (retention 0.997), but the teacher's
  labels scored below Lead-3, and DialogSum chats are not meetings.
- The v2 plan still needed about 10–14 GPU-h of labelling and student runs.

v3 changes the data, the architecture and the teacher signal:
- **Data:** the gold-labelled meeting corpus `meeting-notes-extractive-v1` (separate repo).
- **Architecture:** a hierarchical student that reads whole meetings.
- **Teachers:** a small reasoning LLM (Qwen3-4B) for *why* a sentence matters, plus
  RoBERTa-Large for a second, bidirectional view.

Decisions taken on 2026-10-17:
- one main experiment (E4) plus one control (E1)
- a 21M student that reuses the existing MLM encoder (no new pretraining)
- Qwen3-4B in thinking mode as the LLM teacher, with Qwen3-1.7B as fallback
- RoBERTa-Large hidden-state MSE is dropped: aligning hidden states across tokenizers is
  unreliable

### 10.1 Task and output
Input: a meeting as a list of `(speaker, sentence)`. ASR-style text goes through Model 2
first. Output: **structured extractive notes**:
```
Decisions      – zone = decision,    importance = 1, in meeting order
Action items   – zone = action_item, importance = 1, with owner / deadline spans
Key points     – zone = discussion,  importance = 1
```
- Sentences are selected by an importance threshold tuned on validation, not a fixed top-k.
- No text is generated: no decoder.

### 10.2 Data
- Corpus `meeting-notes-extractive-v1`, pinned to `v1.0`. The Kaggle notebooks attach it as a
  Kaggle dataset; locally it is downloaded from HF. Until v1.0 is out, development uses the
  corpus pilot / tiny files.
- Splits:
  - **Train:** synthetic train + hard negatives (oversampled ×2) + AMI train
  - **Validation:** synthetic validation + AMI validation, used for early stopping, the
    threshold and the teacher quality gate
  - **Test:** synthetic test and **AMI test (headline)**, reported separately
- Input text: `sentence_text` in training. The pipeline row (§10.7) uses
  `sentence_text_asr` → Model 2 → Model 1.
- Tokenizer: the existing 16k WordPiece (§2), unchanged.
- Pre-tokenised `int32` `.npy`: `[n_sentences, 64]` per meeting, plus label and teacher arrays.
  Sentences longer than 64 subwords are truncated; the count is logged.

### 10.3 Notebook 1: teachers (own environment; only cached arrays leave it)

**Qwen3-4B, thinking mode** (vLLM, fp16 on T4, own uv venv as before). Fallback: Qwen3-1.7B.
- **Step 0: compatibility check.** Load the model in vLLM on a T4, run 5 meetings, and log
  throughput. If this fails, use Qwen2.5-3B-Instruct without thinking. That is a change from
  this plan and needs approval.
- **Input:** one meeting, numbered sentences with speakers. Meetings with more than 256
  sentences are split into windows of 256 with a 64-sentence overlap. The teacher **never
  sees the gold labels**.
- **Output:** a thinking block (capped at `max_thinking_tokens`, config), then JSON only. The
  format is shown **without example values**:
  `[{"id": <int>, "score": <int 0-3>, "reason": <string>, "depends_on": <int|null>}, …]`
  - `score`: anchored 0–3 scale, as in v2.
  - `reason`, one of: `agreed_outcome`, `commitment_with_owner`, `risk_or_blocker`,
    `key_fact`, `status_update`, `rejected_proposal`, `tentative`, `question`, `filler`.
  - `depends_on`: the earlier sentence that makes this one meaningful (e.g. the proposal
    behind an agreement), or null.
- **Soft scores from logprobs:**
  - generate with `logprobs=5`
  - for each `score` digit token, read the probabilities of `0`–`3`, renormalise, and take
    the expected value / 3 as the soft score
  - if the digits are missing from the top-5, fall back to the one-hot of the emitted score
- **Validation of each output:**
  - valid JSON, every ID exactly once
  - `score` in range, `reason` in the set
  - `depends_on` is null or an earlier ID

  On failure: one greedy retry with the reason, then drop the meeting. The failure rate is
  logged.
- **Which meetings are labelled:** all validation meetings, AMI train/validation, and up to
  `max_train_meetings` synthetic train meetings (start with 1,000). Meetings without teacher
  labels get zero weight on the teacher losses.
- **Raw outputs kept,** including the thinking text, for analysis and the blog. They are not
  used for training.
- **Quality gate (before Notebook 2):** compare Qwen's labels with gold on validation:
  - importance F1 (soft score > 0.5)
  - reason ↔ zone agreement
  - F1 on the hard negatives

  If importance F1 does not beat the keyword baseline (§10.6), stop and report. Do not
  distil.

**RoBERTa-Large** (KerasHub preset `roberta_large_en`, JAX backend, mixed precision):
- Fine-tuned once on corpus train gold importance labels.
- Input: windows of consecutive sentences up to 512 tokens, separated by `</s>`, with a
  32-token overlap.
- Sentence score: mean of the sentence's token states → `Dense(1)`.
- 2 epochs, AdamW lr 1e-5, warmup 10%, early stopping on validation loss.
- Cached output: per-sentence logits on train/validation/test. Test logits give the ceiling
  row.

**DistilBERT** (KerasHub `distil_bert_base_en_uncased`): the same recipe as RoBERTa, used as
the **baseline row only** (not a teacher).

### 10.4 Student architecture (~21M params)
```
sentence tokens (≤ 64) ─► Sentence encoder: 6L, d384, 6h, FFN 1536   (existing MLM encoder, reused)
                        ─► [CLS] vector per sentence
                        ─► + speaker emb (S1…S8, by first appearance) + sentence-position emb (≤ 256)
                           + speaker-change flag
                        ─► Meeting encoder: 2L, d384, 6h, FFN 1536, pre-LN (src/layers block)
                        ─► per-sentence heads:
                             importance  Dense(1)
                             zone        Dense(5)
                             reason      Dense(9)        (Qwen's reasoning, §10.3)
                             depends_on  pointer: bilinear score to each earlier sentence + "none"
                        ─► per-token head (sentence encoder output): spans Dense(5), BIO {owner, deadline}
```
- **Sentence encoder:** loaded from `checkpoints/mlm/encoder.keras`, using its first 64 learned
  positions.
- **Meeting encoder and heads:** new, random init, about 4M params.
- **Long meetings:** windows of 256 sentences with a 64-sentence overlap; overlapping scores
  are averaged.
- **Static shapes:** batches bucketed by sentence count (32 / 64 / 128 / 256), with a fixed
  64 tokens per sentence.
- Every custom layer and model is registered with `get_config`, as usual.

### 10.5 Losses and experiments
```
L_E4 = 1.0 · BCE(importance, gold; weighted by confidence)
     + 0.5 · KL(importance, Qwen soft score;   T = 2)
     + 0.5 · KL(importance, RoBERTa logits;    T = 2)
     + 0.3 · CE(reason, Qwen)
     + 0.3 · CE(depends_on, Qwen)
     + 0.5 · CE(zone, gold; inverse-sqrt class weights; masked on AMI for context/closing)
     + 0.3 · CE(spans, gold; synthetic only)
L_E1 = L_E4 without the four teacher terms
```
All weights and temperatures live in `configs/model1_v3_E4.yaml` / `model1_v3_E1.yaml`.

| Run | Seeds | Purpose |
|---|---|---|
| **E4**: gold + Qwen (scores, reasons, links) + RoBERTa | 3 | Main result |
| **E1**: gold only, same architecture | 1 | Control: shows whether distillation helped |

- Optimiser: AdamW. lr 1e-4 for the sentence encoder, 5e-4 for the meeting encoder and heads.
  Warmup 10%, cosine schedule.
- Batch of 16 meetings; up to 15 epochs.
- Early stopping on validation loss, patience 2.

### 10.6 Baselines (no extra student runs)
| Row | Params | How |
|---|---|---|
| Lead-k | – | First k sentences; k tuned on validation |
| Keyword rule | – | Trigger words ("decided", "will … by"), tuned on validation |
| Qwen3-4B (teacher) | 4B | Its own labels (soft score > 0.5) |
| RoBERTa-Large (fine-tuned) | 355M | Ceiling |
| DistilBERT (fine-tuned) | 66M | Primary opponent |
| Student E1 / **E4** | ~21M | Ours |

### 10.7 Evaluation (`src/eval/`; ROUGE and BERTScore are evaluation only)
- **Sentence level** (synthetic test, AMI test): importance F1 and PR-AUC; zone macro-F1;
  decision / action_item F1, all against **gold**. Hard-negative precision on synthetic.
- **Spans** (synthetic test): owner and deadline exact-match F1.
- **Summary level** (AMI test): ROUGE-1/2/L and BERTScore of the notes against AMI's human
  summary. Decisions are scored against the decisions section and actions against the actions
  section.
- **Distillation:** agreement between the student and Qwen on reasons and `depends_on`
  (validation), shown with the E4 − E1 difference.
- **Pipeline:** ASR-style text → Model 2 → student, versus clean text.
- **Efficiency:** parameters, file size, CPU latency per meeting (M2) at 100 and 500
  sentences. Each teacher's GPU latency is shown for reference.

### 10.8 Compute (Kaggle 2× T4, wall-clock; updated 2026-10-21 from the first Qwen run)
| Notebook | Work | Time |
|---|---|---|
| 1 | Setup: vLLM venv + Qwen3-0.6B download | ~15–20 min |
| 1 | Qwen3-0.6B labels: all validation + test, 400 synthetic + 30 AMI train meetings, windows of 64 sentences, one shard per T4 | **~6–7 h** (measured: a synthetic meeting is ~2 windows and a batch of 64 takes ~25 min; an AMI meeting is ~16 slow windows and a batch of ~20 takes ~70–75 min, so AMI dominates) |
| 1 | Merge + quality gate | < 1 min |
| 1 | RoBERTa-Large fine-tune (2 epochs) + logits; DistilBERT fine-tune + logits | ~45–75 min (estimate) |
| **1** | **Notebook 1 total** | **~7.5–8 h**: Qwen stops at 60% of the 11 h budget, so the last Qwen meetings may need a short second session (resumes) |
| 2 | Arrays + first-epoch compilation | ~10 min |
| 2 | E4 × 3 seeds + E1 × 1: ~3–5 min per epoch, 6–10 epochs with early stopping (max 15) | ~1.5–3 h (estimate; worst case ~4.5 h) |
| **2** | **Notebook 2 total** | **~1.5–3 h** |
| | **Both notebooks** | **~9–11 h** (~1/3 of the weekly quota) |

Measured so far: Qwen3-4B ran at ~40 output tok/s per T4 (it would have needed 8–17 h), so the
teacher was switched to Qwen3-0.6B (~200 tok/s). The AMI windows cost most of the Qwen time:
`max_ami_train_meetings` 30 → 10 would save ~1.5–2 h in a future relabelling. Notebook 2's epoch time is checked on the first
epoch (Keras prints `s/step`); multiply by ~8 epochs × 4 runs for the total.

Notebook 1 frees all GPU memory before it ends. Notebook 2 never loads a teacher.

### 10.9 Order of work (stop and report after each step)
1. CLAUDE.md and configs for v3. Corpus loader + pre-tokenisation in
   `src/model1_notes/data.py`, against the corpus pilot / tiny files.
2. Student: meeting encoder, heads (including the pointer head), losses, MLM weight loading;
   tests; `configs/model1_v3_tiny.yaml` smoke run (< 2 min on CPU).
3. Teacher reference code in `src/model1_notes/teacher/`: prompt, JSON validator, logprob
   soft scores. Tested locally with a fake teacher; Qwen is never downloaded locally.
4. Notebook 1 (Kaggle): Qwen check → labels → quality gate → RoBERTa / DistilBERT. Stop and
   report the gate result.
5. Notebook 2 (Kaggle): E4 × 3, E1 × 1, evaluation.
6. Results tables, notes renderer (with Model 2 for ASR input), then `hf-publish`.

Steps 4–5 wait for the corpus v1.0 release.

**Progress log**
- 2026-10-17, step 1 done:
  - `configs/model1_v3{,_E4,_E1,_tiny}.yaml`
  - `src/model1_notes/data.py`: corpus Parquet → per-sentence `int32` arrays, plus meeting
    windows of 256 sentences overlapping by 64
  - tests in `tests/test_model1_notes_data.py`
  - `pyarrow` added to the dependencies
- 2026-10-17, step 2 done:
  - `src/model1_notes/{model,losses,train,tiny_corpus}.py`
  - 20.9M params (17.0M reused MLM encoder + 4.0M new); the real `checkpoints/mlm/encoder.keras`
    loads into it
  - smoke run: `python -m src.model1_notes.tiny_corpus`, then `.data`, then
    `.train --config configs/model1_v3_tiny.yaml --seed 0`, about 20 s on CPU
  - tests in `tests/test_model1_notes_model.py`
  - Part of step 5 done early: threshold tuning plus importance / zone F1 per source, in
    `train.py`.
  - The teacher array format (aligned to the processed sentence rows) is fixed in the
    `train.py` docstring. The Notebook 1 export must write it.
- 2026-10-29, **v1.1 results, balanced zone calibration and release choice (approved).**
  - E1 and E4 × 3 seeds on corpus v1.1. AMI test importance F1: **E1 0.578 ± 0.006**,
    E4 0.567 ± 0.006 (DistilBERT 0.580, RoBERTa-Large 0.634). Distillation gave no measurable
    gain on v1.1; the v1.0 single-seed lead of E4 (0.589) was within seed noise. Likely causes:
    Qwen is a weak teacher (AMI 0.318), gold labels already cover every sentence, and the v1.0
    teacher outputs miss the v1.1 addendum meetings.
  - **Released: E1 seed 1** (best AMI validation importance, 0.584). E4 seed 2 (its best on
    validation) was worse on the app examples: its `depends_on` pointer attached replies to the
    wrong sentence, and it filed facts as decisions / action items.
  - **Zone calibration `balanced`** (`inference.zone_calibration`; E1 and E4): decision and
    action thresholds maximise the mean of AMI and synthetic validation F1, so the small AMI
    group weighs as much as the synthetic one. `ami` set the action threshold to ~0.005, so
    clean conversations filed facts as action items; `all` gave AMI decision / action F1 0.
    E1 test, `ami` → `balanced`: synthetic zone F1 0.547 → 0.620, AMI decision F1
    0.083 → 0.081, **AMI action F1 0.091 → 0.016** (the cost). The `ami` metrics are kept as
    `metrics.ami_calibration.json` and `results/tables/model1_v3.ami_calibration.md`.
  - **Post-processing:**
    - `use_pointer: false` for E1 (its `depends_on` head is never trained).
    - Without a pointer, a reply attaches to the nearest kept sentence among the
      `lookback: 3` previous ones, skipping unkept fragments.
    - The speaker-as-owner rule runs before replies attach, so a reply never takes the owner of
      the speaker's own "I …" commitment.
  - **App:**
    - Sentences split only before a capital, digit or quote, so a cleanup slip ("rent.
      increase") stays one sentence.
    - "Restore punctuation & casing first" is ticked by default.
  - **Known limits (model card):**
    - Extractive: a decision spread over several turns ("Monday at ten") is not produced.
    - Deadlines in replies are sometimes missed.
    - AMI decision / action F1 stays below 0.1.
    - Generated (abstractive) notes would be a v4.
- 2026-10-25, **corpus v1.1 and retraining plan (option 2, approved).**
  - **The problem:** the student dropped every first-person task ("I will email him tonight":
    0.01). The corpus marked 93% of first-person commitments unimportant, because a chair always
    restated them.
  - **Corpus v1.1** (corpus repo PLAN §9c):
    - 18 seed relabels (follow-up tasks nobody restates), applied to the v1.0 meetings in place
      (492 rows; text byte-identical, so the Qwen labels stay valid)
    - 300 addendum meetings: first-person tasks, terse decisions, 100 informal conversations
    - quality report PASS
  - **Note_taker:**
    - configs point at `data/raw/meeting_notes_v1_1`
    - Notebook 1 restores merged Qwen tables (no Qwen rerun) and retrains RoBERTa + DistilBERT
    - Notebook 2 is synced with the zone calibration and runs E4 / E1 × 3 seeds
  - **Kaggle uploads** in `data/kaggle_upload/`: corpus v1.1, salvaged Qwen tables, MLM encoder +
    tokenizer.
  - **Regression check after retraining:** the app's "Raw transcript (no punctuation)" example
    (landlord) must yield its first-person tasks.
- 2026-10-25, **Notes post-processing (option 1, no retraining)** in
  `src/model1_notes/postprocess.py`, configured under `inference.postprocess`:
  - Short replies ("Yes, I will…", "Agreed.") are attached under the kept request or decision
    they answer, via the `depends_on` head or the previous sentence. A confirmation joins a
    reply chain; a bare "I'll…" does not.
  - A first-person reply or action item supplies a missing owner (the speaker) and deadline.
  - The app has a keep-threshold slider.
  - Rejected after measurement: keeping unkept first-person commitments. It lowered AMI action
    F1 0.111 → 0.08–0.09; only 11% of such sentences are gold-important.
  - Zone calibration for all three runs, AMI test decision / action F1: E1 0.093 / 0.073,
    E-R 0.083 / 0.023, **E4 0.119 / 0.111**. E4 stays the best run.
- 2026-10-25, **Notebook 2 results and zone calibration.**
  - Each run took ~14 min on one T4. AMI importance F1: E1 0.573, E-R 0.571, **E4 0.589**.
    DistilBERT scores 0.580 and RoBERTa 0.634.
  - Problem: argmax zones never produced decision / action items on AMI, so their F1 was 0.
  - Fix: new `src/model1_notes/calibrate.py` gives each of the two zones its own threshold,
    tuned on AMI validation (`inference.zone_calibration: ami`); training does this
    automatically from now on.
  - E4 on AMI test: decision F1 0 → 0.119, action F1 0 → 0.111. Cost on synthetic: action
    0.98 → 0.88, decision 0.64 → 0.60.
  - The app now serves Model 1 v3 (`release/meeting-notes-en-21m`, kind `notes`): sections
    Decisions / Action items / Key points, and a chart of every sentence's importance.
  - Known weaknesses in the demo: some first-person commitments ("I'll …") are missed, and "I" is
    never extracted as an owner.
- 2026-10-25, **Notebook 2 run list changed (approved).**
  - Runs: E1 (gold only), **E-R** (gold + RoBERTa KL, new: `configs/model1_v3_ER.yaml`) and E4
    (gold + RoBERTa + Qwen), **one seed each**.
  - This replaces E4 × 3 + E1. Qwen3-0.6B only matches Lead-k on AMI, so E-R separates the strong
    teacher's effect from Qwen's.
  - Seeds 1–2 are added later for the best run.
  - Upload for Notebook 2: `data/kaggle_upload/model1-v3-notebook2-inputs.zip` (teacher tables,
    MLM encoder, tokenizer).
- 2026-10-24, **Notebook 1 done.**
  - **Qwen3-0.6B** (~7 h on Kaggle) answered in object form, inside an extra array, with
    `<|im_end|>` left in and often cut at the token budget. The strict reader therefore kept 0.2%
    of sentences.
  - **Fix:** `teacher/salvage.py` re-read the saved raw answers locally with the new lenient reader
    (`answer.parse_entries`) and merged them per sentence (`label.merge_meeting`). Coverage is now
    83% of validation and test and 82% of train. Soft scores are score / 3, because the raw
    answers keep no log-probabilities.
  - **Gate** (validation, 25k sentences): passed, narrowly. Qwen F1 0.365, Lead-k 0.313, keyword
    rule 0.089.
  - **RoBERTa-Large** (19 min) and **DistilBERT** (10 min) fine-tuned on **one T4**: NCCL across
    the two T4s failed (`Cuda failure 500`), hence the new `data_parallel: false`.
  - **Kaggle Models** must be attached as inputs: a batch session cannot attach them, and the
    notebook now checks for them first.
  - **Test results (gold labels), AMI importance F1:** Qwen 0.318 (= Lead-k 0.319), RoBERTa 0.634,
    DistilBERT 0.580. Synthetic: 0.414, 0.939, 0.905.
  - Notebook 1's Qwen code is synced with the lenient reader, but it doesn't need to run again.
- 2026-10-21, **teacher switched to Qwen3-0.6B** (the smallest Qwen3, as the user asked).
  - Why: on Kaggle, Qwen3-4B on T4 ran at about 40 output tokens/s (40 windows per ~17 min),
    which works out to 8–17 h.
  - New budgets: 64-sentence windows (overlap 16), 384 thinking tokens, 1,280 answer tokens,
    context 6,144.
  - Training meetings labelled: 400 synthetic and 30 AMI (new key `max_ami_train_meetings`).
  - Fallback if the gate fails: Qwen3-1.7B.
  - Read "Qwen3-4B" in this section as Qwen3-0.6B.
- 2026-10-21, steps 3, 5 and the notebooks are done (local). Kaggle runs and publishing are left:
  - **Step 3:**
    - Qwen teacher code in `src/model1_notes/teacher/`: prompt, answer validation, soft scores
      from logprobs, windows, fake backend, vLLM backend, merge
    - quality gate in `teacher/gate.py`
    - baselines in `baselines.py`
    - RoBERTa / DistilBERT in `encoder_teachers.py`
  - **Step 5:** `metrics.py`, `evaluate.py` (→ `results/tables/model1_v3.md`) and `notes.py`
    (Markdown notes, optional Model 2 cleanup).
  - **Steps 4–5 notebooks:** `notebooks/kaggle/model1_teachers.ipynb` (Notebook 1) and
    `notebooks/kaggle/model1_notes.ipynb` (Notebook 2). Both are self-contained, and their smoke
    runs pass in `tests/test_notebooks.py`. `fetch_outputs.py` pulls their outputs.
  - **Real-corpus check (CPU, 3 steps, MLM init):** the pipeline runs end to end. AMI test
    importance F1 is 0.46, against 0.32 for Lead-k and 0.03 for the keyword rule.
  - **Deviations from this section** (made while finishing):
    - Batches hold a sentence budget (`sentences_per_batch: 2048`) instead of 16 windows, which
      would be about 260k tokens per step on a T4.
    - Teacher outputs are Parquet tables keyed by `(note_id, sentence_id)`.
    - Qwen answers use compact entries `[id, score, "reason", depends_on]` and teacher windows of
      128 sentences, to save tokens.
    - The gate must beat **both** the keyword rule and Lead-k: the keyword rule alone scores 0.13
      F1 on synthetic validation.
    - Teacher tokenizers come from the `tokenizers` library, because KerasHub's need TensorFlow.
    - The v3 loss is named `NotesImportanceLoss`, to avoid a registry clash with v1.
    - The `TimeGuard` last-epoch fix is applied in `src/training.py`, with a test.
- **Next (on Kaggle):**
  - Notebook 1, with the corpus attached. Read `qwen/gate.json`; if it fails, set the
    `Qwen3-1.7B` fallback or stop.
  - Notebook 2, with the corpus, the v1 `model1_scorer` output and the Notebook 1 output
    attached.
  - `fetch_outputs.py` for both notebooks, then `python -m src.model1_notes.evaluate --latency`,
    then `hf-publish`.

Sources:
- HiBERT (arXiv 1905.06566)
- Distilling Step-by-Step (arXiv 2305.02301)
- MixSumm/PPSL (arXiv 2407.07341)
- Meeting action items (arXiv 2303.16763)
- Cramming (arXiv 2212.14034)
- Cross-tokenizer distillation (arXiv 2602.21669)
