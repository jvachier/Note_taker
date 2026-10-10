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
notebooks/kaggle/   # thin notebooks: pip install repo from GitHub, run one stage
tests/
```

**Data handling**
- Pre-tokenize everything once to `int32` NumPy arrays (`.npy`, memory-mapped). No `tf.data`, no TF dependency at training time.
- Each Kaggle stage writes its outputs (arrays, checkpoints, labels) to a Kaggle Dataset that the next stage reads.
- Checkpoint every epoch (sessions are capped at 12 h, ~30 GPU-h per week).

---

## 1. Data

| Dataset | Use | Notes |
|---|---|---|
| WikiText-103 (`wikitext-103-raw-v1`) | Tokenizer, Model 2 training, MLM pretraining | Use the **raw** version (keeps case and punctuation, no `<unk>`) |
| DialogSum | Tokenizer, Model 2 (in-domain mix + eval), Model 1 everything | ~12.5k train / 500 val / 500 test dialogues; test has 3 reference summaries. Non-commercial licence: state it in model cards |

**Splits**: use the official DialogSum splits. For WikiText use the official train/valid/test.
Test: no dialogue ID appears in more than one split.

---

## 2. Tokenizer (shared by both models)

- **Type**: WordPiece (BERT-style), trained by us with `tokenizers.BertWordPieceTokenizer`.
- **Vocab size**: 16,000.
- **Normalisation**: lowercase, strip accents, NFD. Uncased is deliberate: Model 2 input is lowercase by construction, and Model 1 does not need case.
- **Pre-tokenisation**: whitespace + punctuation split (so `,` `.` `?` are their own tokens for Model 1).
- **Special tokens**: `[PAD]=0, [UNK], [CLS], [SEP], [MASK]`, plus speaker tokens `[S1]`…`[S8]` reserved.
- **Training corpus**: WikiText-103 train + DialogSum train dialogues (no val/test text).
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

### 5.3 Teacher labelling (Kaggle GPU)
- Teachers: **Qwen2.5-7B-Instruct-AWQ** and **Qwen2.5-1.5B-Instruct** (both Apache 2.0). Greedy decoding (temperature 0).
- Prompt: numbered utterances; ask for JSON only:
  `[{"id": 1, "score": 0-10, "tag": "none|key_point|decision|action_item"}, …]`
- Validate: valid JSON, every ID present once, score in range, tag in set. One retry, then drop. Log failure rate per teacher.
- Normalise score to [0, 1].
- Label the full train/val/test splits if throughput allows, else a fixed subset (start with 5,000 train dialogues), identical across teachers.

### 5.4 Oracle labels (no LLM)
Greedy selection per dialogue: repeatedly add the utterance that most increases ROUGE-1 F1 + ROUGE-2 F1 against the reference summary; stop when no gain or 3 utterances. Selected = 1, others = 0. No tags.

### 5.5 Students (identical architecture, init from the MLM checkpoint)
| Student | Targets |
|---|---|
| A | 7B teacher scores + tags |
| B | 1.5B teacher scores + tags |
| C | Oracle 0/1 labels (importance head only) |
| D (ablation) | Same as A but **no MLM pretraining** (random init) |

**Loss**
- Importance: binary cross-entropy with soft targets + λ · listwise KL between softmax(student logits / T) and softmax(teacher scores / T) over the utterances of a dialogue. Start with λ = 1, T = 2.
- Tags (A, B only): masked sparse categorical cross-entropy, weight 0.5.
- AdamW, lr 1e-4 (encoder) / 1e-3 (heads), warmup 10%, batch 16 dialogues, max 10 epochs, early stopping on validation Spearman.
- 3 seeds per student; report mean ± std.

**Inference rule (same for every system)**: select the top-3 utterances by score, output them in dialogue order, grouped by predicted tag.

### 5.6 Evaluation (DialogSum test)
- Summary quality vs the 3 human references: ROUGE-1/2/L, BERTScore F1.
- **Retention** = student score / teacher score (target ≥ 0.90), where the teacher row is the teacher's own top-3 selection.
- Agreement with teacher: per-dialogue Spearman (averaged), top-3 Jaccard overlap.
- Tags: precision/recall/F1 for `decision` and `action_item` vs the 7B teacher's tags (state clearly this is agreement with the teacher, not ground truth).
- Efficiency: parameter count, file size, CPU latency per dialogue (M2), compared with the teachers.
- Reference rows: oracle selection (upper bound for extraction), lead-3 (first three utterances).

---

## 6. Results tables to produce

**Model 2**: rule baseline vs model, both test sets, all noise levels.

**Model 1**:
| System | Params | ROUGE-1/2/L | BERTScore | Retention | Spearman vs 7B | Decision/Action F1 | CPU latency |
|---|---|---|---|---|---|---|---|
| Lead-3 | – | | | | | | |
| Oracle | – | | | | | | |
| Teacher 7B | 7B | | | 1.00 | 1.00 | | |
| Teacher 1.5B | 1.5B | | | | | | |
| Student A | ~17M | | | | | | |
| Student B | ~17M | | | | | | |
| Student C | ~17M | | | | | | |
| Student D | ~17M | | | | | | |

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
7. Students A–D, 3 seeds each → evaluation → results tables.
8. Publish both models to Hugging Face: `.keras` file, `tokenizer.json`, config, model card (data, licence, method, metrics, limitations, usage snippet).
