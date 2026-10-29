---
library_name: keras
language: en
license: cc-by-nc-sa-4.0
datasets:
  - Salesforce/wikitext
  - knkarthick/dialogsum
metrics:
  - f1
  - rouge
pipeline_tag: token-classification
tags:
  - keras
  - jax
  - from-scratch
  - meeting-notes
  - extractive-summarization
  - action-items
---

# meeting-notes-en-21m

A small transformer that turns a meeting transcript into **notes**: it picks the sentences
that matter and sorts them into **Decisions**, **Action items** (with owner and deadline) and
**Key points**.

```text
Anna: Thanks for joining. We need to decide on the launch date today.
Ben: The app is almost ready, but the payment page still has two bugs.
Anna: How long do you need to fix them?
Ben: About a week, if nobody pulls me onto other work.
Chloe: Marketing can be ready by the 15th, but not earlier.
Anna: Then let's launch on the 15th. Ben, please fix the payment bugs by Friday.
Ben: Okay, I'll send an update on Wednesday.
Chloe: I'll book the newsletter for the morning of the 15th.
Anna: Great. Anything else? No? Thanks, everyone.
```

```markdown
## Decisions
- Then let's launch on the 15th. (Anna)

## Action items
- Ben, please fix the payment bugs by Friday. (Anna) — owner: ?, due: Friday
- Okay, I'll send an update on Wednesday. (Ben) — owner: Ben, due: ?
- I'll book the newsletter for the morning of the 15th. (Chloe) — owner: Chloe, due: ?

## Key points
- The app is almost ready, but the payment page still has two bugs. (Ben)
- Marketing can be ready by the 15th, but not earlier. (Chloe)
```

The model is **extractive**: every note is a sentence from the transcript. It does not write
new text.

| | |
|---|---|
| Parameters | 20.9M |
| File size | 84 MB (`model.keras`) |
| CPU latency (Apple M2, one meeting) | ~0.4 s for 100 sentences, ~2.2 s for 500 sentences |
| Outputs per sentence | importance · zone (context / discussion / decision / action item / closing) · owner and deadline spans |

## Built from scratch

No pretrained weights and no pretrained tokenizer. Everything was trained for this project with
**Keras 3 on the JAX backend**:
- a 16k-token uncased WordPiece tokenizer, trained on WikiText-103 and DialogSum (shared with
  [`punctuation-casing-en-7m`](https://huggingface.co/<hf-user>/punctuation-casing-en-7m));
- a sentence encoder, pretrained with masked-language modelling on WikiText-103 and DialogSum
  dialogues;
- the meeting model (custom Keras layers, in `modeling.py`), fine-tuned on a gold-labelled
  meeting corpus.

## Architecture

| | |
|---|---|
| Sentence encoder | 6 pre-LayerNorm transformer blocks, d_model 384, 6 heads, FFN 1536 (GELU); each sentence is encoded on its own (up to 64 tokens) with its speaker embedding |
| Meeting encoder | 2 pre-LN blocks over the sentences' `[CLS]` states, plus sentence-position, speaker and speaker-change embeddings; up to 256 sentences per window |
| Heads (per sentence) | importance (1 logit) · zone (5 classes) · owner / deadline BIO tags per token (5 classes) |
| Unused heads | reason (9 classes) and a `depends_on` pointer, kept for the teacher-distillation runs; untrained in this checkpoint |
| Vocabulary | 16,000 WordPiece tokens; up to 8 speakers |

Meetings longer than 256 sentences are read in windows overlapping by 64 sentences; each
sentence takes its prediction from the window where it is farthest from the edges.

**Inference.** A sentence is kept when its importance is at least **0.40** (tuned on
validation). Its zone is the argmax, except that *decision* and *action item* have their own
thresholds (**0.045** and **0.495**), tuned on validation to maximise the mean of the real-meeting
and synthetic F1. Then a few rules tidy the notes (no change to which sentences are kept):
- a short reply from another speaker ("OK, I will email him tonight.", "Will do.") is attached
  under the kept request or decision just before it;
- a first-person action item ("I'll book the newsletter…") gets its speaker as owner; a
  first-person reply gives the request its owner.

All settings are in `config.json`.

## Training data

**`meeting-notes-extractive-v1`, version 1.1**
(built in the companion repository `meeting-notes-extractive-corpus`): meeting transcripts
with a gold label for every sentence (important or not, zone, owner / deadline spans).

| Layer | Split | Meetings | Sentences | Important |
|---|---|--:|--:|--:|
| Synthetic | train / validation / test | 1,460 / 170 / 170 | 121,566 / 14,956 / 14,854 | 26% |
| AMI (real meetings) | train / validation / test | 97 / 20 / 20 | 75,996 / 16,476 / 15,876 | 18% |

- **Synthetic layer:** meetings composed from 1,460 hand-written, hand-labelled segments
  (standups, project reviews, retrospectives, informal conversations…), including hard
  negatives such as rejected proposals and hypothetical tasks. No language model writes or
  labels this data.
- **AMI layer:** real design-team meetings, labelled from the AMI Meeting Corpus manual
  extractive-summary annotations.
- Splits are the corpus's own; no meeting or seed family is in more than one split.

**Training:** importance (binary cross-entropy, weight 1.0), zone (cross-entropy with
inverse-square-root class weights, 0.5) and spans (0.3). AdamW (lr 1e-4 for the sentence
encoder, 5e-4 for the rest; 10% warmup, cosine), up to 15 epochs with early stopping on
validation loss, one T4 GPU. 3 seeds; this release is the seed with the best AMI validation
importance F1 (seed 1, 0.584).

**Teacher distillation (tried, not used here).** A second run (E4) added distillation from
Qwen3-0.6B (importance, reasons, reply links) and a fine-tuned RoBERTa-Large. It gave no gain
over gold-only training at this scale (AMI importance F1 0.567 ± 0.006 vs 0.578 ± 0.006), so
this checkpoint is trained on the gold labels only. Neither teacher's weights are part of
this model.

## Results

Test sets, against the gold labels. **AMI (real meetings) is the headline.** Students: mean ±
std over 3 seeds. Imp F1 = importance F1 (is the sentence in the notes?). HN FP = share of
synthetic hard negatives wrongly kept (lower is better). ROUGE: kept sentences vs the AMI
human abstractive summary.

| System | Params | Synth imp F1 | Synth zone F1 | Synth HN FP ↓ | **AMI imp F1** | AMI dec F1 | AMI act F1 | AMI ROUGE-1 / 2 / L |
|---|---|---|---|---|---|---|---|---|
| Lead-k (first sentences) | – | 0.337 | – | 0.490 | 0.319 | – | – | 0.148 / 0.041 / 0.076 |
| Keyword rule | – | 0.136 | 0.247 | 0.000 | 0.029 | 0.045 | 0.024 | 0.128 / 0.020 / 0.076 |
| Qwen3-0.6B (prompted) | 600M | 0.403 | 0.194 | 0.585 | 0.318 | 0.030 | 0.000 | 0.143 / 0.041 / 0.073 |
| DistilBERT (fine-tuned) | 66M | 0.870 | – | 0.187 | 0.580 | – | – | 0.149 / 0.050 / 0.080 |
| RoBERTa-Large (fine-tuned) | 355M | 0.903 | – | 0.070 | 0.634 | – | – | 0.166 / 0.054 / 0.087 |
| **This model** | **20.9M** | 0.878 ± 0.013 | 0.620 ± 0.039 | 0.142 ± 0.044 | **0.578 ± 0.006** | 0.081 ± 0.014 | 0.016 ± 0.023 | 0.154 / 0.050 / 0.082 |

- It **matches a fine-tuned DistilBERT** (3× larger, pretrained on far more text) and is 5.6
  F1 points below RoBERTa-Large (17× larger) on real meetings.
- On synthetic meetings, decision / action-item F1 are 0.65 / 0.94 for the released seed. On
  AMI they are below 0.1 for every system: real decisions are rare, implicit and spread over
  many turns.
- Owner / deadline span F1 (synthetic test): 0.81 ± 0.07.
- The DistilBERT and RoBERTa rows only score importance (no zone or span heads). Their
  synthetic scores miss the 20 test meetings added in corpus v1.1.

The full table, with PR-AUC and the distilled run, is in `results.md`.

## Usage

```bash
pip install "keras>=3.6" jax tokenizers huggingface_hub
```

```python
import sys
from huggingface_hub import snapshot_download

folder = snapshot_download("<hf-user>/meeting-notes-en-21m")
sys.path.insert(0, folder)
import modeling  # registers the custom layers

notetaker = modeling.NoteTaker.from_folder(folder)
print(
    notetaker("Anna: Then let's launch on the 15th.\nBen: Okay, I'll send an update on Wednesday.")
)
```

Input: one `Speaker: text` line per turn; turns are split into sentences. For structured
output, `notetaker.annotate(speakers, sentences)` returns every sentence with its score, zone,
`kept` flag, owner, deadline and attached replies. `notetaker(text, threshold=0.3)` keeps more
sentences.

**Raw speech-to-text** (lowercase, no punctuation): restore punctuation and casing first with
[`punctuation-casing-en-7m`](https://huggingface.co/<hf-user>/punctuation-casing-en-7m), via
`notetaker(text, clean=cleaner)`. The model was trained on punctuated, cased text and is much
worse without this step. `example.py` shows both cases.

The model runs on CPU. `modeling.py` selects the JAX backend; set `KERAS_BACKEND=jax` if you
import Keras before `modeling`.

## Limitations

- **Extractive only.** A decision reached over several turns ("Monday is fine" … "I can work
  from home" … "then I'll tell him ten") is not stated in any single sentence, so it does not
  appear as a decision.
- **Weak on real meetings' zones.** Importance is reasonable on AMI (F1 0.58), but decision and
  action-item labels on long, messy real meetings are unreliable (F1 < 0.1). It works best on
  short, clear conversations like the example above.
- Owners named in the sentence ("Ben, please…") and deadlines are sometimes missed, and
  reported facts can be labelled action items ("he said someone will come on Monday").
- The reply rules are heuristics: a reply is attached to the nearest kept request before it,
  which can be the wrong one.
- English only, at most 8 distinct speakers (more share one speaker embedding), sentences cut
  at 62 subwords.

## Licences

- This model: **CC BY-NC-SA 4.0** (non-commercial, share-alike), because its tokenizer and
  sentence encoder are trained on DialogSum.
- DialogSum (`knkarthick/dialogsum`): CC BY-NC-SA 4.0.
- WikiText-103 (`Salesforce/wikitext`): Creative Commons Attribution-ShareAlike.
- `meeting-notes-extractive-v1` (synthetic and AMI layers): CC BY 4.0.
- AMI Meeting Corpus annotations: CC BY 4.0. Carletta, J. et al. (2005). *The AMI Meeting
  Corpus: A Pre-announcement.* MLMI 2005, LNCS 3869, Springer.
- Teachers and baselines (evaluation and the distillation experiment only; not part of this
  model): Qwen3-0.6B (Apache 2.0), RoBERTa-Large (MIT) and DistilBERT (Apache 2.0), the
  latter two via KerasHub.
