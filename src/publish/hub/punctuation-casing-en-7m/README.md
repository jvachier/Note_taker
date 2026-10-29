---
library_name: keras
language: en
license: cc-by-nc-sa-4.0
datasets:
  - Salesforce/wikitext
  - knkarthick/dialogsum
metrics:
  - f1
  - accuracy
pipeline_tag: token-classification
tags:
  - keras
  - jax
  - from-scratch
  - punctuation-restoration
  - truecasing
---

# punctuation-casing-en-7m

A small transformer that **restores punctuation and casing** in lowercase, unpunctuated English
text such as voice-note or speech-to-text output:

> **in:** okay so lets start with the budget we spent about twenty percent more than planned on marketing in march john do you know why that happened i think it was the new campaign on instagram we should check the numbers with the finance team next week
>
> **out:** Okay, so lets start with the budget. We spent about twenty percent more than planned on marketing in March. John, do you know why that happened? I think it was the new campaign on Instagram. We should check the numbers with the finance team next week.

| | |
|---|---|
| Parameters | 7.3M |
| File size | 29 MB (`model.keras`) |
| CPU latency (Apple M2, one text) | ~5 ms for a sentence, ~24 ms for 250 words |
| Outputs per word | casing: lower / Capitalised / UPPER · punctuation after it: none / `,` / `.` / `?` |

## Built from scratch

No pretrained weights and no pretrained tokenizer. Everything was trained for this project with
**Keras 3 on the JAX backend**:
- a 16k-token uncased WordPiece tokenizer trained on WikiText-103 and DialogSum;
- the transformer encoder (custom Keras layers, in `modeling.py`), trained from random
  initialisation.

## Architecture

| | |
|---|---|
| Encoder | 4 pre-LayerNorm transformer blocks, d_model 256, 4 heads, FFN 1024 (GELU) |
| Positions | learned, up to 128 tokens |
| Vocabulary | 16,000 WordPiece tokens |
| Heads | two token-classification heads (casing: 3 classes, punctuation: 4 classes), read at each word's first subword |

Input: `[CLS] word subwords … [SEP]`, at most 64 words per window. Longer texts are cut into
overlapping windows (stride 32) and each word takes its prediction from the window where it is
farthest from the edges.

## Training data

- **WikiText-103** (train, ×1) and **DialogSum** (train dialogues, ×5 to balance the
  conversational domain). Official splits; no dialogue in more than one split.
- Clean text is cut into windows of whole sentences (≤ 64 words). Every word is turned into its
  lowercase form without punctuation; its original casing and trailing punctuation are the
  labels (`!` and `;` count as periods, `:` as a comma).
- **Noise**, so the model copes with speech-to-text errors: 5% of words substituted, 3% deleted
  (their punctuation moves to the previous word), 2% inserted.
- Loss: cross-entropy on both heads, punctuation classes weighted by inverse square-root
  frequency. AdamW (peak lr 5e-4, 2k warmup steps, cosine), batch 128, up to 20 epochs with early
  stopping on validation punctuation macro-F1. 3 seeds; this release is the seed with the best
  validation score (seed 1, punctuation macro-F1 0.759).

## Results

Test sets, word level, **mean ± std over 3 seeds**. Noise = share of corrupted words in the
input. Rule baseline: capitalise the first word of each window and the word "i", and put a
period after the last word.

| Test set | Noise | System | Punct macro-F1 | Comma F1 | Period F1 | Question F1 | Casing acc |
|---|---|---|---|---|---|---|---|
| WikiText | 0% | Rule baseline | 0.191 | 0.000 | 0.574 | 0.000 | 0.814 |
| WikiText | 0% | **This model** | **0.671 ± 0.014** | 0.733 ± 0.001 | 0.896 ± 0.002 | 0.383 ± 0.044 | **0.976 ± 0.000** |
| WikiText | 20% | **This model** | 0.557 ± 0.009 | 0.603 ± 0.002 | 0.784 ± 0.002 | 0.284 ± 0.031 | 0.945 ± 0.000 |
| DialogSum | 0% | Rule baseline | 0.093 | 0.000 | 0.280 | 0.000 | 0.876 |
| DialogSum | 0% | **This model** | **0.789 ± 0.001** | 0.670 ± 0.002 | 0.850 ± 0.001 | 0.846 ± 0.002 | **0.965 ± 0.000** |
| DialogSum | 20% | **This model** | 0.662 ± 0.004 | 0.565 ± 0.003 | 0.740 ± 0.002 | 0.679 ± 0.006 | 0.937 ± 0.000 |

All noise levels (0 / 5 / 10 / 20%) and per-class casing scores are in `results.md`. The
WikiText test set has only 12 question marks, so its question F1 is not reliable.

## Usage

```bash
pip install "keras>=3.6" jax tokenizers huggingface_hub
```

```python
import sys
from huggingface_hub import snapshot_download

folder = snapshot_download("<hf-user>/punctuation-casing-en-7m")
sys.path.insert(0, folder)
import modeling  # registers the custom layers

cleaner = modeling.Cleaner.from_folder(folder)
print(cleaner("are you coming to the party tonight i can pick you up at eight"))
# Are you coming to the party tonight? I can pick you up at eight.
```

`example.py` runs the same thing. The model runs on CPU; set `KERAS_BACKEND=jax` if you import
Keras before `modeling`.

## Limitations

- English only; trained on encyclopaedic text (WikiText) and everyday dialogue (DialogSum).
- It restores **casing and `, . ?` only**: no apostrophes ("its" stays "its"), no `!`, `:`, `;`,
  quotes or brackets, and it never adds, removes or corrects words. Misrecognised words from
  speech-to-text stay as they are.
- Mixed-case words ("iPhone") and words outside the vocabulary's casing patterns can come out
  wrong; capitalisation of rare names depends on what the model saw in training.
- Commas are its weakest common class (F1 0.67–0.73).
- It makes mistakes like any punctuation model. For example, "hi sarah its tom … can you send
  me the updated files by friday" comes out as "Hi, Sarah. its Tom. … Can you send me the
  updated files? by Friday.": a missed capital after a period, and a question mark placed
  before the end of the question.

## Licences

- This model: **CC BY-NC-SA 4.0** (non-commercial, share-alike), because it is trained on
  DialogSum.
- DialogSum (`knkarthick/dialogsum`): CC BY-NC-SA 4.0.
- WikiText-103 (`Salesforce/wikitext`): Creative Commons Attribution-ShareAlike.
