"""Raw-text inference helpers of both models, the human-review helpers, teacher stats layouts."""

import json

import numpy as np

from src.eval.human_eval import best_seed, label_differences
from src.model1_scorer.evaluate import teacher_throughput_ms
from src.model1_scorer.infer import dialogue_notes, score_dialogue
from src.model1_scorer.model import build_student
from src.model2_cleanup.infer import clean_text, input_words, render_word, window_starts
from src.model2_cleanup.labels import CAPITALISED, LOWER, NONE, PERIOD, QUESTION, UPPER
from src.tokenizer.tokenizer import load_tokenizer

SMALL = {
    "vocab_size": 400,
    "max_len": 32,
    "d_model": 16,
    "num_layers": 1,
    "num_heads": 2,
    "ffn_dim": 32,
    "dropout": 0.0,
    "init_std": 0.02,
    "num_speakers": 8,
}


class _EndOfWindowModel:
    """Stub Model 2: capitalises everything, period on the last token before ``[SEP]``."""

    def predict_on_batch(self, token_ids):
        ids = np.asarray(token_ids)
        n, length = ids.shape
        casing = np.zeros((n, length, 3))
        casing[..., CAPITALISED] = 1
        punct = np.zeros((n, length, 4))
        punct[..., NONE] = 1
        for row in range(n):
            last = int(np.nonzero(ids[row])[0].max()) - 1  # token before [SEP]
            punct[row, last] = np.eye(4)[PERIOD]
        return {"casing": casing, "punct": punct}


def test_input_words_and_render_word():
    assert input_words("Hi, John! I'm OK -- 1,000?") == ["hi", "john", "i'm", "ok", "1,000"]
    assert render_word("john", CAPITALISED, PERIOD) == "John."
    assert render_word("asap", UPPER, QUESTION) == "ASAP?"
    assert render_word("ok", LOWER, NONE) == "ok"


def test_window_starts_overlap_and_cover():
    assert window_starts(10, 64) == [0]
    assert window_starts(64, 64) == [0]
    assert window_starts(96, 64) == [0, 32]
    starts = window_starts(200, 64)
    assert starts[-1] + 64 == 200
    assert all(b - a <= 32 for a, b in zip(starts, starts[1:], strict=False))


def test_clean_text_maps_predictions_back_to_words(tiny_assets):
    tok = load_tokenizer(tiny_assets["tokenizer"])
    # Single-token words, so the stub's last token is also the last word's first subword.
    vocab = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy",
        "dog", "it", "was", "sunny", "day", "in", "paris",
    ]  # fmt: skip
    single = [w for w in vocab if len(tok.encode(w, add_special_tokens=False).ids) == 1]
    assert len(single) >= 5
    words = (single * 10)[:20]
    out = clean_text(_EndOfWindowModel(), tok, " ".join(words), window_words=8, max_len=64)
    # Every word comes back once, in order; only the last word of the text gets the
    # window-end period, because each word is read from its most central window.
    assert out.split() == [w.capitalize() for w in words[:-1]] + [words[-1].capitalize() + "."]
    assert clean_text(_EndOfWindowModel(), tok, "  -- ", 8, 64) == ""


def test_score_dialogue_and_notes(tiny_assets):
    tok = load_tokenizer(tiny_assets["tokenizer"])
    model = build_student(SMALL, use_tags=True)
    utts = ["Hi, can you send the report?", "Sure, on Friday.", "Thanks.", "Bye."]
    speakers = [1, 2, 1, 2]
    scored = score_dialogue(model, tok, speakers, utts, max_len=32, top_k=3)
    assert len(scored.scores) == len(scored.tags) == 4 and scored.dropped == 0
    assert all(0 <= s <= 1 for s in scored.scores) and len(scored.selected) == 3
    assert dialogue_notes(scored, speakers, utts).count("- #Person") == 3
    # Utterances that do not fit are scored 0 and tagged none.
    short = score_dialogue(model, tok, speakers, utts, max_len=6, top_k=3)
    assert short.dropped == 1 and short.scores[-1] == 0.0 and short.tags[-1] == "none"
    untagged = score_dialogue(build_student(SMALL, use_tags=False), tok, speakers, utts, 32, 2)
    assert untagged.tags is None and len(untagged.selected) == 2


def test_label_differences_aligns_by_spelling():
    assert label_differences("Hi, John. How are you?", "Hi John, how are you?") == (3, 5)
    # Words missing from the output are skipped, not counted as errors.
    assert label_differences("Send the file.", "Send me the file.") == (0, 3)


def test_best_seed_uses_validation(tmp_path):
    for seed, value in [(0, 0.5), (1, 0.7), (2, 0.6)]:
        d = tmp_path / "exp" / f"seed_{seed}"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"validation": {"spearman": value}}))
    assert best_seed(tmp_path, "exp", ("validation", "spearman")) == (1, 0.7)


def test_teacher_throughput_reads_both_stats_layouts(tmp_path):
    counts = {"labelled": 3, "failed": 1}
    local = {"splits": {"test": {"this_session": counts, "seconds_this_session": 2.0}}}
    kaggle = {"splits": {"test": {"labelled": 3, "failed": 1, "seconds": 2.0}}}
    for i, stats in enumerate([local, kaggle]):
        path = tmp_path / f"stats_{i}.json"
        path.write_text(json.dumps(stats))
        assert teacher_throughput_ms(path) == 500.0
    assert teacher_throughput_ms(tmp_path / "missing.json") is None
