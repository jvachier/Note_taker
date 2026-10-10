"""BertSum-style dialogue encoding: layout, speakers, truncation of the longest utterances."""

import numpy as np

from src.model1_scorer.dialogue import encode_dialogue, length_cap
from src.tokenizer.tokenizer import CLS_ID, PAD_ID, SEP_ID, load_tokenizer


def test_length_cap():
    assert length_cap([3, 5, 2], 100) == 5
    assert length_cap([3, 5, 2], 8) == 3  # 3 + 3 + 2
    assert length_cap([10, 10], 5) == 2
    assert length_cap([], 4) == 0


def test_layout_and_speakers(tiny_assets):
    tok = load_tokenizer(tiny_assets["tokenizer"])
    utts = ["hello there", "how are you", "fine"]
    enc = encode_dialogue(tok, [1, 2, 1], utts, max_len=32)
    lengths = [len(tok.encode(u, add_special_tokens=False).ids) for u in utts]
    expected_cls = np.cumsum([0, *[n + 2 for n in lengths[:-1]]])
    np.testing.assert_array_equal(enc.cls_positions, expected_cls)
    end = int(expected_cls[-1]) + lengths[-1] + 2
    assert (enc.token_ids[enc.cls_positions] == CLS_ID).all()
    assert (enc.token_ids[enc.cls_positions[1:] - 1] == SEP_ID).all()
    assert enc.token_ids[end - 1] == SEP_ID and (enc.token_ids[end:] == PAD_ID).all()
    assert (enc.speaker_ids[end:] == 0).all()
    assert list(enc.speaker_ids[enc.cls_positions]) == [1, 2, 1]
    assert enc.token_ids.dtype == np.int32 and enc.speaker_ids.dtype == np.int8
    assert enc.truncated == 0 and enc.dropped == 0


def test_longest_utterance_is_truncated_first(tiny_assets):
    tok = load_tokenizer(tiny_assets["tokenizer"])
    long = " ".join(["the quick brown fox jumps over the lazy dog"] * 3)
    enc = encode_dialogue(tok, [1, 2], ["hi", long], max_len=16)
    assert enc.truncated == 1 and enc.dropped == 0
    assert enc.token_ids[-1] == SEP_ID  # the budget is used exactly
    second = enc.token_ids[enc.cls_positions[1] :]
    assert len(second) == 16 - (len(tok.encode("hi", add_special_tokens=False).ids) + 2)


def test_too_many_utterances_are_dropped_from_the_end(tiny_assets):
    tok = load_tokenizer(tiny_assets["tokenizer"])
    enc = encode_dialogue(tok, [1] * 5, ["hi"] * 5, max_len=6)
    assert len(enc.cls_positions) == 3 and enc.dropped == 2
    assert (enc.token_ids == [CLS_ID, SEP_ID] * 3).all()
