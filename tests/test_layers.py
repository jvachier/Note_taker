"""Tests for the shared Keras layers and masked-loss utilities."""

import keras
import numpy as np
import pytest

from src.layers.embeddings import TokenAndPositionEmbedding
from src.layers.losses import (
    IGNORE,
    inverse_sqrt_frequency_weights,
    label_weights,
    masked_sparse_ce,
    safe_labels,
)
from src.layers.transformer import TransformerEncoder, TransformerEncoderBlock

ENC = dict(vocab_size=50, max_len=16, d_model=16, num_layers=2, num_heads=2, ffn_dim=32,
           dropout=0.0)  # fmt: skip


def _ids(batch=2, length=10, pad_from=6, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, 50, size=(batch, length)).astype("int32")
    ids[:, pad_from:] = 0
    return ids


def test_embedding_shapes_and_speakers():
    emb = TokenAndPositionEmbedding(50, 16, 8, num_speakers=3, dropout=0.0)
    ids = _ids()
    out = emb(ids, speaker_ids=np.ones_like(ids))
    assert out.shape == (2, 10, 8)
    with pytest.raises(ValueError, match="speaker_ids"):
        emb(ids)
    assert TokenAndPositionEmbedding(50, 16, 8)(ids).shape == (2, 10, 8)


def test_speaker_embedding_changes_output():
    emb = TokenAndPositionEmbedding(50, 16, 8, num_speakers=3, dropout=0.0)
    ids = _ids()
    a = emb(ids, speaker_ids=np.ones_like(ids))
    b = emb(ids, speaker_ids=np.full_like(ids, 2))
    assert not np.allclose(keras.ops.convert_to_numpy(a), keras.ops.convert_to_numpy(b))


def test_block_rejects_bad_heads():
    with pytest.raises(ValueError, match="divisible"):
        TransformerEncoderBlock(10, 3, 20)


def test_encoder_output_shape():
    enc = TransformerEncoder(**ENC)
    assert enc(_ids()).shape == (2, 10, 16)


def test_padding_does_not_affect_real_positions():
    enc = TransformerEncoder(**ENC)
    ids = _ids()
    other = ids.copy()
    other[:, 6:] = 0
    longer = np.concatenate([ids, np.zeros((2, 4), "int32")], axis=1)  # extra padding
    a = keras.ops.convert_to_numpy(enc(ids))
    b = keras.ops.convert_to_numpy(enc(longer))
    np.testing.assert_allclose(a[:, :6], b[:, :6], atol=1e-5)


def test_changing_padded_token_ids_is_masked():
    """Padding is defined by ID 0; real tokens must not attend to padded positions."""
    enc = TransformerEncoder(**ENC)
    ids = _ids(pad_from=6)
    a = keras.ops.convert_to_numpy(enc(ids))
    ids2 = ids.copy()
    ids2[:, 8:] = 0  # same mask, only positions already padded
    b = keras.ops.convert_to_numpy(enc(ids2))
    np.testing.assert_allclose(a[:, :6], b[:, :6], atol=1e-6)
    shorter = ids.copy()
    shorter[:, 5] = 0  # padding one more real token changes real outputs
    c = keras.ops.convert_to_numpy(enc(shorter))
    assert not np.allclose(a[:, :5], c[:, :5], atol=1e-6)


def test_encoder_serialisation_round_trip(tmp_path):
    inp = keras.Input((None,), dtype="int32")
    model = keras.Model(inp, TransformerEncoder(num_speakers=0, **ENC)(inp))
    ids = _ids()
    expected = keras.ops.convert_to_numpy(model(ids))
    path = tmp_path / "m.keras"
    model.save(path)
    loaded = keras.saving.load_model(path)
    np.testing.assert_allclose(keras.ops.convert_to_numpy(loaded(ids)), expected, atol=1e-6)


def test_get_config_round_trip():
    for layer in (
        TokenAndPositionEmbedding(50, 16, 8, num_speakers=2),
        TransformerEncoderBlock(16, 2, 32),
        TransformerEncoder(**ENC),
    ):
        clone = type(layer).from_config(layer.get_config())
        assert clone.get_config() == layer.get_config()


def test_label_weights_zero_on_ignored():
    labels = np.array([[IGNORE, 0, 2, IGNORE], [1, IGNORE, IGNORE, 3]])
    np.testing.assert_array_equal(label_weights(labels), [[0, 1, 1, 0], [1, 0, 0, 1]])
    w = label_weights(labels, [0.5, 1.0, 2.0, 4.0])
    np.testing.assert_array_equal(w, [[0, 0.5, 2.0, 0], [1.0, 0, 0, 4.0]])
    assert safe_labels(labels).min() == 0


def test_masked_loss_ignores_weighted_out_positions():
    labels = np.array([[IGNORE, 1, 2, IGNORE]])
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(1, 4, 3)).astype("float32")
    loss = masked_sparse_ce()
    base = float(loss(safe_labels(labels), logits, sample_weight=label_weights(labels)))
    logits2 = logits.copy()
    logits2[0, [0, 3]] = rng.normal(size=(2, 3)) * 10  # change only ignored positions
    assert float(
        loss(safe_labels(labels), logits2, sample_weight=label_weights(labels))
    ) == pytest.approx(base)
    per_pos = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None)(
        safe_labels(labels), logits
    )
    expected = float(np.mean(keras.ops.convert_to_numpy(per_pos)[0, 1:3]))
    assert base == pytest.approx(expected, rel=1e-5)


def test_inverse_sqrt_frequency_weights():
    w = inverse_sqrt_frequency_weights([900, 100, 0])
    assert w[2] == 0.0
    assert w[1] / w[0] == pytest.approx(3.0)
    assert 0.9 * w[0] + 0.1 * w[1] == pytest.approx(1.0)
