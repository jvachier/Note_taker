"""Tests for shared training utilities."""

import time

import keras
import numpy as np
import pytest

from src.training import NO_DECAY, TimeGuard, best_logged_value, build_optimizer, steps_for


def test_steps_for():
    assert steps_for(1000, 10, 3) == (100, 3, 300)
    assert steps_for(1000, 10, 3, max_steps=50) == (50, 1, 50)
    assert steps_for(1000, 10, 9, max_steps=50, steps_per_epoch=10) == (10, 5, 50)
    with pytest.raises(ValueError):
        steps_for(5, 10, 1)


def test_best_logged_value(tmp_path):
    path = tmp_path / "h.csv"
    assert best_logged_value(path, "f1", "max") is None
    path.write_text("epoch,f1\n0,0.2\n1,0.5\n2,0.4\n")
    assert best_logged_value(path, "f1", "max") == 0.5
    assert best_logged_value(path, "f1", "min") == 0.2


def test_optimizer_schedule_warmup_and_decay():
    opt = build_optimizer(
        {
            "name": "adamw",
            "schedule": "cosine",
            "peak_lr": 1e-3,
            "warmup_steps": 10,
            "weight_decay": 0.01,
        },
        total_steps=110,
    )
    schedule = opt._learning_rate
    assert float(schedule(0)) == pytest.approx(0.0)
    assert float(schedule(10)) == pytest.approx(1e-3)
    assert float(schedule(110)) == pytest.approx(0.0, abs=1e-9)
    assert "bias" in NO_DECAY


def test_optimizer_warmup_fraction():
    opt = build_optimizer(
        {
            "name": "adamw",
            "schedule": "cosine",
            "peak_lr": 1e-4,
            "warmup_fraction": 0.1,
            "weight_decay": 0.01,
        },
        total_steps=100,
    )
    assert float(opt._learning_rate(10)) == pytest.approx(1e-4)


class _Model:
    stop_training = False


def test_time_guard_stops_when_next_epoch_would_overrun():
    guard = TimeGuard(max_hours=1.0, session_start=time.time() - 3500)
    guard.set_model(_Model())
    guard.on_train_begin()
    guard.on_epoch_begin(0)
    guard._epoch_start -= 200  # pretend the epoch took 200 s
    guard.on_epoch_end(0)
    assert guard.stopped and guard.model.stop_training


def test_time_guard_lets_short_runs_continue():
    guard = TimeGuard(max_hours=1.0)
    guard.set_model(_Model())
    guard.on_train_begin()
    guard.on_epoch_begin(0)
    guard.on_epoch_end(0)
    guard.on_train_batch_end(0)
    assert not guard.stopped


def test_sync_epoch_sets_the_dataset_epoch():
    from src.training import SyncEpoch

    class Recorder:
        epochs: list[int] = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    ds = Recorder()
    cb = SyncEpoch(ds)
    for epoch in (3, 4):  # e.g. a run resumed at epoch 3
        cb.on_epoch_begin(epoch)
    assert ds.epochs == [3, 4]


def test_layerwise_lr_scales_encoder_updates():
    """Adam's first step moves each weight by ~lr: encoder weights move 10x less."""
    inp = keras.Input((4,))
    hidden = keras.layers.Dense(4, name="encoder", use_bias=False)(inp)
    model = keras.Model(inp, keras.layers.Dense(1, name="head", use_bias=False)(hidden))
    cfg = {
        "name": "adamw",
        "schedule": "cosine",
        "lr_heads": 1e-2,
        "lr_encoder": 1e-3,
        "warmup_steps": 0,
        "weight_decay": 0.0,
    }
    opt = build_optimizer(cfg, total_steps=100)
    before = [np.array(w) for w in model.get_weights()]
    model.compile(optimizer=opt, loss="mse")
    model.train_on_batch(np.ones((2, 4), "float32"), np.full((2, 1), 5.0, "float32"))
    enc, head = (np.abs(np.array(a) - b) for a, b in zip(model.get_weights(), before, strict=True))
    np.testing.assert_allclose(enc, 1e-3, rtol=0.05)
    np.testing.assert_allclose(head, 1e-2, rtol=0.05)
    assert opt.get_config()["lr_multipliers"] == {"encoder": 0.1}


def test_pad_rows_fills_partial_batches():
    from src.training import pad_rows

    rows, n = pad_rows(np.array([4, 7, 9]), 5)
    assert n == 3 and rows.tolist() == [4, 7, 9, 9, 9]
    rows, n = pad_rows(np.array([1, 2]), 2)
    assert n == 2 and rows.tolist() == [1, 2]


def test_configure_devices_rejects_uneven_batches(monkeypatch):
    import jax

    from src.training import configure_devices

    monkeypatch.setattr(jax, "devices", lambda: ["gpu:0", "gpu:1"])
    monkeypatch.setattr(keras.distribution, "set_distribution", lambda d: None)
    with pytest.raises(ValueError, match="not divisible"):
        configure_devices("float32", 128, 77)
    keras.mixed_precision.set_dtype_policy("float32")


def test_time_guard_hard_limit_fires_on_any_batch_index():
    # With steps_per_execution=32 Keras reports batches 31, 63, ... (never a multiple of 50).
    guard = TimeGuard(max_hours=1.0, session_start=time.time() - 3700)
    guard.set_model(_Model())
    guard.on_train_begin()
    guard.on_train_batch_end(31)
    assert guard.stopped and guard.model.stop_training
