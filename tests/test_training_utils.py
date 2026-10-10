"""Tests for shared training utilities."""

import time

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
        {"name": "adamw", "schedule": "cosine", "peak_lr": 1e-3, "warmup_steps": 10,
         "weight_decay": 0.01}, total_steps=110,
    )  # fmt: skip
    schedule = opt._learning_rate
    assert float(schedule(0)) == pytest.approx(0.0)
    assert float(schedule(10)) == pytest.approx(1e-3)
    assert float(schedule(110)) == pytest.approx(0.0, abs=1e-9)
    assert "bias" in NO_DECAY


def test_optimizer_warmup_fraction():
    opt = build_optimizer(
        {"name": "adamw", "schedule": "cosine", "peak_lr": 1e-4, "warmup_fraction": 0.1,
         "weight_decay": 0.01}, total_steps=100,
    )  # fmt: skip
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
