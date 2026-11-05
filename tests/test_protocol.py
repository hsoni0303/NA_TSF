"""Protocol checks that need no training: causality, baseline-identical anomalies,
train-only scaling, noisy training targets, model shapes."""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "na_tsf"))
import train as T  # noqa: E402


def test_features_ignore_the_future():
    rng = np.random.default_rng(0)
    series = rng.normal(size=300)
    targets = np.array([100, 150, 200])
    before = T.causal_stl(series, targets, seq_len=16, context=64, period=4)
    tampered = series.copy()
    tampered[100:] += 100.0            # change the target and everything after it
    after = T.causal_stl(tampered, targets[:1], seq_len=16, context=64, period=4)
    np.testing.assert_allclose(before[:1], after)


def test_anomalies_match_robusttsf_baseline():
    sys.path.insert(0, str(ROOT / "baselines" / "robusttsf"))
    try:
        upstream = importlib.import_module("dataset")
    except Exception as e:  # heavy optional deps (cvxpy, bokeh, ...)
        pytest.skip(f"baseline deps not installed: {e}")
    x = np.random.default_rng(1).normal(size=500)
    for kind in ["const", "missing", "gaussian", "none"]:
        np.testing.assert_allclose(T.add_noise(x, 0.3, 0.5, kind),
                                   upstream.adding_noise(x, 0.3, 0.5, noise_type=kind))


@pytest.fixture
def args(tmp_path, monkeypatch):
    t = np.arange(1200)
    y = np.sin(2 * np.pi * t / 24) + 0.002 * t
    pd.DataFrame({"OT": y}, index=pd.RangeIndex(1200, name="date")).to_csv(tmp_path / "electricity.csv")
    monkeypatch.setattr(T, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(T, "CACHE_DIR", str(tmp_path / "cache"))
    return argparse.Namespace(**{**T.DEFAULTS, "ano_type": "const", "ano_ratio": 0.3, "verbose": 0})


def test_train_only_scaling_and_noisy_targets(args):
    train, train_noisy, val = T.load_splits(args)
    assert abs(train.mean()) < 1e-9 and abs(train.std() - 1) < 1e-9   # scaler fit on train only
    assert abs(val.mean()) > 0.5                                      # upward trend -> val not re-centred
    tr_loader, va_loader = T.make_loaders(args, seed=0, verbose=False)
    y = tr_loader.dataset.y
    t_tr = np.arange(args.stl_context, len(train) - 1)
    np.testing.assert_allclose(y, train_noisy[t_tr])                  # targets include the anomalies
    assert (train_noisy[t_tr] != train[t_tr]).any()


def test_model_output_shape(args):
    model = T.DecompLSTM(args.hidden, args.layers, args.embed)
    assert model(torch.zeros(5, args.seq_len, 3)).shape == (5,)
