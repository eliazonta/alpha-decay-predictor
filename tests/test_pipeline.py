import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.model import purged_split, train_models, FEATURES, TARGETS, EMBARGO_TICKS
from py_src.evaluate import block_bootstrap_ic, compute_metrics


def make_synthetic_quotes(n=5000, seed=0):
    """Synthetic quotes table with the same columns the q server produces,
    where OBI genuinely predicts forward returns (so IC should be > 0)."""
    rng = np.random.default_rng(seed)
    obi = rng.uniform(-1, 1, n)
    mid = 100 + np.cumsum(rng.normal(0, 0.01, n))

    df = pd.DataFrame({
        'time': np.arange(n, dtype=float),
        'micro_price': mid,
        'obi': obi,
        'obi_5': obi + rng.normal(0, 0.1, n),
        'obi_10': obi + rng.normal(0, 0.2, n),
        'spread': rng.uniform(0.01, 0.05, n),
        'rolling_vol': pd.Series(mid).rolling(100).std().bfill().values,
    })
    # Forward returns driven by current OBI, with noise growing with horizon
    # to mimic alpha decay
    for target, horizon in [('ret_10c', 10), ('ret_50c', 50), ('ret_200c', 200)]:
        noise = rng.normal(0, 0.5 * np.sqrt(horizon / 10), n)
        df[target] = obi + noise
    return df


class TestPurgedSplit:
    def test_embargo_gap_between_train_and_test(self):
        df = make_synthetic_quotes(1000)
        train_df, test_df = purged_split(df, train_frac=0.7, embargo=200)
        # Train must end at least `embargo` rows before test begins
        assert train_df.index.max() + 200 < test_df.index.min() + 1
        assert test_df.index.min() == 700
        assert train_df.index.max() == 499

    def test_no_overlap(self):
        df = make_synthetic_quotes(1000)
        train_df, test_df = purged_split(df)
        assert set(train_df.index).isdisjoint(set(test_df.index))

    def test_zero_embargo_matches_plain_split(self):
        df = make_synthetic_quotes(1000)
        train_df, test_df = purged_split(df, embargo=0)
        assert len(train_df) == 700
        assert len(test_df) == 300


class TestBlockBootstrapIC:
    def test_perfect_correlation(self):
        x = np.arange(1000, dtype=float)
        ic, lo, hi = block_bootstrap_ic(x, x, horizon=10, n_boot=100)
        assert ic == pytest.approx(1.0)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_ci_brackets_point_estimate(self):
        rng = np.random.default_rng(1)
        actuals = rng.normal(0, 1, 2000)
        preds = actuals + rng.normal(0, 1, 2000)
        ic, lo, hi = block_bootstrap_ic(preds, actuals, horizon=50, n_boot=200)
        assert lo <= ic <= hi
        assert 0 < ic < 1

    def test_no_signal_ci_contains_zero(self):
        rng = np.random.default_rng(2)
        preds = rng.normal(0, 1, 2000)
        actuals = rng.normal(0, 1, 2000)
        _, lo, hi = block_bootstrap_ic(preds, actuals, horizon=10, n_boot=200)
        assert lo < 0 < hi

    def test_constant_predictions_return_zero(self):
        preds = np.ones(500)
        actuals = np.random.default_rng(3).normal(0, 1, 500)
        ic, _, _ = block_bootstrap_ic(preds, actuals, horizon=10, n_boot=50)
        assert ic == 0.0


class TestEndToEnd:
    def test_train_and_evaluate_on_synthetic_data(self):
        df = make_synthetic_quotes(5000)
        models, train_df, test_df = train_models(df)

        assert set(models.keys()) == set(TARGETS)
        # Purged split applied
        assert len(train_df) == int(5000 * 0.7) - EMBARGO_TICKS
        assert len(test_df) == 5000 - int(5000 * 0.7)

        metrics = compute_metrics(models, test_df)
        for target in TARGETS:
            m = metrics[target]
            assert m['ci_low'] <= m['ic'] <= m['ci_high']
            assert 0 <= m['obi_importance'] <= 100
        # OBI drives the synthetic signal, so the model should find alpha
        assert metrics['ret_50c']['ic'] > 0.05
