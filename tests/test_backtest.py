import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tradeclf.backtest import (  # noqa: E402
    breakeven_prob, oof_probs, run_backtest, run_group, summarise, time_split, undersample,
)
from tradeclf.data import FEATURE_COLS, make_synthetic_trades  # noqa: E402
from tradeclf.models import make_pipeline  # noqa: E402


def test_synthetic_shape_and_profit_consistency():
    df = make_synthetic_trades(500, seed=1)
    assert len(df) == 500
    good = df["Good_Bad_Flag"] == "Good"
    assert np.allclose(df.loc[good, "Profit"], df.loc[good, "TP_Ratio"])
    assert (df.loc[~good, "Profit"] == -1).all()


def test_time_split_no_overlap_and_embargo():
    df = make_synthetic_trades(800, seed=2)
    train, test = time_split(df, 2017, embargo_days=10)
    assert train["date"].max() <= pd.Timestamp("2017-12-21")  # purged
    assert test["date"].min() > pd.Timestamp("2018-01-10")
    assert set(train.index).isdisjoint(test.index)


def test_undersample_uses_training_rows_only():
    df = make_synthetic_trades(800, seed=3)
    train, test = time_split(df, 2017)
    bal = undersample(train)
    assert set(bal.index) <= set(train.index)
    assert (bal["Good_Bad_Flag"] == "Good").sum() == (bal["Good_Bad_Flag"] == "Bad").sum()


def test_breakeven_formula():
    assert np.allclose(breakeven_prob([1.0, 2.0]), [0.5, 1 / 3])


def test_quantile_cutoff_is_independent_of_test_rows():
    """The cutoff must not change when the test rows change."""
    df = make_synthetic_trades(900, seed=4)
    a = run_group(df, "logreg", 2017, rule="quantile")
    shuffled = df.copy()
    late = shuffled["date"] > pd.Timestamp("2018-01-31")
    shuffled.loc[late, FEATURE_COLS] = np.random.default_rng(0).normal(size=(late.sum(), len(FEATURE_COLS)))
    b = run_group(shuffled, "logreg", 2017, rule="quantile")
    assert a["threshold"].nunique() == 1
    assert a["threshold"].iloc[0] == b["threshold"].iloc[0]


def test_oof_probs_are_not_in_sample():
    df = make_synthetic_trades(900, seed=5, signal_strength=0.0)
    X = df[FEATURE_COLS]
    y = (df["Good_Bad_Flag"] == "Good").astype(int).to_numpy()
    p = oof_probs(make_pipeline("rf"), X, y)
    # In-sample RF probabilities on noise separate the classes; OOF ones must not.
    in_sample = make_pipeline("rf").fit(X, y).predict_proba(X)[:, 1]
    y_oof = y[len(y) - len(p):]
    gap_oof = p[y_oof == 1].mean() - p[y_oof == 0].mean()
    gap_in = in_sample[y == 1].mean() - in_sample[y == 0].mean()
    assert gap_in > 0.1
    assert abs(gap_oof) < 0.05


def test_noise_input_shows_no_edge():
    """Pure-noise features: averaged over seeds, the filter must not add R per trade."""
    lifts_wr, lifts_r = [], []
    for seed in (11, 12, 13, 14):
        df = make_synthetic_trades(2000, seed=seed, signal_strength=0.0)
        s = summarise(run_backtest(df, ["logreg"], 2017), cost_per_trade=0.05)
        lifts_wr.append(float(s["filt_win_rate"].iloc[0] - s["base_win_rate"].iloc[0]))
        lifts_r.append(float(s["filt_R_per_trade"].iloc[0] - s["base_R_per_trade"].iloc[0]))
    assert np.mean(lifts_wr) < 4, lifts_wr
    assert np.mean(lifts_r) < 0.08, lifts_r


def test_signal_is_recovered():
    df = make_synthetic_trades(2000, seed=6, signal_strength=0.8)
    s = summarise(run_backtest(df, ["logreg"], 2017))
    assert (s["filt_R_per_trade"] > s["base_R_per_trade"]).all(), s
