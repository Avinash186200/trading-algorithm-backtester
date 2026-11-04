"""Time-split backtest: train on early years, choose the cutoff on training data, score later years once."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from .data import FEATURE_COLS
from .models import tuned_model

GROUP_COLS = ["Currency", "Condition", "TimeFrame"]


def time_split(df: pd.DataFrame, train_end_year: int, embargo_days: int = 10
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by entry date with a purge and an embargo around 31 Dec of ``train_end_year``.

    The trade log has no exit date, so a trade entered shortly before the cutoff
    may resolve (and get its label) inside the test period. Training rows entered
    in the last ``embargo_days`` before the cutoff are purged, and test rows in the
    first ``embargo_days`` after it are dropped, so no label straddles the boundary
    as long as trades last fewer than ``embargo_days`` calendar days.
    """
    cutoff = pd.Timestamp(f"{train_end_year}-12-31")
    train = df[df["date"] <= cutoff - pd.Timedelta(days=embargo_days)]
    test = df[df["date"] > cutoff + pd.Timedelta(days=embargo_days)]
    return train.sort_values("date"), test.sort_values("date")


def undersample(train: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Balance classes by undersampling the majority class. Training rows only."""
    good = train[train["Good_Bad_Flag"] == "Good"]
    bad = train[train["Good_Bad_Flag"] == "Bad"]
    n = min(len(good), len(bad))
    out = pd.concat([good.sample(n, random_state=seed), bad.sample(n, random_state=seed)])
    return out.sort_values("date")


def breakeven_prob(tp_ratio: pd.Series | np.ndarray) -> np.ndarray:
    """Win probability at which a trade with reward/risk ``tp_ratio`` breaks even: 1 / (RR + 1)."""
    return 1.0 / (np.asarray(tp_ratio, dtype=float) + 1.0)


def oof_probs(estimator, X: pd.DataFrame, y: np.ndarray, n_splits: int = 4) -> np.ndarray:
    """Out-of-fold P(good) on training data with time-ordered folds.

    In-sample probabilities from a random forest are close to 0/1, so a cutoff
    fitted on them is meaningless. These probabilities come from models that
    never saw the rows they score.
    """
    preds = []
    for tr, te in TimeSeriesSplit(n_splits).split(X):
        m = clone(estimator).fit(X.iloc[tr], y[tr])
        preds.append(m.predict_proba(X.iloc[te])[:, 1])
    return np.concatenate(preds)


def run_group(df: pd.DataFrame, model: str, train_end_year: int, rule: str = "breakeven",
              keep: float = 0.6, balance: bool = False, seed: int = 0,
              embargo_days: int = 10) -> pd.DataFrame:
    """Fit on the training years of one group and score the test years.

    ``rule``:
      * ``breakeven``: take a trade if P(good) >= 1/(RR+1). No parameter is fitted.
      * ``quantile``: take a trade if P(good) >= the ``(1-keep)`` quantile of the
        **out-of-fold training** probabilities. ``keep`` is fixed in advance.
    """
    train, test = time_split(df, train_end_year, embargo_days)
    if balance:
        train = undersample(train, seed)
    y_train = (train["Good_Bad_Flag"] == "Good").astype(int).to_numpy()
    X_train, X_test = train[FEATURE_COLS], test[FEATURE_COLS]

    est = tuned_model(model, seed).fit(X_train, y_train)
    out = test.copy()
    out["prob_good"] = est.predict_proba(X_test)[:, 1]

    if rule == "breakeven":
        out["threshold"] = breakeven_prob(out["TP_Ratio"])
    elif rule == "quantile":
        out["threshold"] = float(np.quantile(oof_probs(est.best_estimator_, X_train, y_train),
                                             1 - keep))
    else:
        raise ValueError(f"unknown rule {rule!r}")
    out["take"] = out["prob_good"] >= out["threshold"]
    out["model"] = model
    return out


def run_backtest(df: pd.DataFrame, models: list[str], train_end_year: int,
                 rule: str = "breakeven", keep: float = 0.6, balance: bool = False,
                 seed: int = 0) -> pd.DataFrame:
    """Run every model on every Currency/Condition/TimeFrame group and stack the test rows."""
    parts = []
    for _, g in df.groupby(GROUP_COLS):
        for m in models:
            parts.append(run_group(g, m, train_end_year, rule, keep, balance, seed))
    return pd.concat(parts, ignore_index=True)


def summarise(scored: pd.DataFrame, cost_per_trade: float = 0.05) -> pd.DataFrame:
    """Baseline (take all) vs filtered, per model, after a fixed cost per trade (in risk units)."""
    rows = []
    for m, g in scored.groupby("model"):
        net = g["Profit"] - cost_per_trade
        kept = g[g["take"]]
        rows.append({
            "model": m,
            "test_trades": len(g),
            "kept_pct": 100 * len(kept) / len(g),
            "base_win_rate": 100 * (g["Good_Bad_Flag"] == "Good").mean(),
            "filt_win_rate": 100 * (kept["Good_Bad_Flag"] == "Good").mean() if len(kept) else np.nan,
            "base_net_R": net.sum(),
            "filt_net_R": (kept["Profit"] - cost_per_trade).sum(),
            "base_R_per_trade": net.mean(),
            "filt_R_per_trade": (kept["Profit"] - cost_per_trade).mean() if len(kept) else np.nan,
        })
    return pd.DataFrame(rows).round(3)


def decile_lift(scored: pd.DataFrame, model: str) -> pd.DataFrame:
    """Decile table on the test set (descriptive only; nothing here is used to pick a cutoff)."""
    g = scored[scored["model"] == model].copy()
    g["decile"] = pd.qcut(g["prob_good"].rank(method="first"), 10, labels=False) + 1
    t = g.groupby("decile").agg(trades=("Profit", "size"),
                                win_rate=("Good_Bad_Flag", lambda s: (s == "Good").mean()),
                                profit=("Profit", "sum"))
    return t.sort_index(ascending=False).round(3)
