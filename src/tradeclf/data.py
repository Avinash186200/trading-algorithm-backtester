"""Synthetic trade log with the same shape as the original research data.

The original input was one row per closed trade of a rule-based base strategy,
run across several FX pairs, market conditions (long / short) and timeframes.
Each row carried entry-time indicator features, the take-profit / risk ratio,
how the trade closed (take-profit or stop-loss) and its profit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_FEATURES = 12
FEATURE_COLS = [f"f_{i:02d}" for i in range(N_FEATURES)]


def make_synthetic_trades(
    n_trades: int = 3000,
    start: str = "2011-01-01",
    end: str = "2020-12-31",
    signal_strength: float = 0.8,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a synthetic closed-trade log.

    Args:
        n_trades: number of closed trades.
        start, end: date range for entry dates.
        signal_strength: how strongly ``f_00`` and ``f_01`` drive the win
            probability. ``0`` gives pure noise (no learnable edge).
        seed: RNG seed.

    Returns:
        DataFrame with columns ``date, Currency, Condition, TimeFrame,
        TP_Ratio, f_00..f_11, Close_Flag, Good_Bad_Flag, Profit``.
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="B")
    date = np.sort(rng.choice(days, size=n_trades))
    currency = rng.choice(["PAIR_A", "PAIR_B"], size=n_trades)
    condition = rng.choice(["Long", "Short"], size=n_trades)
    timeframe = np.full(n_trades, "D1")
    tp_ratio = rng.choice([1.0, 1.5, 2.0], size=n_trades)

    X = rng.normal(size=(n_trades, N_FEATURES))
    breakeven = 1.0 / (tp_ratio + 1.0)
    # Base strategy alone sits slightly below breakeven: a small loss before costs.
    logit_base = np.log(breakeven / (1 - breakeven)) - 0.05
    logit = logit_base + signal_strength * (0.7 * X[:, 0] - 0.5 * X[:, 1])
    p_win = 1.0 / (1.0 + np.exp(-logit))
    win = rng.random(n_trades) < p_win

    risk = 1.0  # one unit of risk per trade
    profit = np.where(win, tp_ratio * risk, -risk)

    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df.insert(0, "date", pd.to_datetime(date))
    df.insert(1, "Currency", currency)
    df.insert(2, "Condition", condition)
    df.insert(3, "TimeFrame", timeframe)
    df.insert(4, "TP_Ratio", tp_ratio)
    df["Close_Flag"] = np.where(win, "Take_Profit_Reach", "Stop_Loss_Reach")
    df["Good_Bad_Flag"] = np.where(win, "Good", "Bad")
    df["Profit"] = profit
    return df
