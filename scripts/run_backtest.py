"""Run the time-split backtest on synthetic data (default) or on your own trade log CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tradeclf.backtest import decile_lift, run_backtest, summarise  # noqa: E402
from tradeclf.data import make_synthetic_trades  # noqa: E402
from tradeclf.models import MODEL_NAMES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", help="CSV with the synthetic-data columns; omit for synthetic")
    ap.add_argument("--n-trades", type=int, default=3000)
    ap.add_argument("--signal", type=float, default=0.8, help="synthetic signal strength (0 = noise)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-end-year", type=int, default=2018)
    ap.add_argument("--rule", choices=["breakeven", "quantile"], default="breakeven")
    ap.add_argument("--keep", type=float, default=0.6, help="used by --rule quantile only")
    ap.add_argument("--cost", type=float, default=0.05, help="cost per trade in risk units")
    ap.add_argument("--balance", action="store_true", help="undersample majority class in training")
    ap.add_argument("--models", default=",".join(MODEL_NAMES))
    ap.add_argument("--out", help="directory for CSV outputs")
    a = ap.parse_args()

    if a.trades:
        df = pd.read_csv(a.trades, parse_dates=["date"])
    else:
        df = make_synthetic_trades(a.n_trades, signal_strength=a.signal, seed=a.seed)

    scored = run_backtest(df, a.models.split(","), a.train_end_year, a.rule, a.keep,
                          a.balance, a.seed)
    summary = summarise(scored, a.cost)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print("\nDecile lift (rf, test set):")
    print(decile_lift(scored, "rf").to_string() if "rf" in a.models else "rf not run")

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "summary.csv", index=False)
        scored.to_csv(out / "scored_test_trades.csv", index=False)


if __name__ == "__main__":
    main()
