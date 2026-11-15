# Walkthrough: a first run

This is a guided first run. It assumes you know Python but not trading. Read the "Key concepts" list in the
[README](../README.md#key-concepts) first if a term is new. All numbers below come from running the code on a
synthetic trade log. Each run takes a minute or two.

## 1. Set up and run the tests

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

You should see `8 passed`. Read `tests/test_backtest.py` from the top. Three tests carry the main lessons:

- `test_quantile_cutoff_is_independent_of_test_rows` replaces the test-period features with noise and checks that
  the cutoff does not move. The cutoff is fixed before the test years are seen.
- `test_oof_probs_are_not_in_sample` shows that in-sample random forest probabilities separate pure noise, while
  out-of-fold ones do not. That is why a cutoff picked on in-sample probabilities means nothing.
- `test_noise_input_shows_no_edge` runs four seeds of pure noise and checks that the filter adds nothing on average.

## 2. Run with the planted signal

```bash
python scripts/run_backtest.py
```

The data: 3,000 trades from 2011 to 2020, two pairs, long and short, so four groups and four models per group.
Training is up to 2018, testing is 2019-2020. The test set has 608 trades.

| Column | What it means |
|---|---|
| `kept_pct` | Share of test trades the model says to take |
| `base_win_rate` | Win rate if you take every trade: 38.7% |
| `filt_win_rate` | Win rate of the kept trades |
| `base_R_per_trade` | Average net R per trade, taking everything: -0.10 after the 0.05 R cost |
| `filt_R_per_trade` | Average net R per kept trade |

Logistic regression keeps 51% of trades and lifts the win rate to 49.2% and R per trade to +0.20. The
planted signal is linear, so a linear model is the natural winner here. The MLP keeps 58% and only gets to +0.03,
which is within noise (see the statistical caveat in the README).

Under the table is a decile lift table for the random forest. Decile 10 holds the 61 trades with the highest
P(good): 72% of them won. Decile 1 holds the lowest: 25% won. A monotone-ish slope like this is what a working
model looks like. The table is descriptive only; nothing in it is used to pick a cutoff.

## 3. Run on pure noise

```bash
python scripts/run_backtest.py --signal 0
```

Now the features carry no information. Every model's filtered win rate is *below* the baseline (34-37% vs 39.5%),
and the decile table has no slope: decile 10 wins 43%, decile 7 wins 50%, decile 1 wins 36%. Filtered R per trade
lands on both sides of the baseline of -0.09: -0.07 for logistic regression and the MLP, -0.10 and -0.11 for the
SVM and random forest. Looking at one seed can mislead in either direction, which is why the CI test averages
four seeds.

## 4. Try changing one thing

I ran each of these while writing this page. Add `--out somewhere` to save the CSVs.

| Command | logreg R/trade | rf R/trade | svm R/trade | mlp R/trade | Note |
|---|---|---|---|---|---|
| `python scripts/run_backtest.py` | +0.20 | +0.15 | +0.14 | +0.03 | Break-even rule |
| `python scripts/run_backtest.py --rule quantile --keep 0.6` | +0.12 | +0.15 | +0.14 | -0.04 | Keep the top 60% by OOF training cutoff. The MLP keeps 83%, not 60% |
| `python scripts/run_backtest.py --balance` | +0.08 | -0.01 | +0.04 | +0.08 | Undersample the majority class in training |
| `python scripts/run_backtest.py --signal 0` | -0.07 | -0.11 | -0.10 | -0.07 | No signal. Baseline is -0.09 |

Two things to notice:

- The quantile rule targets 60% kept, but the cutoff comes from training probabilities, so the share kept on the
  test years can differ. For the MLP it is 83%. That is not a bug: the test years simply score higher. It is the
  honest cost of fixing the cutoff in advance.
- Balancing the classes makes every model keep more trades (64-74%) and earn less per trade. Undersampling shifts
  the predicted probabilities upward, so more trades clear the break-even line. If you balance, the break-even rule
  no longer means what it says.

## 5. Where to go next

- [methodology.md](methodology.md) has the full method and the bias audit.
- `src/tradeclf/backtest.py` is about 130 lines and holds the split, the cutoff rules and the summary.
- `make_pipeline` in `src/tradeclf/models.py` has a comment on why the original 8-4-2 network collapsed to a
  constant prediction.
