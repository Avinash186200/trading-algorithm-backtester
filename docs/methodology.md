# Methodology and bias audit

## Problem

A rule-based base strategy produces trades, each with a fixed stop-loss and a take-profit at `TP_Ratio` times the risk. The label is `Good` if the trade closed at take-profit and `Bad` if it closed at stop-loss. A classifier sees entry-time indicator features and outputs P(good). The decision is binary: take or skip.

## Data contract

| Field | Used for |
|---|---|
| `date` | Entry date; drives the time split |
| `Currency`, `Condition`, `TimeFrame` | Grouping: one model per group |
| `TP_Ratio` | Reward/risk; sets the break-even probability 1 / (RR + 1) |
| `f_00..f_11` | Entry-time features (the original used price/volume indicators) |
| `Close_Flag`, `Good_Bad_Flag` | Label |
| `Profit` | Evaluation only, in risk units |

## Validation design

- **Time split** per group: train up to 31 Dec of the chosen year, test after, with a 10-day purge of training trades before the cutoff and a 10-day embargo after it.
- **Tuning**: small grids, `GridSearchCV` with `TimeSeriesSplit(4)`, log-loss, training rows only.
- **Scaling** inside an sklearn `Pipeline`.
- **Cutoff**, one of:
  - break-even: `P(good) >= 1 / (RR + 1)`. Nothing is fitted, so nothing can be fitted to the test years.
  - quantile: the `(1 - keep)` quantile of out-of-fold training probabilities, with `keep` fixed before the run.
- **Evaluation**: win rate and net R per trade, baseline vs filtered, after a fixed cost per trade. The decile table is descriptive only.

## What the original code did, and what changed

1. **Test-set threshold selection.** The original produced percentile tables (10th..90th) and break-even tables with offsets of -20% to +20%, all computed on the 2019-2020 test years, and compared them to pick a cutoff. Fix: one rule, chosen up front.
2. **In-sample training threshold.** A "train rank 1" step picked the profit-maximising threshold from in-sample training probabilities and applied it to test. For a random forest these probabilities are close to 0 or 1, so the threshold is arbitrary. Fix: out-of-fold probabilities (`oof_probs`), checked in a test that shows in-sample probabilities separate pure noise while OOF ones do not.
3. **Pre-standardised features.** Inputs came from a file standardised across all years. Fix: scaler fitted inside each training fold.
4. **Shuffled CV for tuning.** Fix: `TimeSeriesSplit`.
5. **Best-of-N iterations.** An iteration sheet drove runs over models, seeds, oversampling on/off, cluster features and top-K feature subsets, all scored on the same test years. Fix: one configuration, all models reported.
6. **Feature-importance ranks from external files.** It is not clear which data they were computed on. Fix: no feature-selection step.
7. **No embargo.** Fix: 10-day purge before and 10-day embargo after the cutoff.
8. **No costs.** Fix: `--cost` per trade.

## Checks in CI

- Split: no overlap, embargo respected.
- Undersampling draws only from training rows and balances the classes.
- Break-even formula.
- The quantile cutoff does not change when test-period features are replaced with noise.
- OOF probabilities do not separate classes on noise; in-sample ones do.
- **Noise test**: with `signal_strength=0`, the mean win-rate and R-per-trade lift over four seeds stays near zero.
- With a planted signal, the filter beats the baseline.

## Dropped from the original

- Keras networks (replaced with scikit-learn `MLPClassifier`, originally the same 8-4-2 shape; changed to 16-8 with a stronger L2 grid and no early stopping after the 8-4-2 net collapsed to a constant prediction) and pickle dumps of every fitted model.
- Cluster-feature variants, permutation-importance tables, Excel exports and the iteration spreadsheet.
- Hard-coded local paths and `os.chdir` file searches.

## Known limitations

- Synthetic data only. The planted signal is linear, which favours logistic regression.
- One train/test split, not a walk-forward. With more years a yearly expanding walk-forward would be the next step.
- A fixed cost per trade is a simplification; spreads vary by pair and session.
