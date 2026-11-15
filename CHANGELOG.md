# Changelog

All notable changes to this repository are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.0] - 2026-11-15

### Docs: training-grade README and walkthrough
- README: added "What problem this solves", "What you will learn", one-line key concepts, a mermaid
  pipeline diagram, a full repo layout and a clear synthetic-data label on the results.
- New `docs/walkthrough.md`: a guided first run, how to read each output, and one-flag experiments with
  their results.

### Fixed
- README quick start: the `--trades your_trades.csv` example failed on a fresh clone because no such file
  exists. It now writes the synthetic log to that CSV first. The README also says each run takes a minute or two.

## [1.1.0] - 2026-11-15

### Added
- `original/`: the original research code (read-only).
- `CHANGELOG.md`, `CONTRIBUTING.md` and a documentation index in `docs/README.md`.

## [1.0.0] - 2026-11-04

### Added
- First public release: the maintained code, documentation, tests and CI.
