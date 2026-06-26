# Alpha Decay Predictor

> How fast does a high-frequency trading signal die? This measures the decay of
> order-book alpha on real NASDAQ limit-order-book data — feature engineering
> running natively in kdb+/q, modelling in Python (LightGBM), wired together
> over a PyKX IPC bridge.

![Alpha decay report](alpha_decay_report.png)

## What it does

Using full-depth LOBSTER limit-order-book data (NASDAQ: AMZN, AAPL, GOOG, MSFT,
INTC), the pipeline:

1. **Ingests millions of order-book events in KDB+/q** and computes
   microstructure features fully vectorized: micro-price, bid-ask spread,
   rolling volatility, and order-book imbalance (OBI) at 1, 5, and 10 price
   levels of depth.
2. **Streams the engineered data to Python** over a PyKX IPC connection.
3. **Trains one LightGBM model per prediction horizon** (10, 50, 200 ticks
   ahead) to forecast forward returns. The **primary target is the mid-price
   return** — the tradable mid. Micro-price returns are trained too, but only
   as a labelled *mechanical baseline* (see below).
4. **Measures the Information Coefficient (IC)** — rank correlation between
   predictions and realized returns — at each horizon, producing the alpha
   decay curve above.

## Headline results

*These come from a single 2012 NASDAQ session (the free LOBSTER sample), five
symbols cross-sectionally. Read this as a methodology demonstration and a
one-day signal snapshot, not a robust multi-day alpha estimate.*

Primary target — **mid-price** forward returns. `flat%` is the share of
forward returns that are exactly zero (the mid didn't move over the horizon);
a high `flat%` means the rank IC is over a near-degenerate target and overstates
economic value.

| Symbol | spread | IC 10 / 50 / 200 (mid) | flat% 10 / 50 / 200 |
| ------ | :----: | :--------------------: | :-----------------: |
| AMZN   |  ~13¢  |  0.17 / 0.17 / 0.14    |   53 / 13 / 3       |
| AAPL   |  ~15¢  |  0.12 / 0.05 / −0.02   |   39 / 8 / 2        |
| GOOG   |  ~28¢  |  0.15 / 0.19 / 0.10    |   36 / 6 / 1        |
| MSFT   |   1¢   |  0.23 / 0.42 / 0.50 ⚠  |   96 / 81 / 51 ⚠    |
| INTC   |   1¢   |  0.20 / 0.40 / 0.50 ⚠  |   97 / 86 / 61 ⚠    |

- **OBI gives a small but real short-horizon edge on liquid, wide-spread
  names** (AMZN, AAPL, GOOG): mid-price IC on the order of 0.1–0.2 at 10–50
  ticks, fading toward ~200 ticks (AAPL decays to zero; AMZN/GOOG hold a weak
  ~0.1). OBI drives 47–66% of model gain. This is the alpha-decay thesis.
- **The "high" MSFT/INTC numbers are an artifact, not alpha.** These are
  penny-pinned (1-tick spread), so 50–96% of their forward mid-returns are
  *exactly zero*. A rank IC over a target that is mostly ties only orders the
  rare moves and badly overstates a signal you couldn't harvest (a half-tick
  edge that takes 200 events to appear, on a 1¢ spread). The pipeline flags
  these automatically with `flat% > 25%`.

### Micro-price is a mechanical baseline, not the headline

An earlier version reported IC up to **0.33** on *micro-price* returns. That is
mostly mechanical: the micro-price is, by construction, pulled toward the
heavier queue, so it co-moves with OBI almost algebraically
(`corr(OBI, micro − mid) ≈ 0.90`). Predicting micro-price returns from OBI
therefore partly predicts an identity. On the tradable mid the same models give
roughly a third of that IC. Micro-price returns are still trained and printed,
but labelled **`[Baseline] MICRO-price IC (mechanical, for contrast only)`**.

| Symbol | mid IC @ 50 | micro IC @ 50 |
| ------ | :---------: | :-----------: |
| AMZN   |    0.17     |     0.33      |
| AAPL   |    0.05     |     0.33      |
| GOOG   |    0.19     |     0.31      |

## How I tried not to fool myself

- **Tradable target** — IC is reported on **mid-price** returns; the
  micro-price IC is kept only as a labelled mechanical baseline, because the
  micro-price encodes the imbalance it is being predicted from.
- **Degenerate-target guard** — forward returns that are mostly ties (penny-wide
  names) inflate rank IC; the pipeline reports `flat%` per horizon and flags
  any target with >25% zero-return observations.
- **Purged walk-forward validation** — chronological 70/30 split with an
  embargo gap equal to the longest forward-return horizon, so no training
  label overlaps the test period (no look-ahead leakage).
- **Moving-block bootstrap confidence intervals** — overlapping forward
  returns make tick observations heavily serially dependent, so naive
  p-values are wildly optimistic; all reported ICs carry 95% CIs from block
  resampling that preserves the serial dependence structure.
- **Honest caveats** — single trading day, no transaction costs, latency, or
  fill modeling: this measures signal decay, not strategy viability.

## Running it

The pipeline is two processes: a standalone kdb+/q server that ingests a LOBSTER
sample and computes the features, and a Python client that pulls the engineered
table over IPC, trains the models, and renders the report.

```
LOBSTER CSVs ──> kdb+/q server (port 5050) ──> PyKX IPC ──> LightGBM ──> IC decay report
```

```sh
pip install -r requirements.txt

# Terminal 1 — start the feature-engineering server (needs a kdb+ install)
q q_src/lobster_server.q

# Terminal 2 — run the full pipeline across all symbols
python3 main.py
```

The LOBSTER sample files aren't redistributed here; download the 10-level
samples from [lobsterdata.com](https://lobsterdata.com/info/DataSamples.php) and
extract them into `data/` (one folder per symbol, message + orderbook CSV pair).

## Tests

```sh
python3 -m pytest tests/ -v
```

The Python layer is tested independently of q — the suite covers the purged
split, the block-bootstrap IC, and an end-to-end train/evaluate run on synthetic
data where OBI genuinely predicts returns. No kdb+ license required.

## Repo layout

```
q_src/lobster_server.q   q server: ingest LOBSTER, compute features + forward returns
py_src/loader.py         PyKX IPC: trigger ingestion, pull the quotes table to pandas
py_src/model.py          one LightGBM model per horizon, on a purged walk-forward split
py_src/evaluate.py       Spearman IC + block-bootstrap CIs, OBI importance, flat% guard
main.py                  run the pipeline across all symbols and render the report
tests/                   pure-Python tests (no q server needed)
```

## Tech stack

`kdb+/q` · `PyKX (IPC)` · `Python` · `LightGBM` · `NumPy / pandas / SciPy` · `pytest`
