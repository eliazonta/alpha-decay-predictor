import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Horizon (in ticks) of each forward-return target. Consecutive rows share
# most of their forward window, so observations are heavily serially
# dependent — plain spearmanr p-values are wildly optimistic. We report
# block-bootstrap confidence intervals instead.
HORIZON_TICKS = {
    'ret_10c': 10, 'ret_50c': 50, 'ret_200c': 200,                  # mid-price (primary)
    'ret_micro_10c': 10, 'ret_micro_50c': 50, 'ret_micro_200c': 200,  # micro-price (baseline)
}


def block_bootstrap_ic(preds, actuals, horizon, n_boot=500, seed=42):
    """Point IC plus a moving-block bootstrap confidence interval.

    Blocks are longer than the forward-return overlap window so each
    resample preserves the serial dependence structure of the data.
    Returns (ic, ci_low, ci_high).
    """
    preds = np.asarray(preds)
    actuals = np.asarray(actuals)
    n = len(preds)

    ic, _ = spearmanr(preds, actuals)
    if np.isnan(ic):
        ic = 0.0

    block = min(max(2 * horizon, 100), n)
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)

    boot_ics = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        boot_ic, _ = spearmanr(preds[idx], actuals[idx])
        boot_ics[b] = 0.0 if np.isnan(boot_ic) else boot_ic

    ci_low, ci_high = np.percentile(boot_ics, [2.5, 97.5])
    return ic, ci_low, ci_high


def compute_metrics(models, test_df):
    """Per-horizon IC with bootstrap CI, plus normalized OBI gain importance."""
    features = list(models.values())[0].feature_name()
    X_test = test_df[features]

    metrics = {}
    for target, model in models.items():
        preds = model.predict(X_test)
        actuals = test_df[target].values

        ic, ci_low, ci_high = block_bootstrap_ic(
            preds, actuals, HORIZON_TICKS[target])

        importance = pd.Series(
            model.feature_importance(importance_type='gain'),
            index=model.feature_name())
        if importance.sum() > 0:
            importance = importance / importance.sum() * 100

        # Degenerate-target guard. On tick-constrained (penny-wide) names the
        # mid-price often doesn't move over the horizon, so a large share of
        # forward returns are exactly zero. A rank IC over a target that is
        # mostly ties only orders the rare moves and badly overstates economic
        # value -- flag it so the headline IC isn't read at face value.
        frac_flat = float(np.mean(np.isclose(actuals, 0.0)))

        metrics[target] = {
            'ic': ic,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'obi_importance': importance.get('obi', 0.0),
            'frac_flat': frac_flat,
        }
        flag = '  <-- degenerate target (mostly ties)' if frac_flat > 0.25 else ''
        print(f"  {target}: IC = {ic:.4f}  [95% CI {ci_low:.4f}, {ci_high:.4f}]"
              f"  OBI importance = {importance.get('obi', 0.0):.1f}%"
              f"  flat = {frac_flat:.1%}{flag}")

    return metrics


def plot_report(results, output_path):
    """Combined alpha-decay report across symbols.

    results: {symbol: {target: {ic, ci_low, ci_high, obi_importance}}}
    """
    # Plot only the targets actually present in `results` (the primary
    # mid-price set), preserving HORIZON_TICKS order.
    present = next(iter(results.values()))
    horizons = [h for h in HORIZON_TICKS if h in present]
    x = np.arange(len(horizons))
    symbols = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for sym in symbols:
        ics = np.array([results[sym][h]['ic'] for h in horizons])
        lo = np.array([results[sym][h]['ci_low'] for h in horizons])
        hi = np.array([results[sym][h]['ci_high'] for h in horizons])
        # Penny-wide names whose mid barely moves produce a near-degenerate
        # (mostly-tied) target; their IC is inflated and not tradable. Render
        # them as a faint dashed line so they don't read as the strongest signal.
        flats = [results[sym][h].get('frac_flat', 0.0) for h in horizons]
        degenerate = max(flats) > 0.25
        axes[0].errorbar(
            x, ics, yerr=[ics - lo, hi - ics], marker='o', capsize=4,
            label=f'{sym} (degenerate target)' if degenerate else sym,
            linestyle='--' if degenerate else '-',
            alpha=0.4 if degenerate else 1.0)

    axes[0].axhline(0, color='gray', linewidth=0.8, linestyle='--')
    axes[0].set_xticks(x, horizons)
    axes[0].set_title('Alpha Decay: mid-price IC vs Horizon (95% block-bootstrap CI)')
    axes[0].set_xlabel('Prediction Horizon')
    axes[0].set_ylabel('Spearman Rank Correlation (IC)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    width = 0.8 / len(symbols)
    for i, sym in enumerate(symbols):
        obi = [results[sym][h]['obi_importance'] for h in horizons]
        axes[1].bar(x + (i - (len(symbols) - 1) / 2) * width, obi,
                    width=width, label=sym)

    axes[1].set_xticks(x, horizons)
    axes[1].set_title('Feature Importance: OBI across Horizons')
    axes[1].set_xlabel('Prediction Horizon')
    axes[1].set_ylabel('Relative Importance (%)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nSaved visualization to {output_path}")
