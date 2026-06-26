import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from py_src.loader import connect, get_training_data
from py_src.model import train_models, TARGETS, MICRO_TARGETS
from py_src.evaluate import compute_metrics, plot_report

# All symbols with 10-level LOBSTER samples in data/
SYMBOLS = ['AMZN', 'AAPL', 'MSFT', 'INTC', 'GOOG']


def main():
    print("=== Alpha Decay Predictor Pipeline (LOBSTER) ===")

    q_server = connect()
    results = {}

    for symbol in SYMBOLS:
        print(f"\n========== {symbol} ==========")

        print(f"[Load] Ingesting LOBSTER data for {symbol} and computing features in KDB+...")
        df = get_training_data(q_server, symbol)
        print(f"Retrieved {len(df)} rows of engineered data.")

        print("[Train] Training LightGBM models (purged walk-forward split)...")
        models, train_df, test_df = train_models(df, targets=TARGETS)
        print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

        print("[Evaluate] MID-price IC (tradable target) with block-bootstrap CIs:")
        results[symbol] = compute_metrics(models, test_df)

        # Mechanical baseline: same features, micro-price target. Reported only
        # for contrast -- a higher IC here is mostly OBI-micro-price coupling,
        # not extra alpha.
        micro_models, _, micro_test = train_models(df, targets=MICRO_TARGETS)
        print("[Baseline] MICRO-price IC (mechanical, for contrast only):")
        compute_metrics(micro_models, micro_test)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'alpha_decay_report.png')
    plot_report(results, output_path)

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
