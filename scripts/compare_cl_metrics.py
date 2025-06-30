import os
import argparse
from src.utils.compare import aggregate_metrics, plot_comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare CL metrics from experiments")
    parser.add_argument(
        "--aggregated",
        action="store_true",
        help="Use aggregated multi-seed results with confidence intervals",
    )
    args = parser.parse_args()

    if args.aggregated:
        aggregated_files = [
            "logs/test0_aggregated_metrics.json",
            "logs/test1_aggregated_metrics.json",
            "logs/test1b_aggregated_metrics.json",
        ]
        if any(os.path.exists(f) for f in aggregated_files):
            names, bwt, fwt, ret, trans, bwt_err, fwt_err, ret_err, trans_err = (
                aggregate_metrics(use_aggregated=True)
            )
            plot_comparison(
                names, bwt, fwt, ret, trans, bwt_err, fwt_err, ret_err, trans_err
            )
            print("✓ Aggregated comparison completed with confidence intervals.")
        else:
            print(
                "No aggregated results found. Run experiments with enable_averaging=True first."
            )
    else:
        names, bwt, fwt, ret, trans = aggregate_metrics(use_aggregated=False)
        plot_comparison(names, bwt, fwt, ret, trans)
        print("✓ Single-seed comparison completed and plot saved.")
