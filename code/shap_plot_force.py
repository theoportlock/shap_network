#!/usr/bin/env python3
import argparse
import joblib
import shap

def main():
    parser = argparse.ArgumentParser(description="Generate SHAP force plot")
    parser.add_argument("--input", required=True, help="Path to SHAP Explanation joblib file")
    parser.add_argument("--output", required=True, help="Output HTML file (e.g., force_sample_0.html)")
    parser.add_argument("--sample_index", type=int, default=0, help="Sample index for the force plot")
    args = parser.parse_args()

    shap_values = joblib.load(args.input)

    # Save force plot as standalone HTML (force plot is interactive)
    force_plot = shap.plots.force(shap_values[args.sample_index], matplotlib=False)

    with open(args.output, "w") as f:
        f.write(force_plot.html())

if __name__ == "__main__":
    main()

