#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Plot SHAP values (summary, force, decision, interaction, etc.)")
    parser.add_argument("--input", type=str, required=True, help="Path to .joblib SHAP value or interaction file")
    parser.add_argument("--output", type=str, required=True, help="Path to save output plot (e.g., .png)")
    parser.add_argument("--plot_type", type=str, required=True,
                        choices=["summary_dot", "summary_bar", "summary_violin", "dependence",
                                 "force", "waterfall", "decision", "interaction_heatmap"],
                        help="Type of SHAP plot to generate")
    parser.add_argument("--index", type=int, help="Row index (sample) for local plots (force, waterfall, decision)")
    parser.add_argument("--feature", type=str, help="Feature name for dependence plot")
    parser.add_argument("--interaction_feature", type=str, help="Second feature to color dependence plot")
    parser.add_argument("--max_display", type=int, default=20, help="Max features to display in summary plots")
    return parser.parse_args()


def load_shap_explanation(shap_data):
    return shap.Explanation(
        values=shap_data["values"],
        base_values=shap_data.get("base_values", None),
        data=shap_data["data"],
        feature_names=shap_data["feature_names"]
    )


def plot_summary(shap_exp, plot_type, output, max_display):
    shap.summary_plot(shap_exp, plot_type=plot_type.split("_")[1], max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output)
    print(f"✅ Saved {plot_type} plot to {output}")


def plot_violin(shap_exp, output, max_display):
    shap.summary_plot(shap_exp, plot_type="violin", max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output)
    print(f"✅ Saved summary_violin plot to {output}")


def plot_dependence(shap_exp, feature, interaction_feature, output):
    shap.dependence_plot(
        ind=feature,
        shap_values=shap_exp,
        interaction_index=interaction_feature,
        show=False
    )
    plt.tight_layout()
    plt.savefig(output)
    print(f"✅ Saved dependence plot to {output}")


def plot_force(shap_exp, index, output):
    force_plot = shap.plots.force(shap_exp[index], matplotlib=True, show=False)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    print(f"✅ Saved force plot for index {index} to {output}")


def plot_waterfall(shap_exp, index, output):
    shap.plots.waterfall(shap_exp[index], show=False)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    print(f"✅ Saved waterfall plot for index {index} to {output}")


def plot_decision(shap_exp, index, output):
    if index is not None:
        shap.plots.decision(shap_exp[index], show=False)
        print(f"✅ Saved decision plot for index {index} to {output}")
    else:
        shap.plots.decision(shap_exp, show=False)
        print(f"✅ Saved decision plot for all instances to {output}")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")


def plot_interaction_heatmap(shap_data, output):
    interaction_values = shap_data["interaction_values"]
    feature_names = shap_data["feature_names"]

    mean_abs = np.abs(interaction_values).mean(axis=0)
    df = pd.DataFrame(mean_abs, index=feature_names, columns=feature_names)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap="viridis", square=True, cbar_kws={"label": "Mean |Interaction SHAP|"})
    plt.title("Mean Absolute SHAP Interaction Values")
    plt.tight_layout()
    plt.savefig(output)
    print(f"✅ Saved SHAP interaction heatmap to {output}")


def main():
    args = parse_args()
    shap_data = joblib.load(args.input)

    if args.plot_type == "interaction_heatmap":
        if "interaction_values" not in shap_data:
            raise ValueError("Input file does not contain SHAP interaction values.")
        plot_interaction_heatmap(shap_data, args.output)
        return

    if "values" not in shap_data:
        raise ValueError("Input file does not contain SHAP values.")
    shap_exp = load_shap_explanation(shap_data)

    if args.plot_type.startswith("summary"):
        if args.plot_type == "summary_violin":
            plot_violin(shap_exp, args.output, args.max_display)
        else:
            plot_summary(shap_exp, args.plot_type, args.output, args.max_display)

    elif args.plot_type == "dependence":
        if not args.feature:
            raise ValueError("--feature must be provided for dependence plot.")
        plot_dependence(shap_exp, args.feature, args.interaction_feature, args.output)

    elif args.plot_type == "force":
        if args.index is None:
            raise ValueError("--index is required for force plot.")
        plot_force(shap_exp, args.index, args.output)

    elif args.plot_type == "waterfall":
        if args.index is None:
            raise ValueError("--index is required for waterfall plot.")
        plot_waterfall(shap_exp, args.index, args.output)

    elif args.plot_type == "decision":
        plot_decision(shap_exp, args.index, args.output)

    else:
        raise ValueError(f"Unknown plot type: {args.plot_type}")


if __name__ == "__main__":
    main()

