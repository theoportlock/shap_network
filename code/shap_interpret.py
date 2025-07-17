#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SHAP and SHAP interaction scores")
    parser.add_argument("--model", type=str, required=True, help="Path to joblib model file")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with X_train.tsv and/or X_test.tsv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save SHAP results")
    parser.add_argument("--shap_val", action="store_true", help="Compute SHAP values")
    parser.add_argument("--shap_interact", action="store_true", help="Compute SHAP interaction values")
    return parser.parse_args()


def load_tsv(file_path):
    return pd.read_csv(file_path, sep='\t', index_col=0)


def save_tsv(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False)


def compute_shap_values(model, X, label, output_dir):
    explainer = shap.Explainer(model, seed=42)
    shap_values = explainer(X)

    # Save raw SHAP values with joblib
    shap_raw = {
        "values": shap_values.values,
        "base_values": shap_values.base_values,
        "data": shap_values.data,
        "feature_names": list(X.columns),
        "sample_ids": list(X.index)
    }

    shap_path = os.path.join(output_dir, f"shap_values.joblib")
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(shap_raw, shap_path)
    print(f"✅ Saved raw SHAP values with feature names to {shap_path}")

    # Save mean absolute SHAP values
    mean_abs_shap = pd.DataFrame(
        np.abs(shap_values.values).mean(axis=0),
        index=X.columns,
        columns=[f"mean_abs_shap"]
    )
    mean_abs_path = os.path.join(output_dir, f"mean_abs_shap.tsv")
    save_tsv(mean_abs_shap.reset_index().rename(columns={"index": "feature"}), mean_abs_path)
    print(f"✅ Saved {mean_abs_path}")


def compute_shap_interactions(model, X, label, output_dir):
    explainer = shap.Explainer(model)
    shap_inter = explainer.shap_interaction_values(X)

    inter_path = os.path.join(output_dir, f"shap_interaction_values.joblib")
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(inter_raw, inter_path)
    print(f"✅ Saved raw SHAP interaction values with feature names to {inter_path}")

    # Mean absolute interaction matrix
    mean_abs_inter = np.abs(shap_inter).mean(axis=0)
    inter_df = pd.DataFrame(mean_abs_inter, index=X.columns, columns=X.columns)

    # Stack to long format and symmetrize
    long_df = inter_df.stack().reset_index()
    long_df.columns = ["feature1", "feature2", "mean_abs_weight"]
    long_df[["source", "target"]] = np.sort(long_df[["feature1", "feature2"]].values, axis=1)

    grouped = (
        long_df.groupby(["source", "target"], as_index=False)
               .agg({"mean_abs_weight": "sum"})
    )

    total = grouped["mean_abs_weight"].sum()
    grouped["relative_importance"] = grouped["mean_abs_weight"] / total if total > 0 else 0.0

    output_file = os.path.join(output_dir, f"mean_abs_shap_interaction.tsv")
    save_tsv(grouped, output_file)
    print(f"✅ Saved {output_file}")


def main():
    args = parse_args()
    model = joblib.load(args.model)

    for split in ["train", "test"]:
        x_path = Path(args.input_dir) / f"X_{split}.tsv"
        if x_path.exists():
            X = load_tsv(x_path)
            if args.shap_val:
                compute_shap_values(model, X, label=split, output_dir=args.output_dir)
            if args.shap_interact:
                compute_shap_interactions(model, X, label=split, output_dir=args.output_dir)
        else:
            print(f"⚠️ Skipping: {x_path} not found")


if __name__ == "__main__":
    main()

