#!/usr/bin/env python3
from sympy.parsing.sympy_parser import parse_expr
import argparse
import os
import pandas as pd
import sympy as sp

def parse_args():
    parser = argparse.ArgumentParser(description="Compute feature and interaction variance from symbolic formula.")
    parser.add_argument("--formula_file", type=str, default="formula.txt", help="Path to symbolic formula file")
    parser.add_argument("--output_dir", type=str, default="results/formula_EV", help="Directory to save variance files")
    parser.add_argument("--residual_name", type=str, default="noise", help="Variable name to treat as residual error")
    return parser.parse_args()

def collect_term_contributions(expr, residual_name="noise"):
    """
    Walk through expanded formula and extract squared coefficients for each variable and interaction.
    Returns dicts for individual and interaction variance contributions, and the residual (if any).
    """
    individual = {}
    interactions = {}
    residual = 0.0

    expr = sp.expand(expr)
    terms = expr.args if isinstance(expr, sp.Add) else [expr]

    for term in terms:
        symbols = list(term.free_symbols)
        var_names = sorted(str(sym) for sym in symbols)

        try:
            coeff = float(term.as_coeff_mul()[0])
        except Exception:
            continue

        contribution = coeff ** 2

        if len(var_names) == 1:
            var = var_names[0]
            if var == residual_name:
                residual += contribution
            else:
                individual[var] = individual.get(var, 0.0) + contribution
        elif len(var_names) > 1:
            key = tuple(sorted(var_names))
            interactions[key] = interactions.get(key, 0.0) + contribution

    return individual, interactions, residual

def main():
    args = parse_args()

    with open(args.formula_file, "r") as f:
        formula_str = f.read().strip()

    expr = parse_expr(formula_str)
    individual, interactions, residual = collect_term_contributions(expr, residual_name=args.residual_name)

    total = sum(individual.values()) + sum(interactions.values()) + residual

    # Convert to DataFrames
    feature_df = pd.DataFrame([
        {"feature": k, "variance_explained_percent": 100 * v / total}
        for k, v in sorted(individual.items())
    ])

    interaction_df = pd.DataFrame([
        {"source": key[0], "target": key[1], "interaction_percent": 100 * val / total}
        for key, val in interactions.items()
        if len(key) == 2
    ])

    if residual > 0:
        residual_df = pd.DataFrame([{
            "feature": args.residual_name,
            "variance_explained_percent": 100 * residual / total
        }])
        feature_df = pd.concat([feature_df, residual_df], ignore_index=True)

    os.makedirs(args.output_dir, exist_ok=True)
    feature_df.to_csv(f"{args.output_dir}/feature_variance.tsv", sep='\t', index=False)
    interaction_df.to_csv(f"{args.output_dir}/interaction_variance.tsv", sep='\t', index=False)

    explained_pct = 100 * (sum(individual.values()) + sum(interactions.values())) / total
    print(f"✅ Variance decomposition complete. Model explains {explained_pct:.2f}% of variance.")
    print(f"📁 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

