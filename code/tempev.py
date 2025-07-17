#!/usr/bin/env python3
"""variance_tables.py
Compute theoretical explained variance for main effects and interaction terms in a linear-style
formula. The script
  • parses a plain‑text formula (e.g. the one in *formula.txt*)
  • separates main‑effect vs. interaction terms
  • multiplies each term’s coefficient² by a user‑supplied variance (defaults to 1) to get the
    contribution to variance explained
  • writes two tab‑separated files — *feature_variance.txt* and *interaction_variance.txt* —
    into an output directory you specify.

Example
-------
Given *formula.txt* containing
    1.5*(Temperature_C - 25) + 6*((MarketingSpend_USD - 1000)/100) + \
    2*(Temperature_C - 25)*((MarketingSpend_USD - 1000)/100) + \
    8*((TimeOfDay == 'Afternoon') & (Promotion == 1)) + noise

run:
    python variance_tables.py \
        --formula formula.txt \
        --output-dir out \
        --var "Temperature_C - 25"=4 \
        --var "(MarketingSpend_USD - 1000)/100"=1.2 \
        --var "(TimeOfDay == 'Afternoon')"=0.25 \
        --var "(Promotion == 1)"=0.3

If you omit --var options every variable is assumed to have variance 1, so each term’s
ExplainedVariance is simply coefficient².
"""

from __future__ import annotations
import argparse
import os
import re
from typing import Dict, List, Tuple
import pandas as pd


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def _top_level_split(expr: str, sep: str = "+") -> List[str]:
    """Split *expr* on *sep*, but only at the top level (outside any parentheses)."""
    out, depth, token = [], 0, []
    for ch in expr:
        if ch == "(" or ch == "[":
            depth += 1
        elif ch == ")" or ch == "]":
            depth -= 1
        if ch == sep and depth == 0:
            part = "".join(token).strip()
            if part:
                out.append(part)
            token = []
        else:
            token.append(ch)
    final = "".join(token).strip()
    if final:
        out.append(final)
    return out


def _classify(term: str) -> Tuple[str, str, float] | None:
    """Return (kind, inner_expr, coefficient).

    kind ∈ {"feature", "interaction"}
    inner_expr is the expression inside the outermost parentheses
    """
    m = re.match(r"([+-]?\d+(?:\.\d+)?)\*\((.*)\)$", term.strip())
    if not m:
        return None
    coef = float(m.group(1))
    inner = m.group(2).strip()
    if "*" in inner or "&" in inner:
        return ("interaction", inner, coef)
    return ("feature", inner, coef)


def _build_tables(tokens: List[str], variances: Dict[str, float]):
    feat_rows, int_rows = [], []
    for t in tokens:
        c = _classify(t)
        if c is None:
            continue  # skip noise or unsupported pieces
        kind, inner, coef = c
        if kind == "feature":
            var = variances.get(inner, 1.0)
            feat_rows.append(
                {
                    "Term": inner,
                    "Coefficient": coef,
                    "Variance": var,
                    "ExplainedVariance": coef ** 2 * var,
                }
            )
        else:  # interaction
            components = re.split(r"[*&]", inner)
            var = 1.0
            for comp in components:
                comp = comp.strip()
                var *= variances.get(comp, 1.0)
            int_rows.append(
                {
                    "Term": inner,
                    "Coefficient": coef,
                    "Variance": var,
                    "ExplainedVariance": coef ** 2 * var,
                }
            )
    return pd.DataFrame(feat_rows), pd.DataFrame(int_rows)


# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def main():  # noqa: D401
    """Run as a script."""
    ap = argparse.ArgumentParser(
        description="Compute theoretical explained variance from a linear-style formula."
    )
    ap.add_argument("--formula", "-f", required=True, help="Path to formula .txt file")
    ap.add_argument(
        "--output-dir", "-o", required=True, help="Directory in which to write results"
    )
    ap.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="EXPR=VALUE",
        help="Optional variance for an expression inside () — you can repeat this option.",
    )
    args = ap.parse_args()

    # Build variance dict
    variances: Dict[str, float] = {}
    for spec in args.var:
        try:
            key, val = spec.split("=", 1)
            variances[key.strip()] = float(val)
        except ValueError as exc:
            raise SystemExit(f"Invalid --var specification: '{spec}'. Expect EXPR=value.") from exc

    # Read formula and parse tokens
    formula_text = open(args.formula, "r", encoding="utf8").read().strip()
    tokens = _top_level_split(formula_text, "+")

    # Build tables
    feats, inters = _build_tables(tokens, variances)

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Write
    feats.to_csv(os.path.join(args.output_dir, "feature_variance.txt"), sep="\t", index=False)
    inters.to_csv(
        os.path.join(args.output_dir, "interaction_variance.txt"), sep="\t", index=False
    )

    print(
        "Saved:\n  - feature_variance.txt\n  - interaction_variance.txt\ninto",
        os.path.abspath(args.output_dir),
    )


if __name__ == "__main__":
    main()

