#!/usr/bin/env python

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('results/format_merged_interactions.tsv', sep='\t', index_col=[0, 1])

# Create figure and axes
fig, ax = plt.subplots(figsize=(4, 4))

# Scatterplot with identity line and annotations
x_vals = df['interaction_importance.tsv']
y_vals = df['mean_abs_shap_interaction_test.tsv']
ax.scatter(x_vals, y_vals, s=5, alpha=0.8)

# Add identity line
lims = [-0.1, 1]
ax.plot(lims, lims, 'k--', linewidth=0.4, label="y = x")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Theoretical Interaction Importance")
ax.set_ylabel("SHAP Interaction Importance")
ax.legend()

# Annotate points with (source, target)
for (source, target), row in df.iterrows():
    label = f"{source},{target}"
    ax.annotate(label, (row['interaction_importance.tsv'], row['mean_abs_shap_interaction_test.tsv']),
                   fontsize=6, alpha=0.6)

# Save and clear
plt.tight_layout()
plt.savefig('results/plot.pdf')
plt.clf()

