#!/usr/bin/env python3
import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Define causal edges based on the new data generation process
edges = [
    ("Temperature", "IceCreamSales"),
    ("MarketingSpend", "IceCreamSales"),
    ("Temperature × MarketingSpend", "IceCreamSales"),
    ("WindSpeed", "Ignored"),
]

G.add_edges_from(edges)

# Define positions manually for a clean layout
pos = {
    "Temperature": (-2, 1),
    "MarketingSpend": (2, 1),
    "WindSpeed": (0, 2),
    "Temperature × MarketingSpend": (0, 0),
    "Ignored": (0, 1.2),
    "IceCreamSales": (0, -2)
}

# Draw graph
plt.figure(figsize=(10, 6))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=600)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

# Draw edges
nx.draw_networkx_edges(
    G, pos,
    arrows=True,
    arrowstyle='-|>',
    arrowsize=20,
    edge_color='black',
    connectionstyle='arc3,rad=0.1'
)

# Title and layout
plt.title("Causal DAG: Simulated Ice Cream Sales with Interaction")
plt.axis('off')
plt.tight_layout()

# Save
plt.savefig('results/dataset_dag.svg')
print("✅ DAG saved to results/dataset_dag.svg")

