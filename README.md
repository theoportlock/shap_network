
# SHAP Interaction Benchmarking Toolkit

This repository provides a complete pipeline to **compare theoretical vs. model-estimated feature and interaction importances** using SHAP (SHapley Additive exPlanations). It is designed to evaluate how well machine learning models, particularly Random Forests, capture known effects in synthetic datasets generated from symbolic formulas.

## ✨ Features

- Generate toy datasets from user-defined mathematical formulas.
- Compute **theoretical variance explained** by individual features and pairwise interactions.
- Train and evaluate a **random forest model**.
- Compute SHAP values and SHAP interaction values.
- Merge and compare theoretical vs. empirical importances.
- Visualize results with scatter plots and network graphs.

## 🗂 Project Structure

```
.
├── all.sh                       # Master script to run the full pipeline
├── code/                        # Utility scripts for formatting and plotting
├── conf/formula.txt             # Input symbolic formula
├── metatoolkit/                 # Python package used in the pipeline
├── results/                     # Output folder for data, plots, and networks
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🧠 Concept

You define a symbolic formula such as:

```
y = x1 + x2 + x1*x2
```

The toolkit then:
1. Computes the **theoretical variance explained** by each feature and interaction.
2. Simulates data under the assumption of independent inputs.
3. Fits a Random Forest regressor.
4. Uses SHAP to extract feature and interaction importances.
5. Compares theoretical vs. SHAP-derived importances.
6. Generates plots and a network visualization.

## ⚙️ Usage

### 1. Create and activate your virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Define your formula

Edit `conf/formula.txt` with a symbolic expression like:
```
y = x1 + x2**2 + x1*x3
```

### 3. Run the full pipeline

```bash
bash all.sh
```

This will:
- Parse your formula
- Generate data
- Train a model
- Evaluate model performance
- Compute and plot SHAP values
- Compare theoretical and empirical importances
- Visualize the interaction network

### 4. View Results

Output files are written to the `results/` folder:
- **Variance reports**: theoretical and SHAP-based
- **SHAP plots**: summary, dependence, and interaction plots
- **PDF and SVG** visualizations of comparisons
- **GraphML** network of interactions

## 📊 Example Output

- `results/plot.pdf` – Scatter plot comparing theoretical vs. SHAP importance
- `results/network2.svg` – Network of feature interactions
- `results/merged_interactions.tsv` – Merged table of theoretical and SHAP importances

## 🧪 Development Notes

- Core logic is implemented in `metatoolkit`, structured as a Python package
- Utility scripts in `code/` help with formatting, plotting, and layout
- Modular pipeline allows adaptation to other models or interaction methods

## 📝 License

This project is licensed under the MIT License. See [`metatoolkit/LICENSE`](metatoolkit/LICENSE) for details.

## 👤 Author

Theo Portlock  
Liggins Institute, University of Auckland
