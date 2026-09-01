# Financial Risk Prediction

An end-to-end machine learning pipeline for predicting consumer credit default risk. The project covers exploratory data analysis, feature engineering, class imbalance handling, model training, evaluation, explainability, and a business-facing credit scorecard.

## Overview

Financial institutions rely on predictive models to assess the probability that a borrower will default on credit obligations. This project implements a complete credit risk scoring pipeline, from raw tabular data to a deployable, interpretable model.

## Features

- Exploratory data analysis with distribution, correlation, and missing-value diagnostics
- Robust data cleaning and outlier treatment
- Domain-driven feature engineering (delinquency aggregates, income ratios, utilization buckets)
- Class imbalance handling using SMOTE
- Multiple model implementations: Logistic Regression, Random Forest, XGBoost, LightGBM
- Model evaluation via ROC-AUC, Precision-Recall, KS statistic, and confusion matrix
- Feature importance analysis
- Model explainability using SHAP
- Probability-to-credit-score conversion using the industry-standard PDO scaling method

## Project Structure

```
.
├── Financial_Risk_Prediction.ipynb   # Main notebook
├── README.md
└── requirements.txt
```

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
joblib
```

## Installation

```bash
git clone <repository-url>
cd financial-risk-prediction
pip install -r requirements.txt
```

## Usage

1. Place your credit risk dataset (CSV format) in the project directory.
2. Update the file path in the data loading cell of the notebook.
3. Run the notebook cells sequentially.

Expected input schema: a binary target column indicating default status, along with standard credit bureau features such as credit utilization, age, debt ratio, monthly income, number of open credit lines, delinquency history, and number of dependents.

## Methodology

1. **Data Cleaning** — Missing value imputation, outlier capping, and invalid record removal.
2. **Feature Engineering** — Aggregated delinquency counts, income-adjusted ratios, and utilization segmentation.
3. **Resampling** — SMOTE oversampling applied to the training set to address class imbalance.
4. **Modeling** — Comparative training across four classifiers with consistent train/test splits.
5. **Evaluation** — ROC-AUC and average precision for ranking performance; KS statistic for separation quality; confusion matrix and classification report at the chosen decision threshold.
6. **Explainability** — SHAP summary plots for global and local feature contribution analysis.
7. **Scorecard Conversion** — Predicted probabilities mapped to a 300–850 credit score scale.

## Model Output

The final trained model and feature scaler are serialized with `joblib` for downstream deployment or batch scoring.

## License

This project is released under the MIT License.
