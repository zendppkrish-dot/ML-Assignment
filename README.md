# Machine Learning — Practical Assignment 1

A practical implementation of seven machine learning assignments using the Iris, Auto MPG, and Breast Cancer Wisconsin datasets.

## 1. Assignment Overview

This project demonstrates:

- Maximum Likelihood Estimation
- Maximum A Posteriori Estimation
- Linear regression
- Logistic regression
- Gradient descent variants
- K-nearest neighbours
- Decision trees

The implementation uses NumPy, pandas, Matplotlib, SciPy, scikit-learn, and the UCI Machine Learning Repository.

---

## 2. Dataset Information

| Dataset | UCI ID | Task |
|---|---:|---|
| Iris | 53 | Multi-class classification |
| Auto MPG | 9 | Regression |
| Breast Cancer Wisconsin (Diagnostic) | 17 | Binary classification |

The cancer target is encoded as:

- `1` — Malignant
- `0` — Benign

Identifier columns are removed from the predictor sets.

---

## 3. Development Environment

Required Python packages:

```bash
pip install ucimlrepo numpy pandas matplotlib scikit-learn scipy
