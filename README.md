# Machine Learning Algorithms from Scratch & Scikit-Learn Comparison

This repository contains a comprehensive suite of foundational Machine Learning algorithms implemented both **from scratch** using NumPy/Pandas and compared with industry-standard implementations in **scikit-learn**.

##  Key Features

- **Foundational Probability & Parameter Estimation**:
  - Maximum Likelihood Estimation (MLE) for Gaussian distributions on the **Iris** dataset.
  - MLE & Maximum A Posteriori (MAP) with Beta-Bernoulli priors on the **UCI Breast Cancer** dataset to assess estimator stability under small sample sizes.
- **Linear Regression (Ordinary Least Squares)**:
  - Custom OLS solver via the Normal Equation and Pseudoinverse.
  - Simple and Multivariate Linear Regression on the **Auto MPG** dataset.
- **Logistic Regression (Classification)**:
  - From-scratch vectorized Logistic Regression optimizer utilizing custom binary cross-entropy and numerically stable sigmoid functions.
- **Optimization Algorithms (Gradient Descent Variants)**:
  - Empirical comparison of **Batch Gradient Descent**, **Stochastic Gradient Descent (SGD)**, and **Mini-Batch Gradient Descent (sizes 16 & 32)**.
- **Non-Parametric Models**:
  - Custom **$k$-Nearest Neighbors ($k$-NN)** classifier supporting Manhattan and Euclidean distances with strict tie-breaking logic.
- **Decision Trees**:
  - Scikit-learn classification tree optimization, custom root split impurity reduction calculations (Entropy/Gini), and recursive extraction of human-readable decision rules.

---

##  Datasets Used
1. **Iris Dataset**: Species classification ($150$ samples, $4$ features).
2. **Auto MPG Dataset**: Vehicle fuel efficiency regression ($392$ samples, $5$ features).
3. **Breast Cancer Wisconsin Dataset**: Malignant/benign classification ($699$ samples, $9$ features).

---

##  Model Evaluation Summaries

### 1. Regression (Auto MPG Dataset)

| Model Config | Validation RMSE | Test RMSE | Validation $R^2$ |
| :--- | :---: | :---: | :---: |
| **Simple OLS (Weight Only)** | 4.1489 | 4.2630 | 0.6441 |
| **Multiple OLS (All Features)** | 3.0326 | 3.2390 | 0.8101 |

### 2. Optimization Comparison (Linear Regression)

- **Batch Gradient Descent** (lr=0.01): Test RMSE **3.4340**
- **Stochastic Gradient Descent (SGD)** (lr=0.0001): Test RMSE **3.3751**
- **Mini-Batch (Size 16)** (lr=0.01): Test RMSE **3.2469** (Fastest convergence to analytical Least Squares)
- **Mini-Batch (Size 32)** (lr=0.01): Test RMSE **3.2840**

### 3. Classification Metrics (Breast Cancer Dataset)

| Classifier Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Scratch)** | 94.28% | 87.03% | 97.91% | 92.15% | 0.981 |
| **Decision Tree (Entropy, Depth=3)** | 94.28% | 87.03% | 97.91% | 92.15% | -- |

### 4. $k$-Nearest Neighbors (Iris Dataset)
- Selected Optimal Parameters: $k=7$, Metric: **Euclidean**, Scaled: **False**.
- **Test Accuracy**: **96.67%** (Perfect parity achieved with `KNeighborsClassifier` from scikit-learn).

---

##  Structure & Implementation Highlights
```python
# Numerically stable Sigmoid definition implemented from scratch
def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1 + ez)
    return out
```

##  Setup & Requirements
Clone this repository and open the notebook in Google Colab or your local Jupyter environment. Install dependencies via:
```bash
pip install numpy pandas matplotlib scikit-learn scipy ucimlrepo
```
