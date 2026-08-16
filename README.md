# 🏦 Home Credit Default Risk — Credit Scoring Analysis

An end-to-end credit risk analysis and default prediction model built on the Home Credit Default Risk dataset (307,511 clients, 122 original features). The project moves beyond a standard classification exercise into a production-oriented workflow: statistical validation of findings, model stability checks, probability calibration, model interpretability, and a business-facing profit optimization.

## 📋 Executive Summary

**Problem:** predict the probability that a loan applicant will default, and translate that prediction into an actionable, profit-maximizing lending threshold.

**Key EDA findings:**
- Severe class imbalance: 91.9% reliable clients vs. 8.1% defaulters — shaped every downstream modeling decision (metric choice, class weighting, evaluation strategy)
- Found and corrected a systemic data anomaly: `DAYS_EMPLOYED` contained a masked placeholder value (365,243 days ≈ 1,000 years) affecting 18% of clients
- **The correlation trap:** standard Pearson correlation understated the predictive power of the `EXT_SOURCE` credit scores (r ≈ -0.15 to -0.18) due to class imbalance. Effect size analysis via Cohen's d revealed a much stronger true relationship (d = 0.60–0.68) — a reminder that correlation alone can mislead on imbalanced targets
- Education level is a strong risk signal: default rate ranges from 1.8% (academic degree) to 10.9% (lower secondary education)

**Model:** LightGBM classifier with balanced class weighting, trained on a proper three-way train/validation/test split (early stopping on validation, final evaluation on a held-out test set).
- **ROC-AUC (test set): 0.762**

**Stability check (Cross-Validation):** a single train/test split can produce a misleadingly optimistic or pessimistic score. To confirm the result wasn't an artifact of one particular split, the model was re-evaluated using Stratified 5-Fold Cross-Validation.
- **Mean ROC-AUC: 0.7562 ± 0.004** across folds — confirms the model's performance is stable, not a product of a lucky split

**Calibration:** raw model probabilities were poorly calibrated (Brier Score = 0.189) as a side effect of class balancing. Applying isotonic regression improved calibration substantially:
- **Brier Score: 0.189 → 0.068**

**Business translation (Profit Curve):** calibrated probabilities were used to simulate lending outcomes across a range of risk thresholds, converting model output into expected profit in tenge.
- **Optimal risk cutoff: 18.18%**
- **Expected profit at optimal threshold: 7.18 billion KZT on the held-out test set**

**Interpretability:** SHAP values used to identify which features drive risk both globally (across the population) and locally (for individual applicants).

## 🔍 Methodology

The analysis follows a deliberate, defensible sequence:

1. **Target & missing value analysis** — quantify class imbalance; drop features with >60% missingness while preserving high-signal sparse features (e.g., `EXT_SOURCE_1` at 56% missing)
2. **Anomaly detection** — percentile-based scan across all numeric columns to catch masked placeholders and extreme outliers, replacing them with `NaN` rather than dropping rows outright
3. **Statistical validation** — t-test + Cohen's d for numeric predictors (age), chi-square + Cramér's V for categorical predictors (gender), always pairing significance with effect size rather than relying on p-values alone at large sample sizes
4. **Feature engineering** — debt-to-income and annuity-to-income ratios, employment-to-age ratio; raw financial columns showed weak separation between classes, engineered ratios carry more signal
5. **Baseline model** — LightGBM with `class_weight='balanced'`, three-way split (train 65% / validation 16% / test 20%) to keep early stopping and final evaluation fully separated
6. **Stability check** — Stratified 5-Fold Cross-Validation on the train+validation portion to confirm the reported ROC-AUC generalizes beyond a single split, rather than trusting one train/test partition alone
7. **Calibration** — Brier score + reliability curve diagnosis, isotonic regression fit on the validation set only
8. **Interpretability** — SHAP TreeExplainer for global and local feature attribution
9. **Profit curve** — calibrated probabilities converted into a threshold-vs-profit simulation under explicit business assumptions (average loan size, interest margin, loss-given-default)


## ⚠️ Known Limitations
- One-Hot Encoding is applied before the train/test split for baseline simplicity. 
  In a production pipeline, this would be wrapped in `sklearn.pipeline.Pipeline` 
  with `fit` on train and `transform` on test to prevent data leakage.
- Profit Curve uses an average loan amount; individual `AMT_CREDIT` per applicant 
  would give a more precise business estimate.
- No hyperparameter tuning (Optuna / Grid Search) — planned for v2.

  
## 📊 Results at a Glance

| Metric | Value |
|---|---|
| ROC-AUC (test set) | 0.762 |
| ROC-AUC (5-fold CV, mean ± std) | 0.762 ± 0.004 |
| Brier Score (uncalibrated) | 0.189 |
| Brier Score (calibrated, isotonic) | 0.068 |
| Optimal decision threshold | 18.18% predicted risk |
| Strongest predictors | EXT_SOURCE_1/2/3, age, credit-to-income ratio |

## 🛠️ Tech Stack

- **Data manipulation:** Pandas, NumPy
- **Statistics:** SciPy (t-test, chi-square), custom Cohen's d / Cramér's V implementations
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** LightGBM, scikit-learn (`StratifiedKFold` for cross-validation)
- **Calibration:** scikit-learn (`CalibratedClassifierCV`, isotonic regression)
- **Interpretability:** SHAP


## 🚀 Running the Project

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the [Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk/data) from Kaggle (`application_train.csv`) and place it in the project root
4. Open and run `notebooks/home_credit_default_risk_analysis.ipynb`

## 📌 Data Source

This project uses the publicly available [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset from Kaggle. Home Credit is a consumer lender operating across several markets, including Kazakhstan. While the dataset is global, the modeling approach (probability calibration, profit curve in KZT, regulatory-style validation) mirrors the workflow used in Kazakhstani retail banking.

## 👤 Aldiyar Zhakupov

Data Science student, Corvinus University of Budapest — building this as part of a portfolio focused on financial risk modeling for the Kazakhstani market.

---

*This is the second project in a portfolio series; the first is an exploratory and statistical analysis of the Astana real estate market.*
