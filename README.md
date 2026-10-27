# Lightweight Machine Learning Model for Early Diabetes Prediction Using Clinical Data

> **IT41033 – Nature Inspired Algorithms · Mini Project** · Horizon Campus · August 2026
> A leakage-safe, lightweight classification pipeline that predicts diabetes from routine clinical measurements, compressed to **~1.5 KB** with no loss in accuracy.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00)
![Status](https://img.shields.io/badge/Status-Educational%20prototype-lightgrey)

---

## Table of Contents

1. [Overview](#overview)
2. [Key Results](#key-results)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Model Comparison](#model-comparison)
6. [Lightweight Optimization](#lightweight-optimization)
7. [My Contribution](#my-contribution)
8. [Repository Structure](#repository-structure)
9. [How to Run](#how-to-run)
10. [Limitations](#limitations)
11. [Team](#team)
12. [References](#references)

---

## Overview

Diabetes affects hundreds of millions of adults worldwide, and early screening matters. Many published models chase maximum accuracy with heavy, complex architectures that are hard to deploy in small clinics, mobile apps or edge devices.

This project asks a different question: **how small and fast can a diabetes screening model be while staying useful?**

We built and compared four classifiers on the PIMA Indians Diabetes dataset, used feature selection to reduce the required clinical inputs from 8 to 5, and then compressed the best model into a deployable artifact of about 1.5 KB.

### Research Questions

| # | Question | Outcome |
|---|----------|---------|
| RQ1 | Which subset of the 8 clinical attributes gives the strongest predictive value with the fewest inputs? | **Pregnancies, Glucose, BMI, Diabetes Pedigree Function, Age** (5 of 8) |
| RQ2 | Can lightweight models reach ≥ 80% held-out accuracy and stay competitive with a Random Forest baseline? | **Partially.** Best held-out accuracy was 73.4%; Logistic Regression reached 78.5% mean 5-fold CV accuracy and matched or beat Random Forest on CV |
| RQ3 | How small and fast can the model get while losing < 3% accuracy? | **Yes.** 1.5176 KB, ~0.02 ms per patient, 0% accuracy loss |

---

## Key Results

**Selected model: Logistic Regression** (balanced class weights, L2 penalty, C = 0.01, liblinear solver)

| Metric | Value |
|--------|-------|
| Held-out accuracy | 72.73% |
| Recall (diabetic class) | **74.07%** (up from 50.00% before tuning) |
| F1-score | 65.57% |
| Test ROC-AUC | 0.8128 |
| 5-fold CV ROC-AUC | **0.8483** |
| Final model size | **1.5176 KB** (51.1% smaller after Float32 + compression) |
| Inference time | ~0.02 ms per patient |

Random Forest needed roughly **115× more storage** and **10.8× more inference time per patient** than Logistic Regression while delivering no better cross-validated performance.

---

## Dataset

**PIMA Indians Diabetes Dataset** – 768 patient records, 8 numeric predictors and 1 binary target.

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | Two-hour serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Family-history risk score |
| Age | Age in years |
| **Outcome** | **0 = no diabetes, 1 = diabetes** (target) |

**Data quality findings**

- No standard NaN values and no duplicate rows.
- **652 clinically invalid zero values** hidden in Glucose (5), BloodPressure (35), SkinThickness (227), Insulin (374) and BMI (11).
- Moderate class imbalance: 500 non-diabetic (65.1%) vs 268 diabetic (34.9%).
- Glucose had the strongest correlation with Outcome (0.47), followed by BMI, Age and Pregnancies.

---

## Methodology

```
Raw data → Invalid zeros → NaN → Stratified 80/20 split → Median imputation
        → StandardScaler → RFE (8 → 5 features) → Model training
        → 5-fold stratified CV → GridSearchCV (ROC-AUC) → Size / latency benchmark
        → Float32 quantization + joblib compression → Single-patient demo
```

**Leakage control.** Imputation, scaling and RFE are fitted on training data only. During cross-validation and grid search they sit inside a scikit-learn `Pipeline`, so every fold learns its own transformations and no test information leaks into training.

**Models compared**

| Model | Role | Features |
|-------|------|----------|
| Logistic Regression | Lightweight candidate | 5 (RFE) |
| Decision Tree | Lightweight candidate | 5 (RFE) |
| Gaussian Naive Bayes | Lightweight candidate | 5 (RFE) |
| Random Forest (100 trees) | Heavy baseline | all 8 |

**Evaluation:** accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, 5-fold stratified cross-validation, serialized model size and per-patient latency.

---

## Model Comparison

### Tuned models on the held-out test set (154 patients)

| Model | Train Acc. | Test Acc. | Precision | Recall | F1 | Test AUC | Best CV AUC |
|-------|-----------|-----------|-----------|--------|----|----------|-------------|
| **Logistic Regression** | 0.7524 | 0.7273 | 0.5882 | **0.7407** | **0.6557** | 0.8128 | **0.8483** |
| Gaussian Naive Bayes | 0.7834 | 0.7338 | 0.6226 | 0.6111 | 0.6168 | 0.8017 | 0.8427 |
| Random Forest | 0.8371 | 0.7338 | 0.6585 | 0.5000 | 0.5684 | 0.8031 | 0.8422 |
| Decision Tree | 0.7280 | 0.6818 | 0.5287 | 0.8519 | 0.6525 | 0.7840 | 0.7963 |

### Five-fold stratified cross-validation (before tuning)

| Model | CV Accuracy | CV F1 | CV ROC-AUC | Accuracy Std. Dev. |
|-------|-------------|-------|------------|--------------------|
| Logistic Regression | 0.7850 | 0.6512 | 0.8445 | 0.0189 |
| Gaussian Naive Bayes | 0.7773 | 0.6482 | 0.8427 | 0.0230 |
| Random Forest | 0.7606 | 0.6258 | 0.8163 | 0.0188 |
| Decision Tree | 0.6499 | 0.5200 | 0.6259 | 0.0343 |

**Takeaways**

- The unpruned Decision Tree and Random Forest both hit 100% training accuracy, a clear sign of **overfitting**. Reducing tree depth from 15 to 3 closed the gap.
- Logistic Regression gave the best balance of discrimination, recall, stability and resource use.
- Balanced class weights lifted recall from 50% to 74% at the cost of precision (58.8%), a deliberate screening trade-off: fewer missed cases, more false alarms.

---

## Lightweight Optimization

| Version | Size (KB) | Accuracy | ROC-AUC | Time / Patient (ms) |
|---------|-----------|----------|---------|---------------------|
| Original (Float64) | 3.1064 | 0.7273 | 0.8128 | 0.0179 |
| Quantized (Float32) | 2.9658 | 0.7273 | 0.8128 | 0.0200 |
| Float32 + compression | **1.5176** | 0.7273 | 0.8128 | 0.0200 |

Converting parameters from Float64 to Float32 and compressing with joblib cut the file size by **51.1%** with **0% change** in accuracy and ROC-AUC.

**Serialized size and speed by model**

| Model | Size (KB) | Time / Patient (ms) |
|-------|-----------|---------------------|
| Logistic Regression | 3.1064 | 0.0179 |
| Gaussian Naive Bayes | 3.3369 | 0.0187 |
| Decision Tree | 4.8545 | 0.0181 |
| Random Forest | 356.4082 | 0.1932 |

### Sample prediction

The final pipeline accepts all 8 raw clinical inputs, applies imputation, scaling and feature selection internally, and returns a class and probability. For a demo patient (6 pregnancies, glucose 148, BP 72, skin thickness 35, insulin 0 → treated as missing, BMI 33.6, pedigree 0.627, age 50), it predicts **class 1 (diabetic) with 71.39% probability**.

---

## My Contribution

**Member 3 – E.N.S. Wijerathne**

I owned two parts of the shared pipeline:

**1. Data preprocessing and feature selection**
- Detected and converted the 652 invalid zero measurements to missing values.
- Built the stratified 80/20 train/test split (614 / 154 patients) that preserves class proportions.
- Implemented training-only median imputation and `StandardScaler` standardization to prevent data leakage.
- Applied Recursive Feature Elimination (Logistic Regression estimator) to reduce the inputs from 8 to 5 features, consistent with the correlation analysis.

**2. Gaussian Naive Bayes model**
- Trained the initial model (78.34% train / 73.38% test accuracy).
- Tuned it with `GridSearchCV` over `var_smoothing` and the number of RFE-selected features, using 5-fold stratified CV and ROC-AUC scoring.
- Result: CV ROC-AUC 0.8427, test accuracy 73.38%, test ROC-AUC 0.8017, 3.3 KB model, with the smallest train-test gap among the lightweight models.

The rest of the pipeline was built by the other team members (EDA, Logistic Regression and tuning, Decision Tree and tuning, Random Forest baseline, model evaluation and comparison). See [Team](#team).

---

## Repository Structure

```
.
├── 01_Dataset/        # PIMA Indians Diabetes dataset (diabetes.csv)
├── 02_Notebooks/      # Google Colab / Jupyter notebooks
├── 03_Report/         # Full project report (PDF)
└── README.md
```

---

## How to Run

### Option 1: Google Colab (recommended)

1. Open the notebook from `02_Notebooks/` in [Google Colab](https://colab.research.google.com/).
2. Upload `diabetes.csv` from `01_Dataset/` to the Colab session (the notebook reads `/content/diabetes.csv`).
3. Run all cells from top to bottom (**Runtime → Run all**).

### Option 2: Run locally

```bash
git clone https://github.com/n-wijerathne/Lightweight-Machine-Learning-Model-for-Early-Diabetes-Prediction-Using-Clinical-Data.git
cd Lightweight-Machine-Learning-Model-for-Early-Diabetes-Prediction-Using-Clinical-Data

pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
jupyter notebook 02_Notebooks/
```

If running locally, change the data path in the notebook from `/content/diabetes.csv` to `../01_Dataset/diabetes.csv`.

### Tech stack

Python · pandas · NumPy · scikit-learn (Pipeline, RFE, GridSearchCV) · Matplotlib · Seaborn · joblib · Google Colab

---

## Limitations

This is an **educational prototype, not a diagnostic tool.**

- Only 768 records from one specific population; results may not generalize to other populations, including Sri Lankan patients.
- The 80% held-out accuracy target was not reached (best: 73.4%), and the test set has only 154 records, so variance is high.
- Median imputation can hide clinically meaningful missingness, especially for Insulin and SkinThickness.
- Probabilities are not calibrated, performance was not checked across demographic subgroups, and no external validation dataset was used.
- Latency was measured in Google Colab on a small batch and should be re-measured on the target device.

**Future work:** validate on larger, locally relevant clinical data; calibrate probabilities and tune decision thresholds with clinicians; add confidence intervals and subgroup analysis; use nested cross-validation.

---

## Team

| Member | Main responsibilities |
|--------|----------------------|
| R.G.S.M. Rathwella | Model training, hyperparameter tuning, final model comparison |
| R.G.G.M.N. Rajapaksha | Data exploration, Decision Tree and tuning |
| **E.S. Wijerathne** | **Data preprocessing, feature selection, Gaussian Naive Bayes and tuning** |
| D.B. Senarathna | Random Forest baseline, model evaluation and comparison |

Full methodology, tables and discussion are in the [project report](03_Report/).

---

## References

1. International Diabetes Federation, *IDF Diabetes Atlas, 11th edition*, 2025.
2. J. W. Smith et al., "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus," *Proc. Annual Symposium on Computer Application in Medical Care*, 1988.
3. I. Guyon, J. Weston, S. Barnhill and V. Vapnik, "Gene selection for cancer classification using support vector machines," *Machine Learning*, vol. 46, 2002.
4. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, 2011.
5. T. Fawcett, "An introduction to ROC analysis," *Pattern Recognition Letters*, vol. 27, no. 8, 2006.
