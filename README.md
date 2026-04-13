# Student Mental Health Analysis

## Overview

This project explores correlations between students' lifestyle habits and their mental health and academic performance. Two machine learning models are built to predict **CGPA** (academic performance) and **depression risk** based on lifestyle features such as sleep duration, study hours, social media usage, and physical activity.

## Dataset

- **Source:** `student_lifestyle_100k.csv`
- **Size:** 100,000 records, 11 features
- **Features:** Age, Gender, Department, CGPA, SleepDuration, StudyHours, SocialMediaHours, PhysicalActivity, StressLevel
- **Target variables:**
  - `CGPA` — continuous (regression task)
  - `Depression` — binary classification (depression risk)

## Models

| Model | Task | Algorithm | Key Metrics |
|-------|------|-----------|-------------|
| CGPA Predictor | Regression | LinearRegression (sklearn Pipeline) | R² ≈ 0.01, RMSE ≈ 0.53 |
| Depression Classifier | Classification | SGDClassifier (log_loss / Logistic Regression) | Accuracy ≈ 0.61, F1 ≈ 0.19, ROC-AUC ≈ 0.56 |

The classification model was tuned via a grid search over **learning rate** and **number of epochs**, with learning curves plotted to verify the absence of overfitting.

## Pipeline

1. **EDA** — distribution analysis, correlation heatmaps, box plots, outlier detection
2. **Feature Engineering** — derived features (`StudySleepRatio`, `SleepQualityFlag`)
3. **Preprocessing** — imputation, OneHotEncoding, StandardScaler via sklearn Pipelines
4. **Train/Val/Test Split** — 60/20/20 with stratification for classification
5. **Model Training** — Linear Regression + SGDClassifier with hyperparameter tuning
6. **Evaluation** — R², RMSE, Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, ROC/PR curves, residual analysis

## Technologies & Libraries

- **Python 3**
- **pandas** — data manipulation
- **NumPy** — numerical operations
- **matplotlib / seaborn** — data visualization
- **scikit-learn** — machine learning pipelines, metrics, model training

## Project Structure

```
├── notebook.ipynb          # Main analysis notebook
├── student_lifestyle_100k.csv  # Dataset
└── README.md               # This file
```

## Reproducibility

All steps are contained within `notebook.ipynb` and can be reproduced by running all cells sequentially. A fixed `RANDOM_STATE = 42` is used throughout for reproducibility.
