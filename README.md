# Predicting Telco Customer Churn

A classification project that predicts whether a telecom customer will churn based on demographic, account, and service-usage data. The project compares imbalance-handling strategies (SMOTE vs. class weighting) across 11 classification algorithms, then hyperparameter-tunes the top performers to select a final model optimized for recall on the churn class.

## Highlights

* Exploratory data analysis of customer demographics, account details, and subscribed services
* Data cleaning: correcting the `TotalCharges` data type, and standardizing inconsistent categorical entries (e.g. mixed casing/spelling in `gender`, `Partner`, `InternetService`, `Contract`)
* Removal of `tenure = 0` records (churn status undefined at signup) and targeted imputation of `MonthlyCharges` using `TotalCharges` for first-month customers
* Statistical association analysis using the Mann–Whitney U test (numerical features) and Cramér's V (categorical features) against the target variable
* Feature selection: dropped `customerID` and `gender` (no predictive value) and `TotalCharges` (multicollinear with `tenure`)
* Univariate, bivariate, and multivariate analysis of churn drivers (tenure, contract type, payment method, add-on services, senior citizen status)
* Encoding via Label Encoding (binary features) and One-Hot Encoding (nominal features), followed by Robust Scaling and PCA (2 components, ~99% explained variance)
* Comparison of class-imbalance handling strategies: SMOTE vs. class-weighting parameters, across 11 classifiers (Logistic Regression, SVC, Decision Tree, Naive Bayes, KNN, Random Forest, CatBoost, XGBoost, LightGBM, AdaBoost, Gradient Boosting)
* Hyperparameter tuning (GridSearchCV, optimized for recall) on the four best-performing models
* Evaluation focused on churn-class recall and non-churn-class precision, reflecting the real-world cost of missed churners

## Data

The dataset (Kaggle Telco Customer Churn) contains 7,043 customer records and 21 columns, including:

* Demographics (gender, senior citizen status, partner, dependents)
* Account information (tenure, contract type, paperless billing, payment method)
* Subscribed services (phone, multiple lines, internet type, online security/backup, device protection, tech support, streaming TV/movies)
* Billing (monthly charges, total charges)
* `Churn` as the target variable (Yes/No)

## Model Experiments

**Imbalance handling comparison:** All 11 classifiers were trained twice — once on SMOTE-resampled training data, and once using each model's native class-weighting mechanism (`class_weight='balanced'`, `scale_pos_weight`, or sample weights where applicable). SMOTE consistently produced higher non-churn precision and higher churn recall across models, making it the preferred imbalance-handling approach for this dataset.

**Hyperparameter tuning:** Based on the SMOTE comparison, the four strongest models — SVC, Naive Bayes, AdaBoost, and Logistic Regression — were tuned via `GridSearchCV` optimizing for recall (the priority metric for catching potential churners). Tuning did not meaningfully improve SVC, Naive Bayes, or AdaBoost, suggesting these were already near-optimal or insensitive to further tuning on this dataset. Logistic Regression, however, benefited clearly from tuning.

## Results

**Logistic Regression (tuned)** was selected as the final model. The strongly regularized, pure-L1 configuration (C=0.001) effectively performs feature selection, reducing overfitting and improving generalization while achieving the highest churn recall (0.86) among all tuned models — the most important metric for this business problem, since missing an actual churner carries a higher cost than a false alarm.

## How to Run

The notebook was developed using Python and Jupyter Notebook.

1. Clone this repository:

```
git clone https://github.com/feliceeeee/Telco_Customer_Churn.git
```

2. Install the required libraries:

```
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn xgboost lightgbm catboost jupyter
```

3. Ensure the dataset is located at: `data/Telco-Customer-Churn.csv`
4. Open the notebook: `notebook/Telco Customer Churn.ipynb`
5. Run all cells to perform data preprocessing, exploratory data analysis, model comparison (SMOTE vs. class weighting), hyperparameter tuning, and final model selection.
