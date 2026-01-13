# Best Hyperparameters for Models

This file contains the best hyperparameters found through optimized fine-tuning techniques.

---

## Fine-Tuning Techniques Used

### Random Forest

**Fine-Tuning Technique:** RandomizedSearchCV (20 iterations, 3-fold CV)

**Cross-Validation Accuracy:** 0.834189

**Test Set Accuracy:** 0.835607

**Best Hyperparameters:**

```python
{
    'n_estimators': 100,
    'min_samples_split': 10,
    'min_samples_leaf': 2,
    'max_features': 'log2',
    'max_depth': 30,
    'class_weight': None,
    'random_state': 42,
    'n_jobs': -1
}
```

---



## Instructions

Run the following scripts to fine-tune each model and populate this file:

1. **Random Forest**: `python Python_Files/finetune_randomforest.py`
2. **XGBoost**: `python Python_Files/finetune_xgboost.py`
3. **SVM**: `python Python_Files/finetune_svm.py`

Each script will:
- Load the real diabetes dataset
- Apply appropriate optimization technique
- Perform cross-validation
- Evaluate the best model on the test set
- Save the best hyperparameters to this file
- Save the trained model to the models directory

---

## Model Results

Results will appear here after running the fine-tuning scripts.

## XGBoost

**Fine-Tuning Technique:** RandomizedSearchCV (20 iterations, 3-fold CV)

**Cross-Validation Accuracy:** 0.835032

**Test Set Accuracy:** 0.837892

**Best Hyperparameters:**

```python
{
    'subsample': 1.0,
    'n_estimators': 100,
    'min_child_weight': 1,
    'max_depth': 5,
    'learning_rate': 0.1,
    'colsample_bytree': 1.0,
    'objective': 'multi:softmax',
    'num_class': 3,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}
```

---

## SVM (Support Vector Machine)

**Fine-Tuning Technique:** Stratified Sampling (5%) + GridSearchCV (Fast Mode)

**Cross-Validation Accuracy (on 5% sample):** 0.829734

**Test Set Accuracy (on full data):** 0.800030

**Best Hyperparameters:**

```python
{
    'C': 1,
    'class_weight': None,
    'gamma': 'scale',
    'kernel': 'rbf',
    'random_state': 42
}
```

---

