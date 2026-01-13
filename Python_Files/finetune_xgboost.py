"""
XGBoost Hyperparameter Fine-Tuning - INDEPENDENT EXPERIMENT
Uses RandomizedSearchCV + Early Stopping
- This is a standalone experiment and does NOT affect the main project
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
import pickle
import warnings
warnings.filterwarnings('ignore')


def load_data(data_path):
    """Load and prepare dataset"""
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    data = data.drop_duplicates()
    print(f"✓ Loaded {len(data)} samples, {data.shape[1]} features")
    return data


def prepare_data(dataset, target_column='Diabetes_012', test_size=0.2, random_state=42):
    """Prepare training and test sets with scaling"""
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def create_param_distributions():
    """Define hyperparameter distributions for XGBoost (simplified)"""
    param_distributions = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    return param_distributions


def run_randomized_search(X_train, y_train, param_distributions, n_iter=100, n_folds=5, n_jobs=-1, verbose=2):
    """Perform randomized search with cross-validation and early stopping"""
    print("\n" + "="*80)
    print("XGBOOST HYPERPARAMETER TUNING - RANDOMIZED SEARCH + EARLY STOPPING")
    print("="*80)
    print("\nTechnique: RandomizedSearchCV with Early Stopping")
    print("  - Efficient parameter exploration")
    print("  - Early stopping prevents overfitting")
    
    xgb_base = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        n_jobs=n_jobs,
        verbosity=0
    )
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=42,
        return_train_score=True
    )
    
    print(f"\nTesting {n_iter} random parameter combinations with {n_folds}-fold CV...")
    random_search.fit(X_train, y_train)
    
    return random_search


def evaluate_best_model(random_search, X_test, y_test):
    """Evaluate the best model on test set"""
    print("\n" + "="*80)
    print("BEST MODEL EVALUATION")
    print("="*80)
    
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nBest Parameters: {random_search.best_params_}")
    print(f"Best CV Accuracy: {random_search.best_score_:.6f}")
    print(f"Test Set Accuracy: {test_accuracy:.6f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return best_model, test_accuracy


def save_results(random_search, test_accuracy, output_file):
    """Save best hyperparameters to markdown file"""
    best_params = random_search.best_params_
    cv_accuracy = random_search.best_score_
    
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = "# Best Hyperparameters for Models\n\n"
        content += "This file contains the best hyperparameters found through fine-tuning.\n\n---\n\n"
    
    xgb_section = "## XGBoost\n\n"
    xgb_section += f"**Fine-Tuning Technique:** RandomizedSearchCV (20 iterations, 3-fold CV)\n\n"
    xgb_section += f"**Cross-Validation Accuracy:** {cv_accuracy:.6f}\n\n"
    xgb_section += f"**Test Set Accuracy:** {test_accuracy:.6f}\n\n"
    xgb_section += "**Best Hyperparameters:**\n\n```python\n{\n"
    for key, value in best_params.items():
        if isinstance(value, str):
            xgb_section += f"    '{key}': '{value}',\n"
        else:
            xgb_section += f"    '{key}': {value},\n"
    xgb_section += "    'objective': 'multi:softmax',\n"
    xgb_section += "    'num_class': 3,\n"
    xgb_section += "    'random_state': 42,\n"
    xgb_section += "    'n_jobs': -1,\n"
    xgb_section += "    'verbosity': 0\n}\n```\n\n---\n\n"
    
    if "## XGBoost" in content:
        start_idx = content.find("## XGBoost")
        next_section = content.find("##", start_idx + 1)
        if next_section != -1:
            content = content[:start_idx] + xgb_section + content[next_section:]
        else:
            content = content[:start_idx] + xgb_section
    else:
        content += xgb_section
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✓ Results saved to: {output_file}")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("XGBOOST FINE-TUNING - INDEPENDENT EXPERIMENT")
    print("="*80)
    print("\nNOTE: This does NOT affect the main project!")
    print("="*80)
    
    # Get script directory and construct absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    DATA_PATH = os.path.join(project_root, 'Datasets', 'Real', 'diabetes_012_health_indicators_BRFSS2015.csv')
    OUTPUT_FILE = os.path.join(project_root, 'best_hyperparameters.md')
    MODEL_SAVE_PATH = os.path.join(project_root, 'models', 'XGBoost', 'best_xgb_finetuned.pkl')
    
    dataset = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler = prepare_data(dataset)
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    param_distributions = create_param_distributions()
    random_search = run_randomized_search(X_train, y_train, param_distributions, n_iter=20, n_folds=3)
    best_model, test_accuracy = evaluate_best_model(random_search, X_test, y_test)
    save_results(random_search, test_accuracy, OUTPUT_FILE)
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': scaler}, f)
    print(f"✓ Best model saved to: {MODEL_SAVE_PATH}")
    
    print("\n" + "="*80)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("="*80)
    results_df = pd.DataFrame(random_search.cv_results_).sort_values('rank_test_score')
    
    for idx, row in results_df.head(10).iterrows():
        print(f"\nRank {int(row['rank_test_score'])}: CV Accuracy = {row['mean_test_score']:.6f}")
        print(f"  Parameters: {row['params']}")
    
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
