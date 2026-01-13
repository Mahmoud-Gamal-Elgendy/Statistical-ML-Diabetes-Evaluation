"""
SVM Hyperparameter Fine-Tuning - INDEPENDENT EXPERIMENT
Uses Stratified Sampling + RandomizedSearchCV
- This is a standalone experiment and does NOT affect the main project
- Uses stratified sampling (15%) to speed up SVM training
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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


def prepare_data(dataset, target_column='Diabetes_012', test_size=0.2, sample_size=0.05, random_state=42):
    """
    Prepare training and test sets with stratified sampling for SVM
    SVM is O(n^2) to O(n^3), so we use stratified sampling for speed
    Using only 5% sample for very fast tuning
    """
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]
    
    # First split: get test set from full data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: stratified sample of training data for tuning
    _, X_train_sample, _, y_train_sample = train_test_split(
        X_temp, y_temp, test_size=sample_size, random_state=random_state, stratify=y_temp
    )
    
    print(f"\n📊 Data Sampling Strategy:")
    print(f"  Original dataset: {len(dataset)} samples")
    print(f"  Test set: {len(X_test)} samples ({test_size*100:.0f}%)")
    print(f"  Training sample for tuning: {len(X_train_sample)} samples (~{sample_size*100:.0f}% of train)")
    print(f"  Speedup: ~{(len(X_temp) / len(X_train_sample)):.1f}x faster")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sample)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train_sample, y_test, scaler


def create_param_grid():
    """Define hyperparameter grid for SVM (very small for speed)"""
    param_grid = {
        'C': [1, 10],
        'kernel': ['rbf'],
        'gamma': ['scale'],
        'class_weight': [None, 'balanced']
    }
    return param_grid


def run_grid_search(X_train, y_train, param_grid, n_folds=2, n_jobs=-1, verbose=2):
    """Perform grid search with cross-validation"""
    print("\n" + "="*80)
    print("SVM HYPERPARAMETER TUNING - FAST MODE")
    print("="*80)
    print("\nTechnique: Stratified Sampling (5%) + GridSearchCV")
    print("  - Uses only 5% stratified sample for very fast tuning")
    print("  - Small parameter grid for quick results")
    print(f"  - Using {n_folds}-fold CV for speed")
    
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    
    print(f"\nTotal combinations to test: {total_combinations}")
    
    svm_base = SVC(random_state=42, cache_size=2000, max_iter=2000)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=svm_base,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    print(f"\nTesting {total_combinations} combinations with {n_folds}-fold CV...")
    print("Should complete in 2-5 minutes...")
    grid_search.fit(X_train, y_train)
    
    return grid_search


def evaluate_best_model(grid_search, X_test, y_test):
    """Evaluate the best model on test set"""
    print("\n" + "="*80)
    print("BEST MODEL EVALUATION")
    print("="*80)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Accuracy (on 5% sample): {grid_search.best_score_:.6f}")
    print(f"Test Set Accuracy (full data): {test_accuracy:.6f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nNote: Model tuned on 5% sample for speed, evaluated on full test set")
    
    return best_model, test_accuracy


def save_results(grid_search, test_accuracy, output_file):
    """Save best hyperparameters to markdown file"""
    best_params = grid_search.best_params_
    cv_accuracy = grid_search.best_score_
    
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = "# Best Hyperparameters for Models\n\n"
        content += "This file contains the best hyperparameters found through fine-tuning.\n\n---\n\n"
    
    svm_section = "## SVM (Support Vector Machine)\n\n"
    svm_section += f"**Fine-Tuning Technique:** Stratified Sampling (5%) + GridSearchCV (Fast Mode)\n\n"
    svm_section += f"**Cross-Validation Accuracy (on 5% sample):** {cv_accuracy:.6f}\n\n"
    svm_section += f"**Test Set Accuracy (on full data):** {test_accuracy:.6f}\n\n"
    svm_section += "**Best Hyperparameters:**\n\n```python\n{\n"
    for key, value in best_params.items():
        if isinstance(value, str):
            svm_section += f"    '{key}': '{value}',\n"
        elif isinstance(value, float):
            svm_section += f"    '{key}': {value:.6f},\n"
        else:
            svm_section += f"    '{key}': {value},\n"
    svm_section += "    'random_state': 42\n}\n```\n\n---\n\n"
    
    if "## SVM" in content:
        start_idx = content.find("## SVM")
        next_section = content.find("##", start_idx + 1)
        if next_section != -1:
            content = content[:start_idx] + svm_section + content[next_section:]
        else:
            content = content[:start_idx] + svm_section
    else:
        content += svm_section
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✓ Results saved to: {output_file}")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("SVM FINE-TUNING - INDEPENDENT EXPERIMENT")
    print("="*80)
    print("\nNOTE: This does NOT affect the main project!")
    print("="*80)
    
    # Get script directory and construct absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    DATA_PATH = os.path.join(project_root, 'Datasets', 'Real', 'diabetes_012_health_indicators_BRFSS2015.csv')
    OUTPUT_FILE = os.path.join(project_root, 'best_hyperparameters.md')
    MODEL_SAVE_PATH = os.path.join(project_root, 'models', 'SVM', 'best_svm_finetuned.pkl')
    
    dataset = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler = prepare_data(dataset, sample_size=0.10)
    
    print(f"\nTraining sample: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    param_grid = create_param_grid()
    grid_search = run_grid_search(X_train, y_train, param_grid, n_folds=2)
    best_model, test_accuracy = evaluate_best_model(grid_search, X_test, y_test)
    save_results(grid_search, test_accuracy, OUTPUT_FILE)
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': scaler}, f)
    print(f"✓ Best model saved to: {MODEL_SAVE_PATH}")
    
    print("\n" + "="*80)
    print("ALL PARAMETER COMBINATIONS")
    print("="*80)
    results_df = pd.DataFrame(grid_search.cv_results_).sort_values('rank_test_score')
    
    for idx, row in results_df.iterrows():
        print(f"\nRank {int(row['rank_test_score'])}: CV Accuracy = {row['mean_test_score']:.6f}")
        print(f"  Parameters: {row['params']}")
    
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
