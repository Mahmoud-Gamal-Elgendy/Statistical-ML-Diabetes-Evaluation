"""
Model Training Pipeline Module
Automated training pipeline for multiple datasets, models, and parameters
"""

import os
import pickle
import time
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


def get_default_parameters():
    """
    Get default parameter sets for each model group
    
    Returns:
    --------
    dict : Dictionary containing parameter sets for RF, SVM, and XGBoost
    """
    RF_params = {
        'P1': {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 5, 'random_state': 42},
        'P2': {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 10, 'random_state': 42},
        'P3': {'n_estimators': 150, 'max_depth': 30, 'min_samples_split': 15, 'random_state': 42},
        'P4': {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 20, 'random_state': 42}
    }
    
    SVM_params = {
        'P1': {'C': 0.1, 'kernel': 'rbf', 'gamma': 'scale', 'max_iter': 500, 'random_state': 42},
        'P2': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'max_iter': 1000, 'random_state': 42},
        'P3': {'C': 10.0, 'kernel': 'rbf', 'gamma': 'auto', 'max_iter': 1000, 'random_state': 42},
        'P4': {'C': 1.0, 'kernel': 'linear', 'max_iter': 1000, 'random_state': 42}
    }
    
    XGB_params = {
        'P1': {'n_estimators': 50, 'learning_rate': 0.05, 'max_depth': 3, 'subsample': 0.8,
               'objective': 'multi:softmax', 'num_class': 3, 'random_state': 42},
        'P2': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.8,
               'objective': 'multi:softmax', 'num_class': 3, 'random_state': 42},
        'P3': {'n_estimators': 150, 'learning_rate': 0.15, 'max_depth': 9, 'subsample': 0.9,
               'objective': 'multi:softmax', 'num_class': 3, 'random_state': 42},
        'P4': {'n_estimators': 200, 'learning_rate': 0.2, 'max_depth': 12, 'subsample': 1.0,
               'objective': 'multi:softmax', 'num_class': 3, 'random_state': 42}
    }
    
    return {
        'RF': RF_params,
        'SVM': SVM_params,
        'XGB': XGB_params
    }


def preprocess_dataset(dataset, target_column='Diabetes_012', test_size=0.2, random_state=42):
    """
    Preprocess a dataset: remove duplicates, split, and scale
    
    Parameters:
    -----------
    dataset : DataFrame
        Input dataset
    target_column : str
        Name of target column
    test_size : float
        Proportion of test set
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    tuple : X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    # Separate features and target
    X_data = dataset.drop(target_column, axis=1)
    y_data = dataset[target_column]
    
    # Remove duplicates
    combined_data = pd.concat([X_data, y_data], axis=1)
    combined_data = combined_data.drop_duplicates()
    X_data = combined_data.drop(target_column, axis=1)
    y_data = combined_data[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=random_state, stratify=y_data
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_single_model(model_class, params, X_train, y_train, X_test, y_test, 
                       model_group, suppress_verbose=True):
    """
    Train a single model and evaluate it
    
    Parameters:
    -----------
    model_class : class
        Model class (RandomForestClassifier, SVC, XGBClassifier)
    params : dict
        Model parameters
    X_train, y_train : array-like
        Training data
    X_test, y_test : array-like
        Testing data
    model_group : str
        Model group ID (G1, G2, G3)
    suppress_verbose : bool
        Suppress verbose output
    
    Returns:
    --------
    tuple : trained_model, metrics_dict, training_time
    """
    # Prepare parameters
    params_copy = params.copy()
    
    if suppress_verbose:
        if 'verbose' in params_copy and model_group != 'G2':
            params_copy['verbose'] = 0
        if model_group == 'G2':  # SVM
            params_copy['verbose'] = False
    
    # Initialize and train model
    model = model_class(**params_copy)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    return model, metrics, train_time


def run_training_pipeline(datasets_dict, models_dir='models', custom_params=None, verbose=True):
    """
    Run complete automated training pipeline
    
    Parameters:
    -----------
    datasets_dict : dict
        Dictionary with format: {'D1': ('Name', dataframe), 'D2': ...}
    models_dir : str
        Directory to save trained models
    custom_params : dict, optional
        Custom parameter sets (defaults to get_default_parameters())
    verbose : bool
        Print progress
    
    Returns:
    --------
    DataFrame : Results dataframe with all experiments
    """
    # Create models directory
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        if verbose:
            print(f"✓ Created '{models_dir}' directory")
    
    # Create subdirectories for each model type
    model_subdirs = {
        'RandomForest': os.path.join(models_dir, 'RandomForest'),
        'SVM': os.path.join(models_dir, 'SVM'),
        'XGBoost': os.path.join(models_dir, 'XGBoost')
    }
    
    for model_type, subdir in model_subdirs.items():
        if not os.path.exists(subdir):
            os.makedirs(subdir)
            if verbose:
                print(f"✓ Created '{subdir}' subdirectory")
    
    # Get parameters
    if custom_params is None:
        params_dict = get_default_parameters()
        RF_params = params_dict['RF']
        SVM_params = params_dict['SVM']
        XGB_params = params_dict['XGB']
    else:
        RF_params = custom_params.get('RF', {})
        SVM_params = custom_params.get('SVM', {})
        XGB_params = custom_params.get('XGB', {})
    
    # Organize models
    model_groups = {
        'G1': ('RandomForest', RandomForestClassifier, RF_params),
        'G2': ('SVM', SVC, SVM_params),
        'G3': ('XGBoost', XGBClassifier, XGB_params)
    }
    
    # Results storage
    all_results = []
    experiment_count = 0
    total_experiments = len(datasets_dict) * len(model_groups) * 4  # 4 parameter sets
    
    if verbose:
        print("="*80)
        print("STARTING AUTOMATED TRAINING PIPELINE")
        print("="*80)
        print(f"Total Experiments: {len(datasets_dict)} Datasets × {len(model_groups)} Models × 4 Parameters = {total_experiments}")
        print("="*80)
    
    # Main training loop
    for dataset_id, (dataset_name, dataset) in datasets_dict.items():
        if verbose:
            print(f"\n{'='*80}")
            print(f"DATASET: {dataset_id} ({dataset_name})")
            print(f"{'='*80}")
        
        # Preprocess dataset
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_dataset(dataset)
        
        for group_id, (model_name, model_class, params_dict) in model_groups.items():
            if verbose:
                print(f"\n  MODEL GROUP: {group_id} ({model_name})")
                print(f"  {'-'*76}")
            
            for param_id, params in params_dict.items():
                experiment_count += 1
                experiment_id = f"{dataset_id}{group_id}{param_id}"
                
                if verbose:
                    print(f"    [{experiment_count}/{total_experiments}] Training {experiment_id}...", end=" ")
                
                try:
                    # Train model
                    model, metrics, train_time = train_single_model(
                        model_class, params, X_train_scaled, y_train, 
                        X_test_scaled, y_test, group_id
                    )
                    
                    # Save model to specific subdirectory
                    model_subdir = model_subdirs[model_name]
                    model_filename = os.path.join(model_subdir, f"{experiment_id}_model.pkl")
                    with open(model_filename, 'wb') as f:
                        pickle.dump(model, f)
                    
                    # Store results
                    all_results.append({
                        'Experiment_ID': experiment_id,
                        'Dataset': dataset_id,
                        'Dataset_Name': dataset_name,
                        'Model_Group': group_id,
                        'Model_Name': model_name,
                        'Parameter_Set': param_id,
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1_Score': metrics['f1_score'],
                        'Training_Time': train_time,
                        'Model_File': model_filename
                    })
                    
                    if verbose:
                        print(f"✓ Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, Time: {train_time:.2f}s")
                
                except Exception as e:
                    if verbose:
                        print(f"✗ ERROR: {str(e)[:50]}")
                    
                    all_results.append({
                        'Experiment_ID': experiment_id,
                        'Dataset': dataset_id,
                        'Dataset_Name': dataset_name,
                        'Model_Group': group_id,
                        'Model_Name': model_name,
                        'Parameter_Set': param_id,
                        'Accuracy': 0,
                        'Precision': 0,
                        'Recall': 0,
                        'F1_Score': 0,
                        'Training_Time': 0,
                        'Model_File': 'FAILED'
                    })
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"✓ PIPELINE COMPLETED: {experiment_count} experiments finished")
        print(f"✓ All models saved to: {os.path.abspath(models_dir)}")
        print(f"{'='*80}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    return results_df


if __name__ == "__main__":
    # Example usage
    print("Model Training Pipeline Module")
    print("Use: run_training_pipeline(datasets_dict, models_dir='models')")
