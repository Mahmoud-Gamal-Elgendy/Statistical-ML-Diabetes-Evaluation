"""
Model Training Pipeline Module - NEW WORKFLOW
Implements 10-fold cross-validation on 12 data blocks (4 blocks per dataset)
Uses only Real dataset for testing, single parameter set per model
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


def get_single_parameter_set():
    """
    Get single optimized parameter set for each model
    Fine-tuned using RandomizedSearchCV and GridSearchCV
    
    Returns:
    --------
    dict : Dictionary with parameters for RF, SVM, and XGBoost
    """
    params = {
        'RandomForest': {
            'n_estimators': 100,
            'max_depth': 30,
            'min_samples_split': 10,
            'min_samples_leaf': 2,
            'max_features': 'log2',
            'class_weight': None,
            'random_state': 42,
            'n_jobs': -1
        },
        'SVM': {
            'C': 1,
            'kernel': 'rbf',
            'gamma': 'scale',
            'class_weight': None,
            'max_iter': 2000,
            'random_state': 42
        },
        'XGBoost': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'objective': 'multi:softmax',
            'num_class': 3,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
    }
    
    return params


def split_dataset_into_blocks(dataset, n_blocks=4, target_column='Diabetes_012', random_state=42):
    """
    Split a dataset into equal blocks for cross-validation
    
    Parameters:
    -----------
    dataset : DataFrame
        Input dataset
    n_blocks : int
        Number of blocks to create (default: 4)
    target_column : str
        Name of target column
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    list : List of DataFrames, each representing a data block
    """
    # Remove duplicates first
    dataset_clean = dataset.drop_duplicates()
    
    # Shuffle the dataset
    dataset_shuffled = dataset_clean.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate block size
    total_samples = len(dataset_shuffled)
    block_size = total_samples // n_blocks
    
    blocks = []
    for i in range(n_blocks):
        start_idx = i * block_size
        if i == n_blocks - 1:  # Last block gets remaining samples
            block = dataset_shuffled.iloc[start_idx:]
        else:
            end_idx = (i + 1) * block_size
            block = dataset_shuffled.iloc[start_idx:end_idx]
        
        blocks.append(block.reset_index(drop=True))
    
    return blocks


def prepare_real_test_set(real_dataset, test_size=0.2, target_column='Diabetes_012', random_state=42):
    """
    Create test set from Real dataset only
    
    Parameters:
    -----------
    real_dataset : DataFrame
        Real dataset
    test_size : float
        Proportion of test set
    target_column : str
        Name of target column
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    tuple : X_test_scaled, y_test, scaler
    """
    # Remove duplicates
    real_clean = real_dataset.drop_duplicates()
    
    # Separate features and target
    X = real_clean.drop(target_column, axis=1)
    y = real_clean[target_column]
    
    # Split to get test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale test set
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    return X_test_scaled, y_test, scaler


def train_with_cross_validation(model_class, params, data_block, scaler, 
                                  n_folds=10, target_column='Diabetes_012', 
                                  random_state=42, verbose=False):
    """
    Train model using k-fold cross-validation on a data block
    
    Parameters:
    -----------
    model_class : class
        Model class (RandomForestClassifier, SVC, XGBClassifier)
    params : dict
        Model parameters
    data_block : DataFrame
        Training data block
    scaler : StandardScaler
        Pre-fitted scaler from test set
    n_folds : int
        Number of cross-validation folds
    target_column : str
        Name of target column
    random_state : int
        Random state for reproducibility
    verbose : bool
        Print fold-by-fold results
    
    Returns:
    --------
    tuple : (mean_accuracy, final_model)
        mean_accuracy : float - Mean accuracy across all folds
        final_model : trained model - Model trained on entire data block
    """
    # Separate features and target
    X = data_block.drop(target_column, axis=1)
    y = data_block[target_column]
    
    # Scale features using the same scaler as test set
    X_scaled = scaler.transform(X)
    
    # Initialize cross-validation
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_accuracies = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y), 1):
        # Split into train and validation
        X_train_fold = X_scaled[train_idx]
        X_val_fold = X_scaled[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train model
        model = model_class(**params)
        model.fit(X_train_fold, y_train_fold)
        
        # Predict and evaluate
        y_pred = model.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, y_pred)
        fold_accuracies.append(accuracy)
        
        if verbose:
            print(f"      Fold {fold_idx}/{n_folds}: Accuracy = {accuracy:.6f}")
    
    mean_accuracy = np.mean(fold_accuracies)
    
    # Train final model on entire data block
    final_model = model_class(**params)
    final_model.fit(X_scaled, y)
    
    return mean_accuracy, final_model


def run_training_pipeline(datasets_dict, models_dir='models', n_blocks=4, n_folds=10, 
                          test_size=0.2, random_state=42, verbose=True):
    """
    Run complete training pipeline with new workflow
    
    NEW WORKFLOW:
    1. Split each of 3 datasets into 4 blocks → 12 data blocks
    2. Create test set from Real dataset only
    3. Train each model on each block using 10-fold cross-validation
    4. Return 12×3 matrix of average accuracies
    
    Parameters:
    -----------
    datasets_dict : dict
        Dictionary with format: {'Real': dataframe, 'CTGAN': dataframe, 'VAE': dataframe}
    models_dir : str
        Directory to save trained models
    n_blocks : int
        Number of blocks per dataset (default: 4)
    n_folds : int
        Number of cross-validation folds (default: 10)
    test_size : float
        Proportion of test set from Real dataset
    random_state : int
        Random state for reproducibility
    verbose : bool
        Print detailed progress
    
    Returns:
    --------
    DataFrame : 12×3 matrix with rows=blocks, columns=models, values=mean accuracy
    """
    # Create models directory
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        if verbose:
            print(f"✓ Created '{models_dir}' directory")
    
    # Get single parameter set for each model
    params = get_single_parameter_set()
    
    # Model classes
    model_classes = {
        'RandomForest': RandomForestClassifier,
        'SVM': SVC,
        'XGBoost': XGBClassifier
    }
    
    model_names = ['RandomForest', 'SVM', 'XGBoost']
    dataset_names = ['Real', 'CTGAN', 'VAE']
    
    if verbose:
        print("="*90)
        print("NEW TRAINING PIPELINE WITH 10-FOLD CROSS-VALIDATION")
        print("="*90)
        print(f"\nConfiguration:")
        print(f"  - Datasets: {len(dataset_names)} (Real, CTGAN, VAE)")
        print(f"  - Blocks per dataset: {n_blocks}")
        print(f"  - Total blocks: {len(dataset_names) * n_blocks}")
        print(f"  - Models: {len(model_names)} (RandomForest, SVM, XGBoost)")
        print(f"  - Cross-validation folds: {n_folds}")
        print(f"  - Test set: Only from Real dataset ({test_size*100:.0f}% of Real data)")
        print("="*90)
    
    # Step 1: Prepare test set from Real dataset
    if verbose:
        print("\n[1/3] Preparing test set from Real dataset...")
    
    if 'Real' not in datasets_dict:
        raise ValueError("'Real' dataset must be provided in datasets_dict")
    
    real_dataset = datasets_dict['Real']
    X_test, y_test, scaler = prepare_real_test_set(
        real_dataset, test_size=test_size, random_state=random_state
    )
    
    if verbose:
        print(f"  ✓ Test set created: {len(y_test)} samples")
        print(f"  ✓ Scaler fitted on Real data")
    
    # Step 2: Split each dataset into blocks
    if verbose:
        print(f"\n[2/3] Splitting datasets into {n_blocks} blocks each...")
    
    all_blocks = {}
    block_labels = []
    
    for dataset_name in dataset_names:
        if dataset_name not in datasets_dict:
            raise ValueError(f"'{dataset_name}' dataset not found in datasets_dict")
        
        dataset = datasets_dict[dataset_name]
        blocks = split_dataset_into_blocks(dataset, n_blocks=n_blocks, random_state=random_state)
        
        for block_idx, block in enumerate(blocks, 1):
            block_label = f"{dataset_name}_Block{block_idx}"
            all_blocks[block_label] = block
            block_labels.append(block_label)
            
            if verbose:
                print(f"  ✓ {block_label}: {len(block)} samples")
    
    # Step 3: Train models on each block with cross-validation
    if verbose:
        print(f"\n[3/3] Training models with {n_folds}-fold cross-validation...")
        print("="*90)
    
    results_matrix = []
    total_experiments = len(all_blocks) * len(model_names)
    experiment_count = 0
    
    for block_label in block_labels:
        data_block = all_blocks[block_label]
        row_results = []
        
        if verbose:
            print(f"\n{block_label} ({len(data_block)} samples):")
        
        for model_name in model_names:
            experiment_count += 1
            
            if verbose:
                print(f"  [{experiment_count}/{total_experiments}] Training {model_name}...", end=" ")
            
            try:
                # Train with cross-validation
                mean_accuracy, final_model = train_with_cross_validation(
                    model_class=model_classes[model_name],
                    params=params[model_name],
                    data_block=data_block,
                    scaler=scaler,
                    n_folds=n_folds,
                    random_state=random_state,
                    verbose=False  # Set to True to see fold-by-fold results
                )
                
                row_results.append(mean_accuracy)
                
                # Save the trained model
                model_dir = os.path.join(models_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"{block_label}.pkl")
                
                with open(model_path, 'wb') as f:
                    pickle.dump(final_model, f)
                
                if verbose:
                    print(f"✓ Mean Accuracy: {mean_accuracy:.6f} | Model saved: {model_path}")
            
            except Exception as e:
                if verbose:
                    print(f"✗ ERROR: {str(e)[:50]}")
                row_results.append(0.0)
        
        results_matrix.append(row_results)
    
    # Create DataFrame with results
    results_df = pd.DataFrame(
        results_matrix,
        index=block_labels,
        columns=model_names
    )
    results_df.index.name = 'Block'
    
    if verbose:
        print("\n" + "="*90)
        print("✓ TRAINING PIPELINE COMPLETED")
        print("="*90)
        print("\nFinal Results Matrix (12 blocks × 3 models):")
        print("-"*90)
        print(results_df.to_string())
        print("-"*90)
        print(f"\nSummary Statistics:")
        for model_name in model_names:
            mean_acc = results_df[model_name].mean()
            std_acc = results_df[model_name].std()
            print(f"  {model_name:15s}: Mean = {mean_acc:.6f}, Std = {std_acc:.6f}")
    
    return results_df


if __name__ == "__main__":
    print("Model Training Pipeline Module - NEW WORKFLOW")
    print("\nUsage:")
    print("  from model_training_pipeline import run_training_pipeline")
    print("\n  datasets = {")
    print("      'Real': real_df,")
    print("      'CTGAN': ctgan_df,")
    print("      'VAE': vae_df")
    print("  }")
    print("\n  results_matrix = run_training_pipeline(datasets)")
    print("\nOutput: 12×3 DataFrame with average 10-fold CV accuracy")
