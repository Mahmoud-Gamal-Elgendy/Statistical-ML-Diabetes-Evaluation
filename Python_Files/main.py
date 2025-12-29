import sys
import io
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for console output to avoid Unicode errors
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import custom modules
from model_training_pipeline import run_training_pipeline
from statistical_analysis import run_complete_statistical_analysis
from visualization import save_all_visualizations


def main():
    """Main execution function - NEW WORKFLOW"""
    
    print("="*90)
    print("DIABETES CLASSIFICATION - NEW EXPERIMENTAL WORKFLOW")
    print("="*90)
    print("\nWorkflow Overview:")
    print("  1. Load 3 datasets (Real, CTGAN, VAE)")
    print("  2. Split each dataset into 4 blocks → 12 total blocks")
    print("  3. Train 3 models on each block using 10-fold cross-validation")
    print("  4. Generate 12×3 accuracy matrix")
    print("  5. Perform statistical analysis (Friedman + post-hoc tests)")
    print("="*90)
    
    # Step 1: Load Datasets
    print("\n[STEP 1] Loading datasets...")
    real_data_path = 'Datasets/Real/diabetes_012_health_indicators_BRFSS2015.csv'
    ctgan_data_path = 'Datasets/CTGAN/synthetic_1_unscaled.csv'
    vae_data_path = 'Datasets/VAE/synthetic_2_unscaled.csv'
    
    real_data = pd.read_csv(real_data_path)
    ctgan_data = pd.read_csv(ctgan_data_path)
    vae_data = pd.read_csv(vae_data_path)
    
    print(f"  ✓ Real dataset: {len(real_data)} samples, {real_data.shape[1]} features")
    print(f"  ✓ CTGAN dataset: {len(ctgan_data)} samples, {ctgan_data.shape[1]} features")
    print(f"  ✓ VAE dataset: {len(vae_data)} samples, {vae_data.shape[1]} features")
    
    # Step 2: Organize Datasets
    datasets = {
        'Real': real_data,
        'CTGAN': ctgan_data,
        'VAE': vae_data
    }
    
    # Step 3: Run Training Pipeline with 10-Fold Cross-Validation
    print("\n[STEP 2] Running training pipeline with 10-fold cross-validation...")
    print("  This will:")
    print("    - Split each dataset into 4 blocks")
    print("    - Train 3 models (RandomForest, SVM, XGBoost) on each block")
    print("    - Use 10-fold CV to compute average accuracy")
    print("    - Generate a 12×3 accuracy matrix")
    print()
    
    results_matrix = run_training_pipeline(
        datasets_dict=datasets,
        models_dir='models',
        n_blocks=4,
        n_folds=10,
        test_size=0.2,
        random_state=42,
        verbose=True
    )
    
    # Save results matrix
    results_matrix.to_csv('experiment_results_matrix.csv')
    print(f"\n✓ Results matrix saved to: experiment_results_matrix.csv")
    
    # Step 4: Statistical Analysis
    print("\n[STEP 3] Performing statistical analysis...")
    statistical_results = run_complete_statistical_analysis(
        data_matrix=results_matrix,
        model_names=['RandomForest', 'SVM', 'XGBoost'],
        block_labels=list(results_matrix.index),
        save_to_csv=True,
        output_dir='statistical_results'
    )
    
    # Step 5: Generate Visualizations
    print("\n[STEP 4] Generating visualizations...")
    save_all_visualizations(results_matrix, output_dir='visualizations')
    
    # Final Summary
    print("\n" + "="*90)
    print("✓ WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*90)
    print("\nGenerated Files:")
    print("  - experiment_results_matrix.csv (12×3 matrix with average accuracies)")
    print("  - statistical_results/ (Friedman test, post-hoc tests, effect sizes)")
    print("  - visualizations/ (Heatmaps, comparison plots)")
    print("  - models/ (Trained models from cross-validation)")
    print("\nNext Steps:")
    print("  - Review experiment_results_matrix.csv for model performance")
    print("  - Check statistical_results/ for significance tests")
    print("  - View visualizations/ for graphical analysis")
    print("="*90)
    
    return results_matrix, statistical_results


if __name__ == "__main__":
    results_matrix, statistical_results = main()
