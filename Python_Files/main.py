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
from visualization import plot_comparison_summary, save_visualizations


def main():
    """Main execution function"""
    
    print("="*80)
    print("DIABETES CLASSIFICATION - COMPLETE EXPERIMENTAL WORKFLOW")
    print("="*80)
    
    # Step 1: Load Dataset
    print("\n Loading real dataset...")
    real_data_path = 'Datasets/Real/diabetes_012_health_indicators_BRFSS2015.csv'
    real_data = pd.read_csv(real_data_path)
    D2_synthetic_gan = pd.read_csv("Datasets/CTGAN/synthetic_1_unscaled.csv")
    D3_synthetic_vae = pd.read_csv("Datasets/VAE/synthetic_2_unscaled.csv")
    
    # Step 2: Organize Datasets
    print("\n Organizing datasets...")
    datasets = {
        'D1': ('Real', real_data),
        'D2': ('CTGAN', D2_synthetic_gan),
        'D3': ('VAE', D3_synthetic_vae)
    }
    print("✓ Datasets organized: D1 (Real), D2 (CTGAN), D3 (VAE)")
    
    # Step 3: Run Training Pipeline
    print("\n Running automated training pipeline...")
    results_df = run_training_pipeline(datasets, models_dir='models', verbose=True)
    
    # Save results
    results_df.to_csv('experiment_results_complete.csv', index=False)
    print(f"\n✓ Results saved to: experiment_results_complete.csv")
    
    # Display summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nTop 5 Performing Configurations:")
    top_5 = results_df.nlargest(5, 'Accuracy')[['Experiment_ID', 'Dataset_Name', 
                                                   'Model_Name', 'Accuracy', 'F1_Score']]
    print(top_5.to_string(index=False))
    
    print(f"\n\nOverall Statistics:")
    print(f"  Mean Accuracy: {results_df['Accuracy'].mean():.4f}")
    print(f"  Std Accuracy:  {results_df['Accuracy'].std():.4f}")
    print(f"  Max Accuracy:  {results_df['Accuracy'].max():.4f}")
    print(f"  Min Accuracy:  {results_df['Accuracy'].min():.4f}")
    
    # Step 4: Statistical Analysis
    print("\n Performing statistical analysis...")
    statistical_results = run_complete_statistical_analysis(
        results_df, 
        save_to_csv=True, 
        output_dir='Reports'
    )
    
    # Generate Visualizations
    print("\nGenerating visualizations...")
    save_visualizations(results_df, output_dir='visualizations')
    
    print("\n" + "="*80)
    print("✓ WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated Files:")
    print("  - experiment_results_complete.csv (All results)")
    print("  - Reports/ (Statistical analysis CSV files)")
    print("  - models/ (36 trained models)")
    print("  - visualizations/ (Comparison plots)")
    
    return results_df, statistical_results


if __name__ == "__main__":
    results_df, statistical_results = main()
