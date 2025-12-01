"""
Test script for statistical_analysis.py
Creates sample data and runs all statistical tests
"""

import numpy as np
import pandas as pd
from statistical_analysis import (
    perform_friedman_test,
    calculate_effect_size,
    perform_posthoc_test,
    perform_hommel_correction,
    run_complete_statistical_analysis
)

def create_sample_data():
    """
    Create sample experimental results data
    
    Structure:
    - 3 Model Groups (G1=RandomForest, G2=SVM, G3=XGBoost)
    - 3 Dataset Blocks (D1=Real, D2=GAN, D3=VAE)
    - 4 Parameter Sets per group per dataset (P1, P2, P3, P4)
    - Total: 3 groups × 3 datasets × 4 params = 36 runs
    """
    
    np.random.seed(42)  # For reproducibility
    
    data = []
    
    # Define groups and datasets
    group_ids = ['G1', 'G2', 'G3']
    group_names = ['RandomForest', 'SVM', 'XGBoost']
    dataset_ids = ['D1', 'D2', 'D3']
    dataset_names = ['Real', 'GAN', 'VAE']
    param_ids = ['P1', 'P2', 'P3', 'P4']
    
    # Base performance for each group (to create realistic differences)
    # G3 (XGBoost) performs best, G1 (RandomForest) second, G2 (SVM) third
    base_performance = {
        'G1': {'Accuracy': 0.85, 'Precision': 0.83, 'Recall': 0.82, 'F1_Score': 0.82},
        'G2': {'Accuracy': 0.80, 'Precision': 0.78, 'Recall': 0.77, 'F1_Score': 0.77},
        'G3': {'Accuracy': 0.90, 'Precision': 0.89, 'Recall': 0.88, 'F1_Score': 0.88}
    }
    
    # Dataset impact (Real > GAN > VAE)
    dataset_impact = {
        'D1': 0.02,   # Real data: +2%
        'D2': 0.00,   # GAN: baseline
        'D3': -0.03   # VAE: -3%
    }
    
    # Generate data for each combination
    for group_id, group_name in zip(group_ids, group_names):
        for dataset_id, dataset_name in zip(dataset_ids, dataset_names):
            for param_id in param_ids:
                # Get base performance and add dataset impact
                base_acc = base_performance[group_id]['Accuracy'] + dataset_impact[dataset_id]
                base_prec = base_performance[group_id]['Precision'] + dataset_impact[dataset_id]
                base_rec = base_performance[group_id]['Recall'] + dataset_impact[dataset_id]
                base_f1 = base_performance[group_id]['F1_Score'] + dataset_impact[dataset_id]
                
                # Add some random variation (parameter tuning effect)
                noise = np.random.normal(0, 0.01)  # 1% std deviation
                
                data.append({
                    'Dataset': dataset_id,
                    'Dataset_Name': dataset_name,
                    'Model_Group': group_id,
                    'Model_Name': group_name,
                    'Parameter_Set': param_id,
                    'Accuracy': np.clip(base_acc + noise, 0, 1),
                    'Precision': np.clip(base_prec + noise, 0, 1),
                    'Recall': np.clip(base_rec + noise, 0, 1),
                    'F1_Score': np.clip(base_f1 + noise, 0, 1)
                })
    
    df = pd.DataFrame(data)
    return df


def test_individual_functions(results_df):
    """Test individual statistical functions"""
    
    print("\n" + "="*80)
    print("TESTING INDIVIDUAL FUNCTIONS")
    print("="*80)
    
    # Test 1: Friedman Test
    print("\n[TEST 1] Testing Friedman ANOVA Test for Accuracy...")
    friedman_results = perform_friedman_test(results_df, metric='Accuracy')
    
    print(f"\nFriedman Test Results:")
    print(f"  Chi-square: {friedman_results['statistic']:.4f}")
    print(f"  P-value: {friedman_results['p_value']:.6f}")
    print(f"  Significant: {friedman_results['is_significant']}")
    
    # Test 2: Effect Size
    if friedman_results['is_significant']:
        print("\n[TEST 2] Testing Effect Size Calculation...")
        effect_size = calculate_effect_size(friedman_results['statistic'], k=3, n=3)
        print(f"  Kendall's W: {effect_size['kendalls_w']:.4f}")
        print(f"  Interpretation: {effect_size['interpretation']}")
        
        # Test 3: Post-hoc Test
        print("\n[TEST 3] Testing Nemenyi Post-hoc Test...")
        posthoc_results = perform_posthoc_test(
            friedman_results['mean_matrix'],
            friedman_results['group_ids'],
            friedman_results['group_names'],
            metric='Accuracy'
        )
        print(f"\nPost-hoc test completed with {len(posthoc_results['significant_pairs'])} pairwise comparisons")
        
        # Test 4: Hommel Correction
        print("\n[TEST 4] Testing Hommel Correction...")
        hommel_results = perform_hommel_correction(
            friedman_results['mean_matrix'],
            friedman_results['group_ids'],
            friedman_results['group_names'],
            metric='Accuracy'
        )
        print(f"\nHommel correction completed: {hommel_results['n_significant']}/{hommel_results['n_comparisons']} significant")


def test_complete_analysis(results_df):
    """Test complete statistical analysis pipeline"""
    
    print("\n" + "="*80)
    print("TESTING COMPLETE ANALYSIS PIPELINE")
    print("="*80)
    
    # Run complete analysis with CSV saving enabled
    all_results = run_complete_statistical_analysis(
        results_df, 
        save_to_csv=True, 
        output_dir='Reports/Test_Results'
    )
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print(f"\nTotal metrics analyzed: {len(all_results)}")
    
    for metric, results in all_results.items():
        print(f"\n{metric}:")
        print(f"  Friedman test performed: ✓")
        if 'effect_size' in results:
            print(f"  Effect size calculated: ✓")
        if 'posthoc' in results:
            print(f"  Post-hoc test performed: ✓")
        if 'hommel' in results:
            print(f"  Hommel correction performed: ✓")
    
    return all_results


def display_sample_data(df, n=10):
    """Display sample of the data"""
    
    print("\n" + "="*80)
    print("SAMPLE DATA")
    print("="*80)
    
    print(f"\nTotal rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\nFirst {n} rows:")
    print(df.head(n))
    
    print("\nData summary by Model Group:")
    print(df.groupby('Model_Group')[['Accuracy', 'Precision', 'Recall', 'F1_Score']].mean())
    
    print("\nData summary by Dataset:")
    print(df.groupby('Dataset')[['Accuracy', 'Precision', 'Recall', 'F1_Score']].mean())


def main():
    """Main test function"""
    
    print("="*80)
    print("STATISTICAL ANALYSIS TEST SUITE")
    print("="*80)
    
    # Create sample data
    print("\n[STEP 1] Creating sample experimental results...")
    results_df = create_sample_data()
    print(f"✓ Created {len(results_df)} experimental runs")
    
    # Display sample data
    display_sample_data(results_df)
    
    # Test individual functions
    print("\n" + "="*80)
    test_individual_functions(results_df)
    
    # Test complete analysis
    print("\n" + "="*80)
    test_complete_analysis(results_df)
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()
