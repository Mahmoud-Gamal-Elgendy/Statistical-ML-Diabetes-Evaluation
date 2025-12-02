"""
Test script for statistical_analysis.py
Creates sample data and runs all statistical tests
"""

import sys
import io

# Set UTF-8 encoding for console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from statistical_analysis import run_complete_statistical_analysis

def create_sample_data():
    """
    Create sample aggregated data with 3 rows (one per model)
    This simulates the aggregated results after averaging all experiments
    """
    
    # Create aggregated data directly (3 rows - one per model)
    data = {
        'Model_Name': ['RandomForest', 'SVM', 'XGBoost'],
        'Accuracy': [0.8496, 0.7908, 0.8947],
        'Precision': [0.8296, 0.7708, 0.8847],
        'Recall': [0.8196, 0.7608, 0.8747],
        'F1_Score': [0.8196, 0.7608, 0.8747],
        'Training_Time': [45.23, 89.67, 123.45]
    }
    
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


def test_complete_analysis(aggregated_df):
    """Test complete statistical analysis pipeline with aggregated data"""
    
    print("\n" + "="*80)
    print("TESTING COMPLETE ANALYSIS PIPELINE")
    print("="*80)
    
    # Run statistical analysis on aggregated data
    print("\nRunning statistical analysis on aggregated data...")
    statistical_results = run_complete_statistical_analysis(
        aggregated_df, 
        save_to_csv=True, 
        output_dir='statistical_results/Test_Results'
    )
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print(f"\nAggregated data: {len(aggregated_df)} rows (3 models)")
    print(f"\nStatistical analysis completed successfully!")
    print(f"Results saved to: statistical_results/Test_Results/")
    
    return statistical_results


def display_sample_data(df):
    """Display the aggregated data"""
    
    print("\n" + "="*80)
    print("AGGREGATED MODEL DATA (3 ROWS)")
    print("="*80)
    
    print(f"\nTotal models: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    print("\nAggregated Data:")
    print(df.to_string(index=False))


def main():
    """Main test function"""
    
    print("="*80)
    print("STATISTICAL ANALYSIS TEST SUITE")
    print("="*80)
    
    # Create aggregated sample data (3 rows)
    print("\n[STEP 1] Creating aggregated sample data (3 models)...")
    aggregated_df = create_sample_data()
    print(f"[OK] Created aggregated data with {len(aggregated_df)} models")
    
    # Display aggregated data
    display_sample_data(aggregated_df)
    
    # Test complete analysis
    print("\n" + "="*80)
    test_complete_analysis(aggregated_df)
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()
