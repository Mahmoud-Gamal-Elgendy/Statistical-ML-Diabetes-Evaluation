import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from statsmodels.stats.multitest import multipletests
from pathlib import Path


def perform_friedman_test(data_matrix, model_names):
    """
    Perform Friedman ANOVA test on the data matrix
    
    Parameters:
    -----------
    data_matrix : numpy array
        12×3 matrix (blocks × models)
    model_names : list
        Names of the three models
    
    Returns:
    --------
    dict : Friedman test results
    """
    print("\n" + "="*90)
    print("FRIEDMAN ANOVA TEST")
    print("="*90)
    print("\nNull Hypothesis (H₀): All models perform equally across blocks")
    print("Alternative Hypothesis (H₁): At least one model differs significantly")
    print("Significance level: α = 0.05")
    
    # Perform Friedman test (each column is a treatment/model)
    rf_scores = data_matrix[:, 0]
    svm_scores = data_matrix[:, 1]
    xgb_scores = data_matrix[:, 2]
    
    friedman_stat, friedman_p = friedmanchisquare(rf_scores, svm_scores, xgb_scores)
    
    print(f"\nResults:")
    print(f"  Chi-square statistic (χ²): {friedman_stat:.6f}")
    print(f"  P-value: {friedman_p:.6f}")
    print(f"  Degrees of freedom: {len(model_names) - 1}")
    
    is_significant = friedman_p < 0.05
    
    if is_significant:
        print(f"\n  ✓ SIGNIFICANT (p < 0.05)")
        print(f"  → Models show significantly different performance")
    else:
        print(f"\n  ✗ NOT SIGNIFICANT (p ≥ 0.05)")
        print(f"  → No significant difference between models")
    
    return {
        'statistic': friedman_stat,
        'p_value': friedman_p,
        'is_significant': is_significant,
        'df': len(model_names) - 1
    }


def calculate_kendalls_w(friedman_stat, n_blocks, n_models):
    """
    Calculate Kendall's W effect size
    
    Parameters:
    -----------
    friedman_stat : float
        Chi-square statistic from Friedman test
    n_blocks : int
        Number of blocks (12)
    n_models : int
        Number of models (3)
    
    Returns:
    --------
    dict : Effect size results
    """
    print("\n" + "="*90)
    print("KENDALL'S W - Effect Size")
    print("="*90)
    
    # Kendall's W = χ² / (n_blocks * (n_models - 1))
    kendalls_w = friedman_stat / (n_blocks * (n_models - 1))
    
    print(f"\nKendall's W: {kendalls_w:.6f}")
    print(f"Formula: W = χ² / (k × (n - 1)) = {friedman_stat:.6f} / ({n_blocks} × {n_models-1})")
    
    if kendalls_w < 0.3:
        interpretation = "Small effect - Weak agreement"
    elif kendalls_w < 0.5:
        interpretation = "Medium effect - Moderate agreement"
    else:
        interpretation = "Large effect - Strong agreement"
    
    print(f"Interpretation: {interpretation}")
    
    return {
        'kendalls_w': kendalls_w,
        'interpretation': interpretation
    }


def perform_nemenyi_posthoc(data_matrix, model_names):
    """
    Perform Nemenyi post-hoc test
    
    Parameters:
    -----------
    data_matrix : numpy array
        12×3 matrix (blocks × models)
    model_names : list
        Names of the three models
    
    Returns:
    --------
    dict : Post-hoc test results
    """
    print("\n" + "="*90)
    print("NEMENYI POST-HOC TEST")
    print("="*90)
    
    # Perform Nemenyi post-hoc test
    posthoc_matrix = posthoc_nemenyi_friedman(data_matrix)
    
    # Set labels
    posthoc_matrix.columns = model_names
    posthoc_matrix.index = model_names
    
    print("\nPairwise P-values:")
    print(posthoc_matrix.to_string())
    
    print("\n" + "-" * 90)
    print("Pairwise Comparisons:")
    print("-" * 90)
    
    pairwise_results = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            p_val = posthoc_matrix.iloc[i, j]
            model_i = model_names[i]
            model_j = model_names[j]
            is_sig = p_val < 0.05
            
            sig_marker = "✓ SIGNIFICANT" if is_sig else "✗ not significant"
            print(f"  {model_i} vs {model_j}: p = {p_val:.6f} ({sig_marker})")
            
            pairwise_results.append({
                'Model_1': model_i,
                'Model_2': model_j,
                'p_value': p_val,
                'significant': is_sig
            })
    
    return {
        'p_matrix': posthoc_matrix,
        'pairwise_results': pairwise_results
    }


def apply_hommel_correction(posthoc_matrix, model_names):
    """
    Apply Hommel correction for multiple comparisons
    
    Parameters:
    -----------
    posthoc_matrix : DataFrame
        P-value matrix from Nemenyi test
    model_names : list
        Names of the three models
    
    Returns:
    --------
    dict : Hommel correction results
    """
    print("\n" + "="*90)
    print("HOMMEL MULTIPLE COMPARISON CORRECTION")
    print("="*90)
    
    # Extract p-values
    p_values = []
    comparisons = []
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            p_val = posthoc_matrix.iloc[i, j]
            p_values.append(p_val)
            comparisons.append((model_names[i], model_names[j]))
    
    # Apply Hommel correction
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='hommel')
    
    print("\nResults:")
    print("-" * 90)
    print(f"{'Comparison':<30} {'Original p':>15} {'Corrected p':>15} {'Significant':>15}")
    print("-" * 90)
    
    hommel_pairs = []
    n_significant = 0
    
    for idx, (model_i, model_j) in enumerate(comparisons):
        original_p = p_values[idx]
        corrected_p = pvals_corrected[idx]
        is_sig = reject[idx]
        
        sig_marker = "✓ YES" if is_sig else "✗ NO"
        print(f"{model_i} vs {model_j:<20} {original_p:15.6f} {corrected_p:15.6f} {sig_marker:>15}")
        
        if is_sig:
            n_significant += 1
        
        hommel_pairs.append({
            'Model_1': model_i,
            'Model_2': model_j,
            'original_p': original_p,
            'corrected_p': corrected_p,
            'significant': bool(is_sig)
        })
    
    print("-" * 90)
    print(f"Significant pairs: {n_significant}/{len(comparisons)}")
    
    return {
        'comparisons': hommel_pairs,
        'n_significant': n_significant,
        'n_comparisons': len(comparisons)
    }


def calculate_cliffs_delta(group1, group2):
    """
    Calculate Cliff's Delta effect size
    
    Parameters:
    -----------
    group1, group2 : array-like
        Two groups to compare
    
    Returns:
    --------
    tuple : (delta, interpretation)
    """
    n1 = len(group1)
    n2 = len(group2)
    
    greater = 0
    less = 0
    
    for val1 in group1:
        for val2 in group2:
            if val1 > val2:
                greater += 1
            elif val1 < val2:
                less += 1
    
    delta = (greater - less) / (n1 * n2)
    
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = "Negligible"
    elif abs_delta < 0.33:
        interpretation = "Small"
    elif abs_delta < 0.474:
        interpretation = "Medium"
    else:
        interpretation = "Large"
    
    return delta, interpretation


def calculate_pairwise_effect_sizes(data_matrix, model_names):
    """
    Calculate Cliff's Delta for all model pairs
    
    Parameters:
    -----------
    data_matrix : numpy array
        12×3 matrix (blocks × models)
    model_names : list
        Names of the three models
    
    Returns:
    --------
    list : Effect size results for each pair
    """
    print("\n" + "="*90)
    print("PAIRWISE EFFECT SIZES - Cliff's Delta")
    print("="*90)
    
    effect_size_results = []
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model_i = model_names[i]
            model_j = model_names[j]
            
            group_i = data_matrix[:, i]
            group_j = data_matrix[:, j]
            
            delta, interpretation = calculate_cliffs_delta(group_i, group_j)
            
            print(f"\n  {model_i} vs {model_j}:")
            print(f"    Cliff's Delta: {delta:+.6f}")
            print(f"    Interpretation: {interpretation}")
            
            if delta > 0:
                favors = model_i
                print(f"    → {model_i} outperforms {model_j}")
            elif delta < 0:
                favors = model_j
                print(f"    → {model_j} outperforms {model_i}")
            else:
                favors = 'Tie'
                print(f"    → No difference")
            
            effect_size_results.append({
                'Model_1': model_i,
                'Model_2': model_j,
                'Cliffs_Delta': delta,
                'Interpretation': interpretation,
                'Favors': favors
            })
    
    return effect_size_results


def save_results_to_csv(all_results, output_dir='statistical_results'):
    """
    Save all statistical analysis results to CSV files
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing all analysis results
    output_dir : str
        Directory to save CSV files (can be absolute or relative path)
    """
    # Handle both absolute and relative paths
    output_path = Path(output_dir)
    
    # If relative path, make it absolute from current working directory
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_dir
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save data matrix
    matrix_df = pd.DataFrame(
        all_results['data_matrix'],
        columns=all_results['model_names']
    )
    if all_results['block_labels'] is not None:
        matrix_df.index = all_results['block_labels']
        matrix_df.index.name = 'Block'
    matrix_df.to_csv(output_path / 'data_matrix.csv')
    
    # 2. Save Friedman test results
    friedman_df = pd.DataFrame([{
        'Test': 'Friedman ANOVA',
        'Chi_Square': all_results['friedman']['statistic'],
        'P_Value': all_results['friedman']['p_value'],
        'DF': all_results['friedman']['df'],
        'Significant': 'Yes' if all_results['friedman']['is_significant'] else 'No'
    }])
    friedman_df.to_csv(output_path / 'friedman_test_results.csv', index=False)
    
    # 3. Save effect size (if calculated)
    if all_results['effect_size'] is not None:
        effect_df = pd.DataFrame([{
            'Measure': 'Kendalls_W',
            'Value': all_results['effect_size']['kendalls_w'],
            'Interpretation': all_results['effect_size']['interpretation']
        }])
        effect_df.to_csv(output_path / 'effect_size_results.csv', index=False)
    
    # 4. Save post-hoc test (if calculated)
    if all_results['posthoc'] is not None:
        posthoc_df = pd.DataFrame(all_results['posthoc']['pairwise_results'])
        posthoc_df.to_csv(output_path / 'posthoc_nemenyi_results.csv', index=False)
    
    # 5. Save Hommel correction (if calculated)
    if all_results['hommel'] is not None:
        hommel_df = pd.DataFrame(all_results['hommel']['comparisons'])
        hommel_df.to_csv(output_path / 'hommel_correction_results.csv', index=False)
    
    # 6. Save pairwise effect sizes (if calculated)
    if all_results['pairwise_effect_sizes'] is not None:
        effect_sizes_df = pd.DataFrame(all_results['pairwise_effect_sizes'])
        effect_sizes_df.to_csv(output_path / 'pairwise_effect_sizes_cliffs_delta.csv', index=False)
    
    # 7. Save model summary statistics
    summary_data = []
    for i, model in enumerate(all_results['model_names']):
        summary_data.append({
            'Model_Name': model,
            'Mean_Accuracy': all_results['data_matrix'][:, i].mean(),
            'Std_Accuracy': all_results['data_matrix'][:, i].std(),
            'Min_Accuracy': all_results['data_matrix'][:, i].min(),
            'Max_Accuracy': all_results['data_matrix'][:, i].max()
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / 'model_summary_statistics.csv', index=False)
    
    print(f"\n✓ All results saved to '{output_path}' directory:")
    print(f"  - data_matrix.csv")
    print(f"  - friedman_test_results.csv")
    if all_results['effect_size'] is not None:
        print(f"  - effect_size_results.csv")
    if all_results['posthoc'] is not None:
        print(f"  - posthoc_nemenyi_results.csv")
    if all_results['hommel'] is not None:
        print(f"  - hommel_correction_results.csv")
    if all_results['pairwise_effect_sizes'] is not None:
        print(f"  - pairwise_effect_sizes_cliffs_delta.csv")
    print(f"  - model_summary_statistics.csv")


def run_complete_statistical_analysis(data_matrix, model_names=None, block_labels=None, save_to_csv=True, output_dir='statistical_results'):

    # Convert DataFrame to numpy array if needed
    if isinstance(data_matrix, pd.DataFrame):
        if model_names is None:
            model_names = list(data_matrix.columns)
        if block_labels is None:
            block_labels = list(data_matrix.index)
        data_matrix = data_matrix.values
    
    # Set default model names if not provided
    if model_names is None:
        model_names = ['Model_1', 'Model_2', 'Model_3']
    
    # Validate matrix dimensions
    if data_matrix.shape[1] != 3:
        raise ValueError(f"Data matrix must have 3 columns (models), got {data_matrix.shape[1]}")
    if data_matrix.shape[0] != 12:
        raise ValueError(f"Data matrix must have 12 rows (blocks), got {data_matrix.shape[0]}")
    
    print("\n" + "="*90)
    print("STATISTICAL ANALYSIS: FRIEDMAN TEST ON RAW EXPERIMENTAL DATA")
    print("="*90)
    print(f"\nAnalyzing {data_matrix.shape[0]} blocks across {data_matrix.shape[1]} models")
    print("="*90)
    
    # Display data matrix
    print("\nData Matrix (12 blocks × 3 models):")
    print("-" * 90)
    print(f"{'Block':<15} {model_names[0]:>20} {model_names[1]:>20} {model_names[2]:>20}")
    print("-" * 90)
    for i in range(data_matrix.shape[0]):
        if block_labels is not None:
            label = block_labels[i]
        else:
            label = f"Block_{i+1}"
        print(f"{label:<15} {data_matrix[i, 0]:20.6f} {data_matrix[i, 1]:20.6f} {data_matrix[i, 2]:20.6f}")
    print("-" * 90)
    
    # STEP 1: Friedman ANOVA Test
    print("\n" + "="*90)
    print("STEP 1: FRIEDMAN ANOVA TEST")
    print("="*90)
    friedman_results = perform_friedman_test(data_matrix, model_names)
    
    # Initialize results dictionary
    all_results = {
        'data_matrix': data_matrix,
        'block_labels': block_labels,
        'model_names': model_names,
        'friedman': friedman_results,
        'effect_size': None,
        'posthoc': None,
        'hommel': None,
        'pairwise_effect_sizes': None
    }
    
    # Only proceed with further tests if Friedman test is significant
    if friedman_results['is_significant']:
        
        # STEP 2: Kendall's W (Effect Size)
        print("\n" + "="*90)
        print("STEP 2: EFFECT SIZE (Kendall's W)")
        print("="*90)
        n_blocks = data_matrix.shape[0]
        n_models = data_matrix.shape[1]
        effect_size_results = calculate_kendalls_w(
            friedman_results['statistic'],
            n_blocks,
            n_models
        )
        all_results['effect_size'] = effect_size_results
        
        # STEP 3: Nemenyi Post-hoc Test
        print("\n" + "="*90)
        print("STEP 3: POST-HOC TEST (Nemenyi)")
        print("="*90)
        posthoc_results = perform_nemenyi_posthoc(data_matrix, model_names)
        all_results['posthoc'] = posthoc_results
        
        # STEP 4: Hommel Correction
        print("\n" + "="*90)
        print("STEP 4: HOMMEL CORRECTION")
        print("="*90)
        hommel_results = apply_hommel_correction(
            posthoc_results['p_matrix'],
            model_names
        )
        all_results['hommel'] = hommel_results
        
        # STEP 5: Pairwise Effect Sizes (Cliff's Delta)
        print("\n" + "="*90)
        print("STEP 5: PAIRWISE EFFECT SIZES")
        print("="*90)
        pairwise_effect_sizes = calculate_pairwise_effect_sizes(data_matrix, model_names)
        all_results['pairwise_effect_sizes'] = pairwise_effect_sizes
        
    else:
        print("\n" + "="*90)
        print("STEPS 2-5: SKIPPED (Friedman test not significant)")
        print("="*90)
    
    # Summary
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    print("\nModel Performance (Mean Accuracy across 12 blocks):")
    for i, model in enumerate(model_names):
        mean_acc = data_matrix[:, i].mean()
        std_acc = data_matrix[:, i].std()
        print(f"  {model:15s}: {mean_acc:.6f} (±{std_acc:.6f})")
    
    print("\n" + "="*90)
    print("ANALYSIS COMPLETE")
    print("="*90)
    
    # Save results to CSV
    if save_to_csv:
        save_results_to_csv(all_results, output_dir)
    
    return all_results


if __name__ == "__main__":
    # Example usage
    print("Statistical Analysis Module")
    print("\nImport this module and use:")
    print("  from statistical_analysis import run_complete_statistical_analysis")
    print("\nExample:")
    print("  # With numpy array:")
    print("  data_matrix = np.array([[0.73, 0.74, 0.75], ...])  # 12×3 matrix")
    print("  results = run_complete_statistical_analysis(")
    print("      data_matrix,")
    print("      model_names=['RandomForest', 'SVM', 'XGBoost'],")
    print("      block_labels=['Real_P1', 'Real_P2', ..., 'VAE_P4']")
    print("  )")
    print("\n  # With DataFrame:")
    print("  df = pd.DataFrame(data_matrix, columns=['RandomForest', 'SVM', 'XGBoost'])")
    print("  results = run_complete_statistical_analysis(df)")

