import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from statsmodels.stats.multitest import multipletests


def perform_friedman_test(results_df, metric='Accuracy'):
    """
    Perform Friedman ANOVA test to compare model groups across dataset blocks
    Uses mean performance per group per dataset block (3×3 matrix)
    
    Parameters:
    -----------
    results_df : DataFrame
        Results dataframe with columns: Dataset, Model_Group, Parameter_Set, metrics
    metric : str
        Metric to test ('Accuracy', 'Precision', 'Recall', 'F1_Score')
    
    Returns:
    --------
    dict : Dictionary containing test results
    """
    print("="*80)
    print(f"FRIEDMAN ANOVA TEST - {metric}")
    print("Comparing Model Groups across Dataset Blocks (using mean per group per dataset)")
    print("="*80)
    
    group_ids = ['G1', 'G2', 'G3']
    group_names = ['RandomForest', 'SVM', 'XGBoost']
    dataset_ids = ['D1', 'D2', 'D3']
    dataset_names = ['Real', 'GAN', 'VAE']
    
    # Compute mean for each group in each dataset block
    print(f"\nComputing mean {metric} for each group in each dataset block:")
    print("-" * 80)
    
    mean_matrix = []
    mean_data = {}
    
    for dataset_id, dataset_name in zip(dataset_ids, dataset_names):
        row = []
        mean_data[dataset_id] = {}
        
        print(f"\n{dataset_id} ({dataset_name}):")
        for group_id, group_name in zip(group_ids, group_names):
            # Get all runs for this group in this dataset
            mask = (results_df['Dataset'] == dataset_id) & (results_df['Model_Group'] == group_id)
            group_values = results_df[mask][metric].values
            group_mean = group_values.mean()
            
            row.append(group_mean)
            mean_data[dataset_id][group_id] = {
                'mean': group_mean,
                'std': group_values.std(),
                'n_runs': len(group_values),
                'values': group_values
            }
            
            print(f"  {group_id} ({group_name:12s}): Mean = {group_mean:.6f}, Std = {group_values.std():.6f}, n = {len(group_values)}")
        
        mean_matrix.append(row)
    
    # Convert to numpy array for Friedman test
    # Rows = dataset blocks (repeated measures), Columns = model groups (treatments)
    mean_matrix = np.array(mean_matrix)
    
    print(f"\n{metric} Mean Matrix (3 datasets × 3 model groups):")
    print("-" * 80)
    print(f"{'Dataset':<12} {group_ids[0]:>15} {group_ids[1]:>15} {group_ids[2]:>15}")
    for i, (dataset_id, dataset_name) in enumerate(zip(dataset_ids, dataset_names)):
        print(f"{dataset_id} ({dataset_name:4s})  {mean_matrix[i, 0]:15.6f} {mean_matrix[i, 1]:15.6f} {mean_matrix[i, 2]:15.6f}")
    
    # Perform Friedman test on columns (model groups)
    # Each column represents one model group across the 3 dataset blocks
    group_1_means = mean_matrix[:, 0]  # G1 across D1, D2, D3
    group_2_means = mean_matrix[:, 1]  # G2 across D1, D2, D3
    group_3_means = mean_matrix[:, 2]  # G3 across D1, D2, D3
    
    friedman_stat, friedman_p = friedmanchisquare(group_1_means, group_2_means, group_3_means)
    
    print(f"\nFriedman Test Results:")
    print(f"  Metric: {metric}")
    print(f"  Model Groups: G1 (RandomForest), G2 (SVM), G3 (XGBoost)")
    print(f"  Dataset Blocks: D1 (Real), D2 (GAN), D3 (VAE)")
    print(f"  Chi-square statistic: {friedman_stat:.4f}")
    print(f"  P-value: {friedman_p:.6f}")
    print(f"  Significance level: α = 0.05")
    
    is_significant = friedman_p < 0.05
    
    if is_significant:
        print(f"\n  ✓ SIGNIFICANT (p < 0.05): Model groups differ significantly in {metric}")
    else:
        print(f"\n  ✗ NOT SIGNIFICANT (p >= 0.05): No significant differences between model groups")
    
    print("="*80)
    
    return {
        'metric': metric,
        'statistic': friedman_stat,
        'p_value': friedman_p,
        'is_significant': is_significant,
        'group_ids': group_ids,
        'group_names': group_names,
        'mean_matrix': mean_matrix,
        'mean_data': mean_data,
        'group_means_across_datasets': [group_1_means, group_2_means, group_3_means]
    }


def calculate_effect_size(friedman_stat, k=3, n=12):
    """
    Calculate Kendall's W (Effect Size)
    
    Parameters:
    -----------
    friedman_stat : float
        Chi-square statistic from Friedman test
    k : int
        Number of datasets (raters)
    n : int
        Number of model configurations
    
    Returns:
    --------
    dict : Dictionary containing effect size results
    """
    print("\n" + "="*80)
    print("EFFECT SIZE: Kendall's W (Coefficient of Concordance)")
    print("="*80)
    
    # Kendall's W = Chi-square / (k * (n - 1))
    kendalls_w = friedman_stat / (k * (n - 1))
    
    print(f"\nKendall's W: {kendalls_w:.4f}")
    print(f"\nInterpretation:")
    
    if kendalls_w < 0.3:
        interpretation = "Small effect size (W < 0.3)"
    elif kendalls_w < 0.5:
        interpretation = "Medium effect size (0.3 ≤ W < 0.5)"
    else:
        interpretation = "Large effect size (W ≥ 0.5)"
    
    print(f"  {interpretation}")
    print("="*80)
    
    return {
        'kendalls_w': kendalls_w,
        'interpretation': interpretation
    }


def perform_posthoc_test(mean_matrix, group_ids, group_names, metric='Accuracy'):
    """
    Perform Nemenyi post-hoc test for pairwise model group comparisons
    
    Parameters:
    -----------
    mean_matrix : numpy array
        3×3 matrix (dataset blocks × model groups)
    group_ids : list
        Group identifiers (e.g., ['G1', 'G2', 'G3'])
    group_names : list
        Group names (e.g., ['RandomForest', 'SVM', 'XGBoost'])
    metric : str
        Name of the metric being tested
    
    Returns:
    --------
    dict : Post-hoc test results
    """
    print("\n" + "="*80)
    print(f"POST-HOC TEST: Nemenyi-Friedman Test - {metric}")
    print("Pairwise comparisons between Model Groups")
    print("="*80)
    
    # Perform Nemenyi post-hoc test
    # mean_matrix is already in correct format (rows = dataset blocks, columns = model groups)
    print("\nPerforming Nemenyi post-hoc test on model groups...")
    posthoc_results = posthoc_nemenyi_friedman(mean_matrix)
    
    # Set column and index names
    labels = [f'{gid} ({gn})' for gid, gn in zip(group_ids, group_names)]
    posthoc_results.columns = labels
    posthoc_results.index = labels
    
    print("\nPairwise P-values (Nemenyi-Friedman Test):")
    print(posthoc_results)
    
    print("\nSignificant Pairwise Differences (p < 0.05):")
    significant_pairs = []
    
    for i in range(len(posthoc_results)):
        for j in range(i+1, len(posthoc_results)):
            group1 = posthoc_results.index[i]
            group2 = posthoc_results.columns[j]
            p_val = posthoc_results.iloc[i, j]
            
            if p_val < 0.05:
                print(f"  ✓ {group1} vs {group2}: p = {p_val:.6f} (SIGNIFICANT)")
                significant_pairs.append((group1, group2, p_val, True))
            else:
                print(f"  ✗ {group1} vs {group2}: p = {p_val:.6f} (not significant)")
                significant_pairs.append((group1, group2, p_val, False))
    
    print("="*80)
    
    return {
        'metric': metric,
        'posthoc_matrix': posthoc_results,
        'significant_pairs': significant_pairs
    }


def perform_hommel_correction(mean_matrix, group_ids, group_names, metric='Accuracy'):
    """
    Perform Hommel correction for multiple comparisons
    
    Parameters:
    -----------
    mean_matrix : numpy array
        3×3 matrix (dataset blocks × model groups)
    group_ids : list
        Group identifiers (e.g., ['G1', 'G2', 'G3'])
    group_names : list
        Group names (e.g., ['RandomForest', 'SVM', 'XGBoost'])
    metric : str
        Name of the metric being tested
    
    Returns:
    --------
    dict : Hommel correction results
    """
    print("\n" + "="*80)
    print(f"POST-HOC TEST: Hommel Correction - {metric}")
    print("Multiple comparison correction for pairwise tests")
    print("="*80)
    
    # First, perform Nemenyi test to get raw p-values
    print("\nComputing pairwise p-values using Nemenyi test...")
    posthoc_results = posthoc_nemenyi_friedman(mean_matrix)
    
    # Extract upper triangle p-values (avoid duplicates)
    n_groups = len(group_ids)
    raw_pvalues = []
    comparisons = []
    
    for i in range(n_groups):
        for j in range(i+1, n_groups):
            p_val = posthoc_results.iloc[i, j]
            raw_pvalues.append(p_val)
            comparisons.append((group_ids[i], group_names[i], group_ids[j], group_names[j]))
    
    # Apply Hommel correction
    reject, corrected_pvalues, _, _ = multipletests(
        raw_pvalues, 
        alpha=0.05, 
        method='hommel'
    )
    
    print(f"\nHommel Correction Results:")
    print("-" * 80)
    print(f"{'Comparison':<35} {'Raw p-value':>12} {'Corrected p':>12} {'Significant':>12}")
    print("-" * 80)
    
    significant_pairs = []
    
    for idx, ((gid1, gname1, gid2, gname2), raw_p, corr_p, is_sig) in enumerate(
        zip(comparisons, raw_pvalues, corrected_pvalues, reject)
    ):
        comparison_label = f"{gid1} ({gname1}) vs {gid2} ({gname2})"
        sig_label = "✓ YES" if is_sig else "✗ NO"
        
        print(f"{comparison_label:<35} {raw_p:12.6f} {corr_p:12.6f} {sig_label:>12}")
        
        significant_pairs.append({
            'group1_id': gid1,
            'group1_name': gname1,
            'group2_id': gid2,
            'group2_name': gname2,
            'raw_pvalue': raw_p,
            'corrected_pvalue': corr_p,
            'is_significant': is_sig
        })
    
    print("-" * 80)
    n_significant = sum(reject)
    print(f"\nSignificant comparisons (after Hommel correction): {n_significant}/{len(raw_pvalues)}")
    print("="*80)
    
    return {
        'metric': metric,
        'method': 'hommel',
        'alpha': 0.05,
        'comparisons': significant_pairs,
        'n_comparisons': len(raw_pvalues),
        'n_significant': n_significant
    }


def run_complete_statistical_analysis(results_df):
    """
    Run complete statistical analysis pipeline
    Compares MODEL GROUPS ONLY across dataset blocks using mean performance
    
    Parameters:
    -----------
    results_df : DataFrame
        Complete results dataframe with all experiments
    
    Returns:
    --------
    dict : Dictionary containing all statistical test results
    """
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: MODEL GROUP COMPARISON")
    print("Comparing RandomForest vs SVM vs XGBoost across Dataset Blocks")
    print("="*80)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    all_results = {}
    
    # Test each metric
    for metric in metrics:
        print(f"\n{'#'*80}")
        print(f"ANALYZING METRIC: {metric}")
        print(f"{'#'*80}")
        
        # Step 1: Friedman Test
        friedman_results = perform_friedman_test(results_df, metric=metric)
        
        metric_results = {
            'friedman': friedman_results
        }
        
        # Step 2: Effect Size (only if significant)
        if friedman_results['is_significant']:
            k = 3  # 3 model groups
            n = 3  # 3 dataset blocks (repeated measures)
            
            effect_size_results = calculate_effect_size(
                friedman_results['statistic'],
                k=k,
                n=n
            )
            metric_results['effect_size'] = effect_size_results
            
            # Step 3: Post-hoc Test (only if significant)
            posthoc_results = perform_posthoc_test(
                friedman_results['mean_matrix'],
                friedman_results['group_ids'],
                friedman_results['group_names'],
                metric=metric
            )
            metric_results['posthoc'] = posthoc_results
            
            # Step 4: Hommel Correction (only if significant)
            hommel_results = perform_hommel_correction(
                friedman_results['mean_matrix'],
                friedman_results['group_ids'],
                friedman_results['group_names'],
                metric=metric
            )
            metric_results['hommel'] = hommel_results
        else:
            print(f"\n✗ Effect size and post-hoc tests not performed for {metric} (not significant)")
            print("="*80)
        
        all_results[metric] = metric_results
    
    # Print Overall Summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY: MODEL GROUP PERFORMANCE")
    print("="*80)
    
    print("\nStatistical Significance (Friedman Test):")
    print("-" * 80)
    for metric in metrics:
        friedman = all_results[metric]['friedman']
        sig = "✓ SIGNIFICANT" if friedman['is_significant'] else "✗ NOT SIGNIFICANT"
        print(f"  {metric:15s}: χ²={friedman['statistic']:7.4f}, p={friedman['p_value']:.6f} {sig}")
    
    # Determine best model per metric
    print("\n" + "="*80)
    print("BEST PERFORMING MODEL GROUP PER METRIC")
    print("="*80)
    
    for metric in metrics:
        friedman = all_results[metric]['friedman']
        mean_matrix = friedman['mean_matrix']
        
        # Calculate overall mean for each model group across all dataset blocks
        overall_means = mean_matrix.mean(axis=0)  # Mean across dataset blocks
        
        group_ids = friedman['group_ids']
        group_names = friedman['group_names']
        
        print(f"\n{metric}:")
        print("-" * 80)
        
        # Sort by performance
        sorted_indices = np.argsort(overall_means)[::-1]  # Descending order
        
        for rank, idx in enumerate(sorted_indices, 1):
            group_id = group_ids[idx]
            group_name = group_names[idx]
            mean_perf = overall_means[idx]
            
            if rank == 1:
                print(f"BEST:  {group_id} ({group_name:12s}): {mean_perf:.6f}")
            elif rank == 2:
                print(f"2nd:   {group_id} ({group_name:12s}): {mean_perf:.6f}")
            else:
                print(f"3rd:   {group_id} ({group_name:12s}): {mean_perf:.6f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    # Example usage
    print("Statistical Analysis Module")
    print("Import this module and use: run_complete_statistical_analysis(results_df)")
