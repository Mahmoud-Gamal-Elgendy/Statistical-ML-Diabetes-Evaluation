import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from statsmodels.stats.multitest import multipletests


def aggregate_model_results(results_df, output_file='aggregated_model_results.csv'):
    """
    Aggregate experimental results by computing average metrics for each model type.
    
    Parameters:
    -----------
    results_df : DataFrame
        Complete results with all 36 experiments (3 datasets × 3 models × 4 parameters)
    output_file : str
        Output CSV filename
    
    Returns:
    --------
    DataFrame : Aggregated results with one row per model type
    """
    # Determine which metrics to aggregate (handle missing columns)
    available_metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    if 'Training_Time' in results_df.columns:
        available_metrics.append('Training_Time')
    
    # Create aggregation dictionary
    agg_dict = {metric: 'mean' for metric in available_metrics}
    
    # Group by Model_Name and calculate mean of all metrics
    aggregated = results_df.groupby('Model_Name').agg(agg_dict).reset_index()
    
    # Round to 6 decimal places for consistency
    aggregated[available_metrics] = aggregated[available_metrics].round(6)
    
    # Save to CSV
    aggregated.to_csv(output_file, index=False)
    
    print(f"\n✓ Aggregated model results saved to: {output_file}")
    print(f"\nAggregated Results:")
    print(aggregated.to_string(index=False))
    
    return aggregated


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
        'dataset_ids': dataset_ids,
        'dataset_names': dataset_names,
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


def save_statistical_results_to_csv(all_results, output_dir='Reports'):
    """
    Save statistical analysis results to CSV files
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing all statistical test results
    output_dir : str
        Directory to save CSV files (relative to project root, not script location)
    """
    import os
    from pathlib import Path
    
    # Get the project root directory (parent of Python_Files)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Create absolute path to output directory
    output_path = project_root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    
    # 1. Friedman Test Summary
    friedman_data = []
    for metric in metrics:
        friedman = all_results[metric]['friedman']
        friedman_data.append({
            'Metric': metric,
            'Chi_Square': friedman['statistic'],
            'P_Value': friedman['p_value'],
            'Significant': 'YES' if friedman['is_significant'] else 'NO'
        })
    
    friedman_df = pd.DataFrame(friedman_data)
    friedman_df.to_csv(output_path / 'friedman_test_results.csv', index=False)
    
    # 2. Mean Performance by Dataset and Model Group
    mean_performance_data = []
    for metric in metrics:
        friedman = all_results[metric]['friedman']
        mean_matrix = friedman['mean_matrix']
        dataset_ids = friedman['dataset_ids']
        dataset_names = friedman['dataset_names']
        group_ids = friedman['group_ids']
        group_names = friedman['group_names']
        
        for i, (ds_id, ds_name) in enumerate(zip(dataset_ids, dataset_names)):
            for j, (g_id, g_name) in enumerate(zip(group_ids, group_names)):
                mean_performance_data.append({
                    'Metric': metric,
                    'Dataset_ID': ds_id,
                    'Dataset_Name': ds_name,
                    'Model_Group_ID': g_id,
                    'Model_Group_Name': g_name,
                    'Mean_Performance': mean_matrix[i, j]
                })
    
    mean_perf_df = pd.DataFrame(mean_performance_data)
    mean_perf_df.to_csv(output_path / 'mean_performance_by_group.csv', index=False)
    
    # 3. Effect Size Results
    effect_size_data = []
    for metric in metrics:
        if 'effect_size' in all_results[metric]:
            effect_size = all_results[metric]['effect_size']
            effect_size_data.append({
                'Metric': metric,
                'Kendall_W': effect_size['kendalls_w'],
                'Effect_Size_Interpretation': effect_size['interpretation']
            })
    
    if effect_size_data:
        effect_size_df = pd.DataFrame(effect_size_data)
        effect_size_df.to_csv(output_path / 'effect_size_results.csv', index=False)
    
    # 4. Nemenyi Post-hoc Test Results
    posthoc_data = []
    for metric in metrics:
        if 'posthoc' in all_results[metric]:
            posthoc = all_results[metric]['posthoc']
            for pair in posthoc['significant_pairs']:
                posthoc_data.append({
                    'Metric': metric,
                    'Group1': pair[0],
                    'Group2': pair[1],
                    'P_Value': pair[2],
                    'Significant': 'YES' if pair[3] else 'NO'
                })
    
    if posthoc_data:
        posthoc_df = pd.DataFrame(posthoc_data)
        posthoc_df.to_csv(output_path / 'nemenyi_posthoc_results.csv', index=False)
    
    # 5. Hommel Correction Results
    hommel_data = []
    for metric in metrics:
        if 'hommel' in all_results[metric]:
            hommel = all_results[metric]['hommel']
            for comp in hommel['comparisons']:
                hommel_data.append({
                    'Metric': metric,
                    'Group1_ID': comp['group1_id'],
                    'Group1_Name': comp['group1_name'],
                    'Group2_ID': comp['group2_id'],
                    'Group2_Name': comp['group2_name'],
                    'Raw_P_Value': comp['raw_pvalue'],
                    'Corrected_P_Value': comp['corrected_pvalue'],
                    'Significant': 'YES' if comp['is_significant'] else 'NO'
                })
    
    if hommel_data:
        hommel_df = pd.DataFrame(hommel_data)
        hommel_df.to_csv(output_path / 'hommel_correction_results.csv', index=False)
    
    # 6. Best Performing Model Groups Summary
    best_models_data = []
    for metric in metrics:
        friedman = all_results[metric]['friedman']
        mean_matrix = friedman['mean_matrix']
        overall_means = mean_matrix.mean(axis=0)
        group_ids = friedman['group_ids']
        group_names = friedman['group_names']
        
        sorted_indices = np.argsort(overall_means)[::-1]
        
        for rank, idx in enumerate(sorted_indices, 1):
            best_models_data.append({
                'Metric': metric,
                'Rank': rank,
                'Model_Group_ID': group_ids[idx],
                'Model_Group_Name': group_names[idx],
                'Mean_Performance': overall_means[idx]
            })
    
    best_models_df = pd.DataFrame(best_models_data)
    best_models_df.to_csv(output_path / 'best_performing_models.csv', index=False)
    
    print(f"\n✓ Statistical analysis results saved to '{output_path}' directory:")
    print(f"  - friedman_test_results.csv")
    print(f"  - mean_performance_by_group.csv")
    if effect_size_data:
        print(f"  - effect_size_results.csv")
    if posthoc_data:
        print(f"  - nemenyi_posthoc_results.csv")
    if hommel_data:
        print(f"  - hommel_correction_results.csv")
    print(f"  - best_performing_models.csv")


def run_complete_statistical_analysis(aggregated_df, save_to_csv=True, output_dir='statistical_results'):
    """
    Run complete statistical analysis on aggregated model results
    Compares 3 models: RandomForest vs SVM vs XGBoost
    Performs: Friedman ANOVA, Effect Size (Kendall's W), Post-hoc (Nemenyi), Hommel Correction
    
    Parameters:
    -----------
    aggregated_df : DataFrame
        Aggregated results with one row per model (3 rows total)
        Columns: Model_Name, Accuracy, Precision, Recall, F1_Score, Training_Time
    save_to_csv : bool
        Whether to save results to CSV files (default: True)
    output_dir : str
        Directory to save CSV files (default: 'statistical_results')
    
    Returns:
    --------
    dict : Dictionary containing all statistical test results
    """
    from pathlib import Path
    from scipy.stats import friedmanchisquare
    from statsmodels.stats.multitest import multipletests
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: MODEL COMPARISON")
    print("Comparing RandomForest vs SVM vs XGBoost")
    print("="*80)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    
    # Display aggregated results
    print("\n" + "="*80)
    print("AGGREGATED MODEL PERFORMANCE")
    print("="*80)
    print(aggregated_df[['Model_Name'] + metrics].to_string(index=False))
    
    # Note: For 3 models with single aggregated values, we treat each metric as a "repeated measure"
    # This allows us to test if models differ consistently across metrics
    print("\n" + "="*80)
    print("STATISTICAL TESTS")
    print("="*80)
    print("\nTreating each metric as a repeated measure to test model consistency")
    print("across performance indicators (Accuracy, Precision, Recall, F1_Score).")
    
    # Prepare data for Friedman test
    # Create matrix: rows = metrics (repeated measures), columns = models (treatments)
    model_names = aggregated_df['Model_Name'].tolist()
    
    # Build data matrix
    data_matrix = []
    for metric in metrics:
        row = []
        for model in model_names:
            value = aggregated_df[aggregated_df['Model_Name'] == model][metric].values[0]
            row.append(value)
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    print(f"\nData Matrix ({len(metrics)} metrics × {len(model_names)} models):")
    print("-" * 80)
    print(f"{'Metric':<15} {model_names[0]:>15} {model_names[1]:>15} {model_names[2]:>15}")
    for i, metric in enumerate(metrics):
        print(f"{metric:<15} {data_matrix[i, 0]:15.6f} {data_matrix[i, 1]:15.6f} {data_matrix[i, 2]:15.6f}")
    
    # STEP 1: Friedman ANOVA Test
    print("\n" + "="*80)
    print("STEP 1: FRIEDMAN ANOVA TEST")
    print("="*80)
    print("\nNull Hypothesis (H₀): All models perform equally across metrics")
    print("Alternative Hypothesis (H₁): At least one model differs significantly")
    print(f"Significance level: α = 0.05")
    
    # Perform Friedman test (each column is a model, each row is a metric)
    model_1_scores = data_matrix[:, 0]
    model_2_scores = data_matrix[:, 1]
    model_3_scores = data_matrix[:, 2]
    
    friedman_stat, friedman_p = friedmanchisquare(model_1_scores, model_2_scores, model_3_scores)
    
    print(f"\nFriedman Test Results:")
    print(f"  Chi-square statistic (χ²): {friedman_stat:.4f}")
    print(f"  P-value: {friedman_p:.6f}")
    print(f"  Degrees of freedom: {len(model_names) - 1}")
    
    is_significant = friedman_p < 0.05
    
    if is_significant:
        print(f"\n  ✓ SIGNIFICANT (p < 0.05)")
        print(f"  Conclusion: Models show significantly different performance across metrics")
    else:
        print(f"\n  ✗ NOT SIGNIFICANT (p ≥ 0.05)")
        print(f"  Conclusion: No significant difference between models")
    
    friedman_results = {
        'statistic': friedman_stat,
        'p_value': friedman_p,
        'is_significant': is_significant,
        'df': len(model_names) - 1
    }
    
    # STEP 2: Effect Size (Kendall's W)
    print("\n" + "="*80)
    print("STEP 2: EFFECT SIZE - Kendall's W")
    print("="*80)
    
    k = len(metrics)  # number of raters (metrics)
    n = len(model_names)  # number of treatments (models)
    
    # Kendall's W = χ² / (k * (n - 1))
    kendalls_w = friedman_stat / (k * (n - 1))
    
    print(f"\nKendall's W (Coefficient of Concordance): {kendalls_w:.4f}")
    print(f"\nInterpretation:")
    
    if kendalls_w < 0.3:
        w_interpretation = "Small effect size (W < 0.3) - Weak agreement"
    elif kendalls_w < 0.5:
        w_interpretation = "Medium effect size (0.3 ≤ W < 0.5) - Moderate agreement"
    else:
        w_interpretation = "Large effect size (W ≥ 0.5) - Strong agreement"
    
    print(f"  {w_interpretation}")
    print(f"\nThis indicates the degree of consistency in model rankings across metrics.")
    
    effect_size_results = {
        'kendalls_w': kendalls_w,
        'interpretation': w_interpretation
    }
    
    # STEP 3: Post-hoc Test (Nemenyi) - only if significant
    posthoc_results = None
    if is_significant:
        print("\n" + "="*80)
        print("STEP 3: POST-HOC TEST - Nemenyi-Friedman")
        print("="*80)
        print("\nPairwise comparisons between models:")
        
        # Perform Nemenyi post-hoc test
        # data_matrix: rows = metrics, columns = models
        posthoc_matrix = posthoc_nemenyi_friedman(data_matrix)
        
        # Set labels
        posthoc_matrix.columns = model_names
        posthoc_matrix.index = model_names
        
        print("\nP-values for Pairwise Comparisons:")
        print(posthoc_matrix.to_string())
        
        print("\n" + "-" * 80)
        print("Significant Differences (p < 0.05):")
        print("-" * 80)
        
        significant_pairs = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                p_val = posthoc_matrix.iloc[i, j]
                model_i = model_names[i]
                model_j = model_names[j]
                
                if p_val < 0.05:
                    print(f"  ✓ {model_i} vs {model_j}: p = {p_val:.6f} (SIGNIFICANT)")
                    significant_pairs.append({
                        'Model_1': model_i,
                        'Model_2': model_j,
                        'p_value': p_val,
                        'significant': True
                    })
                else:
                    print(f"  ✗ {model_i} vs {model_j}: p = {p_val:.6f} (not significant)")
                    significant_pairs.append({
                        'Model_1': model_i,
                        'Model_2': model_j,
                        'p_value': p_val,
                        'significant': False
                    })
        
        posthoc_results = {
            'p_matrix': posthoc_matrix.to_dict(),
            'significant_pairs': significant_pairs
        }
    else:
        print("\n" + "="*80)
        print("STEP 3: POST-HOC TEST - SKIPPED")
        print("="*80)
        print("\nPost-hoc test not performed (Friedman test was not significant)")
    
    # STEP 4: Hommel Correction - only if significant
    hommel_results = None
    if is_significant:
        print("\n" + "="*80)
        print("STEP 4: HOMMEL MULTIPLE COMPARISON CORRECTION")
        print("="*80)
        print("\nApplying Hommel correction to control family-wise error rate:")
        
        # Extract p-values from post-hoc test
        pairwise_comparisons = []
        p_values = []
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                p_val = posthoc_matrix.iloc[i, j]
                pairwise_comparisons.append((model_names[i], model_names[j]))
                p_values.append(p_val)
        
        # Apply Hommel correction
        reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='hommel')
        
        print("\nHommel-Corrected Results:")
        print("-" * 80)
        
        hommel_pairs = []
        n_significant = 0
        
        for idx, (model_i, model_j) in enumerate(pairwise_comparisons):
            original_p = p_values[idx]
            corrected_p = pvals_corrected[idx]
            is_sig = reject[idx]
            
            if is_sig:
                n_significant += 1
                print(f"  ✓ {model_i} vs {model_j}:")
                print(f"      Original p = {original_p:.6f}, Corrected p = {corrected_p:.6f} (SIGNIFICANT)")
            else:
                print(f"  ✗ {model_i} vs {model_j}:")
                print(f"      Original p = {original_p:.6f}, Corrected p = {corrected_p:.6f} (not significant)")
            
            hommel_pairs.append({
                'Model_1': model_i,
                'Model_2': model_j,
                'original_p': original_p,
                'corrected_p': corrected_p,
                'significant': bool(is_sig)
            })
        
        print(f"\nSummary: {n_significant}/{len(pairwise_comparisons)} comparisons remain significant after correction")
        
        hommel_results = {
            'comparisons': hommel_pairs,
            'n_significant': n_significant,
            'n_comparisons': len(pairwise_comparisons)
        }
    else:
        print("\n" + "="*80)
        print("STEP 4: HOMMEL CORRECTION - SKIPPED")
        print("="*80)
        print("\nHommel correction not performed (Friedman test was not significant)")
    
    # Store all results
    all_results = {
        'model_performance': aggregated_df.to_dict('records'),
        'friedman_test': friedman_results,
        'effect_size': effect_size_results,
        'posthoc_test': posthoc_results,
        'hommel_correction': hommel_results,
        'rankings': [],
        'differences': []
    }
    
    # Calculate rankings and differences
    print("\n" + "="*80)
    print("MODEL RANKINGS AND PERFORMANCE DIFFERENCES")
    print("="*80)
    
    results_summary = []
    difference_summary = []
    
    for metric in metrics:
        print(f"\n{metric}:")
        print("-" * 80)
        
        # Sort models by metric
        sorted_df = aggregated_df.sort_values(by=metric, ascending=False).reset_index(drop=True)
        
        # Display rankings with values
        for rank, row in sorted_df.iterrows():
            model_name = row['Model_Name']
            value = row[metric]
            rank_num = rank + 1
            
            if rank_num == 1:
                print(f"  BEST:  {model_name:15s}: {value:.6f}")
                label = "BEST"
            elif rank_num == 2:
                print(f"  2nd:   {model_name:15s}: {value:.6f}")
                label = "2nd"
            else:
                print(f"  3rd:   {model_name:15s}: {value:.6f}")
                label = "3rd"
            
            results_summary.append({
                'Metric': metric,
                'Rank': rank_num,
                'Model_Name': model_name,
                'Value': value,
                'Label': label
            })
        
        # Calculate pairwise differences
        print(f"\n  Performance Differences:")
        best_value = sorted_df.iloc[0][metric]
        best_model = sorted_df.iloc[0]['Model_Name']
        
        for rank, row in sorted_df.iterrows():
            if rank > 0:  # Skip the best model
                model_name = row['Model_Name']
                value = row[metric]
                diff = best_value - value
                diff_pct = (diff / value) * 100
                
                print(f"    {best_model} vs {model_name}: +{diff:.6f} (+{diff_pct:.2f}%)")
                
                difference_summary.append({
                    'Metric': metric,
                    'Model_1': best_model,
                    'Model_2': model_name,
                    'Difference': diff,
                    'Difference_Percent': diff_pct,
                    'Model_1_Value': best_value,
                    'Model_2_Value': value
                })
    
    all_results['rankings'] = results_summary
    all_results['differences'] = difference_summary
    
    # Summary statistics
    print("\n" + "="*80)
    print("OVERALL MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Calculate average rank for each model across all metrics
    ranking_df = pd.DataFrame(results_summary)
    avg_ranks = ranking_df.groupby('Model_Name')['Rank'].mean().sort_values()
    
    print("\nAverage Rank Across All Metrics (lower is better):")
    for model, avg_rank in avg_ranks.items():
        print(f"  {model:15s}: {avg_rank:.2f}")
    
    # Count how many times each model was best
    best_counts = ranking_df[ranking_df['Label'] == 'BEST'].groupby('Model_Name').size()
    print("\nNumber of Times Ranked BEST:")
    for model in aggregated_df['Model_Name']:
        count = best_counts.get(model, 0)
        print(f"  {model:15s}: {count}/{len(metrics)} metrics")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Save results to CSV if requested
    if save_to_csv:
        # Get the project root directory
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        
        # Create absolute path to output directory
        output_path = project_root / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save aggregated performance summary
        aggregated_df.to_csv(output_path / 'model_performance_summary.csv', index=False)
        
        # 2. Save ranking results
        ranking_df = pd.DataFrame(results_summary)
        ranking_df.to_csv(output_path / 'model_ranking_by_metric.csv', index=False)
        
        # 3. Save comparison matrix
        comparison_df = aggregated_df[['Model_Name'] + metrics].set_index('Model_Name')
        comparison_df.to_csv(output_path / 'model_comparison_matrix.csv')
        
        # 4. Save performance differences
        diff_df = pd.DataFrame(difference_summary)
        diff_df.to_csv(output_path / 'model_performance_differences.csv', index=False)
        
        # 5. Save overall summary
        summary_data = []
        for model in aggregated_df['Model_Name']:
            avg_rank = avg_ranks.get(model, 0)
            best_count = best_counts.get(model, 0)
            summary_data.append({
                'Model_Name': model,
                'Average_Rank': avg_rank,
                'Times_Ranked_Best': best_count,
                'Best_Percentage': (best_count / len(metrics)) * 100
            })
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / 'model_overall_summary.csv', index=False)
        
        # 6. Save Friedman test results
        friedman_df = pd.DataFrame([{
            'Test': 'Friedman ANOVA',
            'Chi_Square': friedman_results['statistic'],
            'P_Value': friedman_results['p_value'],
            'Degrees_of_Freedom': friedman_results['df'],
            'Significant': 'Yes' if friedman_results['is_significant'] else 'No',
            'Conclusion': 'Models differ significantly' if friedman_results['is_significant'] else 'No significant difference'
        }])
        friedman_df.to_csv(output_path / 'friedman_test_results.csv', index=False)
        
        # 7. Save effect size results
        effect_df = pd.DataFrame([{
            'Measure': 'Kendalls_W',
            'Value': effect_size_results['kendalls_w'],
            'Interpretation': effect_size_results['interpretation']
        }])
        effect_df.to_csv(output_path / 'effect_size_results.csv', index=False)
        
        # 8. Save post-hoc test results (if performed)
        if posthoc_results is not None:
            posthoc_df = pd.DataFrame(posthoc_results['significant_pairs'])
            posthoc_df.to_csv(output_path / 'posthoc_nemenyi_results.csv', index=False)
        
        # 9. Save Hommel correction results (if performed)
        if hommel_results is not None:
            hommel_df = pd.DataFrame(hommel_results['comparisons'])
            hommel_df.to_csv(output_path / 'hommel_correction_results.csv', index=False)
        
        print(f"\n✓ Statistical analysis results saved to '{output_path}' directory:")
        print(f"  - model_performance_summary.csv")
        print(f"  - model_ranking_by_metric.csv")
        print(f"  - model_comparison_matrix.csv")
        print(f"  - model_performance_differences.csv")
        print(f"  - model_overall_summary.csv")
        print(f"  - friedman_test_results.csv")
        print(f"  - effect_size_results.csv")
        if posthoc_results is not None:
            print(f"  - posthoc_nemenyi_results.csv")
        if hommel_results is not None:
            print(f"  - hommel_correction_results.csv")
    
    return all_results
    
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
    
    # Save results to CSV if requested
    if save_to_csv:
        save_statistical_results_to_csv(all_results, output_dir=output_dir)
    
    return all_results


if __name__ == "__main__":
    # Example usage
    print("Statistical Analysis Module")
    print("Import this module and use: run_complete_statistical_analysis(results_df)")
