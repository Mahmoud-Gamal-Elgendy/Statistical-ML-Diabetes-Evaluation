"""
Visualization Module - NEW WORKFLOW
Create visualizations for 12×3 results matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_heatmap(results_matrix, figsize=(10, 8), cmap='YlGnBu', annot=True):
    """
    Create heatmap of 12×3 accuracy matrix
    
    Parameters:
    -----------
    results_matrix : DataFrame
        12×3 matrix with block labels as index, model names as columns
    figsize : tuple
        Figure size
    cmap : str
        Color map
    annot : bool
        Show values in cells
    
    Returns:
    --------
    Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    sns.heatmap(
        results_matrix,
        annot=annot,
        fmt='.4f',
        cmap=cmap,
        cbar_kws={'label': 'Accuracy (max=1)'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )
    
    ax.set_title('Model Performance Across Data Blocks (10-Fold CV Accuracy)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Data Block', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_model_comparison(results_matrix, figsize=(12, 6)):
    """
    Bar plot comparing models across all blocks
    
    Parameters:
    -----------
    results_matrix : DataFrame
        12×3 matrix
    figsize : tuple
        Figure size
    
    Returns:
    --------
    Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Average accuracy per model
    ax1 = axes[0]
    model_means = results_matrix.mean()
    model_stds = results_matrix.std()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax1.bar(model_means.index, model_means.values, color=colors, alpha=0.7, 
                   yerr=model_stds.values, capsize=5)
    
    ax1.set_title('Average Accuracy by Model', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Accuracy (max=1)', fontsize=11)
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylim([0.5, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(model_means.values, model_stds.values)):
        ax1.text(i, mean + std + 0.005, f'{mean:.4f}\n±{std:.4f}', 
                ha='center', fontweight='bold', fontsize=9)
    
    # Right: Box plot
    ax2 = axes[1]
    results_matrix.boxplot(ax=ax2, patch_artist=True, grid=False)
    
    # Color boxes
    for patch, color in zip(ax2.artists, colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('Accuracy Distribution by Model', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy (max=1)', fontsize=11)
    ax2.set_xlabel('Model', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_block_comparison(results_matrix, figsize=(14, 8)):
    """
    Compare performance across data blocks
    
    Parameters:
    -----------
    results_matrix : DataFrame
        12×3 matrix
    figsize : tuple
        Figure size
    
    Returns:
    --------
    Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    x = np.arange(len(results_matrix))
    width = 0.25
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    model_names = results_matrix.columns
    
    for i, model in enumerate(model_names):
        offset = width * (i - 1)
        ax.bar(x + offset, results_matrix[model], width, 
               label=model, color=colors[i], alpha=0.8)
    
    ax.set_title('Model Performance Across Data Blocks', fontsize=14, fontweight='bold')
    ax.set_xlabel('Data Block', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (max=1)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_matrix.index, rotation=45, ha='right')
    ax.legend(title='Model', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_dataset_comparison(results_matrix, figsize=(12, 6)):
    """
    Compare performance grouped by dataset (Real, CTGAN, VAE)
    
    Parameters:
    -----------
    results_matrix : DataFrame
        12×3 matrix
    figsize : tuple
        Figure size
    
    Returns:
    --------
    Figure object
    """
    # Group blocks by dataset
    dataset_groups = {
        'Real': [idx for idx in results_matrix.index if idx.startswith('Real')],
        'CTGAN': [idx for idx in results_matrix.index if idx.startswith('CTGAN')],
        'VAE': [idx for idx in results_matrix.index if idx.startswith('VAE')]
    }
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    dataset_colors = {'Real': '#2ecc71', 'CTGAN': '#3498db', 'VAE': '#e74c3c'}
    
    for ax_idx, (dataset_name, blocks) in enumerate(dataset_groups.items()):
        ax = axes[ax_idx]
        dataset_data = results_matrix.loc[blocks]
        
        x = np.arange(len(blocks))
        width = 0.25
        
        for i, model in enumerate(results_matrix.columns):
            offset = width * (i - 1)
            ax.bar(x + offset, dataset_data[model], width,
                   label=model, color=colors[i], alpha=0.8)
        
        ax.set_title(f'{dataset_name} Dataset', fontsize=12, fontweight='bold',
                    color=dataset_colors[dataset_name])
        ax.set_xlabel('Block', fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel('Accuracy (max=1)', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([b.split('_')[-1] for b in blocks])
        ax.grid(axis='y', alpha=0.3)
        
        if ax_idx == 2:
            ax.legend(title='Model', fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('Performance Comparison by Dataset Type', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_who_won_chart(results_matrix, summary_stats=None, figsize=(10, 7)):
    """
    Chart 1: "Who Won?" - Bar chart with error bars showing model comparison
    
    Parameters:
    -----------
    results_matrix : DataFrame
        12×3 matrix
    summary_stats : DataFrame
        Summary statistics with Mean_Accuracy and Std_Accuracy columns
    figsize : tuple
        Figure size
    
    Returns:
    --------
    Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if summary_stats is not None:
        # Use summary statistics
        models = summary_stats['Model_Name'].values
        means = summary_stats['Mean_Accuracy'].values
        stds = summary_stats['Std_Accuracy'].values
    else:
        # Calculate from results_matrix
        models = results_matrix.columns.tolist()
        means = results_matrix.mean().values
        stds = results_matrix.std().values
    
    # Define colors - XGBoost and RF green (winners), SVM red (loser)
    colors = ['#27ae60', '#e74c3c', '#27ae60']  # Green, Red, Green
    
    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, color=colors, 
                  alpha=0.85, edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2},
                  width=0.5)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.01, f'{mean:.4f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax.text(i, mean/2, f'±{std:.4f}', 
                ha='center', va='center', fontweight='bold', fontsize=10, color='white')
    
    ax.set_ylabel('Mean Accuracy (max=1)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Comparison: "Who Won?"', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.set_ylim([0.5, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='0.80 threshold')
    
    # Add legend explaining colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#27ae60', label='Top Performers (RF & XGBoost)'),
                      Patch(facecolor='#e74c3c', label='Poor Performer (SVM)')]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_real_vs_synthetic(results_matrix, figsize=(14, 7)):
    """
    Chart 2: "Real vs. Synthetic" - Grouped bar chart
    
    Parameters:
    -----------
    results_matrix : DataFrame
        12×3 matrix with blocks labeled as Real_BlockX, CTGAN_BlockX, VAE_BlockX
    figsize : tuple
        Figure size
    
    Returns:
    --------
    Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Group blocks by dataset type
    dataset_groups = {}
    for dataset_type in ['Real', 'CTGAN', 'VAE']:
        blocks = [idx for idx in results_matrix.index if idx.startswith(dataset_type)]
        dataset_groups[dataset_type] = results_matrix.loc[blocks].mean()
    
    # Create grouped data
    grouped_data = pd.DataFrame(dataset_groups).T
    
    # Set up bar positions
    x = np.arange(len(grouped_data))
    width = 0.25
    models = grouped_data.columns
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
    model_colors = {'RandomForest': '#3498db', 'SVM': '#e74c3c', 'XGBoost': '#2ecc71'}
    
    # Plot grouped bars
    for i, model in enumerate(models):
        offset = width * (i - 1)
        bars = ax.bar(x + offset, grouped_data[model], width, 
                     label=model, color=colors[i], alpha=0.85, 
                     edgecolor='black', linewidth=1)
        
        # Add value labels
        for j, (bar, val) in enumerate(zip(bars, grouped_data[model])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Mean Accuracy (max=1)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset Type', fontsize=14, fontweight='bold')
    ax.set_title('Performance Comparison: Real vs. Synthetic Data', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped_data.index, fontsize=13, fontweight='bold')
    ax.legend(title='Model', fontsize=11, title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim([0.5, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


def plot_critical_difference_diagram(results_matrix, posthoc_results, figsize=(12, 6)):
    """
    Chart 4: Critical Difference Diagram - Statistical significance visualization
    
    Parameters:
    -----------
    results_matrix : DataFrame
        12×3 matrix
    posthoc_results : DataFrame
        Nemenyi post-hoc test results
    figsize : tuple
        Figure size
    
    Returns:
    --------
    Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Calculate average ranks
    ranks = results_matrix.rank(axis=1, ascending=False).mean()
    sorted_models = ranks.sort_values().index.tolist()
    sorted_ranks = ranks.sort_values().values
    
    # Create diagram
    y_positions = np.arange(len(sorted_models))
    
    # Plot horizontal line and model positions
    ax.plot([0, max(sorted_ranks) + 0.5], [len(sorted_models)/2, len(sorted_models)/2], 
            'k-', linewidth=2, alpha=0.3)
    
    # Plot each model
    colors_map = {'RandomForest': '#3498db', 'SVM': '#e74c3c', 'XGBoost': '#2ecc71'}
    for i, (model, rank) in enumerate(zip(sorted_models, sorted_ranks)):
        ax.plot([rank, rank], [len(sorted_models)/2 - 0.15, len(sorted_models)/2 + 0.15], 
                color=colors_map.get(model, 'gray'), linewidth=4)
        ax.text(rank, len(sorted_models)/2 + 0.5, model, 
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                color=colors_map.get(model, 'gray'))
        ax.text(rank, len(sorted_models)/2 - 0.5, f'Rank: {rank:.2f}', 
                ha='center', va='top', fontsize=10, style='italic')
    
    # Draw CD bars for non-significant pairs
    if posthoc_results is not None:
        cd_height = len(sorted_models)/2 + 1.0
        for _, row in posthoc_results.iterrows():
            if not row['significant']:
                model1_idx = sorted_models.index(row['Model_1'])
                model2_idx = sorted_models.index(row['Model_2'])
                rank1 = sorted_ranks[model1_idx]
                rank2 = sorted_ranks[model2_idx]
                
                # Draw thick line connecting non-significant models
                ax.plot([rank1, rank2], [cd_height, cd_height], 
                       'k-', linewidth=6, alpha=0.6)
                ax.plot([rank1, rank1], [cd_height - 0.1, cd_height + 0.1], 
                       'k-', linewidth=4)
                ax.plot([rank2, rank2], [cd_height - 0.1, cd_height + 0.1], 
                       'k-', linewidth=4)
                ax.text((rank1 + rank2) / 2, cd_height + 0.3, 'Not Significantly Different',
                       ha='center', fontsize=10, style='italic', bbox=dict(boxstyle='round', 
                       facecolor='yellow', alpha=0.5))
    
    ax.set_xlim([0.5, max(sorted_ranks) + 0.5])
    ax.set_ylim([-1, len(sorted_models)/2 + 2])
    ax.set_xlabel('Average Rank (lower is better)', fontsize=14, fontweight='bold')
    ax.set_title('Critical Difference Diagram - Statistical Significance of Model Performance', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add legend
    ax.text(0.02, 0.98, 'Thick black bar = Models are statistically similar\n(No significant difference)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig


def save_all_visualizations(results_matrix, output_dir='visualizations'):
    """
    Generate and save all visualizations
    
    Parameters:
    -----------
    results_matrix : DataFrame
        12×3 accuracy matrix
    output_dir : str
        Directory to save figures
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    
    # Read summary statistics if available
    stats_path = Path('statistical_results/model_summary_statistics.csv')
    posthoc_path = Path('statistical_results/posthoc_nemenyi_results.csv')
    
    summary_stats = None
    posthoc_results = None
    
    if stats_path.exists():
        summary_stats = pd.read_csv(stats_path)
    if posthoc_path.exists():
        posthoc_results = pd.read_csv(posthoc_path)
    
    # 1. Heatmap (Consistency Heatmap)
    print("  [1/7] Creating consistency heatmap...")
    fig1 = plot_heatmap(results_matrix)
    fig1.savefig(output_path / 'heatmap_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. "Who Won?" Chart - Bar chart with error bars
    print("  [2/7] Creating 'Who Won?' chart...")
    fig2 = plot_who_won_chart(results_matrix, summary_stats)
    fig2.savefig(output_path / 'who_won_chart.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Real vs. Synthetic - Grouped bar chart
    print("  [3/7] Creating Real vs. Synthetic comparison...")
    fig3 = plot_real_vs_synthetic(results_matrix)
    fig3.savefig(output_path / 'real_vs_synthetic.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Critical Difference Diagram
    if posthoc_results is not None and summary_stats is not None:
        print("  [4/7] Creating critical difference diagram...")
        fig4 = plot_critical_difference_diagram(results_matrix, posthoc_results)
        fig4.savefig(output_path / 'critical_difference_diagram.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
    else:
        print("  [4/7] Skipping critical difference diagram (statistical results not found)")
    
    # 5. Model comparison (original)
    print("  [5/7] Creating model comparison...")
    fig5 = plot_model_comparison(results_matrix)
    fig5.savefig(output_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    # 6. Block comparison (original)
    print("  [6/7] Creating block comparison...")
    fig6 = plot_block_comparison(results_matrix)
    fig6.savefig(output_path / 'block_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig6)
    
    # 7. Dataset comparison (original)
    print("  [7/7] Creating dataset comparison...")
    fig7 = plot_dataset_comparison(results_matrix)
    fig7.savefig(output_path / 'dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig7)
    
    print(f"\n✓ All visualizations saved to '{output_path}' directory:")
    print(f"  - heatmap_accuracy.png (Consistency Heatmap)")
    print(f"  - who_won_chart.png (Model Comparison with Error Bars)")
    print(f"  - real_vs_synthetic.png (Dataset Type Comparison)")
    if posthoc_results is not None:
        print(f"  - critical_difference_diagram.png (Statistical Significance)")
    print(f"  - model_comparison.png")
    print(f"  - block_comparison.png")
    print(f"  - dataset_comparison.png")


if __name__ == "__main__":
    print("Visualization Module - NEW WORKFLOW")
    print("\nUsage:")
    print("  from visualization import save_all_visualizations")
    print("\n  # results_matrix is a 12×3 DataFrame")
    print("  save_all_visualizations(results_matrix)")
