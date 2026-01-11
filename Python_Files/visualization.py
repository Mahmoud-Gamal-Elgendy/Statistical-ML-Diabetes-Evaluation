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
        cbar_kws={'label': 'Accuracy'},
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
    ax1.set_ylabel('Mean Accuracy', fontsize=11)
    ax1.set_ylim([0.7, 0.85])
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
    ax2.set_ylabel('Accuracy', fontsize=11)
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
    ax.set_ylabel('Accuracy (10-Fold CV)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_matrix.index, rotation=45, ha='right')
    ax.legend(title='Model', fontsize=10)
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
            ax.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([b.split('_')[-1] for b in blocks])
        ax.grid(axis='y', alpha=0.3)
        
        if ax_idx == 2:
            ax.legend(title='Model', fontsize=9, loc='lower right')
    
    plt.suptitle('Performance Comparison by Dataset Type', 
                 fontsize=14, fontweight='bold', y=1.02)
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
    
    # 1. Heatmap
    print("  [1/4] Creating heatmap...")
    fig1 = plot_heatmap(results_matrix)
    fig1.savefig(output_path / 'heatmap_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Model comparison
    print("  [2/4] Creating model comparison...")
    fig2 = plot_model_comparison(results_matrix)
    fig2.savefig(output_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Block comparison
    print("  [3/4] Creating block comparison...")
    fig3 = plot_block_comparison(results_matrix)
    fig3.savefig(output_path / 'block_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Dataset comparison
    print("  [4/4] Creating dataset comparison...")
    fig4 = plot_dataset_comparison(results_matrix)
    fig4.savefig(output_path / 'dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    print(f"\n✓ All visualizations saved to '{output_path}' directory:")
    print(f"  - heatmap_accuracy.png")
    print(f"  - model_comparison.png")
    print(f"  - block_comparison.png")
    print(f"  - dataset_comparison.png")


if __name__ == "__main__":
    print("Visualization Module - NEW WORKFLOW")
    print("\nUsage:")
    print("  from visualization import save_all_visualizations")
    print("\n  # results_matrix is a 12×3 DataFrame")
    print("  save_all_visualizations(results_matrix)")
