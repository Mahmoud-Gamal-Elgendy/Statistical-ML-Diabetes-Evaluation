"""
Visualization Module
Create comprehensive visualizations for experimental results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_all_experiments_accuracy(results_df, figsize=(18, 6)):
    """
    Plot accuracy for all experiments
    
    Parameters:
    -----------
    results_df : DataFrame
        Results dataframe
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    x_pos = np.arange(len(results_df))
    colors = ['#2ecc71' if d == 'D1' else '#3498db' if d == 'D2' else '#e74c3c' 
              for d in results_df['Dataset']]
    
    ax.bar(x_pos, results_df['Accuracy'], color=colors, alpha=0.7)
    ax.set_title('Accuracy Across All Experiments', fontsize=14, fontweight='bold')
    ax.set_xlabel('Experiment ID', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_xticks(x_pos[::3])
    ax.set_xticklabels(results_df['Experiment_ID'][::3], rotation=45, ha='right', fontsize=8)
    ax.axhline(y=results_df['Accuracy'].mean(), color='black', linestyle='--',
               label=f'Mean: {results_df["Accuracy"].mean():.4f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_comparison_summary(results_df, figsize=(18, 16)):
    """
    Create comprehensive comparison plots
    
    Parameters:
    -----------
    results_df : DataFrame
        Results dataframe
    figsize : tuple
        Figure size
    
    Returns:
    --------
    Figure object
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # 1. Overall Accuracy Comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(results_df))
    colors = ['#2ecc71' if d == 'D1' else '#3498db' if d == 'D2' else '#e74c3c'
              for d in results_df['Dataset']]
    ax1.bar(x_pos, results_df['Accuracy'], color=colors, alpha=0.7)
    ax1.set_title('Accuracy Across All Experiments', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Experiment ID', fontsize=10)
    ax1.set_ylabel('Accuracy', fontsize=10)
    ax1.set_xticks(x_pos[::3])
    ax1.set_xticklabels(results_df['Experiment_ID'][::3], rotation=45, ha='right', fontsize=8)
    ax1.axhline(y=results_df['Accuracy'].mean(), color='black', linestyle='--',
                label=f'Mean: {results_df["Accuracy"].mean():.4f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Average Accuracy by Dataset
    ax2 = axes[0, 1]
    dataset_avg = results_df.groupby('Dataset')['Accuracy'].mean()
    ax2.bar(dataset_avg.index, dataset_avg.values, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax2.set_title('Average Accuracy by Dataset', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Accuracy', fontsize=10)
    for i, v in enumerate(dataset_avg.values):
        ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Average Accuracy by Model Group
    ax3 = axes[1, 0]
    model_avg = results_df.groupby('Model_Group')['Accuracy'].mean()
    ax3.bar(model_avg.index, model_avg.values, color=['#9b59b6', '#f39c12', '#1abc9c'])
    ax3.set_title('Average Accuracy by Model Group', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Average Accuracy', fontsize=10)
    for i, v in enumerate(model_avg.values):
        ax3.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Average Accuracy by Parameter Set
    ax4 = axes[1, 1]
    param_avg = results_df.groupby('Parameter_Set')['Accuracy'].mean()
    ax4.bar(param_avg.index, param_avg.values, color=['#e67e22', '#16a085', '#8e44ad', '#c0392b'])
    ax4.set_title('Average Accuracy by Parameter Set', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Average Accuracy', fontsize=10)
    for i, v in enumerate(param_avg.values):
        ax4.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Heatmap: Model Group vs Dataset
    ax5 = axes[2, 0]
    pivot_heatmap = results_df.groupby(['Model_Group', 'Dataset'])['Accuracy'].mean().unstack()
    sns.heatmap(pivot_heatmap, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax5,
                cbar_kws={'label': 'Accuracy'})
    ax5.set_title('Model Group vs Dataset (Average Accuracy)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Dataset', fontsize=10)
    ax5.set_ylabel('Model Group', fontsize=10)
    
    # 6. Box Plot: Accuracy Distribution by Dataset
    ax6 = axes[2, 1]
    data_for_box = [results_df[results_df['Dataset'] == 'D1']['Accuracy'],
                    results_df[results_df['Dataset'] == 'D2']['Accuracy'],
                    results_df[results_df['Dataset'] == 'D3']['Accuracy']]
    bp = ax6.boxplot(data_for_box, labels=['D1 (Real)', 'D2 (GAN)', 'D3 (VAE)'],
                      patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#3498db', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax6.set_title('Accuracy Distribution by Dataset', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Accuracy', fontsize=10)
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def save_visualizations(results_df, output_dir='visualizations'):
    """
    Save all visualizations to files
    
    Parameters:
    -----------
    results_df : DataFrame
        Results dataframe
    output_dir : str
        Directory to save visualizations
    """
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create and save comprehensive comparison
    fig = plot_comparison_summary(results_df)
    fig.savefig(f'{output_dir}/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Visualizations saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    print("Visualization Module")
    print("Functions available:")
    print("  - plot_all_experiments_accuracy(results_df)")
    print("  - plot_comparison_summary(results_df)")
    print("  - save_visualizations(results_df, output_dir='visualizations')")
