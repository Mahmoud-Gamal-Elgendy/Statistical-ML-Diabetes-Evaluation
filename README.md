# Statistical ML Diabetes Evaluation

A comprehensive machine learning research project evaluating diabetes classification performance across real and synthetic datasets using rigorous cross-validation and statistical validation.

## 📋 Project Overview

This project implements a complete experimental workflow to:
- Generate synthetic health indicator data using **CTGAN** and **VAE**
- Train and evaluate **3 machine learning models** (Random Forest, SVM, XGBoost)
- Use **10-fold stratified cross-validation** on **12 data blocks** (4 blocks per dataset)
- Compare performance across **3 datasets** (Real, CTGAN-generated, VAE-generated)
- Perform rigorous **statistical analysis** using Friedman tests, Nemenyi post-hoc tests, and effect size calculations
- Generate comprehensive visualizations and save all trained models

**Experimental Design**: 12 blocks × 3 models = 36 model configurations  
**Total CV Operations**: 360 model trainings (36 configs × 10 folds)

## ✅ Generated Output Files

All results are automatically saved when running the workflow:

### ✓ Classification Models (36 total)
- **Location**: `models/RandomForest/`, `models/SVM/`, `models/XGBoost/`
- **Files**: 12 models per classifier (Real_Block1-4.pkl, CTGAN_Block1-4.pkl, VAE_Block1-4.pkl)
- **Format**: Pickle (.pkl) files containing trained scikit-learn/XGBoost models
- **Total**: 36 trained classification models ready for predictions

### ✓ Experiment Results Matrix
- **File**: `statistical_results/experiment_results_matrix.csv`
- **Content**: 12×3 accuracy matrix from 10-fold cross-validation
- **Format**: CSV with block labels as rows, model names as columns

### ✓ Statistical Analysis (8 CSV files)
- **Location**: `statistical_results/`
- **Files**:
  - `experiment_results_matrix.csv` - 12×3 accuracy matrix
  - `data_matrix.csv` - Formatted accuracy data for analysis
  - `friedman_test_results.csv` - Friedman ANOVA test results (χ², p-value)
  - `effect_size_results.csv` - Kendall's W effect size
  - `posthoc_nemenyi_results.csv` - Nemenyi pairwise comparisons
  - `hommel_correction_results.csv` - Hommel-corrected p-values
  - `pairwise_effect_sizes_cliffs_delta.csv` - Cliff's Delta effect sizes
  - `model_summary_statistics.csv` - Mean/std accuracy per model

### ✓ Visualizations (4 PNG files)
- **Location**: `visualizations/`
- **Files**:
  - `heatmap_accuracy.png` - Heatmap of all 12×3 accuracies
  - `model_comparison.png` - Bar plots and box plots comparing models
  - `block_comparison.png` - Bar chart across all data blocks
  - `dataset_comparison.png` - Performance grouped by dataset type
- **Resolution**: 300 DPI, publication-quality PNG format

## 🗂️ Project Structure

```
Statistical ML Diabetes Evaluation/
├── .git/                                # Git version control
├── .gitignore                           # Git ignore configuration
├── .venv_py311/                         # Python 3.11 virtual environment
├── .vscode/                             # VS Code settings
│
├── Datasets/                            # Training datasets
│   ├── Real/
│   │   └── diabetes_012_health_indicators_BRFSS2015.csv  # Original BRFSS 2015 dataset (253,680 samples)
│   ├── CTGAN/
│   │   ├── synthetic_1.csv              # Scaled CTGAN synthetic data
│   │   └── synthetic_1_unscaled.csv     # Unscaled CTGAN synthetic data (229,781 samples)
│   └── VAE/
│       ├── synthetic_2.csv              # Scaled VAE synthetic data
│       └── synthetic_2_unscaled.csv     # Unscaled VAE synthetic data (229,781 samples)
│
├── models/                              # Saved models
│   ├── RandomForest/                    # Random Forest classifiers (12 models)
│   │   ├── Real_Block1.pkl              # Real dataset, Block 1
│   │   ├── Real_Block2.pkl              # Real dataset, Block 2
│   │   ├── Real_Block3.pkl              # Real dataset, Block 3
│   │   ├── Real_Block4.pkl              # Real dataset, Block 4
│   │   ├── CTGAN_Block1.pkl             # CTGAN dataset, Block 1
│   │   ├── CTGAN_Block2.pkl             # CTGAN dataset, Block 2
│   │   ├── CTGAN_Block3.pkl             # CTGAN dataset, Block 3
│   │   ├── CTGAN_Block4.pkl             # CTGAN dataset, Block 4
│   │   ├── VAE_Block1.pkl               # VAE dataset, Block 1
│   │   ├── VAE_Block2.pkl               # VAE dataset, Block 2
│   │   ├── VAE_Block3.pkl               # VAE dataset, Block 3
│   │   └── VAE_Block4.pkl               # VAE dataset, Block 4
│   ├── SVM/                             # SVM classifiers (12 models, same structure)
│   │   └── [Real|CTGAN|VAE]_Block[1-4].pkl
│   ├── XGBoost/                         # XGBoost classifiers (12 models, same structure)
│   │   └── [Real|CTGAN|VAE]_Block[1-4].pkl
│   ├── CTGAN/
│   │   ├── ctgan_final.pkl              # Final CTGAN generative model
│   │   └── checkpoints/                 # CTGAN training checkpoints (10 files)
│   └── VAE/
│       ├── vae_final.pkl                # Final VAE generative model
│       └── checkpoints/                 # VAE training checkpoints (10 files)
│
├── Python_Files/                        # Source code
│   ├── main.py                          # Main execution script
│   ├── CTGAN_model.py                   # CTGAN synthetic data generator
│   ├── VAE_model.py                     # VAE synthetic data generator
│   ├── model_training_pipeline.py       # 10-fold CV training pipeline with model saving
│   ├── statistical_analysis.py          # Friedman, Nemenyi, effect size analysis
│   ├── visualization.py                 # Plotting and visualization functions
│   ├── unscale_synthetic_data.py        # Data preprocessing utilities
│   └── test_statistical_analysis.py     # Unit tests (27 tests) for statistical functions
│
├── statistical_results/                 # Statistical analysis outputs (8 CSV files)
│   ├── experiment_results_matrix.csv    # Main 12×3 accuracy matrix
│   ├── data_matrix.csv                  # Formatted data for analysis
│   ├── friedman_test_results.csv        # Friedman ANOVA results
│   ├── effect_size_results.csv          # Kendall's W effect size
│   ├── posthoc_nemenyi_results.csv      # Pairwise post-hoc tests
│   ├── hommel_correction_results.csv    # Multiple testing correction
│   ├── pairwise_effect_sizes_cliffs_delta.csv  # Pairwise effect sizes
│   └── model_summary_statistics.csv     # Mean/std accuracy summary
│
├── visualizations/                      # Generated plots (4 PNG files)
│   ├── heatmap_accuracy.png             # 12×3 accuracy heatmap
│   ├── model_comparison.png             # Model performance bar/box plots
│   ├── block_comparison.png             # Block-wise comparison chart
│   └── dataset_comparison.png           # Dataset-wise performance plots
│
├── Project_workflow/                    # Documentation and diagrams
│   ├── Diagram.excalidraw               # Project workflow diagram
│   └── Workflow Diagram.excalidraw      # Detailed workflow visualization
│
├── Reports/                             # Analysis reports and documentation
│
├── MODEL_CONFIGURATIONS.md              # Detailed hyperparameter documentation
├── WORKFLOW_SUMMARY.md                  # Task completion checklist
└── README.md                            # This file
```

**Summary**:
- **Total Files**: ~60+ files
- **Classification Models**: 36 trained models (.pkl)
- **Generative Models**: 2 final models + 20 checkpoints
- **Statistical Results**: 8 CSV files
- **Visualizations**: 4 PNG files (300 DPI)
- **Source Code**: 8 Python modules
- **Documentation**: 3 markdown files

## 🎯 Research Methodology

### NEW WORKFLOW: Block-Based Cross-Validation

#### Datasets (3)
- **Real**: BRFSS 2015 diabetes health indicators dataset (~253,680 samples)
- **CTGAN**: Synthetic data generated using Conditional Tabular GAN
- **VAE**: Synthetic data generated using Variational Autoencoder

Each dataset is split into **4 equal blocks** → **12 total data blocks**

#### Models (3)
- **RandomForest**: Ensemble decision tree classifier
- **SVM**: Support Vector Machine with RBF kernel
- **XGBoost**: Gradient boosting classifier

#### Model Hyperparameters & Configuration

All models use a **single optimized parameter set** (no grid search) to ensure fair comparison.

##### 1. Random Forest Classifier
**Algorithm**: Ensemble of decision trees with bootstrap aggregation  
**Optimizer**: N/A (deterministic algorithm, no gradient descent)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 100 | Number of trees in the forest |
| `max_depth` | 20 | Maximum depth of each tree |
| `min_samples_split` | 10 | Minimum samples required to split a node |
| `random_state` | 42 | Random seed for reproducibility |
| `n_jobs` | -1 | Use all CPU cores for parallel training |

**Training Method**: Bootstrap sampling with out-of-bag validation  
**Prediction**: Majority voting across all trees

##### 2. Support Vector Machine (SVM)
**Algorithm**: Support Vector Classification with kernel trick  
**Optimizer**: Sequential Minimal Optimization (SMO) - built into sklearn's SVC

| Parameter | Value | Description |
|-----------|-------|-------------|
| `C` | 1.0 | Regularization parameter (penalty for misclassification) |
| `kernel` | 'rbf' | Radial Basis Function (Gaussian) kernel |
| `gamma` | 'scale' | Kernel coefficient = 1 / (n_features × variance) |
| `max_iter` | 1000 | Maximum iterations for solver convergence |
| `random_state` | 42 | Random seed for reproducibility |

**Training Method**: Quadratic programming with kernel transformation  
**Prediction**: Sign of decision function in transformed feature space

##### 3. XGBoost Classifier
**Algorithm**: Gradient Boosted Decision Trees (GBDT)  
**Optimizer**: Second-order gradient descent (Newton-Raphson)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 100 | Number of boosting rounds (trees) |
| `learning_rate` | 0.1 | Step size for gradient descent (η) |
| `max_depth` | 6 | Maximum depth of each tree |
| `subsample` | 0.8 | Fraction of samples for each tree (80%) |
| `objective` | 'multi:softmax' | Multi-class classification with softmax |
| `num_class` | 3 | Number of diabetes classes (0, 1, 2) |
| `random_state` | 42 | Random seed for reproducibility |
| `n_jobs` | -1 | Use all CPU cores for parallel training |
| `verbosity` | 0 | Silent mode (no training logs) |

**Training Method**: Additive boosting with regularized objective  
**Prediction**: Argmax of softmax probabilities across classes

##### Synthetic Data Generation Models

**CTGAN (Conditional Tabular GAN)**
- **Architecture**: Conditional Generative Adversarial Network
- **Optimizer**: Adam (Generator & Discriminator)
- **Learning Rate**: Default CTGAN settings
- **Output**: `Datasets/CTGAN/synthetic_1_unscaled.csv`

**VAE (Variational Autoencoder)**
- **Architecture**: Encoder-Decoder with latent space sampling
- **Optimizer**: Adam
- **Learning Rate**: Default VAE settings
- **Output**: `Datasets/VAE/synthetic_2_unscaled.csv`

### Cross-Validation Strategy

**10-Fold Stratified Cross-Validation** on each of the 12 data blocks:

1. **Block Creation**:
   - Each dataset (Real, CTGAN, VAE) → Split into 4 equal blocks
   - Total: 12 blocks (~57K samples each)

2. **Test Set Preparation** (Only from Real dataset):
   - 20% of Real dataset → Held-out test set (45,957 samples)
   - 80% of Real dataset → Split into 4 blocks
   - StandardScaler fitted on Real training data

3. **Training Process** (Per Block):
   - 10-fold stratified CV preserves class distribution
   - Each fold: 90% training, 10% validation
   - Final model: Trained on full block after CV
   - Output: Mean accuracy across 10 folds

4. **Total Operations**: 360 model trainings (12 blocks × 3 models × 10 folds)

### Dataset Splits

| Dataset | Total Samples | Test Set | Training Blocks | Samples/Block |
|---------|--------------|----------|-----------------|---------------|
| **Real** | 253,680 | 45,957 (20%) | 4 blocks | ~57,445 each |
| **CTGAN** | 229,781 | - | 4 blocks | ~57,445 each |
| **VAE** | 229,781 | - | 4 blocks | ~57,445 each |

**Note**: Test set is created only from Real dataset. Synthetic datasets (CTGAN, VAE) are used exclusively for training to evaluate their quality in producing useful synthetic samples.

### Statistical Analysis

The project performs comprehensive non-parametric statistical testing:

#### 1. Friedman ANOVA Test
- **Purpose**: Compare 3 models across 12 blocks (12×3 matrix)
- **Null Hypothesis**: All models perform equally across blocks
- **Output**: Chi-square statistic, p-value, degrees of freedom
- **Interpretation**: p < 0.05 indicates significant differences between models

#### 2. Kendall's W (Effect Size)
- **Purpose**: Measure agreement/consistency across blocks
- **Range**: 0 (no agreement) to 1 (complete agreement)
- **Interpretation**: Quantifies how consistently models rank across blocks

#### 3. Nemenyi Post-Hoc Test
- **Purpose**: Pairwise comparisons between models
- **Method**: Controls family-wise error rate
- **Output**: P-values for all model pairs (RF vs SVM, RF vs XGBoost, SVM vs XGBoost)
- **Interpretation**: Identifies which specific model pairs differ significantly

#### 4. Hommel Correction
- **Purpose**: Multiple testing correction
- **Method**: Adjusts p-values to control Type I error
- **Output**: Corrected p-values and significance decisions

#### 5. Cliff's Delta (Pairwise Effect Sizes)
- **Purpose**: Measure magnitude of differences between model pairs
- **Range**: -1 to +1
- **Interpretation**: 
  - |δ| < 0.147: Negligible
  - 0.147 ≤ |δ| < 0.33: Small
  - 0.33 ≤ |δ| < 0.474: Medium
  - |δ| ≥ 0.474: Large

### Visualization Outputs

All visualizations are automatically generated and saved to `Python_Files/visualizations/`:

1. **heatmap_accuracy.png**: 12×3 heatmap showing accuracy for each block-model combination
2. **model_comparison.png**: Bar plots (mean ± std) and box plots comparing the 3 models
3. **block_comparison.png**: Grouped bar chart showing all 12 blocks with 3 models side-by-side
4. **dataset_comparison.png**: Three subplots comparing performance within Real, CTGAN, and VAE datasets

**Format**: 300 DPI PNG files suitable for publication

## 🚀 Usage

### Complete Workflow Execution

Run the entire experimental pipeline:

```bash
cd Python_Files
python main.py
```

**What happens:**
1. ✓ Load 3 datasets (Real: 253,680 samples, CTGAN: 229,781, VAE: 229,781)
2. ✓ Create test set (45,957 samples from Real dataset, 20%)
3. ✓ Split each dataset into 4 blocks → 12 total blocks (~57K samples each)
4. ✓ Train 3 models on each block using 10-fold cross-validation
5. ✓ Save all 36 trained models to `models/RandomForest/`, `models/SVM/`, `models/XGBoost/`
6. ✓ Generate 12×3 accuracy matrix → saved to `statistical_results/experiment_results_matrix.csv`
7. ✓ Perform statistical analysis (Friedman, Nemenyi, Hommel, Cliff's Delta)
8. ✓ Save 8 statistical CSV files to `statistical_results/`
9. ✓ Generate 4 visualizations → saved to `visualizations/`

**Estimated Runtime**: 30-60 minutes (depending on hardware)

**Output Files Created**:
- 36 classification models → `models/RandomForest/*.pkl`, `models/SVM/*.pkl`, `models/XGBoost/*.pkl`
- 8 statistical CSV files → `statistical_results/*.csv`
- 4 visualization PNG files → `visualizations/*.png`

### Installation

**Prerequisites**: Python 3.8+

**Required packages**:
```bash
pip install pandas numpy scikit-learn xgboost
pip install scipy statsmodels scikit-posthocs
pip install matplotlib seaborn
pip install joblib
```

For synthetic data generation (optional):
```bash
pip install sdv torch
```

### Quick Setup

1. **Clone or download this repository**
2. **Install dependencies** (see above)
3. **Run the main workflow**:
   ```bash
   cd Python_Files
   python main.py
   ```

### Alternative Workflows

#### Run Statistical Analysis Only

```python
from statistical_analysis import run_complete_statistical_analysis
import pandas as pd

# Load the 12×3 results matrix
results_matrix = pd.read_csv('../statistical_results/experiment_results_matrix.csv', index_col=0)

# Run full statistical analysis
stats = run_complete_statistical_analysis(
    data_matrix=results_matrix,
    model_names=['RandomForest', 'SVM', 'XGBoost'],
    block_labels=list(results_matrix.index),
    save_to_csv=True,
    output_dir='../statistical_results'
)
```

#### Generate Visualizations Only

```python
from visualization import save_all_visualizations
import pandas as pd

# Load results
results_matrix = pd.read_csv('../statistical_results/experiment_results_matrix.csv', index_col=0)

# Generate all plots
save_all_visualizations(results_matrix, output_dir='../visualizations')
```

#### Load and Use Trained Models

```python
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load a trained model
with open('../models/RandomForest/Real_Block1.pkl', 'rb') as f:
    model = pickle.load(f)

# Load test data and make predictions
X_test = pd.read_csv('your_test_data.csv')
predictions = model.predict(X_test)
```

## 🎨 Key Features

### 1. Automated Training Pipeline
- Block-based data splitting with stratified sampling
- 10-fold cross-validation for robust evaluation
- Automatic model persistence (all 36 models saved as .pkl files)
- Progress monitoring with detailed logging
- Consistent scaling using StandardScaler

### 2. Rigorous Statistical Validation
- **Friedman ANOVA**: Non-parametric test for overall model differences
- **Kendall's W**: Effect size measuring consistency across blocks
- **Nemenyi Post-Hoc**: Pairwise model comparisons
- **Hommel Correction**: Multiple testing correction
- **Cliff's Delta**: Pairwise effect sizes
- Comprehensive CSV outputs for all statistical tests (7 files)

### 3. Synthetic Data Generation
- **CTGAN**: Conditional Tabular GAN with Adam optimizer
- **VAE**: Variational Autoencoder with Adam optimizer
- Preserves statistical properties of original data
- Generates balanced synthetic samples

### 4. Rich Visualizations
- Performance heatmaps (12×3 accuracy matrix)
- Model comparison bar plots with error bars
- Box plots showing distribution of accuracies
- Dataset-specific comparison charts
- 300 DPI publication-quality PNG files

## 🧪 Testing

The project includes comprehensive unit tests for statistical analysis functions:

```bash
cd Python_Files
python test_statistical_analysis.py
```

**Test Coverage**:
- 27 total tests covering all statistical functions
- Tests for Friedman test with various data patterns
- Kendall's W calculation validation
- Nemenyi post-hoc test verification
- Hommel correction accuracy
- Cliff's Delta effect size computation
- CSV output generation
- Complete workflow integration tests

## 📊 Results Summary

**Typical Results** (from completed training run):

| Dataset | RandomForest | SVM | XGBoost |
|---------|-------------|-----|---------|
| **Real** | 0.833 ± 0.001 | 0.617 ± 0.063 | 0.835 ± 0.001 |
| **CTGAN** | 0.840 ± 0.001 | 0.727 ± 0.036 | 0.841 ± 0.001 |
| **VAE** | 0.809 ± 0.002 | 0.592 ± 0.029 | 0.813 ± 0.002 |

**Key Findings**:
- ✓ RandomForest and XGBoost significantly outperform SVM
- ✓ CTGAN synthetic data performs comparably to real data
- ✓ VAE synthetic data shows slightly lower performance
- ✓ Statistical tests confirm significant differences (p < 0.05)

## 📝 Research Applications

This framework is suitable for:
- **Synthetic data quality evaluation**: Compare CTGAN vs VAE performance against real data
- **Model robustness testing**: Evaluate model stability across different data blocks
- **Cross-validation methodology**: Rigorous 10-fold CV with stratified sampling
- **Statistical validation**: Non-parametric testing with effect size reporting
- **Healthcare ML research**: Diabetes classification using BRFSS health indicators
- **Reproducible experiments**: All models saved for future predictions and analysis

## ✅ Confirmation: All Results Saved

This workflow has been verified to save all outputs correctly:

✓ **Classification Models**: 36 .pkl files in `models/RandomForest/`, `models/SVM/`, `models/XGBoost/`  
✓ **Experiment Results**: `statistical_results/experiment_results_matrix.csv` (12×3 accuracy matrix)  
✓ **Statistical Analysis**: 8 CSV files in `statistical_results/`  
✓ **Visualizations**: 4 PNG files (300 DPI) in `visualizations/`  
✓ **Generative Models**: 2 final models + 20 checkpoints in `models/CTGAN/` and `models/VAE/`

All files are created automatically when running `main.py`.

## 📞 Project Information

**Project Type**: Machine Learning Research - Diabetes Classification  
**Methodology**: 10-Fold Stratified Cross-Validation  
**Statistical Testing**: Non-parametric (Friedman ANOVA + Post-hoc)  
**Dataset**: BRFSS 2015 Health Indicators  
**Synthetic Data**: CTGAN and VAE Generated  
**Documentation**: Complete with hyperparameters and configurations

#### Generate Visualizations Only

```python
from visualization import save_visualizations
import pandas as pd

results_df = pd.read_csv('experiment_results_complete.csv')
save_visualizations(results_df, output_dir='visualizations')
```


## 🔬 Key Features

### 1. Automated Training Pipeline
- Seamless training across all dataset-model-parameter combinations
- Automatic model persistence and metadata tracking
- Progress monitoring and verbose logging

### 2. Comprehensive Metrics
- **Accuracy**: Overall classification correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity/true positive detection
- **F1-Score**: Harmonic mean of precision and recall
- **Training Time**: Model training duration

### 3. Statistical Validation
- Friedman ANOVA test for group comparisons
- Nemenyi post-hoc test for pairwise differences
- Multiple testing correction
- Effect size reporting

### 4. Rich Visualizations
- Performance comparison box plots
- Model group comparison charts
- Dataset performance analysis
- Parameter sensitivity plots


## 🧪 Testing

Run tests:

```bash
python test_statistical_analysis.py
```

## 📝 Research Applications

This framework is suitable for:
- Synthetic data quality evaluation
- Model robustness testing across data distributions
- Hyperparameter sensitivity analysis
- Comparative machine learning studies
- Healthcare ML research

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👥 Author

**Mahmoud Youssef**
- GitHub: [@Mahmoud-Gamal-Elgendy](https://github.com/Mahmoud-Gamal-Elgendy)
- Project: Statistical ML Diabetes Evaluation

## 🙏 Acknowledgments

- **Dataset**: BRFSS 2015 Diabetes Health Indicators
- **Libraries**: scikit-learn, XGBoost, SDV, PyTorch

## 📚 References

- CTGAN: Conditional Tabular GAN for synthetic data generation
- VAE: Variational Autoencoder for data synthesis
- Friedman Test: Non-parametric ANOVA for repeated measures
- Nemenyi Test: Post-hoc pairwise comparison test

---

**Last Updated**: December 2025  
**Status**: Active Development
