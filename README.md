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

## 🗂️ Project Structure

```
Statistical ML Diabetes Evaluation/
├── .git/                                # Git version control
├── .gitignore                           # Git ignore file
├── .venv/                               # Python virtual environment
├── Datasets/
│   ├── Real/
│   │   └── diabetes_012_health_indicators_BRFSS2015.csv  # Original BRFSS 2015 dataset
│   ├── CTGAN/
│   │   ├── synthetic_1.csv              # Scaled CTGAN synthetic data
│   │   └── synthetic_1_unscaled.csv     # Unscaled CTGAN synthetic data
│   └── VAE/
│       ├── synthetic_2.csv              # Scaled VAE synthetic data
│       └── synthetic_2_unscaled.csv     # Unscaled VAE synthetic data
├── models/                              # Trained classification models (36 total)
│   ├── RandomForest/                    # Random Forest models (12 models)
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
│   ├── SVM/                             # SVM models (12 models, same structure)
│   │   └── [Real|CTGAN|VAE]_Block[1-4].pkl
│   ├── XGBoost/                         # XGBoost models (12 models, same structure)
│   │   └── [Real|CTGAN|VAE]_Block[1-4].pkl
│   ├── CTGAN/
│   │   └── checkpoints/                 # CTGAN training checkpoints
│   └── VAE/
│       └── checkpoints/                 # VAE training checkpoints
├── Python_Files/
│   ├── main.py                          # Main execution script
│   ├── CTGAN_model.py                   # CTGAN synthetic data generator (uses Adam optimizer)
│   ├── VAE_model.py                     # VAE synthetic data generator (uses Adam optimizer)
│   ├── model_training_pipeline.py       # 10-fold CV training pipeline with model saving
│   ├── statistical_analysis.py          # Friedman, Nemenyi, effect size analysis
│   ├── visualization.py                 # Plotting and visualization functions
│   ├── unscale_synthetic_data.py        # Data preprocessing utilities
│   └── test_statistical_analysis.py     # Unit tests (27 tests) for statistical functions
├── Reports/                             # Analysis reports and documentation
├── statistical_results/                 # Statistical analysis outputs
│   ├── friedman_test_results.csv        # Friedman ANOVA test results
│   ├── kendalls_w_results.csv           # Effect size (Kendall's W)
│   ├── nemenyi_posthoc_results.csv      # Pairwise post-hoc comparisons
│   ├── hommel_correction_results.csv    # Hommel multiple testing correction
│   └── cliffs_delta_results.csv         # Pairwise effect sizes (Cliff's Delta)
├── visualizations/                      # Generated plots and charts
│   ├── performance_heatmap.png          # 12×3 accuracy heatmap
│   ├── model_comparison_boxplot.png     # Model performance distributions
│   └── dataset_comparison_violinplot.png # Dataset-wise performance
├── Project_workflow/                    # Documentation and workflows
│   ├── Diagram.excalidraw               # Project workflow diagram
│   └── Workflow Diagram.excalidraw      # Detailed workflow visualization
├── experiment_results_matrix.csv        # 12×3 accuracy matrix (main results)
└── README.md                            # This file
```

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

#### Model Parameters (Single Optimized Set)

**Random Forest**:
```python
n_estimators=100, max_depth=20, min_samples_split=10, random_state=42
```

**SVM**:
```python
C=1.0, kernel='rbf', gamma='scale', max_iter=1000, random_state=42
```

**XGBoost**:
```python
n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8,
objective='multi:softmax', num_class=3, random_state=42
```
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

#### Friedman ANOVA Test
- Non-parametric test comparing model groups across dataset blocks
### Complete Workflow Execution

Run the entire experimental pipeline:

```bash
cd Python_Files
python main.py
```

This will:
1. Load 3 datasets (Real, CTGAN, VAE)
2. Split each dataset into 4 blocks → 12 total blocks
3. Prepare test set (20% of Real dataset) and fit StandardScaler
4. Train 3 models on each block using 10-fold cross-validation
5. Save all 36 trained models to `models/RandomForest/`, `models/SVM/`, `models/XGBoost/`
6. Generate 12×3 accuracy matrix
7. Perform statistical analysis (Friedman, Nemenyi, effect sizes)
8. Generate visualizations (heatmaps, boxplots, comparison charts)
9. Save all results to CSV files

**Estimated Runtime**: 30-60 minutes (depending on hardware)

```bash
pip install pandas numpy scikit-learn xgboost
pip install scipy statsmodels scikit-posthocs
pip install matplotlib seaborn
pip install sdv torch joblib
```

### Quick Setup
#### Run Statistical Analysis Only

```python
from statistical_analysis import run_complete_statistical_analysis
import pandas as pd

# Load the 12×3 results matrix
results_matrix = pd.read_csv('experiment_results_matrix.csv', index_col=0)

# Run full statistical analysis
stats = run_complete_statistical_analysis(
    data_matrix=results_matrix,
    model_names=['RandomForest', 'SVM', 'XGBoost'],
    block_labels=list(results_matrix.index),
    save_to_csv=True,
    output_dir='statistical_results'
)
```

#### Generate Visualizations Only

```python
from visualization import save_all_visualizations
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
- **Cliff's Delta**: Pairwise effect size calculation
- Comprehensive CSV outputs for all statistical tests

### 3. Synthetic Data Generation
- **CTGAN**: Conditional Tabular GAN with Adam optimizer
- **VAE**: Variational Autoencoder with Adam optimizer (lr=1e-3)
- Preserves statistical properties of original data
- Generates balanced synthetic samples

### 4. Rich Visualizations
- Performance heatmaps (12×3 accuracy matrix)
- Model comparison boxplots
- Dataset performance violin plots
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
## 📝 Research Applications

This framework is suitable for:
- **Synthetic data quality evaluation**: Compare CTGAN vs VAE performance against real data
- **Model robustness testing**: Evaluate model stability across different data blocks
- **Cross-validation methodology**: Rigorous 10-fold CV with stratified sampling
- **Statistical validation**: Non-parametric testing with effect size reporting
- **Healthcare ML research**: Diabetes classification using BRFSS health indicators
- **Reproducible experiments**: All models saved for future predictions and analysis
```bash
python VAE_model.py
```

#### Run Statistical Analysis Only

```python
from statistical_analysis import run_complete_statistical_analysis
import pandas as pd

results_df = pd.read_csv('experiment_results_complete.csv')
stats = run_complete_statistical_analysis(results_df)
```

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
