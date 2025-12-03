# Statistical ML Diabetes Evaluation

A comprehensive machine learning research project evaluating diabetes classification performance across real and synthetic datasets using multiple machine learning algorithms with statistical validation.

## 📋 Project Overview

This project implements a complete experimental workflow to:
- Generate synthetic health indicator data using **CTGAN** and **VAE**
- Train and evaluate **3 machine learning models** (Random Forest, SVM, XGBoost)
- Compare performance across **3 datasets** (Real, GAN-generated, VAE-generated)
- Test **4 parameter configurations** per model
- Perform rigorous **statistical analysis** using Friedman tests and post-hoc comparisons
- Generate comprehensive visualizations and reports

**Total Experiments**: 36 (3 datasets × 3 models × 4 parameter sets)

## 🗂️ Project Structure

```
Statistical ML Diabetes Evaluation/
├── .git/                                # Git version control
├── .gitignore                           # Git ignore file
├── .venv/                               # Python virtual environment
├── Datasets/
│   ├── CTGAN/
│   │   ├── synthetic_1.csv              # Scaled CTGAN synthetic data
│   │   └── synthetic_1_unscaled.csv     # Unscaled CTGAN synthetic data
│   ├── VAE/
│   │   ├── synthetic_2.csv              # Scaled VAE synthetic data
│   │   └── synthetic_2_unscaled.csv     # Unscaled VAE synthetic data
│   └── Real/
│       └── diabetes_012_health_indicators_BRFSS2015.csv
├── models/                              # All trained models
│   ├── RandomForest/                    # Random Forest models (12 total)
│   │   ├── D1G1P1_model.pkl to D1G1P4_model.pkl  # Real dataset models (4)
│   │   ├── D2G1P1_model.pkl to D2G1P4_model.pkl  # CTGAN dataset models (4)
│   │   └── D3G1P1_model.pkl to D3G1P4_model.pkl  # VAE dataset models (4)
│   ├── SVM/                             # SVM models (12 total)
│   │   ├── D1G2P1_model.pkl to D1G2P4_model.pkl  # Real dataset models (4)
│   │   ├── D2G2P1_model.pkl to D2G2P4_model.pkl  # CTGAN dataset models (4)
│   │   └── D3G2P1_model.pkl to D3G2P4_model.pkl  # VAE dataset models (4)
│   ├── XGBoost/                         # XGBoost models (12 total)
│   │   ├── D1G3P1_model.pkl to D1G3P4_model.pkl  # Real dataset models (4)
│   │   ├── D2G3P1_model.pkl to D2G3P4_model.pkl  # CTGAN dataset models (4)
│   │   └── D3G3P1_model.pkl to D3G3P4_model.pkl  # VAE dataset models (4)
│   ├── CTGAN/
│   │   ├── ctgan_final.pkl              # Trained CTGAN generator model
│   │   └── checkpoints/                 # CTGAN training checkpoints
│   └── VAE/
│       ├── vae_final.pkl                # Trained VAE generator model
│       └── checkpoints/                 # VAE training checkpoints
├── Python_Files/
│   ├── main.py                          # Main execution script
│   ├── CTGAN_model.py                   # CTGAN synthetic data generator
│   ├── VAE_model.py                     # VAE synthetic data generator
│   ├── model_training_pipeline.py       # Automated ML training pipeline
│   ├── statistical_analysis.py          # Friedman & post-hoc tests
│   ├── visualization.py                 # Plotting and visualization
│   ├── unscale_synthetic_data.py        # Data preprocessing utilities
│   └── test_statistical_analysis.py     # Statistical analysis tests
├── Reports/                             # Analysis reports and documentation
│   └── Experimental_Results_Analysis.ipynb  # Jupyter notebook with comprehensive analysis
├── statistical_results/                 # Statistical analysis results
│   ├── friedman_test_results.csv        # Friedman ANOVA test summary
│   ├── aggregated_model_results.csv     # Mean metrics per model across datasets
│   ├── effect_size_results.csv          # Kendall's W effect sizes
│   ├── nemenyi_posthoc_results.csv      # Pairwise comparison results (p-values)
│   ├── nemenyi_posthoc_mean_ranks.csv   # Mean ranks for Nemenyi test
│   ├── hommel_correction_results.csv    # Multiple comparison corrections
│   ├── model_rankings_by_metric.csv     # Model rankings per metric
│   ├── performance_differences.csv      # Pairwise performance differences
│   └── dataset_performance_summary.csv  # Performance breakdown by dataset
├── visualizations/                      # Generated plots and charts
│   └── comprehensive_comparison.png     # Model performance visualizations
├── Project workflow/                    # Documentation and workflows
│   ├── Diagram.excalidraw               # Project workflow diagram
│   └── Workflow Diagram.excalidraw      # Detailed workflow visualization
├── experiment_results_complete.csv      # Complete experimental results (36 runs)
└── README.md                            # This file
```

## 🎯 Research Methodology

### Datasets (3)
- **D1 (Real)**: BRFSS 2015 diabetes health indicators dataset
- **D2 (GAN)**: Synthetic data generated using CTGAN
- **D3 (VAE)**: Synthetic data generated using Variational Autoencoder

### Models (3)
- **G1 (Random Forest)**: Ensemble decision tree classifier
- **G2 (SVM)**: Support Vector Machine with RBF/Linear kernels
- **G3 (XGBoost)**: Gradient boosting classifier

### Parameter Sets (4 per model)
Each model group is tested with 4 different parameter configurations (P1-P4):

#### Random Forest Parameters
- **P1**: `n_estimators=50, max_depth=10, min_samples_split=5`
- **P2**: `n_estimators=100, max_depth=20, min_samples_split=10`
- **P3**: `n_estimators=150, max_depth=30, min_samples_split=15`
- **P4**: `n_estimators=200, max_depth=None, min_samples_split=20`

#### SVM Parameters
- **P1**: `C=0.1, kernel=rbf, gamma=scale, max_iter=500`
- **P2**: `C=1.0, kernel=rbf, gamma=scale, max_iter=1000`
- **P3**: `C=10.0, kernel=rbf, gamma=auto, max_iter=1000`
- **P4**: `C=1.0, kernel=linear, max_iter=1000`

#### XGBoost Parameters
- **P1**: `n_estimators=50, learning_rate=0.05, max_depth=3`
- **P2**: `n_estimators=100, learning_rate=0.1, max_depth=6`
- **P3**: `n_estimators=150, learning_rate=0.15, max_depth=9`
- **P4**: `n_estimators=200, learning_rate=0.2, max_depth=12`

### Statistical Analysis

#### Friedman ANOVA Test
- Non-parametric test comparing model groups across dataset blocks
- Tests null hypothesis: all model groups perform equally
- Uses mean performance per group per dataset (3×3 matrix)

#### Post-hoc Nemenyi Test
- Pairwise comparisons between model groups
- Controls family-wise error rate
- Identifies which specific groups differ significantly

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Required Libraries

```bash
pip install pandas numpy scikit-learn xgboost
pip install scipy statsmodels scikit-posthocs
pip install matplotlib seaborn
pip install sdv torch joblib
```

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/Mahmoud-Gamal-Elgendy/Statistical-ML-Diabetes-Evaluation.git
cd Statistical-ML-Diabetes-Evaluation
```

## 💻 Usage

### Complete Workflow Execution

Run the entire experimental pipeline:

```bash
cd Python_Files
python main.py
```

This will:
1. Train 36 model configurations
2. Save trained models to `models/` directory
3. Perform statistical analysis
4. Generate visualizations
5. Export results to `experiment_results_complete.csv`

### Individual Components

#### Generate Synthetic Data (CTGAN)

```bash
python CTGAN_model.py
```

#### Generate Synthetic Data (VAE)

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
