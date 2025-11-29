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
├── Datasets/
│   ├── CTGAN/
│   │   ├── synthetic_1.csv              # Scaled GAN synthetic data
│   │   └── synthetic_1_unscaled.csv     # Unscaled GAN synthetic data
│   ├── VAE/
│   │   ├── synthetic_2.csv              # Scaled VAE synthetic data
│   │   └── synthetic_2_unscaled.csv     # Unscaled VAE synthetic data
│   └── Real/
│       └── diabetes_012_health_indicators_BRFSS2015.csv
├── models/
│   ├── CTGAN/
│   │   └── checkpoints/                 # CTGAN model checkpoints
│   └── VAE/
│       └── checkpoints/                 # VAE model checkpoints
├── Python_Files/
│   ├── main.py                          # Main execution script
│   ├── CTGAN_model.py                   # CTGAN synthetic data generator
│   ├── VAE_model.py                     # VAE synthetic data generator
│   ├── model_training_pipeline.py       # Automated ML training pipeline
│   ├── statistical_analysis.py          # Friedman & post-hoc tests
│   ├── visualization.py                 # Plotting and visualization
│   ├── unscale_synthetic_data.py        # Data preprocessing utilities
│   └── test_statistical_analysis.py     # Unit tests
├── Reports/                             # Generated reports and results
├── visualizations/                      # Generated plots and charts
├── Project workflow/                    # Documentation and workflows
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

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Complete Workflow Execution

Run the entire experimental pipeline:

```bash
cd Python_Files
python main.py
```

This will:
1. Load all three datasets (Real, GAN, VAE)
2. Train 36 model configurations
3. Save trained models to `models/` directory
4. Perform statistical analysis
5. Generate visualizations
6. Export results to `experiment_results_complete.csv`

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

## 📊 Output Files

### Generated Results

| File/Directory | Description |
|---------------|-------------|
| `experiment_results_complete.csv` | Complete results for all 36 experiments |
| `models/` | 36 trained model pickle files |
| `visualizations/` | Comparison plots and performance charts |
| `Datasets/CTGAN/` | GAN-generated synthetic datasets |
| `Datasets/VAE/` | VAE-generated synthetic datasets |

### Results CSV Schema

```
Experiment_ID, Dataset, Dataset_Name, Model_Group, Model_Name, Parameter_Set,
Accuracy, Precision, Recall, F1_Score, Training_Time, Model_Path
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

## 📈 Example Results

```
Top 5 Performing Configurations:
Experiment_ID  Dataset_Name  Model_Name      Accuracy  F1_Score
D1_G3_P4       Real          XGBoost         0.8532    0.8421
D1_G1_P4       Real          RandomForest    0.8498    0.8387
D2_G3_P3       GAN           XGBoost         0.8376    0.8265
D1_G3_P3       Real          XGBoost         0.8359    0.8248
D3_G1_P4       VAE           RandomForest    0.8287    0.8176

Overall Statistics:
  Mean Accuracy: 0.8145
  Std Accuracy:  0.0234
  Max Accuracy:  0.8532
  Min Accuracy:  0.7654
```

## 🧪 Testing

Run unit tests:

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

## 📄 License

This project is part of academic research at the Faculty of Information and Communication Technology, WUST (M.Sc).

## 👥 Author

**Mahmoud Youssef**
- GitHub: [@Mahmoud-Gamal-Elgendy](https://github.com/Mahmoud-Gamal-Elgendy)
- Project: Statistical ML Diabetes Evaluation

## 🙏 Acknowledgments

- **Dataset**: BRFSS 2015 Diabetes Health Indicators
- **Libraries**: scikit-learn, XGBoost, SDV, PyTorch
- **Institution**: Faculty of Information and Communication Technology, WUST

## 📚 References

- CTGAN: Conditional Tabular GAN for synthetic data generation
- VAE: Variational Autoencoder for data synthesis
- Friedman Test: Non-parametric ANOVA for repeated measures
- Nemenyi Test: Post-hoc pairwise comparison test

---

**Last Updated**: November 2025  
**Status**: Active Development
