# Statistical Evaluation of Machine Learning Models Trained on Real and Synthetic Data for Diabetes Classification

**Project title:** Statistical Evaluation of Machine Learning Models Trained on Real and Synthetic Data for Diabetes Classification  
**Date:** 11 January 2026  

## Abstract
This report presents a controlled experimental workflow for evaluating diabetes classification performance across one real-world tabular dataset and two synthetic datasets generated using CTGAN and a Variational Autoencoder (VAE). Three supervised learning models—Random Forest, Support Vector Machine (SVM), and XGBoost—were trained and evaluated using a block-based design and 10-fold cross-validation (CV). Model performance was summarized in a 12×3 matrix (12 data blocks × 3 models) of mean CV accuracies and was analyzed using non-parametric statistical tests, including the Friedman test with Nemenyi post-hoc comparisons, Hommel correction for multiple testing, Kendall’s W effect size, and pairwise Cliff’s Delta effect sizes. The provided results indicate statistically significant differences between models (Friedman χ² = 24.0, p = 0.000006) with strong rank agreement across blocks (Kendall’s W = 1.0). XGBoost achieved the highest overall mean accuracy (0.829654) with relatively low variability across blocks, while SVM exhibited markedly lower mean accuracy (0.645233) and substantially higher variability.

**Important scope note (data provenance):** All statements in this report are derived strictly from the experimental logs and results provided by the user. Where a configuration detail was not explicitly provided, it is clearly indicated as requiring confirmation.

---

## 1. Introduction
Diabetes mellitus is a prevalent chronic disease associated with severe complications and substantial public health burdens. Predictive models based on health indicators can support screening, risk stratification, and population-level surveillance. Machine learning (ML) approaches are well suited to such tasks because they can model non-linear relationships and complex feature interactions common in clinical and epidemiological tabular data.

Despite the promise of ML, medical data access is frequently constrained by privacy regulations, consent limitations, institutional data silos, and potential re-identification risks. Additionally, real-world healthcare datasets may exhibit class imbalance and dataset shift across populations. Synthetic data generation methods aim to mitigate these constraints by producing artificial datasets that preserve statistical properties of the real data while reducing direct exposure of patient records.

In this project, two synthetic data generation approaches—CTGAN (a generative adversarial model tailored to tabular data) and a Variational Autoencoder (VAE)—are used to create synthetic health-indicator datasets. The central objective is not only to compare predictive accuracy but also to conduct a rigorous statistical comparison of models across multiple data blocks. This is essential because reporting a single accuracy value can obscure variability, instability, and the consistency of model rankings across experimental conditions.

**Objectives and contributions (as supported by the provided materials):**
- Evaluate three ML models (Random Forest, SVM, XGBoost) on three datasets (Real, CTGAN, VAE).
- Use a block-based experimental design (3 datasets × 4 blocks = 12 blocks) and 10-fold cross-validation.
- Produce a 12×3 matrix of mean CV accuracies.
- Perform non-parametric statistical analysis (Friedman test, Kendall’s W, Nemenyi post-hoc, Hommel correction, Cliff’s Delta).
- Generate summary visualizations (heatmap_accuracy.png, model_comparison.png, block_comparison.png, dataset_comparison.png).

---

## 2. Datasets Description
### 2.1 Real dataset
According to the provided execution log:
- **Dataset name:** BRFSS 2015 diabetes health indicators dataset (file referenced in the workflow as `diabetes_012_health_indicators_BRFSS2015.csv`).
- **Number of samples:** 253,680.
- **Number of features:** 22.

The dataset is used for diabetes classification with a multi-class target `Diabetes_012` (the presence of `num_class = 3` is implied by downstream XGBoost results and the three-class context in the logs; however, the exact target coding and class distribution were not explicitly provided and should be confirmed).

### 2.2 CTGAN synthetic dataset
From the provided log:
- **Number of samples:** 229,781.
- **Number of features:** 22.

This dataset is referred to as CTGAN-generated synthetic data and is used as one of the training sources in the block-based design.

### 2.3 VAE synthetic dataset
From the provided log:
- **Number of samples:** 229,781.
- **Number of features:** 22.

This dataset is referred to as VAE-generated synthetic data and is used as the second synthetic training source.

### 2.4 Feature dimensionality consistency
All three datasets were reported to contain **22 features**, indicating dimensional consistency across Real, CTGAN, and VAE datasets. This consistency is critical for ensuring models can be trained under comparable feature representations.

### 2.5 Rationale for using synthetic data
The provided project description motivates synthetic data through:
- **Privacy-preserving analysis** (reducing direct reliance on identifiable real records).
- **Data access constraints** in medical contexts.
- **Potential utility for model development** under limited real data access.

While the logs confirm the use of CTGAN and VAE to generate synthetic datasets, the report does not infer additional properties (e.g., fidelity metrics or privacy guarantees) beyond what was explicitly provided.

### 2.6 Dataset splitting strategy (training blocks vs. test set)
The experiment applies:
- **Block splitting:** Each dataset is split into 4 blocks, producing 12 total blocks.
- **Test set:** A test set is created from the Real dataset only:
  - **Test set size:** 45,957 samples.
  - The log states: “Scaler fitted on Real data” and “Test set: Only from Real dataset (20% of Real data).”

**Critical clarification:** The provided results (accuracy matrix) are explicitly reported as **10-fold cross-validation accuracy** per block. The logs do not report performance on the Real test set as a final evaluation metric. Therefore, the “test set” appears to be used for preprocessing and/or workflow consistency, but not as the primary reported performance measure. This should be confirmed if a held-out Real evaluation is intended.

---

## 3. Experimental Design
### 3.1 Block-based data splitting
The experiment splits each of the 3 datasets into 4 approximately equal-sized blocks, yielding 12 blocks total:

| Dataset | Blocks | Approx. samples per block (reported) |
|---|---:|---:|
| Real | 4 | 57,445–57,446 |
| CTGAN | 4 | 57,445–57,446 |
| VAE | 4 | 57,445–57,446 |

Block labels in the results matrix are:
- Real_Block1–Real_Block4
- CTGAN_Block1–CTGAN_Block4
- VAE_Block1–VAE_Block4

### 3.2 10-fold cross-validation
For each block and each model, 10-fold cross-validation is performed and the mean accuracy across folds is recorded. This produces:
- **12 blocks × 3 models = 36 model–block configurations**.
- **36 configurations × 10 folds = 360 cross-validation training/evaluation operations** (as described in the project overview).

### 3.3 Justification for using Real data only as the test set
The workflow states: “Test set: Only from Real dataset (20% of Real data).” A plausible methodological motivation is to reserve a real-data reference split to avoid synthetic-data contamination and to maintain an anchor for preprocessing. However, the logs do not include an explicit justification beyond this design choice, and the downstream reported metrics are CV accuracies on the blocks.

**This design choice should be confirmed:** If the research goal includes generalization to real-world deployment, a common evaluation is to train on synthetic and evaluate on held-out real data (often called Train-on-Synthetic Test-on-Real, TSTR). The provided logs do not include TSTR results.

### 3.4 Scaling and preprocessing strategy
The logs state:
- “✓ Scaler fitted on Real data.”

The specific scaling method (e.g., standardization, normalization), whether scaling was fit on training data or the held-out portion, and how it was applied per fold were **not explicitly provided and should be confirmed**.

### 3.5 Reproducibility considerations
Reproducibility-relevant details explicitly present in the logs:
- Fixed data split proportion for the Real test set: 20%.
- The experimental structure (3 datasets, 4 blocks each, 10 folds).

Other reproducibility-critical details—such as random seeds for each learning algorithm, the randomization used during block generation, and library versions—were **not provided and should be confirmed**.

---

## 4. Machine Learning Models
This section provides model background and configuration reporting. Where hyperparameters are not explicitly provided in the user materials, this is stated explicitly.

### 4.1 Random Forest
#### 4.1.1 Model description
Random Forest is an ensemble method that aggregates predictions from multiple decision trees trained on bootstrapped samples and randomized feature subsets. It is well suited to tabular medical data because it:
- Captures non-linearities and interaction effects.
- Is robust to monotonic transformations of features.
- Provides strong baseline performance without extensive feature engineering.

#### 4.1.2 Model configuration
The specific Random Forest hyperparameters used in training were **not explicitly provided in the user logs** and should be confirmed (e.g., number of trees, maximum depth, minimum samples per split/leaf, class weighting, and feature subsampling strategy).

### 4.2 Support Vector Machine (SVM)
#### 4.2.1 Model description
SVMs construct decision boundaries by maximizing the margin between classes, potentially in a non-linear feature space via kernel functions. For tabular medical data, SVMs can be competitive when:
- Features are appropriately scaled.
- Kernel selection and regularization are well tuned.

However, SVMs can be sensitive to hyperparameters and scaling choices, and computational cost can increase with dataset size.

#### 4.2.2 Model configuration
The specific SVM configuration (e.g., kernel type, regularization parameter C, gamma, tolerance, maximum iterations, and class weighting) was **not explicitly provided in the user logs** and should be confirmed.

### 4.3 XGBoost
#### 4.3.1 Model description
XGBoost is a gradient-boosted decision tree method that sequentially builds trees to correct errors made by previous trees, employing regularization and optimized training routines. It is frequently strong on tabular clinical and epidemiological data due to:
- High capacity to model complex feature interactions.
- Built-in regularization mechanisms.
- Robustness across a broad range of tabular tasks.

#### 4.3.2 Model configuration
The exact XGBoost hyperparameters (e.g., number of estimators, learning rate, maximum depth, subsampling, objective, number of classes, and regularization terms) were **not explicitly provided in the user logs** and should be confirmed.

---

## 5. Training Procedure
### 5.1 Cross-validation process
For each of the 12 blocks:
- Each of the 3 models is trained and evaluated using **10-fold cross-validation**.
- The performance metric is the **mean accuracy** across the 10 folds.

### 5.2 Total number of trained configurations
- **36 trained model configurations** are produced (12 blocks × 3 models).
- The workflow includes model persistence (“Model saved: …”) for each configuration.

**Model saving strategy clarification:** The logs confirm that a model artifact is saved for each block and model (e.g., `Real_Block1.pkl`). It is not explicitly stated whether the saved model corresponds to a refit on the full block after CV or to a particular fold model; this should be confirmed.

### 5.3 Performance metric: Accuracy
Accuracy was used as the sole reported metric in the provided materials:
- Reported as mean cross-validation accuracy per block.

**Rationale:** Accuracy is straightforward to interpret and suitable when class distribution is approximately balanced or when misclassification costs are symmetric. However, the class distribution and clinical cost asymmetry were not provided; therefore, the adequacy of accuracy as the primary metric should be confirmed. In medical classification settings, complementary metrics (e.g., macro-F1, sensitivity/specificity, AUC) are often important.

---

## 6. Results
### 6.1 Full accuracy table (12 blocks × 3 models)
Table 1 reproduces the reported mean 10-fold CV accuracy matrix.

**Table 1. Mean 10-fold cross-validation accuracy by data block and model.**

| Block | RandomForest | SVM | XGBoost |
|---|---:|---:|---:|
| Real_Block1 | 0.834659 | 0.699450 | 0.835216 |
| Real_Block2 | 0.833806 | 0.551607 | 0.835077 |
| Real_Block3 | 0.833615 | 0.579599 | 0.834381 |
| Real_Block4 | 0.833148 | 0.638528 | 0.834262 |
| CTGAN_Block1 | 0.839638 | 0.735817 | 0.839847 |
| CTGAN_Block2 | 0.839777 | 0.770183 | 0.842110 |
| CTGAN_Block3 | 0.840891 | 0.719973 | 0.841292 |
| CTGAN_Block4 | 0.839554 | 0.680359 | 0.841556 |
| VAE_Block1 | 0.807311 | 0.603950 | 0.811402 |
| VAE_Block2 | 0.808895 | 0.564699 | 0.811872 |
| VAE_Block3 | 0.812290 | 0.627821 | 0.815284 |
| VAE_Block4 | 0.809700 | 0.570814 | 0.813547 |

### 6.2 Mean and standard deviation per model
Two slightly different standard deviation values are present in the provided logs, which is expected if one computation uses sample standard deviation (ddof=1) and another uses population standard deviation (ddof=0). Both are reported below exactly as provided.

**From “Summary Statistics” after training pipeline completion:**
- RandomForest: Mean = 0.827774, Std = 0.013764
- SVM: Mean = 0.645233, Std = 0.074301
- XGBoost: Mean = 0.829654, Std = 0.012632

**From the statistical analysis “SUMMARY” section:**
- RandomForest: 0.827774 (±0.013178)
- SVM: 0.645233 (±0.071138)
- XGBoost: 0.829654 (±0.012094)

### 6.3 Observed trends across datasets (Real vs CTGAN vs VAE)
Based on Table 1:
- **CTGAN blocks** exhibit the highest accuracies across all three models relative to Real and VAE blocks.
- **VAE blocks** exhibit the lowest accuracies across all three models.
- **Real blocks** are intermediate between CTGAN and VAE.

These observations are descriptive and do not, by themselves, establish causal reasons for differences.

### 6.4 Stability analysis across blocks
- **XGBoost** shows the strongest stability (smallest reported standard deviation among the three models).
- **RandomForest** is similarly stable, with slightly lower mean accuracy than XGBoost.
- **SVM** exhibits much higher variability across blocks, including a notably low result on Real_Block2 (0.551607) and a notably higher result on CTGAN_Block2 (0.770183).

---

## 7. Statistical Analysis
All statistical results in this section are reproduced from the provided logs.

### 7.1 Friedman test
The Friedman test is a non-parametric test for detecting differences in treatments (models) across multiple blocks (datasets/conditions). Here:
- **Blocks** are the repeated-measures units (12 blocks).
- **Treatments** are the models (3 models).

**Hypotheses (as stated in the log):**
- H₀: All models perform equally across blocks.
- H₁: At least one model differs significantly.

**Results:**
- χ² = 24.000000
- p = 0.000006
- df = 2

**Interpretation:** At α = 0.05, the null hypothesis is rejected. There is strong evidence that model performance differs across blocks.

### 7.2 Kendall’s W (effect size)
Kendall’s W quantifies the degree of agreement in rankings across blocks.

**Results:**
- W = 1.000000
- Interpretation in the log: “Large effect – Strong agreement”

**Interpretation:** The model ranking is maximally consistent across all 12 blocks. This indicates that the relative ordering of the models does not fluctuate across blocks.

### 7.3 Nemenyi post-hoc test
The Nemenyi test performs pairwise comparisons between models following a significant Friedman result.

**Pairwise p-values (Nemenyi):**
- RandomForest vs SVM: p = 0.038035 (significant)
- RandomForest vs XGBoost: p = 0.038035 (significant)
- SVM vs XGBoost: p = 0.000003 (significant)

**Interpretation:** All model pairs differ significantly under the Nemenyi procedure at α = 0.05.

### 7.4 Hommel correction
Hommel correction adjusts p-values to control family-wise error rate under multiple comparisons.

**Corrected results (Hommel):**
- RandomForest vs SVM: original p = 0.038035, corrected p = 0.038035 (significant)
- RandomForest vs XGBoost: original p = 0.038035, corrected p = 0.038035 (significant)
- SVM vs XGBoost: original p = 0.000003, corrected p = 0.000009 (significant)

**Interpretation:** After correction, all pairwise differences remain statistically significant.

### 7.5 Pairwise Cliff’s Delta effect sizes
Cliff’s Delta measures the probability that one method outperforms another across paired observations.

**Results:**
- RandomForest vs SVM: δ = +1.000000 (Large) → RandomForest outperforms SVM
- RandomForest vs XGBoost: δ = −0.263889 (Small) → XGBoost outperforms RandomForest
- SVM vs XGBoost: δ = −1.000000 (Large) → XGBoost outperforms SVM

**Interpretation:**
- The difference between **XGBoost and SVM** is extremely large and consistent across blocks.
- The difference between **RandomForest and SVM** is also extremely large and consistent.
- The difference between **XGBoost and RandomForest** is statistically significant but small in magnitude.

---

## 8. Visual Analysis
The workflow states that the following figures were generated and saved:
- `heatmap_accuracy.png`
- `model_comparison.png`
- `block_comparison.png`
- `dataset_comparison.png`

Because the underlying accuracy matrix (Table 1) is available, the expected qualitative content of each figure can be interpreted without making assumptions about stylistic elements.

### 8.1 heatmap_accuracy.png
This heatmap visualizes the 12×3 accuracy matrix.
- Expected observation: higher accuracy values concentrated in CTGAN rows for RandomForest and XGBoost, with consistently lower values for SVM across most blocks.
- The heatmap format supports rapid identification of dataset-level patterns (CTGAN > Real > VAE) and model-level patterns (XGBoost ≳ RandomForest ≫ SVM).

### 8.2 model_comparison.png
This figure compares models aggregated across blocks.
- Expected observation: XGBoost and RandomForest have similar means (~0.83), while SVM has a substantially lower mean (~0.65).
- Expected variability: SVM should show a wider spread than the tree-based methods, consistent with its higher reported standard deviation.

### 8.3 block_comparison.png
This plot compares model performance across individual blocks.
- Expected observation: for each block, XGBoost and RandomForest bars should be close, while SVM is consistently lower.
- It should highlight that the CTGAN blocks are uniformly higher than the VAE blocks across models.

### 8.4 dataset_comparison.png
This figure groups blocks by dataset type.
- Expected observation: within each dataset group, model ordering should be consistent, consistent with Kendall’s W = 1.0.
- Dataset-level effect should be visible: CTGAN blocks highest, VAE blocks lowest.

**Logging note:** The provided console output shows repeated messages about visualization generation and a duplicated filename listing. This appears to be an execution/logging artifact rather than a methodological component and does not affect the numerical results reported in Table 1.

---

## 9. Discussion
This discussion is grounded in the reported results and statistical tests. Explanatory claims about *why* patterns occur are presented as hypotheses when not directly tested in the provided materials.

### 9.1 Why XGBoost outperformed other models
Empirically, XGBoost achieved the highest overall mean accuracy (0.829654) and outperformed RandomForest by a small effect size (Cliff’s δ = −0.263889, favoring XGBoost). A plausible explanation is that gradient-boosted trees often capture fine-grained non-linear feature interactions more effectively than bagged trees, particularly when regularization and boosting dynamics reduce bias. However, the extent to which this mechanism drove the observed differences was **not explicitly investigated in the provided logs and should be confirmed** via model diagnostics (e.g., learning curves, feature interaction analysis).

### 9.2 Why CTGAN data led to superior performance
Across all models, CTGAN blocks yielded higher accuracies than Real and VAE blocks. Plausible hypotheses include:
- CTGAN may produce synthetic samples that are smoother and less noisy than real-world data, which can simplify decision boundaries.
- CTGAN may implicitly amplify separability between classes, increasing CV accuracy within the synthetic distribution.

However, without explicit fidelity analyses (e.g., distributional similarity metrics, conditional distributions, label–feature dependency checks), it cannot be concluded that CTGAN data is “better” in a real-world sense. Higher CV accuracy on synthetic data may reflect reduced complexity rather than improved realism.

### 9.3 Why SVM underperformed
SVM demonstrated both lower mean accuracy and high variability across blocks. Possible contributing factors include:
- Sensitivity to scaling and hyperparameter tuning (kernel choice, C, gamma), which were **not explicitly provided and should be confirmed**.
- Potential mismatch between SVM decision boundaries and multi-class structure.
- Optimization constraints (e.g., iteration limits), if present.

Because none of these configuration details are included in the user-provided materials, these remain hypotheses.

### 9.4 Practical implications for medical ML systems
The strong statistical separation between models suggests that model choice materially impacts performance in this experimental setting. Nevertheless, clinical deployment decisions must consider:
- Generalization to real-world, out-of-sample data.
- Robustness under population shift.
- Calibration and error costs.
- Interpretability and auditability.

### 9.5 Limitations of the current study
Based strictly on the provided workflow and outputs:
- Primary reported performance is **within-block cross-validation accuracy**, not held-out Real test performance.
- The use and role of the Real test set in evaluation is unclear from the reported results.
- Hyperparameter configurations are not explicitly documented in the logs and should be confirmed.
- Only accuracy is reported; additional clinically relevant metrics are absent from the provided outputs.

---

## 10. Conclusion
This project implements a block-based evaluation framework to compare RandomForest, SVM, and XGBoost across Real, CTGAN-synthetic, and VAE-synthetic datasets for diabetes classification.

Key findings from the provided results:
- Model differences are statistically significant (Friedman χ² = 24.0, p = 0.000006).
- Model ranking is perfectly consistent across blocks (Kendall’s W = 1.0).
- XGBoost achieves the best mean accuracy (0.829654) and significantly outperforms RandomForest (small effect) and SVM (large effect).
- CTGAN blocks consistently yield higher cross-validation accuracies than Real and VAE blocks.

**Recommendation for future work (requires additional experimentation beyond provided materials):** Evaluate Train-on-Synthetic Test-on-Real (TSTR) performance and incorporate clinically relevant metrics to assess whether synthetic-data-trained models generalize to real-world decision-making contexts.

---

## 11. Reproducibility and File Structure
This section documents the outputs referenced in the logs.

### 11.1 experiment_results_matrix.csv
- **Purpose:** Stores the 12×3 matrix of mean 10-fold CV accuracies.
- **Role:** Primary numerical result used for all subsequent statistical analyses and visualizations.

### 11.2 statistical_results/ directory
The logs report that the following files are saved:
- `data_matrix.csv`: Copy of the analyzed matrix.
- `friedman_test_results.csv`: Friedman test statistic and p-value.
- `effect_size_results.csv`: Kendall’s W effect size.
- `posthoc_nemenyi_results.csv`: Pairwise Nemenyi post-hoc p-values and significance.
- `hommel_correction_results.csv`: Multiplicity-adjusted p-values using Hommel correction.
- `pairwise_effect_sizes_cliffs_delta.csv`: Pairwise Cliff’s Delta effect sizes.
- `model_summary_statistics.csv`: Mean, standard deviation, min, max per model.

### 11.3 visualizations/ directory
The logs report that the following visual outputs are saved:
- `heatmap_accuracy.png`
- `model_comparison.png`
- `block_comparison.png`
- `dataset_comparison.png`

### 11.4 models/ directory
The logs report that models are saved per block and per model family:
- `models/RandomForest/<Block>.pkl`
- `models/SVM/<Block>.pkl`
- `models/XGBoost/<Block>.pkl`

**Note:** Filenames such as `Real_Block1.pkl` indicate a model trained on the corresponding block.

---

## Appendix A: Consolidated Numerical Summary
### A.1 Overall model means (from logs)
- RandomForest mean accuracy: 0.827774
- SVM mean accuracy: 0.645233
- XGBoost mean accuracy: 0.829654

### A.2 Key statistical outcomes (from logs)
- Friedman χ²: 24.000000
- Friedman p-value: 0.000006
- Kendall’s W: 1.000000
- Nemenyi p-values:
  - RandomForest vs SVM: 0.038035
  - RandomForest vs XGBoost: 0.038035
  - SVM vs XGBoost: 0.000003
- Hommel-corrected p-values:
  - RandomForest vs SVM: 0.038035
  - RandomForest vs XGBoost: 0.038035
  - SVM vs XGBoost: 0.000009
- Cliff’s Delta:
  - RandomForest vs SVM: +1.000000 (Large)
  - RandomForest vs XGBoost: −0.263889 (Small)
  - SVM vs XGBoost: −1.000000 (Large)
