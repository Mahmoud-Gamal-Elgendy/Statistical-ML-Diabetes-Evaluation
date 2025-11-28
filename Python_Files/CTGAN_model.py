from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import os
import warnings
from scipy import stats
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")


# ==================== DATA PREPROCESSING ====================
print("Starting data preprocessing...")

# Load the dataset
dataset_path = r"e:\Study (B.Sc & M.Sc )\Faculty Of Information and Communication Technology, WUST (M.Sc)\S 2\AI\Project\Project Implementaion v2\Datasets\Real\diabetes_012_health_indicators_BRFSS2015.csv"
dataset = pd.read_csv(dataset_path)
print(f"Loaded dataset with shape: {dataset.shape}")

# Separate features and target
target_column = 'Diabetes_012'
X_data = dataset.drop(target_column, axis=1)
y_data = dataset[target_column]

# Remove duplicates
combined_data = pd.concat([X_data, y_data], axis=1)
print(f"Before removing duplicates: {combined_data.shape[0]} rows")
combined_data = combined_data.drop_duplicates()
print(f"After removing duplicates: {combined_data.shape[0]} rows")
X_data = combined_data.drop(target_column, axis=1)
y_data = combined_data[target_column]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)
print(f"Train set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame with original column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_data.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_data.columns)

# Create complete training and test datasets with target
train_df = pd.concat([X_train_scaled.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
test_df = pd.concat([X_test_scaled.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

# Create real_processed_df (full scaled dataset for reference)
X_data_scaled = scaler.fit_transform(X_data)
X_data_scaled = pd.DataFrame(X_data_scaled, columns=X_data.columns)
real_processed_df = pd.concat([X_data_scaled.reset_index(drop=True), y_data.reset_index(drop=True)], axis=1)

print(f"Preprocessing complete!")
print(f"train_df shape: {train_df.shape}")
print(f"test_df shape: {test_df.shape}")
print(f"real_processed_df shape: {real_processed_df.shape}")

# ==================== CTGAN MODEL TRAINING ====================
print("\nTraining CTGAN model with checkpoint continuation...")

for col in train_df.select_dtypes('category').columns:
    train_df[col] = train_df[col].astype('object')


# PERFECT METADATA
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_df)

# Numerical columns (continuous values)
numerical_cols = ['BMI', 'MentHlth', 'PhysHlth']
for col in numerical_cols:
    if col in train_df.columns:
        metadata.update_column(column_name=col, sdtype='numerical')

# Ordinal columns (ordered categories)
ordinal_cols = ['GenHlth', 'Age', 'Education', 'Income']
for col in ordinal_cols:
    if col in train_df.columns:
        metadata.update_column(column_name=col, sdtype='categorical', order_by='numerical_value')

# Binary/Categorical columns
categorical_cols = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
    'DiffWalk', 'Sex', 'Diabetes_012'
]

for col in categorical_cols:
    if col in train_df.columns:
        metadata.update_column(column_name=col, sdtype='categorical')

print(f"Metadata configured for {len(train_df.columns)} columns")

ctgan = CTGANSynthesizer(
    metadata=metadata,
    enforce_rounding=False,
    batch_size=1500,
    epochs=30,
    verbose=True,
    generator_lr=2e-4,
    discriminator_lr=2e-4,
)

# TRAINING LOOP
# Setup paths
checkpoint_dir = r"e:\Study (B.Sc & M.Sc )\Faculty Of Information and Communication Technology, WUST (M.Sc)\S 2\AI\Project\Project Implementaion v2\models\CTGAN\checkpoints"
models_dir = r"e:\Study (B.Sc & M.Sc )\Faculty Of Information and Communication Technology, WUST (M.Sc)\S 2\AI\Project\Project Implementaion v2\models\CTGAN"
output_dir = r"e:\Study (B.Sc & M.Sc )\Faculty Of Information and Communication Technology, WUST (M.Sc)\S 2\AI\Project\Project Implementaion v2\Datasets\CTGAN"

# Create directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Detect the latest checkpoint (if any)
latest_checkpoint = 0
for chk in range(30, 301, 30):
    checkpoint_path = os.path.join(checkpoint_dir, f"ctgan_epoch_{chk:03d}.pkl")
    if os.path.exists(checkpoint_path):
        latest_checkpoint = chk

if latest_checkpoint > 0:
    print(f"Found existing checkpoint at epoch {latest_checkpoint}. Loading it...")
    checkpoint_path = os.path.join(checkpoint_dir, f"ctgan_epoch_{latest_checkpoint:03d}.pkl")
    ctgan = joblib.load(checkpoint_path)
else:
    print("No checkpoint found. Starting from scratch.")

# Continue training from the next checkpoint
for checkpoint in range(latest_checkpoint + 30, 301, 30):
    print(f"\nTraining to epoch {checkpoint}...")
    ctgan.fit(train_df)
    checkpoint_path = os.path.join(checkpoint_dir, f"ctgan_epoch_{checkpoint:03d}.pkl")
    joblib.dump(ctgan, checkpoint_path)
    print(f"Checkpoint {checkpoint} saved to: {checkpoint_path}")


# Generate synthetic data
print("\nGenerating synthetic data...")
synthetic_data_1 = ctgan.sample(len(real_processed_df))
synthetic_output = os.path.join(output_dir, "synthetic_1.csv")
synthetic_data_1.to_csv(synthetic_output, index=False)
print(f"Synthetic data saved to: {synthetic_output}")

# Save final model
final_model_path = os.path.join(models_dir, "ctgan_final.pkl")
joblib.dump(ctgan, final_model_path)
print(f"Final model saved to: {final_model_path}")
print("CTGAN TRAINING DONE - 300 EPOCHS - READY FOR EVALUATION")