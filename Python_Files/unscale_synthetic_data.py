import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

synthetic_data_path = r"E:\Study (B.Sc & M.Sc )\Faculty Of Information and Communication Technology, WUST (M.Sc)\S 2\AI\Project\Project Implementaion v2\Datasets\CTGAN\synthetic_1.csv"

# Load real dataset to fit scalers
real_dataset_path = r"e:\Study (B.Sc & M.Sc )\Faculty Of Information and Communication Technology, WUST (M.Sc)\S 2\AI\Project\Project Implementaion v2\Datasets\Real\diabetes_012_health_indicators_BRFSS2015.csv"
real_dataset = pd.read_csv(real_dataset_path)
real_dataset = real_dataset.drop_duplicates()

target_column = 'Diabetes_012'
X_real = real_dataset.drop(target_column, axis=1)
y_real = real_dataset[target_column].values.reshape(-1, 1)

# Fit feature scaler on real features
feature_scaler = StandardScaler()
feature_scaler.fit(X_real)

# Fit target scaler on real target
target_scaler = StandardScaler()
target_scaler.fit(y_real)

# Load synthetic data
synthetic_data = pd.read_csv(synthetic_data_path)

# Unscale features
synthetic_features_scaled = synthetic_data.drop(target_column, axis=1).values
synthetic_features_unscaled = feature_scaler.inverse_transform(synthetic_features_scaled)

# Unscale target
synthetic_target_scaled = synthetic_data[target_column].values.reshape(-1, 1)
synthetic_target_unscaled = target_scaler.inverse_transform(synthetic_target_scaled)

# Round and clip target to valid range {0, 1, 2}
synthetic_target_corrected = np.round(synthetic_target_unscaled).astype(int)
synthetic_target_corrected = np.clip(synthetic_target_corrected, 0, 2).flatten()

# Create unscaled dataframe
synthetic_unscaled = pd.DataFrame(synthetic_features_unscaled, columns=X_real.columns)
synthetic_unscaled[target_column] = synthetic_target_corrected

# Define column types and constraints
binary_columns = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 
                  'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                  'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']

ordinal_columns = {
    'GenHlth': (1, 5),      # 1-5 scale
    'Age': (1, 13),         # 1-13 scale
    'Education': (1, 6),    # 1-6 scale
    'Income': (1, 8)        # 1-8 scale
}

continuous_columns = {
    'BMI': (12, 98),        # Reasonable BMI range
    'MentHlth': (0, 30),    # 0-30 days
    'PhysHlth': (0, 30)     # 0-30 days
}

# Apply constraints to binary columns
for col in binary_columns:
    synthetic_unscaled[col] = np.clip(np.round(synthetic_unscaled[col]), 0, 1).astype(int)

# Apply constraints to ordinal columns
for col, (min_val, max_val) in ordinal_columns.items():
    synthetic_unscaled[col] = np.clip(np.round(synthetic_unscaled[col]), min_val, max_val).astype(int)

# Apply constraints to continuous columns
for col, (min_val, max_val) in continuous_columns.items():
    synthetic_unscaled[col] = np.clip(synthetic_unscaled[col], min_val, max_val)

# Reorder columns to match original
synthetic_unscaled = synthetic_unscaled[real_dataset.columns]

# Generate output path with number suffix
base_path, ext = os.path.splitext(synthetic_data_path)
output_path = f"{base_path}_unscaled{ext}"

# Save unscaled data
synthetic_unscaled.to_csv(output_path, index=False)

print(f"Done! Unscaled data saved to: {output_path}")
