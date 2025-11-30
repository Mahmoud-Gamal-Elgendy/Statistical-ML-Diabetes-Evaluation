import os
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")


# ==================== DATA PREPROCESSING ====================
print("Starting data preprocessing...")

# Load the dataset
dataset_path = r"Datasets\Real\diabetes_012_health_indicators_BRFSS2015.csv"
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


# ==================== VAE MODEL DEFINITION ====================
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=64):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(x_recon, x, mu, logvar):
    """VAE loss = Reconstruction loss + KL divergence"""
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# ==================== VAE MODEL TRAINING ====================
print("\nTraining VAE model with checkpoint continuation...")

# Setup paths
checkpoint_dir = r"models\VAE\checkpoints"
models_dir = r"models\VAE"
output_dir = r"Datasets\VAE"

# Create directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Prepare data for PyTorch
train_data = torch.FloatTensor(train_df.values)
train_dataset = TensorDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Initialize model
input_dim = train_df.shape[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

vae_model = VAE(input_dim=input_dim, hidden_dim=256, latent_dim=64).to(device)
optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

# Detect the latest checkpoint (if any)
latest_checkpoint = 0
for chk in range(30, 301, 30):
    checkpoint_path = os.path.join(checkpoint_dir, f"vae_epoch_{chk:03d}.pkl")
    if os.path.exists(checkpoint_path):
        latest_checkpoint = chk

if latest_checkpoint > 0:
    print(f"Found existing checkpoint at epoch {latest_checkpoint}. Loading it...")
    checkpoint_path = os.path.join(checkpoint_dir, f"vae_epoch_{latest_checkpoint:03d}.pkl")
    checkpoint = torch.load(checkpoint_path)
    vae_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = latest_checkpoint
else:
    print("No checkpoint found. Starting from scratch.")
    start_epoch = 0

# Training loop
print(f"\nStarting training from epoch {start_epoch + 1} to 300...")
vae_model.train()

for epoch in range(start_epoch + 1, 301):
    epoch_loss = 0
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        x_recon, mu, logvar = vae_model(data)
        loss = vae_loss(x_recon, data, mu, logvar)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader.dataset)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/300 - Loss: {avg_loss:.4f}")
    
    # Save checkpoint every 30 epochs
    if epoch % 30 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"vae_epoch_{epoch:03d}.pkl")
        torch.save({
            'epoch': epoch,
            'model_state_dict': vae_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint {epoch} saved to: {checkpoint_path}")

print("\nTraining complete!")


# ==================== GENERATE SYNTHETIC DATA ====================
print("\nGenerating synthetic data...")
vae_model.eval()

# Generate synthetic samples
num_samples = len(real_processed_df)
with torch.no_grad():
    # Sample from latent space
    z = torch.randn(num_samples, 64).to(device)
    synthetic_data = vae_model.decode(z).cpu().numpy()

# Convert to DataFrame
synthetic_df = pd.DataFrame(synthetic_data, columns=train_df.columns)

# Save synthetic data
synthetic_output = os.path.join(output_dir, "synthetic_2.csv")
synthetic_df.to_csv(synthetic_output, index=False)
print(f"Synthetic data saved to: {synthetic_output}")

# Save final model
final_model_path = os.path.join(models_dir, "vae_final.pkl")
torch.save({
    'model_state_dict': vae_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'input_dim': input_dim,
    'hidden_dim': 256,
    'latent_dim': 64
}, final_model_path)
print(f"Final model saved to: {final_model_path}")
print("VAE TRAINING DONE - 300 EPOCHS - READY FOR EVALUATION")
