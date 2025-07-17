import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import gaussian_kde

df_train = pd.read_csv('data/evaluation/train.csv')

# Train Test Split
X = df_train.drop(columns=['strength'])
y = df_train['strength']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the CVAE model
class CVAE(nn.Module):
    def __init__(self, x_dim, cond_dim=1, latent_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + cond_dim, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 32),
            nn.ReLU(),
            nn.Linear(32, x_dim)
        )

    def encode(self, x, c):
        xc = torch.cat([x, c], dim=1)
        h = self.encoder(xc)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        zc = torch.cat([z, c], dim=1)
        return self.decoder(zc)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar
    
def loss_function(x_recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


# Initialize the model, optimizer, and training parameters
x_dim = X.shape[1]
model = CVAE(x_dim=x_dim, cond_dim=1, latent_dim=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 250
last_values = [0] + [1] * 5
threshold = 0.002

for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x_batch, y_batch.unsqueeze(1))
        loss = loss_function(x_recon, x_batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        last_values[epoch%len(last_values)] = total_loss
    
    absolute_errors = [abs(x - max(last_values)) for x in last_values]
    mae = sum(absolute_errors) / len(absolute_errors)
    if mae < threshold:
        print(f"Early stopping at epoch {epoch} with MAE: {mae:.4f}")
        break

# Generate new Strength values
strength = df_train['strength']
    
kde = gaussian_kde(strength)
zufallswerte = kde.resample(400)
zufallswerte[zufallswerte < 3] = 3
zufallswerte = zufallswerte.ravel()


# Generate new data
augemented_data = []

for i in range(zufallswerte.shape[0]):
    desired_strength = torch.tensor([[zufallswerte[i]]], dtype=torch.float32)
    z = torch.randn(1, 3)

    model.eval()
    with torch.no_grad():
        generated = model.decode(z, desired_strength)
        generated_original = scaler.inverse_transform(generated.numpy())

        generated_original = np.append(generated_original[0], zufallswerte[i])

    augemented_data.append(generated_original)

augemented_data_df = pd.DataFrame(augemented_data, columns=df_train.columns)
augemented_data_train_data_df = pd.concat([df_train, augemented_data_df], ignore_index=True)

augemented_data_train_data_df.to_csv('data/evaluation/cvae_train.csv', index=False) 


