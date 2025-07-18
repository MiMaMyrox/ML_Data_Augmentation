import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

DATA_PATH = "./data/evaluation/"


def gaussian():
    # Load dataset
    df = pd.read_csv(f"{DATA_PATH}train.csv")  # 🔁 Replace with your actual file path

    # Split into features and target
    X = df.drop(columns=["strength"]).values  # 8 features
    y = df["strength"].values  # Target

    # (Optional) Standardize features for better GP performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define Gaussian Process with RBF kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True)

    # Fit GP model
    gp.fit(X_scaled, y)

    # Generate synthetic samples
    n_synthetic = 300
    X_synth_scaled = np.random.normal(loc=0.0, scale=1.0, size=(n_synthetic, X.shape[1]))  # Standard normal in 8D
    y_mean, y_std = gp.predict(X_synth_scaled, return_std=True)

    # Sample synthetic outputs from the posterior
    y_synthetic = y_mean + y_std * np.random.randn(n_synthetic)

    # Inverse transform features to original scale (optional)
    X_synth = scaler.inverse_transform(X_synth_scaled)

    feature_cols = df.drop(columns=["strength"]).columns
    df_synth = pd.DataFrame(X_synth, columns=feature_cols)
    df_synth["strength"] = y_synthetic

    df_combined = pd.concat([df, df_synth], ignore_index=True)
    df_combined.to_csv(f"{DATA_PATH}gaussian.csv", index=False)

if __name__ == "__main__":
    gaussian()
