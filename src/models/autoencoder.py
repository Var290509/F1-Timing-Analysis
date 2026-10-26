"""
F1 Sentinel PyTorch Feedforward Telemetry Autoencoder
Compresses windowed feature vectors into a low-dimensional bottleneck
and reconstructs the signal. Unseen anomalies produce high reconstruction error.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


class TelemetryAutoencoderNN(nn.Module):
    """
    Lightweight feedforward autoencoder with bottleneck compression.
    Fast CPU inference: < 1ms per window.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class AutoencoderDetector:
    """
    PyTorch Autoencoder Anomaly Detector for F1 Telemetry.
    Computes mean-squared reconstruction error as continuous anomaly score.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        latent_dim: int = 8,
        lr: float = 1e-3,
        epochs: int = 15,
        batch_size: int = 64,
        random_state: int = 42
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.name = "PyTorch_Autoencoder"

        self.scaler = StandardScaler()
        self.model: Optional[TelemetryAutoencoderNN] = None
        self.is_fitted = False
        self._baseline_mse_mean = 0.05
        self._baseline_mse_std = 0.02
        self._mse_95 = 0.1

    def fit(self, X: np.ndarray):
        """
        Train autoencoder on normal baseline feature windows.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        input_dim = X.shape[1]
        self.model = TelemetryAutoencoderNN(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        )

        X_scaled = self.scaler.fit_transform(X)
        tensor_data = torch.tensor(X_scaled, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(tensor_data)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        self.model.train()
        for epoch in range(self.epochs):
            for batch in loader:
                x_batch = batch[0]
                optimizer.zero_grad()
                recon = self.model(x_batch)
                loss = criterion(recon, x_batch)
                loss.backward()
                optimizer.step()

        self.model.eval()
        self.is_fitted = True

        # Compute baseline MSE distribution on training set
        with torch.no_grad():
            recons = self.model(tensor_data).numpy()
            train_mse = np.mean((X_scaled - recons) ** 2, axis=1)
            self._baseline_mse_mean = float(np.mean(train_mse))
            self._baseline_mse_std = max(float(np.std(train_mse)), 1e-5)
            self._mse_95 = float(np.percentile(train_mse, 95))

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute continuous [0, 1] anomaly scores based on reconstruction error.
        Higher score = higher reconstruction distortion = anomalous.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Autoencoder is not trained yet.")

        self.model.eval()
        X_scaled = self.scaler.transform(X)
        tensor_data = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            recons = self.model(tensor_data).numpy()

        mse = np.mean((X_scaled - recons) ** 2, axis=1)
        # Ratio relative to 95th percentile of normal baseline
        thresh_ref = max(self._mse_95, self._baseline_mse_mean * 1.5, 1e-4)
        ratio = mse / thresh_ref
        # Sigmoid calibration: 0.1 at ratio=0, 0.5 at ratio=1.0, >0.85 at ratio >= 2.0
        scores = 1.0 / (1.0 + np.exp(-2.5 * (ratio - 1.0)))
        return np.clip(scores, 0.0, 1.0)

    def get_raw_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Return raw uncalibrated MSE reconstruction error."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Autoencoder is not trained yet.")

        self.model.eval()
        X_scaled = self.scaler.transform(X)
        tensor_data = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            recons = self.model(tensor_data).numpy()
        return np.mean((X_scaled - recons) ** 2, axis=1)

    def score_single(self, feature_vector: np.ndarray) -> Tuple[float, float]:
        """
        Score a single feature vector in streaming mode.
        Returns (continuous_anomaly_score, raw_reconstruction_mse).
        """
        feat_2d = feature_vector.reshape(1, -1)
        score = float(self.score(feat_2d)[0])
        mse = float(self.get_raw_reconstruction_error(feat_2d)[0])
        return score, mse
