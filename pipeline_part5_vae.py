"""
=============================================================================
PIPELINE — Part 5: Variational Autoencoder for IV Surface Compression
=============================================================================
Trains a VAE to learn a low-dimensional latent representation of the
154-dimensional implied volatility surface. Compresses each day's surface
to a small number of latent factors (default: 5).

Architecture follows Kingma & Welling (2013) with beta-VAE formulation.
Encoder/Decoder are feedforward networks with 3 hidden layers (128 nodes).

Reference: Zhang et al. (2021) Section 3.1 Method 3 (VAE).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import time


# ============================================================
# 1. VAE ARCHITECTURE
# ============================================================
class IVSurfaceEncoder(nn.Module):
    """
    Encoder: maps 154-dim IV surface to latent mean and log-variance.

    Architecture: 154 -> 128 -> 128 -> 128 -> (mu_d, logvar_d)
    """

    def __init__(self, input_dim=154, hidden_dim=128, latent_dim=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.shared(x)
        return self.fc_mu(h), self.fc_logvar(h)


class IVSurfaceDecoder(nn.Module):
    """
    Decoder: maps latent vector to reconstructed 154-dim IV surface.

    Architecture: d -> 128 -> 128 -> 128 -> 154
    No activation on output layer (IV values are continuous).
    """

    def __init__(self, latent_dim=5, hidden_dim=128, output_dim=154):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class IVSurfaceVAE(nn.Module):
    """
    Full VAE combining encoder and decoder.
    """

    def __init__(self, input_dim=154, hidden_dim=128, latent_dim=5):
        super().__init__()
        self.encoder = IVSurfaceEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = IVSurfaceDecoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        """z = mu + sigma * epsilon, epsilon ~ N(0, I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def encode(self, x):
        """Deterministic encoding (returns mu only, for inference)."""
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        """Decode latent vector to IV surface."""
        return self.decoder(z)


# ============================================================
# 2. LOSS FUNCTION
# ============================================================
def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    Beta-VAE loss = MSE reconstruction + beta * KL divergence.

    MSE: mean over batch and features
    KL: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)), averaged over batch

    Args:
        x_recon: reconstructed surface, shape (batch, 154)
        x: original surface, shape (batch, 154)
        mu: latent mean, shape (batch, d)
        logvar: latent log-variance, shape (batch, d)
        beta: KL weight (< 1.0 emphasizes reconstruction)

    Returns:
        total_loss, recon_loss, kl_loss (all scalars)
    """
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ============================================================
# 3. TRAINING
# ============================================================
def train_vae(
    iv_matrix,
    latent_dim=5,
    hidden_dim=128,
    beta=0.1,
    beta_warmup_fraction=0.4,
    epochs=1000,
    batch_size=64,
    lr=1e-3,
    val_fraction=0.15,
    patience=100,
):
    """
    Train VAE on IV surface matrix.

    Args:
        iv_matrix: np.ndarray shape (n_dates, 154), implied volatilities
        latent_dim: dimension of latent space
        hidden_dim: neurons per hidden layer
        beta: target KL weight
        beta_warmup_fraction: fraction of epochs for beta warmup (0 -> beta)
        epochs: max training epochs
        batch_size: minibatch size
        lr: learning rate
        val_fraction: fraction of data for validation (random split)
        patience: early stopping patience

    Returns:
        vae: trained IVSurfaceVAE
        scaler: fitted StandardScaler (for inverse-transforming decoder output)
        latent_factors: np.ndarray shape (n_dates, latent_dim)
        history: dict with training curves
    """
    print("\n" + "=" * 60)
    print("VAE TRAINING")
    print("=" * 60)
    print(f"  Input dim: {iv_matrix.shape[1]}, Latent dim: {latent_dim}")
    print(f"  Samples: {iv_matrix.shape[0]}, Beta: {beta}")

    # --- Standardize ---
    scaler = StandardScaler()
    iv_scaled = scaler.fit_transform(iv_matrix).astype(np.float32)

    # --- Train/Val Split (TEMPORAL — last val_fraction used for validation) ---
    # Using temporal split prevents data leakage: VAE never sees future surfaces
    # during training, which is critical since IV surfaces are autocorrelated.
    n_total = len(iv_scaled)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_total)

    train_data = torch.tensor(iv_scaled[train_idx])
    val_data = torch.tensor(iv_scaled[val_idx])
    print(f"  Temporal split: Train={n_train} (earliest), Val={n_val} (latest)")

    train_loader = DataLoader(
        TensorDataset(train_data), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_data), batch_size=batch_size, shuffle=False
    )

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = iv_matrix.shape[1]
    vae = IVSurfaceVAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience // 3, factor=0.5, min_lr=1e-6
    )

    n_params = sum(p.numel() for p in vae.parameters())
    print(f"  VAE parameters: {n_params:,}")

    # --- Training Loop ---
    beta_warmup_epochs = int(epochs * beta_warmup_fraction)
    history = {"train_loss": [], "val_loss": [], "recon_loss": [], "kl_loss": []}
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    start_time = time.time()

    for epoch in range(epochs):
        # Beta schedule: linear warmup
        if epoch < beta_warmup_epochs:
            current_beta = beta * (epoch / max(1, beta_warmup_epochs))
        else:
            current_beta = beta

        # --- Train ---
        vae.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            x_recon, mu, logvar = vae(batch_x)
            loss, recon, kl = vae_loss(x_recon, batch_x, mu, logvar, beta=current_beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            n_batches += 1

        epoch_loss /= n_batches
        epoch_recon /= n_batches
        epoch_kl /= n_batches

        # --- Validate ---
        vae.eval()
        val_loss_total = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                x_recon, mu, logvar = vae(batch_x)
                loss, _, _ = vae_loss(x_recon, batch_x, mu, logvar, beta=current_beta)
                val_loss_total += loss.item()
                n_val_batches += 1
        val_loss = val_loss_total / max(1, n_val_batches)

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["recon_loss"].append(epoch_recon)
        history["kl_loss"].append(epoch_kl)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in vae.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch+1}/{epochs} | "
                f"Train: {epoch_loss:.6f} (Recon: {epoch_recon:.6f}, KL: {epoch_kl:.4f}) | "
                f"Val: {val_loss:.6f} | Beta: {current_beta:.4f} | LR: {current_lr:.6f}"
            )

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        vae.load_state_dict(best_state)

    train_time = time.time() - start_time
    print(f"\n  Training complete in {train_time:.1f}s | Best val loss: {best_val_loss:.6f}")

    # --- Extract Latent Factors ---
    latent_factors = extract_latent_factors(vae, iv_matrix, scaler, device)

    # Report latent factor statistics
    print(f"\n[LATENT FACTOR SUMMARY]")
    for k in range(latent_dim):
        z_k = latent_factors[:, k]
        print(f"  z_{k}: mean={z_k.mean():.4f}, std={z_k.std():.4f}, "
              f"range=[{z_k.min():.4f}, {z_k.max():.4f}]")

    # Reconstruction quality
    recon_rmse = _compute_reconstruction_rmse(vae, iv_matrix, scaler, device)
    print(f"\n  Reconstruction RMSE (IV space): {recon_rmse:.6f}")

    return vae, scaler, latent_factors, history


# ============================================================
# 4. LATENT FACTOR EXTRACTION
# ============================================================
def extract_latent_factors(vae, iv_matrix, scaler, device=None):
    """
    Extract latent factors for all dates (deterministic: uses mu only).

    Args:
        vae: trained IVSurfaceVAE
        iv_matrix: np.ndarray shape (n_dates, 154)
        scaler: fitted StandardScaler
        device: torch device

    Returns:
        latent_factors: np.ndarray shape (n_dates, latent_dim)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae.eval()
    iv_scaled = scaler.transform(iv_matrix).astype(np.float32)
    x_tensor = torch.tensor(iv_scaled).to(device)

    with torch.no_grad():
        latent = vae.encode(x_tensor).cpu().numpy()

    return latent


def _compute_reconstruction_rmse(vae, iv_matrix, scaler, device):
    """Compute reconstruction RMSE in original IV space."""
    vae.eval()
    iv_scaled = scaler.transform(iv_matrix).astype(np.float32)
    x_tensor = torch.tensor(iv_scaled).to(device)

    with torch.no_grad():
        mu, logvar = vae.encoder(x_tensor)
        x_recon_scaled = vae.decoder(mu).cpu().numpy()

    x_recon = scaler.inverse_transform(x_recon_scaled)
    rmse = np.sqrt(np.mean((iv_matrix - x_recon) ** 2))
    return rmse


def decode_latent_factors(vae, latent_factors, scaler, device=None):
    """
    Decode latent factors back to IV surfaces.

    Args:
        vae: trained IVSurfaceVAE
        latent_factors: np.ndarray shape (n, latent_dim)
        scaler: fitted StandardScaler
        device: torch device

    Returns:
        iv_surfaces: np.ndarray shape (n, 154) in original IV space
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae.eval()
    z_tensor = torch.tensor(latent_factors, dtype=torch.float32).to(device)

    with torch.no_grad():
        iv_recon_scaled = vae.decoder(z_tensor).cpu().numpy()

    iv_recon = scaler.inverse_transform(iv_recon_scaled)
    return iv_recon


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Stage 5 requires IV surface matrix from Stage 4.")
    print("Run via main_pipeline.py or load iv_surfaces.pkl.")
