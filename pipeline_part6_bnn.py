"""
=============================================================================
PIPELINE — Part 6: Bayesian Neural Network for Latent Factor Prediction
=============================================================================
Predicts next-day latent factors z_{t+1} given today's latent factors z_t
plus market features (spot price, forward curve), with uncertainty
quantification via Bayesian weight distributions.

Architecture: Bayes by Backprop (Blundell et al. 2015) with the local
reparameterization trick (Kingma et al. 2015) for variance reduction.

Novel contribution over Zhang et al. (2021): replaces LSTM with BNN
to obtain calibrated prediction uncertainty.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import time


# ============================================================
# 1. BAYESIAN LINEAR LAYER
# ============================================================
class BayesianLinear(nn.Module):
    """
    Linear layer with Gaussian weight distributions.

    Each weight w_ij ~ N(mu_ij, sigma_ij^2) where sigma = log(1 + exp(rho)).

    Uses the local reparameterization trick: instead of sampling weights,
    sample the pre-activation directly from its induced distribution.
    This reduces gradient variance compared to naive weight sampling.

    KL divergence: KL(q(w|mu,sigma) || p(w|0,prior_sigma))
    """

    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Variational parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))

        # Initialize mu with Kaiming
        nn.init.kaiming_normal_(self.weight_mu, nonlinearity="relu")

    def forward(self, x):
        """
        Local reparameterization trick:
            act_mu  = x @ mu.T + bias_mu
            act_var = x^2 @ sigma^2.T + bias_sigma^2
            act     = act_mu + sqrt(act_var) * eps
        """
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        act_mu = F.linear(x, self.weight_mu, self.bias_mu)
        act_var = F.linear(x.pow(2), weight_sigma.pow(2), bias_sigma.pow(2))
        act_std = torch.sqrt(act_var + 1e-8)

        eps = torch.randn_like(act_mu)
        return act_mu + act_std * eps

    def kl_divergence(self):
        """
        KL(N(mu, sigma^2) || N(0, prior_sigma^2)) summed over all parameters.
        """
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        kl_w = self._gaussian_kl(self.weight_mu, weight_sigma)
        kl_b = self._gaussian_kl(self.bias_mu, bias_sigma)
        return kl_w + kl_b

    def _gaussian_kl(self, mu, sigma):
        """KL divergence between N(mu, sigma^2) and N(0, prior_sigma^2)."""
        prior = self.prior_sigma
        return (
            torch.log(torch.tensor(prior)) - torch.log(sigma)
            + (sigma.pow(2) + mu.pow(2)) / (2 * prior**2)
            - 0.5
        ).sum()


# ============================================================
# 2. BAYESIAN NEURAL NETWORK
# ============================================================
class BayesianNN(nn.Module):
    """
    Bayesian Neural Network for next-day latent factor prediction.

    Input:  [z_t (latent_dim), S_t (1), F_t (n_fwd_tenors)] = input_dim
    Output: z_{t+1} (latent_dim)

    All layers are BayesianLinear with ReLU activations.
    """

    def __init__(self, input_dim=14, hidden_dim=64, output_dim=5,
                 num_layers=3, prior_sigma=1.0):
        super().__init__()

        dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(BayesianLinear(dims[i], dims[i + 1], prior_sigma))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def kl_divergence(self):
        """Total KL divergence across all Bayesian layers."""
        return sum(layer.kl_divergence() for layer in self.layers)

    def predict(self, x, n_samples=100):
        """
        Monte Carlo prediction with uncertainty estimation.

        Args:
            x: input features, shape (batch, input_dim)
            n_samples: number of stochastic forward passes

        Returns:
            mean: shape (batch, output_dim)
            std:  shape (batch, output_dim)
            samples: shape (n_samples, batch, output_dim)
        """
        self.train()  # Keep stochastic for sampling
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(self(x))
        samples = torch.stack(samples)
        return samples.mean(0), samples.std(0), samples


# ============================================================
# 3. DATA PREPARATION
# ============================================================
def prepare_bnn_data(latent_factors, St_df, fwd_df, surface_dates):
    """
    Prepare (X, y) pairs for BNN training.

    For each day t (where t+1 also exists in the dataset):
        X_t = [z_t, S_t, F_t_0d, F_t_7d, ..., F_t_350d]
        y_t = z_{t+1}

    Args:
        latent_factors: np.ndarray shape (n_dates, latent_dim)
        St_df: DataFrame with spot prices (index = dates)
        fwd_df: DataFrame with forward prices at 8 tenors (index = dates)
        surface_dates: list of Timestamps (aligned with latent_factors rows)

    Returns:
        X: np.ndarray shape (n_pairs, input_dim)
        y: np.ndarray shape (n_pairs, latent_dim)
        feature_scaler: StandardScaler fitted on X
        dates_X: list of Timestamps for each row (the "today" date)
    """
    print("\n  Preparing BNN training data...")

    # Build date-indexed lookups
    # St_df may have a single column or be a Series
    if isinstance(St_df, pd.DataFrame):
        spot_vals = St_df.iloc[:, 0]
    else:
        spot_vals = St_df

    # Ensure date indices are comparable
    spot_index = pd.DatetimeIndex(spot_vals.index)
    fwd_index = pd.DatetimeIndex(fwd_df.index)

    X_list = []
    y_list = []
    dates_X = []

    n_dates = len(surface_dates)
    latent_dim = latent_factors.shape[1]
    n_fwd_tenors = fwd_df.shape[1]
    skipped = 0

    for i in range(n_dates - 1):
        d_today = surface_dates[i]
        d_tomorrow = surface_dates[i + 1]

        # Look up spot price
        # Find the closest matching date in spot_index
        spot_match = spot_index.get_indexer([d_today], method="nearest")[0]
        if spot_match < 0 or spot_match >= len(spot_vals):
            skipped += 1
            continue
        S_t = float(spot_vals.iloc[spot_match])

        # Look up forward curve
        fwd_match = fwd_index.get_indexer([d_today], method="nearest")[0]
        if fwd_match < 0 or fwd_match >= len(fwd_df):
            skipped += 1
            continue
        F_t = fwd_df.iloc[fwd_match].values.astype(float)

        # Build feature vector: [z_t, S_t, F_t]
        z_t = latent_factors[i]
        x_row = np.concatenate([z_t, [S_t], F_t])

        X_list.append(x_row)
        y_list.append(latent_factors[i + 1])
        dates_X.append(d_today)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)

    if skipped > 0:
        print(f"  Skipped {skipped} dates due to missing spot/fwd data")

    print(f"  Total pairs: {len(X)}")
    print(f"  Feature dim: {X.shape[1]} "
          f"(latent={latent_dim}, spot=1, fwd={n_fwd_tenors})")

    # Normalize features
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)

    return X_scaled, y, feature_scaler, dates_X


# ============================================================
# 4. TRAINING
# ============================================================
def train_bnn(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim=14,
    hidden_dim=64,
    output_dim=5,
    num_layers=3,
    prior_sigma=1.0,
    epochs=500,
    batch_size=64,
    lr=1e-3,
    kl_weight=None,
    n_train_samples=5,
    patience=50,
):
    """
    Train BNN using ELBO loss.

    ELBO = E_q[log p(y|x,w)] - kl_weight * KL(q(w)||p(w))

    The log-likelihood is approximated as -MSE (Gaussian likelihood
    with fixed variance). KL weight is set to 1/N_train by default
    (Graves scaling).

    Args:
        X_train, y_train: training data (already scaled)
        X_val, y_val: validation data (already scaled)
        input_dim: input feature dimension
        hidden_dim: hidden layer width
        output_dim: output dimension (latent_dim)
        num_layers: number of hidden layers
        prior_sigma: prior standard deviation for weights
        epochs: max training epochs
        batch_size: minibatch size
        lr: learning rate
        kl_weight: KL term weight (default: 1/N_train)
        n_train_samples: MC samples per training step
        patience: early stopping patience

    Returns:
        bnn: trained BayesianNN
        history: dict with loss curves
    """
    print("\n" + "=" * 60)
    print("BNN TRAINING")
    print("=" * 60)

    n_train = len(X_train)
    if kl_weight is None:
        kl_weight = 1.0 / n_train

    print(f"  Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"  Hidden: {hidden_dim} x {num_layers} layers")
    print(f"  Train: {n_train}, Val: {len(X_val)}")
    print(f"  KL weight: {kl_weight:.6f} (1/N_train)")
    print(f"  Prior sigma: {prior_sigma}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Model
    bnn = BayesianNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        prior_sigma=prior_sigma,
    ).to(device)

    n_params = sum(p.numel() for p in bnn.parameters())
    print(f"  BNN parameters: {n_params:,}")

    optimizer = optim.Adam(bnn.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience // 3, factor=0.5, min_lr=1e-6
    )

    # Training loop
    history = {"train_loss": [], "val_loss": [], "mse_loss": [], "kl_loss": []}
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    start_time = time.time()

    for epoch in range(epochs):
        # --- Train ---
        bnn.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()

            # Multiple MC samples for variance reduction
            mse_accum = 0.0
            for _ in range(n_train_samples):
                pred = bnn(batch_x)
                mse_accum += F.mse_loss(pred, batch_y, reduction="mean")
            mse_loss = mse_accum / n_train_samples

            kl_loss = bnn.kl_divergence()
            loss = mse_loss + kl_weight * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(bnn.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1

        epoch_loss /= n_batches
        epoch_mse /= n_batches
        epoch_kl /= n_batches

        # --- Validate ---
        bnn.eval()
        with torch.no_grad():
            # Deterministic validation: use weight means only
            val_pred_mu = _deterministic_forward(bnn, val_x)
            val_mse = F.mse_loss(val_pred_mu, val_y, reduction="mean").item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_mse)
        history["mse_loss"].append(epoch_mse)
        history["kl_loss"].append(epoch_kl)

        scheduler.step(val_mse)

        # Early stopping
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = {k: v.clone() for k, v in bnn.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch+1}/{epochs} | "
                f"Train ELBO: {epoch_loss:.6f} (MSE: {epoch_mse:.6f}, "
                f"KL: {epoch_kl:.1f}) | "
                f"Val MSE: {val_mse:.6f} | LR: {current_lr:.6f}"
            )

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        bnn.load_state_dict(best_state)

    train_time = time.time() - start_time
    print(f"\n  Training complete in {train_time:.1f}s | Best val MSE: {best_val_loss:.6f}")

    return bnn, history


def _deterministic_forward(bnn, x):
    """
    Deterministic forward pass using weight means only (no sampling).
    Used for validation to get a stable loss estimate.
    """
    for i, layer in enumerate(bnn.layers):
        x = F.linear(x, layer.weight_mu, layer.bias_mu)
        if i < len(bnn.layers) - 1:
            x = F.relu(x)
    return x


# ============================================================
# 5. PREDICTION WITH UNCERTAINTY
# ============================================================
def predict_with_uncertainty(
    bnn,
    vae_decoder,
    scaler_features,
    scaler_iv,
    X_test_raw,
    n_mc_samples=200,
    device=None,
):
    """
    Full prediction pipeline with uncertainty quantification.

    BNN predicts latent factors -> VAE decoder -> IV surfaces.
    Uncertainty comes from MC sampling over BNN weight distributions.

    Args:
        bnn: trained BayesianNN
        vae_decoder: VAE decoder module (IVSurfaceDecoder)
        scaler_features: StandardScaler for BNN inputs (already applied if X_test_raw is scaled)
        scaler_iv: StandardScaler for IV surfaces (from VAE training)
        X_test_raw: test features, shape (n_test, input_dim) — already scaled
        n_mc_samples: number of MC forward passes
        device: torch device

    Returns:
        dict with:
            'latent_mean': (n_test, latent_dim)
            'latent_std':  (n_test, latent_dim)
            'iv_mean':     (n_test, 154)
            'iv_std':      (n_test, 154)
            'iv_samples':  (n_mc_samples, n_test, 154) — in original IV space
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n  Generating predictions with uncertainty...")
    print(f"  Test samples: {len(X_test_raw)}, MC samples: {n_mc_samples}")

    x_tensor = torch.tensor(X_test_raw, dtype=torch.float32).to(device)

    # --- MC sampling over BNN ---
    bnn.train()  # Keep stochastic
    vae_decoder.eval()

    latent_samples = []
    iv_samples = []

    with torch.no_grad():
        for s in range(n_mc_samples):
            # Sample latent prediction
            z_pred = bnn(x_tensor)  # (n_test, latent_dim)
            latent_samples.append(z_pred.cpu().numpy())

            # Decode to IV surface (in scaled space)
            iv_scaled = vae_decoder(z_pred)  # (n_test, 154)
            iv_np = iv_scaled.cpu().numpy()

            # Inverse-transform to original IV space
            iv_original = scaler_iv.inverse_transform(iv_np)
            iv_samples.append(iv_original)

    latent_samples = np.array(latent_samples)  # (n_mc, n_test, latent_dim)
    iv_samples = np.array(iv_samples)          # (n_mc, n_test, 154)

    results = {
        "latent_mean": latent_samples.mean(axis=0),
        "latent_std": latent_samples.std(axis=0),
        "iv_mean": iv_samples.mean(axis=0),
        "iv_std": iv_samples.std(axis=0),
        "iv_samples": iv_samples,
    }

    # Summary
    print(f"  Latent factor prediction std (mean across test): "
          f"{results['latent_std'].mean():.4f}")
    print(f"  IV prediction std (mean across test): "
          f"{results['iv_std'].mean():.6f}")

    return results


# ============================================================
# 6. EVALUATION METRICS
# ============================================================
def evaluate_predictions(predictions, y_test_latent, iv_test_actual=None):
    """
    Evaluate BNN prediction quality.

    Args:
        predictions: dict from predict_with_uncertainty
        y_test_latent: actual next-day latent factors, shape (n_test, latent_dim)
        iv_test_actual: actual next-day IV surfaces (optional), shape (n_test, 154)

    Returns:
        metrics: dict with evaluation results
    """
    metrics = {}

    # Latent space metrics
    latent_mean = predictions["latent_mean"]
    latent_std = predictions["latent_std"]

    latent_mse = np.mean((latent_mean - y_test_latent) ** 2)
    latent_rmse = np.sqrt(latent_mse)
    metrics["latent_rmse"] = latent_rmse

    # Calibration: fraction of true values within predicted intervals
    for ci_level in [0.50, 0.90, 0.95]:
        z_score = {0.50: 0.6745, 0.90: 1.6449, 0.95: 1.960}[ci_level]
        lower = latent_mean - z_score * latent_std
        upper = latent_mean + z_score * latent_std
        coverage = np.mean((y_test_latent >= lower) & (y_test_latent <= upper))
        metrics[f"latent_coverage_{int(ci_level*100)}"] = coverage

    print(f"\n[BNN EVALUATION]")
    print(f"  Latent RMSE: {latent_rmse:.6f}")
    for ci_level in [50, 90, 95]:
        cov = metrics[f"latent_coverage_{ci_level}"]
        print(f"  {ci_level}% CI coverage: {cov:.3f} (ideal: {ci_level/100:.2f})")

    # IV space metrics (if actual surfaces provided)
    if iv_test_actual is not None:
        iv_mean = predictions["iv_mean"]
        iv_rmse = np.sqrt(np.mean((iv_mean - iv_test_actual) ** 2))
        metrics["iv_rmse"] = iv_rmse
        print(f"  IV RMSE: {iv_rmse:.6f}")

    return metrics


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Stage 6 requires latent factors from Stage 5.")
    print("Run via main_pipeline.py or load vae_results.pkl.")
