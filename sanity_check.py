"""
=============================================================================
COMPREHENSIVE SANITY CHECK & DIAGNOSTICS (v2 — post-fixes)
=============================================================================
Deep validation of every pipeline stage, paper benchmark comparison,
backtesting, and arbitrage verification.

Usage:
  python sanity_check.py
  python sanity_check.py --output_dir ./outputs
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm as norm_dist

# Import from pipeline to use CURRENT grid definitions
from pipeline_part4_surface_sampling import (
    MONEYNESS_STRIKES, MONEYNESS_GRID, MATURITY_GRID, MATURITY_DAYS_GRID,
    N_MONEYNESS, N_MATURITY, N_GRID,
    bs_normalized_call, implied_vol_newton
)
from pipeline_part5_vae import IVSurfaceVAE, decode_latent_factors
from pipeline_part6_bnn import BayesianNN, _deterministic_forward


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_fig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ============================================================
# 1. IV SURFACE CHECKS
# ============================================================
def check_iv_surfaces(iv_matrix, surface_dates, plots_dir):
    print("\n" + "=" * 60)
    print("[CHECK 1] IV Surface Quality (Stage 4)")
    print("=" * 60)

    n_dates, n_grid = iv_matrix.shape
    print(f"  Shape: {iv_matrix.shape} ({n_dates} dates x {n_grid} grid pts)")
    print(f"  Grid: {N_MONEYNESS} moneyness x {N_MATURITY} maturity = {N_GRID}")
    print(f"  Date range: {surface_dates[0].date()} to {surface_dates[-1].date()}")

    print(f"\n  IV statistics:")
    print(f"    Mean:   {np.mean(iv_matrix):.4f}")
    print(f"    Std:    {np.std(iv_matrix):.4f}")
    print(f"    Min:    {np.min(iv_matrix):.4f}")
    print(f"    Max:    {np.max(iv_matrix):.4f}")
    print(f"    Median: {np.median(iv_matrix):.4f}")
    print(f"    NaN:    {np.isnan(iv_matrix).sum()}")
    print(f"    Inf:    {np.isinf(iv_matrix).sum()}")

    # Percentile distribution
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"    P{p:02d}:    {np.percentile(iv_matrix, p):.4f}")

    # Reshape to 3D for structural checks
    iv_3d = iv_matrix.reshape(n_dates, N_MONEYNESS, N_MATURITY)

    # CHECK: Price convexity (the TRUE no-arbitrage condition)
    # The ISNN guarantees d²C/dK² >= 0 (price convex in strike).
    # IV convexity is NOT required for no-arbitrage and frequently fails
    # for short-dated SPX options — this is a known market feature, not a bug.
    # We compute normalised BS call prices from the IV grid and check d²C/dK² >= 0.
    atm_idx = np.argmin(np.abs(MONEYNESS_STRIKES - 1.0))

    # Price convexity check across maturities
    print(f"\n  Price convexity check (no-arbitrage condition d²C/dK² >= 0):")
    kf = MONEYNESS_STRIKES  # K/F values
    m_vals = MONEYNESS_GRID  # log(K/F)
    for mat_i, mat_days in enumerate(MATURITY_DAYS_GRID):
        T = mat_days / 365.0
        n_convex = 0
        for i in range(n_dates):
            iv_slice = iv_3d[i, :, mat_i]
            # Compute normalised BS prices from IVs
            c_norm = bs_normalized_call(m_vals, np.full(N_MONEYNESS, T), iv_slice)
            # Second difference w.r.t. K/F (finite difference for convexity)
            d2c = np.diff(c_norm, 2)
            if np.all(d2c >= -1e-6):
                n_convex += 1
        pct_c = 100 * n_convex / n_dates
        print(f"    {mat_days:4d}d: {n_convex}/{n_dates} ({pct_c:.1f}%) pass")

    # Also load and report ISNN autograd validation if available
    try:
        val_results = load_pkl(Path("outputs/stage3/validation_results.pkl"))
        n_valid = sum(1 for v in val_results.values() if v["is_valid"])
        n_total_v = len(val_results)
        print(f"\n  ISNN autograd validation (continuous model, 2500-point grid):")
        print(f"    Arbitrage-free: {n_valid}/{n_total_v} ({100*n_valid/n_total_v:.1f}%)")
        n_mono_viol = sum(v["time_monotonicity_violations"] for v in val_results.values())
        n_conv_viol = sum(v["strike_convexity_violations"] for v in val_results.values())
        print(f"    Total monotonicity violations: {n_mono_viol}")
        print(f"    Total convexity violations: {n_conv_viol}")
        if n_valid == n_total_v:
            print(f"    ✓ 100% arbitrage-free — ISNN architecture guarantees hold")
    except Exception:
        print(f"    (ISNN validation results not available)")

    # IV smile shape check (informational only — NOT a no-arbitrage requirement)
    print(f"\n  IV smile shape (informational — IV convexity is NOT required for no-arb):")
    for mat_i, mat_days in [(1, 30), (4, 122), (8, 365)]:
        n_iv_convex = 0
        for i in range(n_dates):
            lo = max(0, atm_idx - 3)
            hi = min(N_MONEYNESS, atm_idx + 4)
            smile = iv_3d[i, lo:hi, mat_i]
            d2 = np.diff(smile, 2)
            if np.all(d2 >= -0.02):
                n_iv_convex += 1
        pct_iv = 100 * n_iv_convex / n_dates
        print(f"    {mat_days:4d}d IV convexity near ATM: {n_iv_convex}/{n_dates} ({pct_iv:.1f}%)")

    # CHECK: Term structure smoothness
    ts_smooth = 0
    for i in range(n_dates):
        atm_ts = iv_3d[i, atm_idx, :]
        jumps = np.abs(np.diff(atm_ts))
        if np.max(jumps) < 0.3:
            ts_smooth += 1
    pct = 100 * ts_smooth / n_dates
    print(f"  Term structure smoothness (ATM): {ts_smooth}/{n_dates} ({pct:.1f}%)")

    # CHECK: Put-call parity / no-arb bounds
    # C_norm should be in [max(0, 1-K/F), 1] for calls
    # This translates to: IV should be positive
    n_negative = np.sum(iv_matrix <= 0)
    print(f"  Positive IV check: {n_dates * n_grid - n_negative}/{n_dates * n_grid} positive")

    # PLOTS
    # 1a. Sample IV surfaces as heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    sample_idx = np.linspace(0, n_dates - 1, 6, dtype=int)
    for ax, idx in zip(axes.ravel(), sample_idx):
        s = iv_3d[idx]
        im = ax.imshow(s.T, aspect="auto", origin="lower", cmap="viridis", vmin=0.05, vmax=0.6)
        ax.set_title(f"{surface_dates[idx].date()}", fontsize=10)
        ax.set_xlabel("K/F")
        ax.set_ylabel("Maturity (days)")
        ax.set_xticks(range(0, N_MONEYNESS, max(1, N_MONEYNESS // 6)))
        ax.set_xticklabels([f"{MONEYNESS_STRIKES[j]:.2f}" for j in range(0, N_MONEYNESS, max(1, N_MONEYNESS // 6))], rotation=45, fontsize=7)
        ax.set_yticks(range(N_MATURITY))
        ax.set_yticklabels(MATURITY_DAYS_GRID, fontsize=7)
    fig.colorbar(im, ax=axes, label="IV", shrink=0.8)
    fig.suptitle("Sample IV Surfaces (Stage 4 — post fix)", fontsize=14)
    plt.tight_layout()
    save_fig(fig, plots_dir / "check1_iv_surfaces.png")

    # 1b. ATM IV time series
    fig, ax = plt.subplots(figsize=(14, 5))
    for mat_i, label in [(1, "30d"), (4, "122d"), (8, "365d")]:
        ax.plot([d.date() for d in surface_dates], iv_3d[:, atm_idx, mat_i], label=label, alpha=0.8)
    ax.set_title("ATM Implied Volatility Over Time")
    ax.set_ylabel("IV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, plots_dir / "check1_atm_timeseries.png")

    # 1c. IV smiles for 3 dates
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, idx in zip(axes, [0, n_dates // 2, n_dates - 1]):
        for mat_i, label in [(1, "30d"), (3, "91d"), (8, "365d")]:
            ax.plot(MONEYNESS_STRIKES, iv_3d[idx, :, mat_i], "o-", label=label, markersize=3)
        ax.set_title(f"{surface_dates[idx].date()}")
        ax.set_xlabel("K/F")
        ax.set_ylabel("IV")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("IV Smile Cross-sections", fontsize=14)
    plt.tight_layout()
    save_fig(fig, plots_dir / "check1_iv_smiles.png")

    # 1d. IV distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(iv_matrix.ravel(), bins=150, alpha=0.7, density=True, edgecolor="none")
    ax.axvline(np.mean(iv_matrix), color="r", ls="--", label=f"Mean={np.mean(iv_matrix):.3f}")
    ax.axvline(np.median(iv_matrix), color="g", ls="--", label=f"Median={np.median(iv_matrix):.3f}")
    ax.set_title("IV Distribution (all grid points, all dates)")
    ax.set_xlabel("IV")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, plots_dir / "check1_iv_distribution.png")

    return iv_3d


# ============================================================
# 2. VAE CHECKS
# ============================================================
def check_vae(iv_matrix, surface_dates, out_dir, plots_dir):
    print("\n" + "=" * 60)
    print("[CHECK 2] VAE Reconstruction Quality (Stage 5)")
    print("=" * 60)

    vae_data = load_pkl(out_dir / "stage5/vae_results.pkl")
    latent_factors = vae_data["latent_factors"]
    scaler_iv = vae_data["scaler"]
    history = vae_data["history"]

    hp = load_pkl(out_dir / "stage5/vae_hyperparams.pkl")
    vae = IVSurfaceVAE(input_dim=hp["input_dim"], hidden_dim=hp["hidden_dim"], latent_dim=hp["latent_dim"])
    vae.load_state_dict(torch.load(out_dir / "stage5/vae_model.pt", map_location="cpu", weights_only=True))
    vae.eval()

    # Reconstruct
    iv_recon = decode_latent_factors(vae, latent_factors, scaler_iv)
    error = iv_matrix - iv_recon
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    max_err = np.max(np.abs(error))
    per_date_rmse = np.sqrt(np.mean(error ** 2, axis=1))

    total_var = np.var(iv_matrix)
    explained_var = 1 - np.var(error) / total_var

    print(f"  Latent dim: {latent_factors.shape[1]}")
    print(f"  Training epochs: {len(history['train_loss'])}")
    print(f"\n  Reconstruction:")
    print(f"    RMSE:           {rmse:.6f}")
    print(f"    MAE:            {mae:.6f}")
    print(f"    Max abs error:  {max_err:.6f}")
    print(f"    Per-date RMSE mean: {per_date_rmse.mean():.6f}")
    print(f"    Per-date RMSE p95:  {np.percentile(per_date_rmse, 95):.6f}")
    print(f"    Explained variance: {explained_var:.4f} ({100 * explained_var:.2f}%)")

    # Benchmark comparison
    # Zhang et al. (2021) report VAE reconstruction RMSE ~0.005-0.02 for SPX
    if rmse < 0.01:
        print(f"    ✓ Excellent — within best paper benchmarks")
    elif rmse < 0.02:
        print(f"    ✓ Good — within paper benchmark range")
    elif rmse < 0.05:
        print(f"    ⚠ Acceptable but above paper benchmark")
    else:
        print(f"    ✗ High — significantly above paper benchmark")

    # Latent factor analysis
    print(f"\n  Latent factor analysis:")
    for k in range(latent_factors.shape[1]):
        z = latent_factors[:, k]
        ac1 = np.corrcoef(z[:-1], z[1:])[0, 1]
        print(f"    z_{k}: mean={z.mean():.3f}, std={z.std():.3f}, "
              f"range=[{z.min():.2f}, {z.max():.2f}], autocorr(1)={ac1:.3f}")

    # Cross-correlation matrix
    print(f"\n  Latent factor correlations:")
    corr = np.corrcoef(latent_factors.T)
    for i in range(corr.shape[0]):
        row = " ".join(f"{corr[i, j]:6.3f}" for j in range(corr.shape[1]))
        print(f"    z_{i}: [{row}]")

    # PLOTS
    # 2a. Training curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("VAE Total Loss"); axes[0].legend(); axes[0].set_yscale("log")
    axes[1].plot(history["recon_loss"])
    axes[1].set_title("Reconstruction Loss")
    axes[2].plot(history["kl_loss"])
    axes[2].set_title("KL Divergence")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, plots_dir / "check2_vae_training.png")

    # 2b. Reconstruction comparison
    n_dates = len(surface_dates)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    sample_idx = [0, n_dates // 2, n_dates - 1]
    for col, idx in enumerate(sample_idx):
        for row, (data, label) in enumerate([(iv_matrix, "Original"), (iv_recon, "Reconstructed")]):
            s = data[idx].reshape(N_MONEYNESS, N_MATURITY)
            im = axes[row, col].imshow(s.T, aspect="auto", origin="lower", cmap="viridis", vmin=0.05, vmax=0.6)
            axes[row, col].set_title(f"{label} — {surface_dates[idx].date()}", fontsize=10)
    fig.colorbar(im, ax=axes, label="IV", shrink=0.8)
    fig.suptitle("VAE Reconstruction Comparison", fontsize=14)
    plt.tight_layout()
    save_fig(fig, plots_dir / "check2_vae_reconstruction.png")

    # 2c. Per-date RMSE
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot([d.date() for d in surface_dates], per_date_rmse, alpha=0.7)
    ax.axhline(rmse, color="r", ls="--", label=f"Mean={rmse:.4f}")
    ax.set_title("VAE Per-Date Reconstruction RMSE")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, plots_dir / "check2_per_date_rmse.png")

    # 2d. Latent factors over time
    n_factors = latent_factors.shape[1]
    fig, axes = plt.subplots(n_factors, 1, figsize=(14, 3 * n_factors), sharex=True)
    for k in range(n_factors):
        axes[k].plot([d.date() for d in surface_dates], latent_factors[:, k], alpha=0.7)
        axes[k].set_ylabel(f"z_{k}")
        axes[k].grid(True, alpha=0.3)
    axes[0].set_title("Latent Factors Over Time")
    plt.tight_layout()
    save_fig(fig, plots_dir / "check2_latent_factors.png")

    # 2e. Reconstruction error heatmap (average across dates)
    avg_error = np.mean(np.abs(error), axis=0).reshape(N_MONEYNESS, N_MATURITY)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(avg_error.T, aspect="auto", origin="lower", cmap="Reds")
    ax.set_title("Average |Reconstruction Error| by Grid Point")
    ax.set_xlabel("K/F")
    ax.set_ylabel("Maturity (days)")
    ax.set_xticks(range(N_MONEYNESS))
    ax.set_xticklabels([f"{s:.2f}" for s in MONEYNESS_STRIKES], rotation=45, fontsize=7)
    ax.set_yticks(range(N_MATURITY))
    ax.set_yticklabels(MATURITY_DAYS_GRID, fontsize=7)
    fig.colorbar(im, label="Mean |Error|")
    plt.tight_layout()
    save_fig(fig, plots_dir / "check2_error_heatmap.png")

    return latent_factors, scaler_iv, vae, rmse, explained_var


# ============================================================
# 3. BNN CHECKS
# ============================================================
def check_bnn(iv_matrix, surface_dates, latent_factors, out_dir, plots_dir):
    print("\n" + "=" * 60)
    print("[CHECK 3] BNN Prediction Quality (Stage 6)")
    print("=" * 60)

    bnn_data = load_pkl(out_dir / "stage6/bnn_results.pkl")
    history = bnn_data["history"]
    predictions = bnn_data["predictions"]
    metrics = bnn_data["metrics"]
    test_dates = bnn_data["test_dates"]
    train_dates = bnn_data["train_dates"]
    val_dates = bnn_data["val_dates"]
    y_test = bnn_data["y_test"]

    print(f"  Train:  {len(train_dates)} dates ({train_dates[0].date()} to {train_dates[-1].date()})")
    print(f"  Val:    {len(val_dates)} dates ({val_dates[0].date()} to {val_dates[-1].date()})")
    print(f"  Test:   {len(test_dates)} dates ({test_dates[0].date()} to {test_dates[-1].date()})")
    print(f"  Training epochs: {len(history['train_loss'])}")

    if metrics:
        print(f"\n  Latent space metrics:")
        print(f"    RMSE:            {metrics.get('latent_rmse', 'N/A'):.6f}")
        for ci in [50, 90, 95]:
            cov = metrics.get(f'latent_coverage_{ci}', 0)
            ideal = ci / 100
            gap = cov - ideal
            status = "✓" if abs(gap) < 0.10 else "⚠"
            direction = "over" if gap > 0 else "under"
            print(f"    {ci}% CI coverage:  {cov:.3f} (ideal {ideal:.2f}, {direction} by {abs(gap):.3f}) {status}")

        if "iv_rmse" in metrics:
            print(f"\n  IV space metrics:")
            print(f"    IV RMSE: {metrics['iv_rmse']:.6f}")
            # Zhang et al. report next-day IV RMSE around 0.01-0.03
            iv_rmse = metrics["iv_rmse"]
            if iv_rmse < 0.02:
                print(f"    ✓ Excellent — within paper benchmarks")
            elif iv_rmse < 0.04:
                print(f"    ⚠ Acceptable")
            else:
                print(f"    ✗ High — needs improvement")

    if predictions is None or y_test is None:
        print("  No predictions available — skipping detailed analysis")
        return

    latent_mean = predictions["latent_mean"]
    latent_std = predictions["latent_std"]
    iv_mean = predictions["iv_mean"]
    iv_std = predictions["iv_std"]

    # Per-factor quality
    print(f"\n  Per-factor prediction:")
    print(f"    {'Factor':>8s} {'RMSE':>8s} {'Corr':>8s} {'Mean Std':>10s} {'R²':>8s}")
    for k in range(y_test.shape[1]):
        rmse_k = np.sqrt(np.mean((latent_mean[:, k] - y_test[:, k]) ** 2))
        corr_k = np.corrcoef(latent_mean[:, k], y_test[:, k])[0, 1]
        std_k = latent_std[:, k].mean()
        ss_res = np.sum((y_test[:, k] - latent_mean[:, k]) ** 2)
        ss_tot = np.sum((y_test[:, k] - y_test[:, k].mean()) ** 2)
        r2_k = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"    z_{k}:    {rmse_k:8.4f} {corr_k:8.4f} {std_k:10.4f} {r2_k:8.4f}")

    # BASELINE: Persistence (random walk z_{t+1} = z_t)
    print(f"\n  Baseline comparison:")
    # Find z_t for test dates
    date_to_idx = {d: i for i, d in enumerate(surface_dates)}
    z_t_test = []
    for d in test_dates:
        idx = date_to_idx.get(d)
        if idx is not None:
            z_t_test.append(latent_factors[idx])
    if len(z_t_test) == len(test_dates):
        z_t_test = np.array(z_t_test)
        persistence_rmse = np.sqrt(np.mean((y_test - z_t_test) ** 2))
        bnn_rmse = metrics.get("latent_rmse", float("inf"))
        improvement = (persistence_rmse - bnn_rmse) / persistence_rmse * 100

        print(f"    Persistence RMSE: {persistence_rmse:.6f}")
        print(f"    BNN RMSE:         {bnn_rmse:.6f}")
        if improvement > 0:
            print(f"    ✓ BNN beats persistence by {improvement:.1f}%")
        else:
            print(f"    ⚠ BNN worse than persistence by {-improvement:.1f}%")
            print(f"      Latent factors may be near-random-walk")

        # Per-factor persistence comparison
        print(f"\n    Per-factor persistence comparison:")
        print(f"    {'Factor':>8s} {'BNN RMSE':>10s} {'Persist RMSE':>14s} {'Improvement':>12s}")
        for k in range(y_test.shape[1]):
            bnn_k = np.sqrt(np.mean((latent_mean[:, k] - y_test[:, k]) ** 2))
            pers_k = np.sqrt(np.mean((z_t_test[:, k] - y_test[:, k]) ** 2))
            imp_k = (pers_k - bnn_k) / pers_k * 100 if pers_k > 0 else 0
            marker = "✓" if imp_k > 0 else "✗"
            print(f"    z_{k}:    {bnn_k:10.4f} {pers_k:14.4f} {imp_k:11.1f}% {marker}")
    else:
        print("    Could not align test dates for persistence baseline")

    # BACKTESTING: directional accuracy
    print(f"\n  Directional accuracy (does BNN predict direction of change?):")
    if len(z_t_test) == len(test_dates):
        actual_change = y_test - z_t_test
        predicted_change = latent_mean - z_t_test
        correct_direction = np.sign(actual_change) == np.sign(predicted_change)
        overall_acc = np.mean(correct_direction)
        print(f"    Overall: {overall_acc:.3f} ({100 * overall_acc:.1f}%)")
        for k in range(y_test.shape[1]):
            acc_k = np.mean(correct_direction[:, k])
            print(f"    z_{k}: {acc_k:.3f} ({100 * acc_k:.1f}%)")
        if overall_acc > 0.55:
            print(f"    ✓ Better than random (50%)")
        elif overall_acc > 0.48:
            print(f"    ~ Near random — difficult to predict direction")
        else:
            print(f"    ⚠ Worse than random — BNN may be learning noise")

    # PLOTS
    # 3a. Training curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(history["train_loss"], label="Train ELBO")
    axes[0].set_title("BNN Train ELBO"); axes[0].legend()
    axes[1].plot(history["val_loss"])
    axes[1].set_title("BNN Val MSE")
    axes[2].plot(history["kl_loss"])
    axes[2].set_title("BNN KL")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, plots_dir / "check3_bnn_training.png")

    # 3b. Predicted vs actual latent factors
    n_factors = y_test.shape[1]
    fig, axes = plt.subplots(n_factors, 1, figsize=(14, 3 * n_factors), sharex=True)
    td = [d.date() for d in test_dates]
    for k in range(n_factors):
        ax = axes[k]
        ax.plot(td, y_test[:, k], "b-", label="Actual", alpha=0.8, linewidth=1.5)
        ax.plot(td, latent_mean[:, k], "r--", label="Predicted", alpha=0.8, linewidth=1.5)
        ax.fill_between(td,
                        latent_mean[:, k] - 2 * latent_std[:, k],
                        latent_mean[:, k] + 2 * latent_std[:, k],
                        alpha=0.15, color="red", label="95% CI")
        ax.set_ylabel(f"z_{k}")
        if k == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("BNN: Predicted vs Actual Latent Factors")
    plt.tight_layout()
    save_fig(fig, plots_dir / "check3_bnn_predictions.png")

    # 3c. Calibration plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ci_levels = np.linspace(0.05, 0.99, 50)
    empirical = []
    for ci in ci_levels:
        z_score = norm_dist.ppf(0.5 + ci / 2)
        lo = latent_mean - z_score * latent_std
        hi = latent_mean + z_score * latent_std
        empirical.append(np.mean((y_test >= lo) & (y_test <= hi)))
    ax.plot(ci_levels, empirical, "b-", linewidth=2, label="BNN")
    ax.plot([0, 1], [0, 1], "r--", label="Perfect")
    ax.set_xlabel("Nominal CI Level"); ax.set_ylabel("Empirical Coverage")
    ax.set_title("BNN Calibration Plot")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    plt.tight_layout()
    save_fig(fig, plots_dir / "check3_calibration.png")

    # 3d. Scatter: predicted vs actual per factor
    fig, axes = plt.subplots(1, n_factors, figsize=(4 * n_factors, 4))
    for k in range(n_factors):
        ax = axes[k]
        ax.scatter(y_test[:, k], latent_mean[:, k], alpha=0.3, s=10)
        lims = [min(y_test[:, k].min(), latent_mean[:, k].min()),
                max(y_test[:, k].max(), latent_mean[:, k].max())]
        ax.plot(lims, lims, "r--", alpha=0.5)
        corr_k = np.corrcoef(y_test[:, k], latent_mean[:, k])[0, 1]
        ax.set_title(f"z_{k} (r={corr_k:.3f})")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_aspect("equal")
    fig.suptitle("BNN: Predicted vs Actual Scatter", fontsize=14)
    plt.tight_layout()
    save_fig(fig, plots_dir / "check3_scatter.png")

    # 3e. IV prediction error
    if "iv_rmse" in metrics:
        n_test = len(test_dates)
        iv_actual = iv_matrix[-n_test:]
        iv_errors = iv_mean - iv_actual
        per_date_iv_rmse = np.sqrt(np.mean(iv_errors ** 2, axis=1))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(iv_errors.ravel(), bins=100, alpha=0.7, density=True, edgecolor="none")
        axes[0].axvline(0, color="r", ls="--")
        axes[0].set_title("IV Prediction Error Distribution")
        axes[0].set_xlabel("Error")

        axes[1].plot(td, per_date_iv_rmse, alpha=0.7)
        axes[1].axhline(metrics["iv_rmse"], color="r", ls="--", label=f"Mean={metrics['iv_rmse']:.4f}")
        axes[1].set_title("Per-Date IV RMSE")
        axes[1].set_ylabel("RMSE"); axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, plots_dir / "check3_iv_errors.png")

    return metrics


# ============================================================
# 4. IV INVERSION ROUND-TRIP
# ============================================================
def check_inversion():
    print("\n" + "=" * 60)
    print("[CHECK 4] IV Inversion Round-Trip")
    print("=" * 60)

    # ATM test
    test_ivs = np.array([0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00, 1.50])
    test_m = np.zeros(len(test_ivs))
    test_T = np.full(len(test_ivs), 0.25)

    prices = bs_normalized_call(test_m, test_T, test_ivs)
    recovered = implied_vol_newton(prices, test_m, test_T)

    print(f"  ATM round-trip (3-month):")
    print(f"  {'Input IV':>10s} {'BS Price':>10s} {'Recovered':>10s} {'Error':>10s}")
    for orig, p, rec in zip(test_ivs, prices, recovered):
        err = abs(orig - rec) if not np.isnan(rec) else float("inf")
        print(f"  {orig:10.4f} {p:10.6f} {rec:10.6f} {err:10.1e}")

    # OTM test
    print(f"\n  OTM round-trip (1-year):")
    test_m2 = np.array([-0.5, -0.2, -0.1, 0.1, 0.2, 0.5])
    test_iv2 = np.full(len(test_m2), 0.20)
    test_T2 = np.full(len(test_m2), 1.0)
    prices2 = bs_normalized_call(test_m2, test_T2, test_iv2)
    recovered2 = implied_vol_newton(prices2, test_m2, test_T2)

    max_err = 0
    print(f"  {'m':>6s} {'Price':>10s} {'Recovered':>10s} {'Error':>10s}")
    for m, p, rec in zip(test_m2, prices2, recovered2):
        err = abs(0.20 - rec) if not np.isnan(rec) else float("inf")
        max_err = max(max_err, err) if not np.isinf(err) else max_err
        print(f"  {m:6.2f} {p:10.6f} {rec:10.6f} {err:10.1e}")

    if max_err < 1e-6:
        print(f"  ✓ All inversions accurate to < 1e-6")
    else:
        print(f"  ⚠ Max round-trip error: {max_err:.1e}")


# ============================================================
# 5. SUMMARY
# ============================================================
def print_summary(rmse_vae, explained_var, metrics):
    print("\n" + "=" * 60)
    print("[SUMMARY] Pipeline Health Report")
    print("=" * 60)

    issues = []
    suggestions = []

    if rmse_vae > 0.05:
        issues.append(f"VAE RMSE={rmse_vae:.4f} is high")
        suggestions.append("Increase VAE latent_dim (8-10) or reduce beta (0.01)")
    elif rmse_vae > 0.02:
        suggestions.append(f"VAE RMSE={rmse_vae:.4f} — consider latent_dim=8")

    if explained_var < 0.90:
        issues.append(f"VAE explains only {100 * explained_var:.1f}% variance")

    if metrics:
        cov_50 = metrics.get("latent_coverage_50", 0.5)
        cov_90 = metrics.get("latent_coverage_90", 0.9)
        cov_95 = metrics.get("latent_coverage_95", 0.95)
        bnn_rmse = metrics.get("latent_rmse", 0)

        if cov_50 < 0.35:
            issues.append(f"50% CI coverage={cov_50:.3f} too low (over-confident)")
            suggestions.append("Increase BNN prior_sigma (e.g., 2.0 or 5.0)")
        if cov_95 < 0.80:
            issues.append(f"95% CI coverage={cov_95:.3f} too low")
            suggestions.append("Increase prior_sigma or reduce BNN hidden_dim")
        if cov_50 > 0.65:
            suggestions.append("50% coverage is high (under-confident); try lower prior_sigma (0.5)")
        if bnn_rmse > 1.5:
            suggestions.append("BNN RMSE is high — try more training epochs or wider network")

    print(f"\n  Issues: {len(issues)}")
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. ⚠ {issue}")

    print(f"\n  Suggestions: {len(suggestions)}")
    for i, sug in enumerate(suggestions, 1):
        print(f"    {i}. {sug}")

    if not issues:
        print("\n  ✓ All checks passed — pipeline is healthy!")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    out = Path(args.output_dir)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)

    print("=" * 70)
    print("  COMPREHENSIVE SANITY CHECK (post-fix)")
    print("=" * 70)

    # Load shared data
    iv_data = load_pkl(out / "stage4/iv_surfaces.pkl")
    iv_matrix = iv_data["iv_matrix"]
    surface_dates = iv_data["dates"]

    # Run all checks
    check_iv_surfaces(iv_matrix, surface_dates, plots)
    latent_factors, scaler_iv, vae, rmse_vae, explained_var = check_vae(
        iv_matrix, surface_dates, out, plots
    )
    metrics = check_bnn(iv_matrix, surface_dates, latent_factors, out, plots)
    check_inversion()
    print_summary(rmse_vae, explained_var, metrics)

    print(f"\n  All plots saved to: {plots}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
