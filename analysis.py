"""
=============================================================================
ANALYSIS — Post-Run Plotting & Diagnostics
=============================================================================
Reads outputs from the pipeline's outputs/ folder and generates comprehensive
diagnostic plots in outputs/plots/.

Usage:
  python analysis.py                          # Analyze all available stages
  python analysis.py --output_dir ./outputs   # Custom output directory

Automatically detects which stages have been run and generates all
available plots.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

# Grid definition (must match pipeline_part4_surface_sampling.py)
MONEYNESS_STRIKES = np.array([
    0.6, 0.8, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0
])
MATURITY_DAYS = np.array([10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730])
N_MONEYNESS = len(MONEYNESS_STRIKES)
N_MATURITY = len(MATURITY_DAYS)


# ============================================================
# HELPERS
# ============================================================
def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _stage_exists(dirs, stage):
    """Check if a stage has outputs available."""
    checks = {
        1: "surface_dict.pkl",
        2: "filtered_dict.pkl",
        3: "model_index.pkl",
        4: "iv_surfaces.pkl",
        5: "vae_results.pkl",
        6: "bnn_results.pkl",
    }
    return (dirs[stage] / checks[stage]).exists()


def _save_fig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# STAGE 2: MOUSSA FILTER PLOTS
# ============================================================
def plot_stage2(dirs):
    """Moussa filter diagnostics."""
    print("\n[Stage 2] Moussa Filter Diagnostics")

    filtered_dict = _load_pkl(dirs[2] / "filtered_dict.pkl")
    dates = sorted(filtered_dict.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Stage 2: Moussa Arbitrage Filter", fontsize=14, fontweight="bold")

    # 1. Daily adjustment rate
    adj_rates = [filtered_dict[d]["is_changed"].mean() * 100 for d in dates]
    ax = axes[0, 0]
    ax.plot(dates, adj_rates, linewidth=0.5, alpha=0.7)
    ax.axhline(np.mean(adj_rates), color="red", ls="--", alpha=0.5,
               label=f"Mean: {np.mean(adj_rates):.1f}%")
    ax.set_title("Daily Adjustment Rate")
    ax.set_ylabel("% quotes adjusted")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Quote count
    counts = [len(filtered_dict[d]) for d in dates]
    ax = axes[0, 1]
    ax.plot(dates, counts, linewidth=0.5, alpha=0.7)
    ax.set_title("Daily Quote Count (post-filter)")
    ax.set_ylabel("# quotes")
    ax.grid(True, alpha=0.3)

    # 3. Adjustment magnitude distribution
    all_diffs = []
    for d in dates:
        df = filtered_dict[d]
        changed = df[df["is_changed"]]
        if len(changed) > 0:
            all_diffs.extend((changed["midP"] - changed["moussa_price"]).abs().values)
    ax = axes[1, 0]
    if all_diffs:
        ax.hist(all_diffs, bins=100, edgecolor="none", alpha=0.7, color="steelblue")
        ax.set_yscale("log")
        ax.set_title(f"Adjustment Magnitudes (n={len(all_diffs):,})")
        ax.set_xlabel("Absolute adjustment ($)")
    ax.grid(True, alpha=0.3)

    # 4. Example slice
    sample_date = dates[len(dates) // 2]
    df = filtered_dict[sample_date]
    exp_counts = df.groupby("exdate").size()
    best_exp = exp_counts.idxmax()
    sl = df[df["exdate"] == best_exp].sort_values("K_norm")
    ax = axes[1, 1]
    ax.scatter(sl["K_norm"], sl["C_norm"], s=15, alpha=0.6, label="Original")
    ch = sl[sl["is_changed"]]
    if len(ch) > 0:
        ax.scatter(ch["K_norm"], ch["moussa_C_norm"], s=20, color="red",
                   marker="x", label="Adjusted")
    ax.set_title(f"Example: {sample_date.date()} (T={sl['maturity_days'].iloc[0]}d)")
    ax.set_xlabel("K/F")
    ax.set_ylabel("C_norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, dirs["plots"] / "stage2_moussa_filter.png")


# ============================================================
# STAGE 3: ISNN PLOTS
# ============================================================
def plot_stage3(dirs):
    """ISNN fitting diagnostics."""
    print("\n[Stage 3] ISNN Surface Fitting Diagnostics")

    model_index = _load_pkl(dirs[3] / "model_index.pkl")
    dates = sorted(model_index.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Stage 3: ISNN-2 Surface Fitting", fontsize=14, fontweight="bold")

    # 1. Final loss per date
    final_losses = [model_index[d]["final_loss"] for d in dates]
    ax = axes[0, 0]
    ax.bar(range(len(dates)), final_losses, color="steelblue", alpha=0.7)
    ax.set_yscale("log")
    ax.set_title("Final Training Loss per Date")
    ax.set_xlabel("Date index")
    ax.set_ylabel("MSE (log)")
    ax.grid(True, alpha=0.3)

    # 2. Contracts per date
    n_contracts = [model_index[d]["n_contracts"] for d in dates]
    ax = axes[0, 1]
    ax.bar(range(len(dates)), n_contracts, color="coral", alpha=0.7)
    ax.set_title("Contracts per Date")
    ax.set_xlabel("Date index")
    ax.set_ylabel("# contracts")
    ax.grid(True, alpha=0.3)

    # 3. Validation results
    val_results = _load_pkl(dirs[3] / "validation_results.pkl")
    time_viol = [val_results.get(d, {}).get("time_monotonicity_violations", 0) for d in dates]
    conv_viol = [val_results.get(d, {}).get("strike_convexity_violations", 0) for d in dates]
    ax = axes[1, 0]
    x_pos = np.arange(len(dates))
    ax.bar(x_pos - 0.15, time_viol, 0.3, label="Time monotonicity", alpha=0.7)
    ax.bar(x_pos + 0.15, conv_viol, 0.3, label="Strike convexity", alpha=0.7)
    ax.set_title("Arbitrage Constraint Violations")
    ax.set_xlabel("Date index")
    ax.set_ylabel("# violations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Sample training loss curve (first model)
    first_date = dates[0]
    checkpoint = torch.load(model_index[first_date]["path"],
                            map_location="cpu", weights_only=False)
    losses = checkpoint.get("losses", [])
    ax = axes[1, 1]
    if losses:
        ax.plot(losses, color="steelblue")
        ax.set_yscale("log")
        ax.set_title(f"Training Curve: {first_date.date()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (log)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, dirs["plots"] / "stage3_isnn_fitting.png")


# ============================================================
# STAGE 4: IV SURFACE PLOTS
# ============================================================
def plot_stage4(dirs):
    """IV surface sampling diagnostics."""
    print("\n[Stage 4] IV Surface Sampling Diagnostics")

    iv_data = _load_pkl(dirs[4] / "iv_surfaces.pkl")
    iv_matrix = iv_data["iv_matrix"]
    dates = iv_data["dates"]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Stage 4: Sampled IV Surfaces", fontsize=14, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig)

    # 1. Mean IV surface as heatmap
    ax = fig.add_subplot(gs[0, 0])
    mean_surface = iv_matrix.mean(axis=0).reshape(N_MONEYNESS, N_MATURITY)
    im = ax.imshow(mean_surface, aspect="auto", cmap="viridis",
                   extent=[MATURITY_DAYS[0], MATURITY_DAYS[-1],
                           MONEYNESS_STRIKES[-1], MONEYNESS_STRIKES[0]])
    plt.colorbar(im, ax=ax, label="IV")
    ax.set_title("Mean IV Surface")
    ax.set_xlabel("Maturity (days)")
    ax.set_ylabel("K/F")

    # 2. IV std surface (volatility of volatility)
    ax = fig.add_subplot(gs[0, 1])
    std_surface = iv_matrix.std(axis=0).reshape(N_MONEYNESS, N_MATURITY)
    im = ax.imshow(std_surface, aspect="auto", cmap="hot",
                   extent=[MATURITY_DAYS[0], MATURITY_DAYS[-1],
                           MONEYNESS_STRIKES[-1], MONEYNESS_STRIKES[0]])
    plt.colorbar(im, ax=ax, label="IV Std")
    ax.set_title("IV Std (vol-of-vol)")
    ax.set_xlabel("Maturity (days)")
    ax.set_ylabel("K/F")

    # 3. ATM IV time series
    ax = fig.add_subplot(gs[0, 2])
    # ATM = moneyness index 5 (K/F=1.0), multiple maturities
    atm_idx = 5  # K/F = 1.0
    for j, mat_days in enumerate([30, 91, 182, 365]):
        mat_idx = list(MATURITY_DAYS).index(mat_days)
        grid_idx = atm_idx * N_MATURITY + mat_idx
        ax.plot(dates, iv_matrix[:, grid_idx], linewidth=0.8,
                label=f"{mat_days}d", alpha=0.8)
    ax.set_title("ATM Implied Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("IV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

    # 4. Smile at first date
    ax = fig.add_subplot(gs[1, 0])
    first_surface = iv_matrix[0].reshape(N_MONEYNESS, N_MATURITY)
    for j, mat_days in enumerate([30, 91, 182, 365, 730]):
        mat_idx = list(MATURITY_DAYS).index(mat_days)
        ax.plot(MONEYNESS_STRIKES, first_surface[:, mat_idx],
                marker="o", markersize=3, label=f"{mat_days}d")
    ax.set_title(f"IV Smile: {dates[0].date()}")
    ax.set_xlabel("K/F")
    ax.set_ylabel("IV")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. IV distribution
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(iv_matrix.ravel(), bins=100, edgecolor="none", alpha=0.7, color="steelblue")
    ax.set_title(f"IV Distribution (all points)")
    ax.set_xlabel("Implied Volatility")
    ax.set_ylabel("Frequency")
    ax.axvline(iv_matrix.mean(), color="red", ls="--", label=f"Mean: {iv_matrix.mean():.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. NaN check
    ax = fig.add_subplot(gs[1, 2])
    nan_per_date = np.isnan(iv_matrix).sum(axis=1)
    ax.bar(range(len(dates)), nan_per_date, color="coral", alpha=0.7)
    ax.set_title("NaN Count per Date")
    ax.set_xlabel("Date index")
    ax.set_ylabel("# NaN grid points")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, dirs["plots"] / "stage4_iv_surfaces.png")


# ============================================================
# STAGE 5: VAE PLOTS
# ============================================================
def plot_stage5(dirs):
    """VAE diagnostics."""
    print("\n[Stage 5] VAE Diagnostics")

    vae_data = _load_pkl(dirs[5] / "vae_results.pkl")
    latent = vae_data["latent_factors"]
    dates = vae_data["dates"]
    history = vae_data["history"]
    d = latent.shape[1]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Stage 5: Variational Autoencoder", fontsize=14, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig)

    # 1. Training loss
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(history["train_loss"], label="Train", alpha=0.8)
    ax.plot(history["val_loss"], label="Val", alpha=0.8)
    ax.set_yscale("log")
    ax.set_title("VAE Total Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Recon vs KL
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(history["recon_loss"], label="Recon MSE", alpha=0.8)
    ax.plot(history["kl_loss"], label="KL Divergence", alpha=0.8)
    ax.set_yscale("log")
    ax.set_title("Loss Components")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Latent factor time series
    ax = fig.add_subplot(gs[0, 2])
    for k in range(d):
        ax.plot(dates, latent[:, k], label=f"z_{k}", linewidth=0.8, alpha=0.8)
    ax.set_title("Latent Factors Over Time")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

    # 4. Latent factor distributions
    ax = fig.add_subplot(gs[1, 0])
    for k in range(d):
        ax.hist(latent[:, k], bins=30, alpha=0.5, label=f"z_{k}")
    ax.set_title("Latent Factor Distributions")
    ax.set_xlabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Latent factor correlation
    ax = fig.add_subplot(gs[1, 1])
    corr = np.corrcoef(latent.T)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(d))
    ax.set_yticks(range(d))
    ax.set_xticklabels([f"z_{k}" for k in range(d)])
    ax.set_yticklabels([f"z_{k}" for k in range(d)])
    ax.set_title("Latent Factor Correlation")

    # 6. Latent factor autocorrelation (lag-1)
    ax = fig.add_subplot(gs[1, 2])
    for k in range(d):
        ac = np.corrcoef(latent[:-1, k], latent[1:, k])[0, 1]
        ax.bar(k, ac, alpha=0.7, label=f"z_{k}: {ac:.3f}")
    ax.set_title("Lag-1 Autocorrelation")
    ax.set_xlabel("Latent Factor")
    ax.set_ylabel("Autocorrelation")
    ax.set_xticks(range(d))
    ax.set_xticklabels([f"z_{k}" for k in range(d)])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, dirs["plots"] / "stage5_vae.png")


# ============================================================
# STAGE 6: BNN PLOTS
# ============================================================
def plot_stage6(dirs):
    """BNN prediction diagnostics."""
    print("\n[Stage 6] BNN Prediction Diagnostics")

    bnn_data = _load_pkl(dirs[6] / "bnn_results.pkl")
    history = bnn_data["history"]
    predictions = bnn_data["predictions"]
    metrics = bnn_data["metrics"]
    test_dates = bnn_data["test_dates"]
    y_test = bnn_data["y_test"]

    if predictions is None or y_test is None:
        print("  No predictions available — skipping BNN plots")
        return

    latent_dim = y_test.shape[1]

    fig = plt.figure(figsize=(18, 16))
    fig.suptitle("Stage 6: Bayesian Neural Network Predictions", fontsize=14, fontweight="bold")
    gs = GridSpec(3, 3, figure=fig)

    # 1. Training loss
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(history["train_loss"], label="Train ELBO", alpha=0.8)
    ax.plot(history["val_loss"], label="Val MSE", alpha=0.8)
    ax.set_yscale("log")
    ax.set_title("BNN Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. MSE vs KL decomposition
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(history["mse_loss"], label="MSE", alpha=0.8)
    ax.plot([kl * (1.0 / len(history["mse_loss"])) for kl in history["kl_loss"]],
            label="Scaled KL", alpha=0.8)
    ax.set_yscale("log")
    ax.set_title("ELBO Components")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Predicted vs Actual scatter (all factors)
    ax = fig.add_subplot(gs[0, 2])
    z_pred = predictions["latent_mean"]
    for k in range(latent_dim):
        ax.scatter(y_test[:, k], z_pred[:, k], s=5, alpha=0.5, label=f"z_{k}")
    lims = [min(y_test.min(), z_pred.min()), max(y_test.max(), z_pred.max())]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_title("Predicted vs Actual (all factors)")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)

    # 4-6. Time series for first 3 latent factors
    z_std = predictions["latent_std"]
    t_idx = np.arange(len(test_dates))
    for k in range(min(3, latent_dim)):
        ax = fig.add_subplot(gs[1, k])
        ax.plot(t_idx, y_test[:, k], "k-", label="Actual", linewidth=1)
        ax.plot(t_idx, z_pred[:, k], "b-", label="Predicted", linewidth=1, alpha=0.8)
        ax.fill_between(t_idx,
                        z_pred[:, k] - 2 * z_std[:, k],
                        z_pred[:, k] + 2 * z_std[:, k],
                        alpha=0.2, color="blue", label="95% CI")
        ax.set_title(f"z_{k}: Prediction vs Actual")
        ax.set_xlabel("Test Day")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 7. IV prediction at ATM sample points
    if "iv_mean" in predictions and predictions["iv_mean"] is not None:
        iv_mean = predictions["iv_mean"]
        iv_std = predictions["iv_std"]
        atm_idx = 5  # K/F = 1.0

        for j_idx, (mat_days, mat_j) in enumerate([(91, 3), (182, 6), (365, 8)]):
            if j_idx >= 3:
                break
            ax = fig.add_subplot(gs[2, j_idx])
            grid_pt = atm_idx * N_MATURITY + mat_j
            iv_m = iv_mean[:, grid_pt]
            iv_s = iv_std[:, grid_pt]
            ax.plot(t_idx, iv_m, "b-", linewidth=1)
            ax.fill_between(t_idx, iv_m - 2 * iv_s, iv_m + 2 * iv_s,
                            alpha=0.2, color="blue", label="95% CI")
            ax.set_title(f"IV Prediction (ATM, {mat_days}d)")
            ax.set_xlabel("Test Day")
            ax.set_ylabel("IV")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, dirs["plots"] / "stage6_bnn_predictions.png")

    # --- Calibration plot ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ci_levels = np.linspace(0.05, 0.99, 50)
    from scipy.stats import norm as norm_dist
    empirical_coverages = []
    for ci in ci_levels:
        z_score = norm_dist.ppf(0.5 + ci / 2)
        lower = z_pred - z_score * z_std
        upper = z_pred + z_score * z_std
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        empirical_coverages.append(coverage)

    ax.plot(ci_levels, empirical_coverages, "b-", linewidth=2, label="BNN")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.set_title("Uncertainty Calibration")
    ax.set_xlabel("Nominal CI Level")
    ax.set_ylabel("Empirical Coverage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    _save_fig(fig, dirs["plots"] / "stage6_calibration.png")


# ============================================================
# SUMMARY REPORT
# ============================================================
def print_summary(dirs):
    """Print a summary of all available pipeline outputs."""
    print("\n" + "=" * 70)
    print("  PIPELINE OUTPUT SUMMARY")
    print("=" * 70)

    for stage in range(1, 7):
        exists = _stage_exists(dirs, stage)
        status = "AVAILABLE" if exists else "NOT FOUND"
        marker = "+" if exists else "-"
        stage_names = {
            1: "Preprocessing",
            2: "Moussa Filter",
            3: "ISNN-2 Fitting",
            4: "IV Surface Sampling",
            5: "VAE Compression",
            6: "BNN Prediction",
        }
        print(f"  [{marker}] Stage {stage}: {stage_names[stage]} — {status}")

        if exists:
            # Print stage-specific stats
            try:
                if stage == 3:
                    idx = _load_pkl(dirs[3] / "model_index.pkl")
                    print(f"      Models: {len(idx)}")
                elif stage == 4:
                    iv = _load_pkl(dirs[4] / "iv_surfaces.pkl")
                    print(f"      IV matrix: {iv['iv_matrix'].shape}")
                    print(f"      IV range: [{iv['iv_matrix'].min():.4f}, {iv['iv_matrix'].max():.4f}]")
                elif stage == 5:
                    vae = _load_pkl(dirs[5] / "vae_results.pkl")
                    print(f"      Latent factors: {vae['latent_factors'].shape}")
                    print(f"      Final recon loss: {vae['history']['recon_loss'][-1]:.6f}")
                elif stage == 6:
                    bnn = _load_pkl(dirs[6] / "bnn_results.pkl")
                    if bnn["metrics"]:
                        print(f"      Latent RMSE: {bnn['metrics'].get('latent_rmse', 'N/A'):.6f}")
                        print(f"      95% CI coverage: {bnn['metrics'].get('latent_coverage_95', 'N/A'):.3f}")
                    print(f"      Test dates: {len(bnn.get('test_dates', []))}")
            except Exception as e:
                print(f"      (Could not load details: {e})")

    print()


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Pipeline Output Analysis & Plotting")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Pipeline outputs directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"ERROR: Output directory '{output_dir}' not found.")
        print("Run the pipeline first, or specify --output_dir")
        sys.exit(1)

    dirs = {}
    for stage in range(1, 7):
        dirs[stage] = output_dir / f"stage{stage}"
    dirs["plots"] = output_dir / "plots"
    dirs["plots"].mkdir(parents=True, exist_ok=True)

    # Summary
    print_summary(dirs)

    # Generate plots for each available stage
    if _stage_exists(dirs, 2):
        plot_stage2(dirs)

    if _stage_exists(dirs, 3):
        plot_stage3(dirs)

    if _stage_exists(dirs, 4):
        plot_stage4(dirs)

    if _stage_exists(dirs, 5):
        plot_stage5(dirs)

    if _stage_exists(dirs, 6):
        plot_stage6(dirs)

    print(f"\n  All plots saved to: {dirs['plots']}")
    print("  Done.")


if __name__ == "__main__":
    main()
