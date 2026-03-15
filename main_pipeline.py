"""
=============================================================================
PIPELINE — Main Runner
=============================================================================
Unified entry point that runs the full pipeline with stage-based execution.

Stages:
  1. Load raw data & preprocess
  2. Moussa arbitrage filter
  3. ISNN-2 surface fitting
  4. Surface sampling & IV inversion
  5. VAE compression to latent factors
  6. BNN prediction with uncertainty

Each stage saves its outputs to the outputs/ directory. You can run
individual stages by loading outputs from previous stages:

  python main_pipeline.py                       # Run all stages
  python main_pipeline.py --stage 6             # Run only stage 6
  python main_pipeline.py --stage 4 5 6         # Run stages 4, 5, 6

Place your CSV files (op_df, fwd_df, discount_df, St_df, list_exp, list_mny)
in the data directory.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Local imports (same directory)
from pipeline_part1_preprocessing import load_all_data, preprocess
from pipeline_part2_moussa_filter import moussa_filter_surface, run_diagnostics
from pipeline_part3_isnn import (
    fit_surface_for_date,
    validate_model_constraints,
    fit_all_surfaces,
    ISNN2_OptionSurface,
    OptionSurfaceData,
)
from pipeline_part4_surface_sampling import sample_all_surfaces
from pipeline_part5_vae import (
    train_vae,
    extract_latent_factors,
    decode_latent_factors,
    IVSurfaceVAE,
)
from pipeline_part6_bnn import (
    prepare_bnn_data,
    train_bnn,
    predict_with_uncertainty,
    evaluate_predictions,
    BayesianNN,
)


# ============================================================
# SAVE / LOAD HELPERS
# ============================================================
def _save_pkl(obj, path):
    """Save object to pickle file."""
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved: {path} ({path.stat().st_size / 1024:.0f} KB)")


def _load_pkl(path):
    """Load object from pickle file."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"  Loaded: {path}")
    return obj


def _ensure_outputs_dir(output_dir):
    """Create stage subdirectories in the outputs folder."""
    dirs = {}
    for stage in range(1, 7):
        d = output_dir / f"stage{stage}"
        d.mkdir(parents=True, exist_ok=True)
        dirs[stage] = d
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    dirs["plots"] = plots_dir
    return dirs


# ============================================================
# STAGE EXECUTION
# ============================================================
def run_stage1(args, dirs):
    """Stage 1: Load & Preprocess."""
    print("\n" + "=" * 70)
    print("[STAGE 1] Loading and preprocessing data...")
    print("=" * 70)

    data = load_all_data(args.data_dir)
    surface_dict = preprocess(data, min_volume=args.min_volume)

    # Save outputs
    _save_pkl(surface_dict, dirs[1] / "surface_dict.pkl")

    # Save raw data references needed later (St_df, fwd_df, discount_df)
    _save_pkl({
        "St_df": data["St_df"],
        "fwd_df": data["fwd_df"],
        "discount_df": data["discount_df"],
    }, dirs[1] / "market_data.pkl")

    return surface_dict, data


def run_stage2(args, dirs, surface_dict=None):
    """Stage 2: Moussa Arbitrage Filter."""
    print("\n" + "=" * 70)
    print(f"[STAGE 2] Running Moussa filter (λ={args.lambda_param})...")
    print("=" * 70)

    # Load inputs if not provided
    if surface_dict is None:
        surface_dict = _load_pkl(dirs[1] / "surface_dict.pkl")

    filtered_dict, filter_stats = moussa_filter_surface(
        surface_dict, lambda_param=args.lambda_param
    )
    run_diagnostics(filtered_dict, sample_dates=10)

    # Save outputs
    _save_pkl(filtered_dict, dirs[2] / "filtered_dict.pkl")
    _save_pkl(filter_stats, dirs[2] / "filter_stats.pkl")

    return filtered_dict, filter_stats


def run_stage3(args, dirs, filtered_dict=None):
    """Stage 3: ISNN-2 Surface Fitting."""
    print("\n" + "=" * 70)
    print("[STAGE 3] Fitting ISNN-2 surfaces...")
    print("=" * 70)

    if filtered_dict is None:
        filtered_dict = _load_pkl(dirs[2] / "filtered_dict.pkl")

    models_dict, val_results = fit_all_surfaces(
        filtered_dict,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_dates=args.max_dates,
        validate=True,
    )

    # Save each model individually (they can be large)
    models_dir = dirs[3] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_index = {}
    for d, (model, proc, losses) in models_dict.items():
        date_str = d.strftime("%Y%m%d")
        model_path = models_dir / f"isnn_{date_str}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "losses": losses,
            "n_contracts": len(filtered_dict[d]),
        }, model_path)
        model_index[d] = {
            "path": str(model_path),
            "final_loss": losses[-1] if losses else None,
            "n_contracts": len(filtered_dict[d]),
            "validation": val_results.get(d, {}),
        }

    _save_pkl(model_index, dirs[3] / "model_index.pkl")
    _save_pkl(val_results, dirs[3] / "validation_results.pkl")

    # Save hyperparams for reloading
    _save_pkl({
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
    }, dirs[3] / "hyperparams.pkl")

    return models_dict, val_results


def _load_models_dict(dirs):
    """Reload trained ISNN models from disk."""
    model_index = _load_pkl(dirs[3] / "model_index.pkl")
    hyperparams = _load_pkl(dirs[3] / "hyperparams.pkl")

    models_dict = {}
    for d, info in model_index.items():
        checkpoint = torch.load(info["path"], map_location="cpu", weights_only=False)
        model = ISNN2_OptionSurface(
            hidden_dim=checkpoint.get("hidden_dim", hyperparams["hidden_dim"]),
            num_layers=checkpoint.get("num_layers", hyperparams["num_layers"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        losses = checkpoint.get("losses", [])
        # Processor is not saved (it's data-dependent), pass None
        models_dict[d] = (model, None, losses)

    print(f"  Loaded {len(models_dict)} ISNN models")
    return models_dict


def run_stage4(args, dirs, models_dict=None):
    """Stage 4: Surface Sampling & IV Inversion."""
    print("\n" + "=" * 70)
    print("[STAGE 4] Sampling IV surfaces from ISNN models...")
    print("=" * 70)

    if models_dict is None:
        models_dict = _load_models_dict(dirs)

    iv_matrix, surface_dates = sample_all_surfaces(models_dict)

    # Save outputs
    _save_pkl({
        "iv_matrix": iv_matrix,
        "dates": surface_dates,
    }, dirs[4] / "iv_surfaces.pkl")

    return iv_matrix, surface_dates


def run_stage5(args, dirs, iv_matrix=None, surface_dates=None):
    """Stage 5: VAE Compression."""
    print("\n" + "=" * 70)
    print(f"[STAGE 5] Training VAE (latent_dim={args.latent_dim})...")
    print("=" * 70)

    if iv_matrix is None:
        loaded = _load_pkl(dirs[4] / "iv_surfaces.pkl")
        iv_matrix = loaded["iv_matrix"]
        surface_dates = loaded["dates"]

    vae, scaler_iv, latent_factors, vae_history = train_vae(
        iv_matrix,
        latent_dim=args.latent_dim,
        hidden_dim=args.vae_hidden_dim,
        beta=args.vae_beta,
        epochs=args.vae_epochs,
        batch_size=args.vae_batch_size,
        lr=args.vae_lr,
    )

    # Save outputs
    torch.save(vae.state_dict(), dirs[5] / "vae_model.pt")
    _save_pkl({
        "latent_factors": latent_factors,
        "dates": surface_dates,
        "scaler": scaler_iv,
        "history": vae_history,
    }, dirs[5] / "vae_results.pkl")
    _save_pkl({
        "input_dim": iv_matrix.shape[1],
        "hidden_dim": args.vae_hidden_dim,
        "latent_dim": args.latent_dim,
    }, dirs[5] / "vae_hyperparams.pkl")

    return vae, scaler_iv, latent_factors, surface_dates, vae_history


def _load_vae(dirs):
    """Reload trained VAE from disk."""
    hp = _load_pkl(dirs[5] / "vae_hyperparams.pkl")
    vae = IVSurfaceVAE(
        input_dim=hp["input_dim"],
        hidden_dim=hp["hidden_dim"],
        latent_dim=hp["latent_dim"],
    )
    vae.load_state_dict(torch.load(dirs[5] / "vae_model.pt", map_location="cpu", weights_only=True))
    vae.eval()
    print(f"  Loaded VAE (latent_dim={hp['latent_dim']})")
    return vae


def run_stage6(args, dirs, latent_factors=None, surface_dates=None,
               vae=None, scaler_iv=None, iv_matrix=None, data=None):
    """Stage 6: BNN Prediction."""
    print("\n" + "=" * 70)
    print("[STAGE 6] Training BNN for latent factor prediction...")
    print("=" * 70)

    # Load dependencies as needed
    if latent_factors is None or surface_dates is None or scaler_iv is None:
        vae_data = _load_pkl(dirs[5] / "vae_results.pkl")
        latent_factors = vae_data["latent_factors"]
        surface_dates = vae_data["dates"]
        scaler_iv = vae_data["scaler"]

    if vae is None:
        vae = _load_vae(dirs)

    if data is None:
        data = _load_pkl(dirs[1] / "market_data.pkl")

    if iv_matrix is None:
        iv_data = _load_pkl(dirs[4] / "iv_surfaces.pkl")
        iv_matrix = iv_data["iv_matrix"]

    # Prepare data
    X, y, scaler_features, dates_X = prepare_bnn_data(
        latent_factors, data["St_df"], data["fwd_df"], surface_dates
    )

    # Temporal split
    test_start = pd.Timestamp(f"{args.test_year}-01-01")
    val_start = test_start - pd.DateOffset(years=1)

    train_mask = np.array([d < val_start for d in dates_X])
    val_mask = np.array([(d >= val_start) & (d < test_start) for d in dates_X])
    test_mask = np.array([d >= test_start for d in dates_X])

    print(f"  Split: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        print("  ERROR: Insufficient data for train/val split.")
        print(f"  Date range: {dates_X[0].date()} to {dates_X[-1].date()}")
        print(f"  Val start: {val_start.date()}, Test start: {test_start.date()}")
        print(f"  Hint: Try --test_year with a year within your date range.")
        return None, None, None

    input_dim = X.shape[1]
    output_dim = y.shape[1]

    bnn, bnn_history = train_bnn(
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        input_dim=input_dim,
        hidden_dim=args.bnn_hidden_dim,
        output_dim=output_dim,
        num_layers=args.bnn_num_layers,
        prior_sigma=args.bnn_prior_sigma,
        epochs=args.bnn_epochs,
        batch_size=args.bnn_batch_size,
        lr=args.bnn_lr,
    )

    # Predict on test set
    predictions = None
    metrics = None
    if test_mask.sum() > 0:
        predictions = predict_with_uncertainty(
            bnn, vae.decoder, scaler_features, scaler_iv,
            X[test_mask],
            n_mc_samples=args.bnn_mc_samples,
        )

        # Get actual IV surfaces for test dates
        # y[i] = latent_factors[i+1], so test IV is at indices test_mask_indices + 1
        test_indices = np.where(test_mask)[0]
        iv_test_actual = None
        if iv_matrix is not None and (test_indices + 1).max() < len(iv_matrix):
            iv_test_actual = iv_matrix[test_indices + 1]

        metrics = evaluate_predictions(
            predictions, y[test_mask],
            iv_test_actual=iv_test_actual,
        )
    else:
        print("  No test data available — skipping prediction evaluation")

    # Save outputs
    torch.save(bnn.state_dict(), dirs[6] / "bnn_model.pt")
    _save_pkl({
        "history": bnn_history,
        "predictions": predictions,
        "metrics": metrics,
        "test_dates": [d for d, m in zip(dates_X, test_mask) if m],
        "train_dates": [d for d, m in zip(dates_X, train_mask) if m],
        "val_dates": [d for d, m in zip(dates_X, val_mask) if m],
        "scaler_features": scaler_features,
        "y_test": y[test_mask] if test_mask.sum() > 0 else None,
    }, dirs[6] / "bnn_results.pkl")
    _save_pkl({
        "input_dim": input_dim,
        "hidden_dim": args.bnn_hidden_dim,
        "output_dim": output_dim,
        "num_layers": args.bnn_num_layers,
        "prior_sigma": args.bnn_prior_sigma,
    }, dirs[6] / "bnn_hyperparams.pkl")

    return bnn, predictions, metrics


# ============================================================
# MAIN
# ============================================================
def main(args):
    print("=" * 70)
    print("  OPTION SURFACE PIPELINE")
    print("  Moussa Filter + ISNN-2 + VAE + BNN")
    print("=" * 70)

    dirs = _ensure_outputs_dir(args.output_dir)

    # Determine which stages to run
    if args.stage is None:
        stages_to_run = [1, 2, 3, 4, 5, 6]
    else:
        stages_to_run = sorted(set(args.stage))

    print(f"\n  Stages to run: {stages_to_run}")
    print(f"  Output dir: {args.output_dir}")

    # Track in-memory objects to pass between stages
    surface_dict = None
    data = None
    filtered_dict = None
    filter_stats = None
    models_dict = None
    iv_matrix = None
    surface_dates = None
    vae = None
    scaler_iv = None
    latent_factors = None
    vae_history = None

    # --- Stage 1 ---
    if 1 in stages_to_run:
        surface_dict, data = run_stage1(args, dirs)

    # --- Stage 2 ---
    if 2 in stages_to_run:
        filtered_dict, filter_stats = run_stage2(args, dirs, surface_dict)

    # --- Stage 3 ---
    if 3 in stages_to_run:
        models_dict, _ = run_stage3(args, dirs, filtered_dict)

    # --- Stage 4 ---
    if 4 in stages_to_run:
        iv_matrix, surface_dates = run_stage4(args, dirs, models_dict)

    # --- Stage 5 ---
    if 5 in stages_to_run:
        vae, scaler_iv, latent_factors, surface_dates, vae_history = run_stage5(
            args, dirs, iv_matrix, surface_dates
        )

    # --- Stage 6 ---
    if 6 in stages_to_run:
        run_stage6(args, dirs, latent_factors, surface_dates,
                   vae, scaler_iv, iv_matrix, data)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Option Surface Pipeline: Moussa + ISNN + VAE + BNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_pipeline.py                          # Run all stages
  python main_pipeline.py --stage 6                # Run only stage 6
  python main_pipeline.py --stage 4 5 6            # Run stages 4-6
  python main_pipeline.py --max_dates 20           # Quick test with 20 dates
  python main_pipeline.py --stage 6 --test_year 2015 --bnn_epochs 1000
        """,
    )

    # General
    parser.add_argument("--stage", type=int, nargs="+", default=None,
                        help="Stage(s) to run (1-6). Default: all stages.")
    parser.add_argument("--data_dir", type=str, default=".",
                        help="Directory containing raw CSV files")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory for all outputs")

    # Stage 1: Preprocessing
    parser.add_argument("--min_volume", type=int, default=10,
                        help="Minimum volume filter")

    # Stage 2: Moussa filter
    parser.add_argument("--lambda_param", type=float, default=0.01,
                        help="Lambda for Moussa adjustment (0 = minimal)")

    # Stage 3: ISNN
    parser.add_argument("--epochs", type=int, default=2000,
                        help="ISNN training epochs per date")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="ISNN learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="ISNN hidden layer width")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="ISNN network depth")
    parser.add_argument("--max_dates", type=int, default=None,
                        help="Limit number of dates (None = all)")

    # Stage 5: VAE
    parser.add_argument("--latent_dim", type=int, default=5,
                        help="VAE latent space dimension")
    parser.add_argument("--vae_hidden_dim", type=int, default=128,
                        help="VAE hidden layer width")
    parser.add_argument("--vae_epochs", type=int, default=1000,
                        help="VAE training epochs")
    parser.add_argument("--vae_lr", type=float, default=1e-3,
                        help="VAE learning rate")
    parser.add_argument("--vae_beta", type=float, default=0.1,
                        help="VAE KL weight (beta)")
    parser.add_argument("--vae_batch_size", type=int, default=64,
                        help="VAE minibatch size")

    # Stage 6: BNN
    parser.add_argument("--bnn_epochs", type=int, default=500,
                        help="BNN training epochs")
    parser.add_argument("--bnn_lr", type=float, default=1e-3,
                        help="BNN learning rate")
    parser.add_argument("--bnn_hidden_dim", type=int, default=64,
                        help="BNN hidden layer width")
    parser.add_argument("--bnn_num_layers", type=int, default=3,
                        help="BNN number of hidden layers")
    parser.add_argument("--bnn_prior_sigma", type=float, default=1.0,
                        help="BNN prior standard deviation")
    parser.add_argument("--bnn_batch_size", type=int, default=64,
                        help="BNN minibatch size")
    parser.add_argument("--bnn_mc_samples", type=int, default=200,
                        help="MC samples for BNN prediction uncertainty")
    parser.add_argument("--test_year", type=int, default=2019,
                        help="Start year for test set (val = test_year - 1)")

    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main(args)
