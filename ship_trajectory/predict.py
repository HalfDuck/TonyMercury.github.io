"""Inference helper to evaluate and visualise trajectory forecasts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in (None, ""):
    # Allow running as a standalone script (e.g. `python predict.py`).
    import sys

    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    from ship_trajectory.config import ExperimentConfig
    from ship_trajectory.data import (
        METERS_PER_DEGREE_LAT,
        TrajectoryDataModule,
        meters_per_degree_lon,
    )
    from ship_trajectory.landmask import LandMask
    from ship_trajectory.model import TrajectoryTransformer
    from ship_trajectory.train import haversine_distance, load_config, set_seed
else:
    from .config import ExperimentConfig
    from .data import METERS_PER_DEGREE_LAT, TrajectoryDataModule, meters_per_degree_lon
    from .landmask import LandMask
    from .model import TrajectoryTransformer
    from .train import haversine_distance, load_config, set_seed

NAUTICAL_MILE_METERS = 1852.0


@dataclass
class PredictionResult:
    """Container with prediction arrays and evaluation metrics."""

    history_lon: np.ndarray
    history_lat: np.ndarray
    target_lon: np.ndarray
    target_lat: np.ndarray
    pred_lon: np.ndarray
    pred_lat: np.ndarray
    step_errors: np.ndarray
    mean_error_m: float
    final_error_m: float
    within_one_nm: bool


def _prepare_land_mask(config: ExperimentConfig) -> Optional[LandMask]:
    if config.data.geojson_path:
        return LandMask(Path(config.data.geojson_path))
    return None


def _load_model(
    experiment: ExperimentConfig,
    datamodule: TrajectoryDataModule,
    checkpoint: Path,
    device: torch.device,
) -> TrajectoryTransformer:
    model = TrajectoryTransformer(experiment.model, datamodule.encoder.embedding_dims)
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("model_state", payload)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _reconstruct_history(
    datamodule: TrajectoryDataModule,
    split: str,
    index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = getattr(datamodule, f"{split}_dataset")
    if dataset is None:
        raise ValueError(f"Dataset for split '{split}' has not been initialised")
    if index < 0 or index >= len(dataset.samples):
        raise IndexError(
            f"Sample index {index} out of range for split '{split}' with {len(dataset.samples)} samples"
        )

    sample = dataset.samples[index]
    history = sample.history.clone()
    if dataset.normalization is not None:
        history = dataset.normalization.inverse(history)

    history_positions = history[:, :2].numpy()
    lon_factor = meters_per_degree_lon(sample.anchor_lat)
    history_lon = sample.anchor_lon + history_positions[:, 0] / lon_factor
    history_lat = sample.anchor_lat + history_positions[:, 1] / METERS_PER_DEGREE_LAT
    return history_lon, history_lat


def predict_sequence(
    config_path: Path,
    checkpoint_path: Path,
    split: str = "test",
    sample_index: int = 0,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> PredictionResult:
    """Run inference on a single sample and optionally visualise the trajectory."""

    experiment = load_config(config_path)
    set_seed(experiment.data.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    land_mask = _prepare_land_mask(experiment)

    datamodule = TrajectoryDataModule(experiment.data, land_mask)
    datamodule.setup()

    dataset = getattr(datamodule, f"{split}_dataset")
    if dataset is None:
        raise ValueError(f"Split '{split}' is not available. Choose from train/val/test.")
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f"Sample index {sample_index} out of range for split '{split}' with {len(dataset)} samples"
        )

    model = _load_model(experiment, datamodule, checkpoint_path, device)

    sample = dataset[sample_index]
    history = sample["history"].unsqueeze(0).to(device)
    static = sample["static"].unsqueeze(0).to(device)
    anchor_lon = sample["anchor_lon"].unsqueeze(0).to(device)
    anchor_lat = sample["anchor_lat"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model.autoregressive_predict(
            history,
            static,
            anchor_lat=anchor_lat,
            anchor_lon=anchor_lon,
        )
    pred_lon = outputs["pred_lon"].squeeze(0).cpu().numpy()
    pred_lat = outputs["pred_lat"].squeeze(0).cpu().numpy()

    target = sample["future_absolute"].cpu().numpy()
    target_lon = target[:, 0]
    target_lat = target[:, 1]

    error_tensor = haversine_distance(
        torch.from_numpy(pred_lon),
        torch.from_numpy(pred_lat),
        torch.from_numpy(target_lon),
        torch.from_numpy(target_lat),
    )
    step_errors = error_tensor.cpu().numpy()
    mean_error = float(step_errors.mean())
    final_error = float(step_errors[-1])
    within_one_nm = bool(final_error < NAUTICAL_MILE_METERS)

    history_lon, history_lat = _reconstruct_history(datamodule, split, sample_index)

    result = PredictionResult(
        history_lon=history_lon,
        history_lat=history_lat,
        target_lon=target_lon,
        target_lat=target_lat,
        pred_lon=pred_lon,
        pred_lat=pred_lat,
        step_errors=step_errors,
        mean_error_m=mean_error,
        final_error_m=final_error,
        within_one_nm=within_one_nm,
    )

    if output_path is not None or show:
        _visualise_prediction(result, output_path, show)

    return result


def _visualise_prediction(result: PredictionResult, output_path: Optional[Path], show: bool) -> None:
    history_future_lon = np.concatenate(([result.history_lon[-1]], result.target_lon))
    history_future_lat = np.concatenate(([result.history_lat[-1]], result.target_lat))
    pred_future_lon = np.concatenate(([result.history_lon[-1]], result.pred_lon))
    pred_future_lat = np.concatenate(([result.history_lat[-1]], result.pred_lat))

    plt.figure(figsize=(8, 6))
    plt.plot(result.history_lon, result.history_lat, "-o", label="History", color="#1f77b4")
    plt.plot(history_future_lon, history_future_lat, "-o", label="Ground truth", color="#2ca02c")
    plt.plot(pred_future_lon, pred_future_lat, "-o", label="Prediction", color="#d62728")
    plt.scatter(result.pred_lon[-1], result.pred_lat[-1], color="#d62728", marker="x", s=80)
    plt.scatter(result.target_lon[-1], result.target_lat[-1], color="#2ca02c", marker="s", s=60)

    title = (
        f"Mean error: {result.mean_error_m:.1f} m | Final: {result.final_error_m:.1f} m | "
        f"Within 1 NM: {'yes' if result.within_one_nm else 'no'}"
    )
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a trained trajectory transformer")
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML configuration")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--index", type=int, default=0, help="Sample index within the chosen split")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the plot image")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
    args = parser.parse_args()

    result = predict_sequence(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        sample_index=args.index,
        output_path=args.output,
        show=args.show,
    )

    print(
        f"Per-step mean error: {result.mean_error_m:.2f} m | Final error: {result.final_error_m:.2f} m | "
        f"Within 1 NM: {'yes' if result.within_one_nm else 'no'}"
    )


if __name__ == "__main__":
    main()
