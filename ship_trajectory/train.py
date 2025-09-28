"""Training script for the trajectory transformer model."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm.auto import tqdm

from .config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from .data import METERS_PER_DEGREE_LAT, TrajectoryDataModule
from .landmask import LandMask
from .model import TrajectoryTransformer

NAUTICAL_MILE_METERS = 1852.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    data_cfg = DataConfig(**payload["data"])
    model_cfg = ModelConfig(**payload.get("model", {}))
    training_cfg = TrainingConfig(**payload.get("training", {}))
    return ExperimentConfig(data=data_cfg, model=model_cfg, training=training_cfg)


def haversine_distance(
    lon1: torch.Tensor,
    lat1: torch.Tensor,
    lon2: torch.Tensor,
    lat2: torch.Tensor,
) -> torch.Tensor:
    """Compute great-circle distance in meters."""

    lon1_rad, lat1_rad = torch.deg2rad(lon1), torch.deg2rad(lat1)
    lon2_rad, lat2_rad = torch.deg2rad(lon2), torch.deg2rad(lat2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return 6378137.0 * c


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: TrainingConfig,
    total_steps: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if training_cfg.lr_scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_steps)
    if training_cfg.lr_scheduler == "linear":
        def lr_lambda(step: int) -> float:
            return max(0.0, 1 - step / float(total_steps))

        return LambdaLR(optimizer, lr_lambda)
    if training_cfg.lr_scheduler in (None, "none"):
        return None
    raise ValueError(f"Unsupported scheduler {training_cfg.lr_scheduler}")


def compute_land_penalty(
    land_mask: Optional[LandMask],
    pred_lon: torch.Tensor,
    pred_lat: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    if land_mask is None or weight <= 0:
        return torch.zeros(1, device=pred_lon.device)

    penalty = 0.0
    total_points = 0
    lon_np = pred_lon.detach().cpu().numpy()
    lat_np = pred_lat.detach().cpu().numpy()
    for lon_seq, lat_seq in zip(lon_np, lat_np):
        for lon, lat in zip(lon_seq, lat_seq):
            total_points += 1
            if land_mask.intersects(float(lon), float(lat)):
                penalty += 1.0
            else:
                distance_deg = land_mask.distance_to_land(float(lon), float(lat))
                distance_m = distance_deg * METERS_PER_DEGREE_LAT
                penalty += max(0.0, 1.0 - distance_m / 500.0)
    if total_points == 0:
        return torch.zeros(1, device=pred_lon.device)
    penalty_tensor = torch.tensor(penalty / total_points, device=pred_lon.device)
    return weight * penalty_tensor


def convert_to_absolute(
    delta_xy: torch.Tensor,
    anchor_lon: torch.Tensor,
    anchor_lat: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    cumulative = delta_xy.cumsum(dim=1)
    meters_per_deg_lon = METERS_PER_DEGREE_LAT * torch.cos(torch.deg2rad(anchor_lat))
    meters_per_deg_lon = torch.clamp(meters_per_deg_lon, min=1e-3)
    lon_offset = cumulative[..., 0] / meters_per_deg_lon.unsqueeze(1)
    lat_offset = cumulative[..., 1] / METERS_PER_DEGREE_LAT
    pred_lon = anchor_lon.unsqueeze(1) + lon_offset
    pred_lat = anchor_lat.unsqueeze(1) + lat_offset
    return {"lon": pred_lon, "lat": pred_lat}


def train_one_epoch(
    model: TrajectoryTransformer,
    datamodule: TrajectoryDataModule,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    training_cfg: TrainingConfig,
    land_mask: Optional[LandMask],
) -> Dict[str, float]:
    model.train()
    dataloader = datamodule.dataloader(
        "train", batch_size=training_cfg.batch_size, num_workers=training_cfg.num_workers
    )
    progress = tqdm(dataloader, desc="train", leave=False)

    total_loss = 0.0
    total_batches = 0

    scaler = torch.cuda.amp.GradScaler(enabled=training_cfg.mixed_precision and device.type == "cuda")

    for batch in progress:
        history = batch["history"].to(device)
        static = batch["static"].to(device)
        future_delta = batch["future_delta"].to(device)
        future_abs = batch["future_absolute"].to(device)
        anchor_lon = batch["anchor_lon"].to(device)
        anchor_lat = batch["anchor_lat"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(history, static)
            delta_pred = outputs["delta_xy"]
            delta_loss = F.smooth_l1_loss(delta_pred, future_delta)
            loss = delta_loss

            absolute_pred = convert_to_absolute(delta_pred, anchor_lon, anchor_lat)
            pred_lon = absolute_pred["lon"]
            pred_lat = absolute_pred["lat"]

            target_lon = future_abs[..., 0]
            target_lat = future_abs[..., 1]
            geo_error = haversine_distance(pred_lon, pred_lat, target_lon, target_lat)
            geodesic_loss = geo_error.mean()
            loss = loss + model.config.geodesic_loss_weight * geodesic_loss

            if "speed" in outputs and "future_speed" in batch:
                speed_loss = F.smooth_l1_loss(outputs["speed"], batch["future_speed"].to(device))
                loss = loss + model.config.velocity_loss_weight * speed_loss

            if "heading" in outputs and "future_course" in batch:
                course_rad = torch.deg2rad(batch["future_course"].to(device))
                target_heading = torch.stack([torch.sin(course_rad), torch.cos(course_rad)], dim=-1)
                heading_loss = F.mse_loss(outputs["heading"], target_heading)
                loss = loss + model.config.velocity_loss_weight * 0.5 * heading_loss

            land_penalty = compute_land_penalty(land_mask, pred_lon, pred_lat, model.config.land_penalty_weight)
            loss = loss + land_penalty

            if "destination_logits" in outputs and "destination" in batch:
                destination_loss = F.cross_entropy(
                    outputs["destination_logits"], batch["destination"].to(device)
                )
                loss = loss + model.config.destination_loss_weight * destination_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if training_cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.detach())
        total_batches += 1
        progress.set_postfix({"loss": total_loss / total_batches})

    return {"loss": total_loss / max(total_batches, 1)}


def evaluate(
    model: TrajectoryTransformer,
    datamodule: TrajectoryDataModule,
    split: str,
    device: torch.device,
    land_mask: Optional[LandMask],
) -> Dict[str, float]:
    model.eval()
    dataloader = datamodule.dataloader(split, batch_size=128, num_workers=0)
    total_loss = 0.0
    total_geo = 0.0
    total_final = 0.0
    total_batches = 0
    within_one_nm = 0
    total_sequences = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=split, leave=False):
            history = batch["history"].to(device)
            static = batch["static"].to(device)
            future_delta = batch["future_delta"].to(device)
            future_abs = batch["future_absolute"].to(device)
            anchor_lon = batch["anchor_lon"].to(device)
            anchor_lat = batch["anchor_lat"].to(device)

            outputs = model(history, static)
            delta_pred = outputs["delta_xy"]
            delta_loss = F.smooth_l1_loss(delta_pred, future_delta)

            absolute_pred = convert_to_absolute(delta_pred, anchor_lon, anchor_lat)
            pred_lon = absolute_pred["lon"]
            pred_lat = absolute_pred["lat"]
            target_lon = future_abs[..., 0]
            target_lat = future_abs[..., 1]

            geo_error = haversine_distance(pred_lon, pred_lat, target_lon, target_lat)
            geodesic_loss = geo_error.mean()

            final_error = geo_error[:, -1].mean()
            within_one_nm += (geo_error[:, -1] < NAUTICAL_MILE_METERS).sum().item()
            total_sequences += geo_error.shape[0]

            total_loss += float(delta_loss)
            total_geo += float(geodesic_loss)
            total_final += float(final_error)
            total_batches += 1

    metrics = {
        "loss": total_loss / max(total_batches, 1),
        "geodesic": total_geo / max(total_batches, 1),
        "final_error": total_final / max(total_batches, 1),
        "within_1nm": within_one_nm / max(total_sequences, 1),
    }
    return metrics


def save_checkpoint(path: Path, model: TrajectoryTransformer, optimizer: torch.optim.Optimizer) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(model.config),
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train trajectory transformer")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Directory for checkpoints")
    args = parser.parse_args()

    experiment = load_config(args.config)
    set_seed(experiment.data.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    land_mask = None
    if experiment.data.geojson_path:
        land_mask = LandMask(Path(experiment.data.geojson_path))

    datamodule = TrajectoryDataModule(experiment.data, land_mask)
    datamodule.setup()

    if datamodule.train_dataset is None:
        raise RuntimeError("Training dataset could not be created")
    if experiment.model.predict_steps != datamodule.train_dataset.predict_steps:
        raise ValueError(
            "Model predict_steps must match data forecast horizon; "
            f"got model={experiment.model.predict_steps} data={datamodule.train_dataset.predict_steps}"
        )

    model = TrajectoryTransformer(experiment.model, datamodule.encoder.embedding_dims)
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=experiment.training.learning_rate,
        weight_decay=experiment.training.weight_decay,
    )

    total_steps = len(datamodule.train_dataset) // experiment.training.batch_size
    total_steps = max(total_steps, 1) * experiment.training.num_epochs
    scheduler = create_scheduler(optimizer, experiment.training, total_steps)

    args.output.mkdir(parents=True, exist_ok=True)
    best_val_error = math.inf
    best_checkpoint = args.output / "best.pt"

    for epoch in range(1, experiment.training.num_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            datamodule,
            optimizer,
            scheduler,
            device,
            experiment.training,
            land_mask,
        )
        val_metrics = evaluate(model, datamodule, "val", device, land_mask)

        tqdm.write(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} val_geo={val_metrics['geodesic']:.2f}m "
            f"val_final={val_metrics['final_error']:.2f}m within_1nm={val_metrics['within_1nm']*100:.1f}%"
        )

        if val_metrics["final_error"] < best_val_error:
            best_val_error = val_metrics["final_error"]
            save_checkpoint(best_checkpoint, model, optimizer)

    test_metrics = evaluate(model, datamodule, "test", device, land_mask)
    tqdm.write(
        f"Test: geo={test_metrics['geodesic']:.2f}m final={test_metrics['final_error']:.2f}m within_1nm={test_metrics['within_1nm']*100:.1f}%"
    )


if __name__ == "__main__":
    main()
