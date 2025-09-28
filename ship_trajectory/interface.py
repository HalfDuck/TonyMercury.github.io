"""Runtime inference helpers for external integrations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .data import (
    METERS_PER_DEGREE_LAT,
    KNOT_TO_MPS,
    TrajectoryDataModule,
    meters_per_degree_lon,
)
from .landmask import LandMask
from .model import TrajectoryTransformer
from .train import haversine_distance, load_config, set_seed


def _prepare_land_mask(path: Optional[str]) -> Optional[LandMask]:
    if path:
        return LandMask(Path(path))
    return None


def _safe_float(value: Optional[object]) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _parse_time(value: Optional[object]) -> Optional[pd.Timestamp]:
    if value is None or value == "":
        return None
    try:
        ts = pd.to_datetime(value)
    except (TypeError, ValueError):
        return None
    return ts


def _normalize_angle(angle_deg: np.ndarray) -> np.ndarray:
    return np.mod(angle_deg + 360.0, 360.0)


@dataclass
class PredictionStep:
    lon: float
    lat: float
    mbH: float
    mbV: float


class TrajectoryInferenceEngine:
    """Loads a trained model and exposes a lightweight prediction API."""

    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        device: Optional[str] = None,
    ) -> None:
        self.config = load_config(Path(config_path))
        set_seed(self.config.data.random_seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.land_mask = _prepare_land_mask(self.config.data.geojson_path)
        self.datamodule = TrajectoryDataModule(self.config.data, self.land_mask)
        self.datamodule.setup()

        if self.datamodule.train_dataset is None:
            raise RuntimeError("Training dataset failed to initialise; check CSV path")

        self.normalization = self.datamodule.train_dataset.normalization
        self.history_window = self.datamodule.train_dataset.history_window
        self.step_minutes = self.config.data.sample_rate_minutes

        self.model = TrajectoryTransformer(
            self.config.model, self.datamodule.encoder.embedding_dims
        )
        payload = torch.load(Path(checkpoint_path), map_location=self.device)
        state_dict = payload.get("model_state", payload)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        history: Sequence[Dict[str, object]],
        metadata: Optional[Dict[str, object]] = None,
    ) -> List[Dict[str, float]]:
        """Forecast future trajectory steps for an arbitrary history window."""

        if len(history) < self.history_window:
            raise ValueError(
                f"Expected at least {self.history_window} history points, "
                f"got {len(history)}"
            )

        history_tensor, anchor_lon, anchor_lat = self._encode_history(history)
        static_tensor = self._encode_static(metadata or history[-1])

        history_tensor = history_tensor.unsqueeze(0).to(self.device)
        static_tensor = static_tensor.unsqueeze(0).to(self.device)
        anchor_lon_tensor = torch.tensor([anchor_lon], dtype=torch.float32, device=self.device)
        anchor_lat_tensor = torch.tensor([anchor_lat], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            outputs = self.model.autoregressive_predict(
                history_tensor,
                static_tensor,
                anchor_lat=anchor_lat_tensor,
                anchor_lon=anchor_lon_tensor,
            )

        pred_lon = outputs["pred_lon"].squeeze(0).cpu().numpy()
        pred_lat = outputs["pred_lat"].squeeze(0).cpu().numpy()

        heading_deg = None
        if "heading" in outputs:
            heading_vec = outputs["heading"].squeeze(0).cpu().numpy()
            heading_deg = np.degrees(np.arctan2(heading_vec[:, 0], heading_vec[:, 1]))
            heading_deg = _normalize_angle(heading_deg)

        speed_knots = None
        if "speed" in outputs:
            speed_mps = outputs["speed"].squeeze(0).cpu().numpy()
            speed_knots = speed_mps / KNOT_TO_MPS

        steps = self._format_predictions(
            pred_lon,
            pred_lat,
            heading_deg,
            speed_knots,
            anchor_lon,
            anchor_lat,
        )
        return [step.__dict__ for step in steps]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _encode_history(
        self, history: Sequence[Dict[str, object]]
    ) -> tuple[torch.Tensor, float, float]:
        records = list(history)[-self.history_window :]

        lons = np.array([
            _safe_float(item.get(self.config.data.longitude_column, item.get("rpLong")))
            for item in records
        ], dtype=np.float32)
        lats = np.array([
            _safe_float(item.get(self.config.data.latitude_column, item.get("lat")))
            for item in records
        ], dtype=np.float32)
        speeds = np.array([
            _safe_float(item.get(self.config.data.speed_column, item.get("v")))
            for item in records
        ], dtype=np.float32)
        courses = np.array([
            _safe_float(item.get(self.config.data.course_column, item.get("mbc")))
            for item in records
        ], dtype=np.float32)
        headings = np.array([
            _safe_float(item.get(self.config.data.heading_column, item.get("h")))
            for item in records
        ], dtype=np.float32)

        times = [_parse_time(item.get("time")) for item in records]
        delta_minutes = np.zeros(len(records), dtype=np.float32)
        for idx in range(1, len(records)):
            t_curr, t_prev = times[idx], times[idx - 1]
            if t_curr is not None and t_prev is not None:
                delta = (t_curr - t_prev).total_seconds() / 60.0
                if not math.isfinite(delta) or delta <= 0:
                    delta = float(self.step_minutes)
            else:
                delta = float(self.step_minutes)
            delta_minutes[idx] = float(delta)

        anchor_lat = float(lats[-1])
        anchor_lon = float(lons[-1])
        lon_factor = meters_per_degree_lon(anchor_lat)

        courses = np.where(np.isfinite(courses), courses, headings)
        headings = np.where(np.isfinite(headings), headings, courses)

        if not np.all(np.isfinite(courses)):
            derived = np.copy(courses)
            for idx in range(len(courses) - 1):
                if math.isfinite(derived[idx]):
                    continue
                dlon = (lons[idx + 1] - lons[idx]) * meters_per_degree_lon(float(lats[idx]))
                dlat = (lats[idx + 1] - lats[idx]) * METERS_PER_DEGREE_LAT
                if dlon == 0.0 and dlat == 0.0:
                    continue
                angle = math.degrees(math.atan2(dlat, dlon))
                derived[idx] = angle
            if len(derived) > 1 and not math.isfinite(derived[-1]):
                derived[-1] = derived[-2]
            courses = np.where(np.isfinite(courses), courses, derived)

        if not np.all(np.isfinite(headings)):
            headings = np.where(np.isfinite(headings), headings, courses)

        courses = np.where(np.isfinite(courses), courses, 0.0)
        headings = np.where(np.isfinite(headings), headings, 0.0)

        courses = _normalize_angle(courses.astype(np.float32))
        headings = _normalize_angle(headings.astype(np.float32))

        speed_mps = speeds * KNOT_TO_MPS
        if not np.all(np.isfinite(speed_mps)):
            derived_speed = np.zeros_like(speed_mps)
            for idx in range(1, len(records)):
                lon_fac = meters_per_degree_lon(float(lats[idx - 1]))
                dx = (lons[idx] - lons[idx - 1]) * lon_fac
                dy = (lats[idx] - lats[idx - 1]) * METERS_PER_DEGREE_LAT
                distance = math.hypot(dx, dy)
                dt = max(delta_minutes[idx], 1e-3) * 60.0
                derived_speed[idx] = distance / dt
            if len(derived_speed) > 1:
                derived_speed[0] = derived_speed[1]
            speed_mps = np.where(np.isfinite(speed_mps), speed_mps, derived_speed)

        history_positions = np.stack(
            [
                (lons - anchor_lon) * lon_factor,
                (lats - anchor_lat) * METERS_PER_DEGREE_LAT,
            ],
            axis=-1,
        )

        course_rad = np.deg2rad(courses)
        heading_rad = np.deg2rad(headings)
        history_velocity = np.stack(
            [np.cos(course_rad) * speed_mps, np.sin(course_rad) * speed_mps],
            axis=-1,
        )

        acceleration = np.zeros_like(history_velocity)
        if len(records) > 1:
            delta_seconds = np.maximum(delta_minutes[1:, None], 1e-3) * 60.0
            acceleration[1:] = (history_velocity[1:] - history_velocity[:-1]) / delta_seconds
            acceleration[0] = acceleration[1]

        heading_vector = np.stack([np.sin(heading_rad), np.cos(heading_rad)], axis=-1)
        course_vector = np.stack([np.sin(course_rad), np.cos(course_rad)], axis=-1)
        time_feature = delta_minutes[:, None] / float(self.step_minutes)
        speed_feature = speed_mps[:, None]

        history_features = np.concatenate(
            [
                history_positions,
                history_velocity,
                acceleration,
                speed_feature,
                heading_vector,
                course_vector,
                time_feature,
            ],
            axis=-1,
        )

        history_tensor = torch.from_numpy(history_features).float()
        if self.normalization is not None:
            history_tensor = self.normalization.transform(history_tensor)

        return history_tensor, anchor_lon, anchor_lat

    def _encode_static(self, metadata: Dict[str, object]) -> torch.Tensor:
        type_key = self.config.data.vessel_type_column
        nation_key = self.config.data.nation_column
        name_key = self.config.data.vessel_name_column

        vessel_type = metadata.get(type_key) or metadata.get("type") or "<unk>"
        nation = metadata.get(nation_key) or metadata.get("gjdq") or "<unk>"
        name = metadata.get(name_key) or metadata.get("mbmc") or "<unk>"

        encoder = self.datamodule.encoder
        type_idx = encoder.type_to_idx.get(str(vessel_type), 0)
        nation_idx = encoder.nation_to_idx.get(str(nation), 0)
        name_idx = encoder.name_to_idx.get(str(name), 0)

        static_tensor = torch.tensor([type_idx, nation_idx, name_idx], dtype=torch.long)
        return static_tensor

    def _format_predictions(
        self,
        pred_lon: np.ndarray,
        pred_lat: np.ndarray,
        heading_deg: Optional[np.ndarray],
        speed_knots: Optional[np.ndarray],
        anchor_lon: float,
        anchor_lat: float,
    ) -> List[PredictionStep]:
        steps: List[PredictionStep] = []

        if heading_deg is None:
            prev_lon = np.concatenate(([anchor_lon], pred_lon[:-1]))
            prev_lat = np.concatenate(([anchor_lat], pred_lat[:-1]))
            headings = []
            for lon_prev, lat_prev, lon_curr, lat_curr in zip(
                prev_lon, prev_lat, pred_lon, pred_lat
            ):
                lon_fac = meters_per_degree_lon(float(lat_prev))
                dx = (lon_curr - lon_prev) * lon_fac
                dy = (lat_curr - lat_prev) * METERS_PER_DEGREE_LAT
                angle = math.degrees(math.atan2(dx, dy))
                headings.append((angle + 360.0) % 360.0)
            heading_deg = np.asarray(headings, dtype=np.float32)

        if speed_knots is None:
            prev_lon = np.concatenate(([anchor_lon], pred_lon[:-1]))
            prev_lat = np.concatenate(([anchor_lat], pred_lat[:-1]))
            lon_t = torch.from_numpy(prev_lon)
            lat_t = torch.from_numpy(prev_lat)
            next_lon_t = torch.from_numpy(pred_lon)
            next_lat_t = torch.from_numpy(pred_lat)
            distances = haversine_distance(lon_t, lat_t, next_lon_t, next_lat_t).numpy()
            speed_mps = distances / (self.step_minutes * 60.0)
            speed_knots = speed_mps / KNOT_TO_MPS

        heading_deg = _normalize_angle(heading_deg.astype(np.float32))
        speed_knots = speed_knots.astype(np.float32)

        for lon, lat, hdg, spd in zip(pred_lon, pred_lat, heading_deg, speed_knots):
            steps.append(
                PredictionStep(
                    lon=float(lon),
                    lat=float(lat),
                    mbH=float(hdg),
                    mbV=float(spd),
                )
            )
        return steps

