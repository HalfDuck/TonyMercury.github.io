"""Dataset and dataloaders for trajectory forecasting."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig
from .landmask import LandMask

METERS_PER_DEGREE_LAT = 111_320.0
KNOT_TO_MPS = 0.514444


def meters_per_degree_lon(lat: float) -> float:
    """Approximate meters per degree of longitude at a specific latitude."""

    return METERS_PER_DEGREE_LAT * math.cos(math.radians(lat))


@dataclass
class NormalizationStats:
    mean: torch.Tensor
    std: torch.Tensor

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / (self.std + 1e-6)

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * (self.std + 1e-6) + self.mean


class VesselFeatureEncoder:
    """Encodes categorical vessel metadata into embedding indices."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.type_to_idx: Dict[str, int] = {}
        self.nation_to_idx: Dict[str, int] = {}
        self.name_to_idx: Dict[str, int] = {}

    def fit(self, df: pd.DataFrame) -> None:
        type_values = sorted(df[self.config.vessel_type_column].fillna("<unk>").unique())
        nation_values = sorted(df[self.config.nation_column].fillna("<unk>").unique())
        name_values = sorted(df[self.config.vessel_name_column].fillna("<unk>").unique())

        self.type_to_idx = {value: i for i, value in enumerate(["<unk>"] + type_values)}
        self.nation_to_idx = {value: i for i, value in enumerate(["<unk>"] + nation_values)}
        self.name_to_idx = {value: i for i, value in enumerate(["<unk>"] + name_values)}

    def encode(self, row: pd.Series) -> Dict[str, int]:
        return {
            "type": self.type_to_idx.get(row.get(self.config.vessel_type_column), 0),
            "nation": self.nation_to_idx.get(row.get(self.config.nation_column), 0),
            "name": self.name_to_idx.get(row.get(self.config.vessel_name_column), 0),
        }

    @property
    def embedding_dims(self) -> Dict[str, int]:
        return {
            "type": max(self.type_to_idx.values(), default=0) + 1,
            "nation": max(self.nation_to_idx.values(), default=0) + 1,
            "name": max(self.name_to_idx.values(), default=0) + 1,
        }


class DestinationClusterer:
    """Clusters historical endpoints to provide destination classes."""

    def __init__(self, num_clusters: int, random_state: int = 42):
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.model: Optional[MiniBatchKMeans] = None

    def fit(self, coordinates: np.ndarray) -> None:
        if coordinates.shape[0] < self.num_clusters:
            raise ValueError(
                "Number of destination clusters exceeds the number of unique samples"
            )
        self.model = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            random_state=self.random_state,
            batch_size=min(2048, coordinates.shape[0]),
            max_iter=200,
        )
        self.model.fit(coordinates)

    def predict(self, coordinates: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DestinationClusterer must be fit before predicting")
        return self.model.predict(coordinates)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("DestinationClusterer must be fit before saving")
        payload = {
            "cluster_centers": self.model.cluster_centers_.tolist(),
            "n_clusters": self.model.n_clusters,
            "random_state": self.random_state,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DestinationClusterer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        clusterer = cls(num_clusters=payload["n_clusters"], random_state=payload["random_state"])
        clusterer.model = MiniBatchKMeans(n_clusters=payload["n_clusters"])
        clusterer.model.cluster_centers_ = np.array(payload["cluster_centers"], dtype=np.float32)
        return clusterer


@dataclass
class TrajectorySample:
    history: torch.Tensor  # (history_window, feature_dim)
    static: torch.Tensor  # (static_feature_dim,)
    future_delta: torch.Tensor  # (predict_steps, 2) in meters
    future_absolute: torch.Tensor  # (predict_steps, 2) in degrees (lon, lat)
    anchor_lon: float
    anchor_lat: float
    future_speed: Optional[torch.Tensor]
    future_course: Optional[torch.Tensor]
    destination: Optional[int]


class TrajectoryDataset(Dataset):
    """PyTorch dataset for historical trajectory sequences."""

    def __init__(
        self,
        config: DataConfig,
        dataframe: pd.DataFrame,
        encoder: VesselFeatureEncoder,
        land_mask: Optional[LandMask] = None,
        destination_clusterer: Optional[DestinationClusterer] = None,
        split: str = "train",
        normalization: Optional[NormalizationStats] = None,
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.land_mask = land_mask
        self.destination_clusterer = destination_clusterer
        self.split = split
        self.history_window = config.history_window
        self.predict_steps = config.forecast_horizon_minutes // config.sample_rate_minutes
        self.samples: List[TrajectorySample] = []
        self.normalization: Optional[NormalizationStats] = normalization

        self._prepare_samples(dataframe)
        if config.normalize_numeric and self.samples:
            if self.normalization is None:
                self._compute_normalization()
            self._apply_normalization()

    def _prepare_samples(self, df: pd.DataFrame) -> None:
        df = df.copy()
        df[self.config.time_column] = pd.to_datetime(df[self.config.time_column])
        df.sort_values([self.config.vessel_id_column, self.config.time_column], inplace=True)

        track_groups = df.groupby(self.config.vessel_id_column)
        random.seed(self.config.random_seed)

        endpoint_coords: List[Tuple[float, float]] = []

        for _, track in track_groups:
            if len(track) < self.config.minimum_track_length:
                continue
            track = track.reset_index(drop=True)
            track["delta_minutes"] = track[self.config.time_column].diff().dt.total_seconds().fillna(0) / 60.0

            lats = track[self.config.latitude_column].to_numpy(dtype=np.float32)
            lons = track[self.config.longitude_column].to_numpy(dtype=np.float32)
            speeds = track[self.config.speed_column].to_numpy(dtype=np.float32)
            courses = track[self.config.course_column].to_numpy(dtype=np.float32)
            headings = track[self.config.heading_column].fillna(courses).to_numpy(dtype=np.float32)

            static_encoded = self.encoder.encode(track.iloc[0])
            static_tensor = torch.tensor([
                static_encoded["type"],
                static_encoded["nation"],
                static_encoded["name"],
            ], dtype=torch.long)

            for start_idx in range(0, len(track) - self.history_window - self.predict_steps + 1):
                history_slice = slice(start_idx, start_idx + self.history_window)
                future_slice = slice(
                    start_idx + self.history_window,
                    start_idx + self.history_window + self.predict_steps,
                )
                history_lat = lats[history_slice]
                history_lon = lons[history_slice]
                future_lat = lats[future_slice]
                future_lon = lons[future_slice]
                future_speed = speeds[future_slice]
                future_course = courses[future_slice]
                history_speed = speeds[history_slice]
                history_course = courses[history_slice]
                history_heading = headings[history_slice]
                history_delta_minutes = track["delta_minutes"].to_numpy(dtype=np.float32)[history_slice]

                # Local tangent plane relative to last history point
                anchor_lat = history_lat[-1]
                anchor_lon = history_lon[-1]

                lon_factor = meters_per_degree_lon(anchor_lat)
                lat_factor = METERS_PER_DEGREE_LAT

                history_positions = np.stack(
                    [
                        (history_lon - anchor_lon) * lon_factor,
                        (history_lat - anchor_lat) * lat_factor,
                    ],
                    axis=-1,
                )
                course_rad = np.deg2rad(history_course)
                heading_rad = np.deg2rad(history_heading)
                history_speed_mps = history_speed * KNOT_TO_MPS
                history_velocity = np.stack(
                    [
                        np.cos(course_rad) * history_speed_mps,
                        np.sin(course_rad) * history_speed_mps,
                    ],
                    axis=-1,
                )
                acceleration = np.zeros_like(history_velocity)
                delta_minutes = np.maximum(history_delta_minutes[1:, None], 1e-3)
                acceleration[1:] = (history_velocity[1:] - history_velocity[:-1]) / (delta_minutes * 60.0)
                acceleration[0] = acceleration[1]
                heading_vector = np.stack([np.sin(heading_rad), np.cos(heading_rad)], axis=-1)
                course_vector = np.stack([np.sin(course_rad), np.cos(course_rad)], axis=-1)
                time_feature = history_delta_minutes[:, None] / self.config.sample_rate_minutes

                history_features = np.concatenate(
                    [
                        history_positions,
                        history_velocity,
                        acceleration,
                        history_speed_mps[:, None],
                        heading_vector,
                        course_vector,
                        time_feature,
                    ],
                    axis=-1,
                )

                future_positions = np.stack(
                    [
                        (future_lon - anchor_lon) * lon_factor,
                        (future_lat - anchor_lat) * lat_factor,
                    ],
                    axis=-1,
                )
                future_delta = future_positions - history_positions[-1]
                future_absolute = np.stack([future_lon, future_lat], axis=-1)
                future_speed_mps = future_speed * KNOT_TO_MPS

                sample = TrajectorySample(
                    history=torch.from_numpy(history_features).float(),
                    static=static_tensor.clone(),
                    future_delta=torch.from_numpy(future_delta).float(),
                    future_absolute=torch.from_numpy(future_absolute).float(),
                    anchor_lon=float(anchor_lon),
                    anchor_lat=float(anchor_lat),
                    future_speed=torch.from_numpy(future_speed_mps).float(),
                    future_course=torch.from_numpy(future_course).float(),
                    destination=None,
                )

                if self.destination_clusterer is not None:
                    endpoint_coords.append((future_lon[-1], future_lat[-1]))
                    # temporarily store, assign later after fitting clusterer
                    sample.destination = -1

                self.samples.append(sample)

        if self.destination_clusterer is not None and endpoint_coords:
            coords = np.asarray(endpoint_coords, dtype=np.float32)
            if self.destination_clusterer.model is None and self.split == "train":
                self.destination_clusterer.fit(coords)
            if self.destination_clusterer.model is None:
                raise RuntimeError(
                    "Destination clusterer has not been fit; provide a trained model or run on train split first."
                )
            labels = self.destination_clusterer.predict(coords)
            for sample, label in zip(self.samples, labels):
                sample.destination = int(label)

    def _compute_normalization(self) -> None:
        histories = torch.stack([sample.history for sample in self.samples])
        mean = histories.mean(dim=(0, 1))
        std = histories.std(dim=(0, 1))
        self.normalization = NormalizationStats(mean=mean, std=std)

    def _apply_normalization(self) -> None:
        assert self.normalization is not None
        for sample in self.samples:
            sample.history = self.normalization.transform(sample.history)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        payload: Dict[str, torch.Tensor] = {
            "history": sample.history,
            "static": sample.static,
            "future_delta": sample.future_delta,
            "future_absolute": sample.future_absolute,
            "anchor_lon": torch.tensor(sample.anchor_lon, dtype=torch.float32),
            "anchor_lat": torch.tensor(sample.anchor_lat, dtype=torch.float32),
        }
        if sample.future_speed is not None:
            payload["future_speed"] = sample.future_speed
        if sample.future_course is not None:
            payload["future_course"] = sample.future_course
        if sample.destination is not None and sample.destination >= 0:
            payload["destination"] = torch.tensor(sample.destination, dtype=torch.long)
        return payload


def collate_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    collated: Dict[str, List[torch.Tensor]] = {key: [] for key in keys}
    for sample in batch:
        for key in keys:
            collated[key].append(sample[key])
    return {key: torch.stack(value) for key, value in collated.items()}


class TrajectoryDataModule:
    """Creates train/validation/test dataloaders for trajectory forecasting."""

    def __init__(
        self,
        config: DataConfig,
        land_mask: Optional[LandMask] = None,
    ) -> None:
        self.config = config
        self.land_mask = land_mask
        self.encoder = VesselFeatureEncoder(config)
        self.destination_clusterer: Optional[DestinationClusterer] = None
        if config.cluster_destinations:
            self.destination_clusterer = DestinationClusterer(
                num_clusters=config.destination_cluster_count,
                random_state=config.random_seed,
            )
        self.train_dataset: Optional[TrajectoryDataset] = None
        self.val_dataset: Optional[TrajectoryDataset] = None
        self.test_dataset: Optional[TrajectoryDataset] = None

    def load_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.csv_path)
        return df

    def setup(self) -> None:
        df = self.load_dataframe()
        self.encoder.fit(df)

        if self.destination_clusterer is not None:
            # fit on entire dataset within TrajectoryDataset
            pass

        vessel_ids = df[self.config.vessel_id_column].unique().tolist()
        random.Random(self.config.random_seed).shuffle(vessel_ids)
        num_vessels = len(vessel_ids)
        num_test = int(num_vessels * self.config.test_fraction)
        num_val = int(num_vessels * self.config.validation_fraction)

        test_ids = set(vessel_ids[:num_test])
        val_ids = set(vessel_ids[num_test : num_test + num_val])

        train_df = df[~df[self.config.vessel_id_column].isin(test_ids | val_ids)]
        val_df = df[df[self.config.vessel_id_column].isin(val_ids)]
        test_df = df[df[self.config.vessel_id_column].isin(test_ids)]

        self.train_dataset = TrajectoryDataset(
            config=self.config,
            dataframe=train_df,
            encoder=self.encoder,
            land_mask=self.land_mask,
            destination_clusterer=self.destination_clusterer,
            split="train",
        )
        normalization = self.train_dataset.normalization
        self.val_dataset = TrajectoryDataset(
            config=self.config,
            dataframe=val_df,
            encoder=self.encoder,
            land_mask=self.land_mask,
            destination_clusterer=self.destination_clusterer,
            split="val",
            normalization=normalization,
        )
        self.test_dataset = TrajectoryDataset(
            config=self.config,
            dataframe=test_df,
            encoder=self.encoder,
            land_mask=self.land_mask,
            destination_clusterer=self.destination_clusterer,
            split="test",
            normalization=normalization,
        )

    def dataloader(self, split: str, batch_size: int, num_workers: int = 0) -> DataLoader:
        dataset_map = {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test": self.test_dataset,
        }
        dataset = dataset_map.get(split)
        if dataset is None:
            raise ValueError(f"Unknown split {split}")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_batch,
            drop_last=(split == "train"),
        )

