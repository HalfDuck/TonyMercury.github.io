"""Configuration dataclasses for the trajectory forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Hyper-parameters that control training loops."""

    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    gradient_clip_norm: float = 1.0
    num_workers: int = 8
    lr_scheduler: str = "cosine"
    label_smoothing: float = 0.0
    max_grad_norm: float = 1.0
    mixed_precision: bool = True


@dataclass
class ModelConfig:
    """Top-level configuration for the trajectory transformer."""

    input_dim: int = 12
    static_feature_dim: int = 8
    hidden_dim: int = 256
    ff_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    activation: str = "gelu"
    predict_steps: int = 6
    # destination head (cluster classification)
    num_destination_clusters: int = 128
    destination_loss_weight: float = 0.3
    # land avoidance weighting
    land_penalty_weight: float = 10.0
    geodesic_loss_weight: float = 1.0
    velocity_loss_weight: float = 0.3
    # optional environment context embedding
    environment_dim: Optional[int] = None
    # optional auxiliary tasks (e.g., future speed prediction)
    predict_speed: bool = True
    predict_heading: bool = True


@dataclass
class DataConfig:
    """Configuration describing data preprocessing behavior."""

    csv_path: str
    geojson_path: Optional[str] = None
    time_column: str = "time"
    vessel_id_column: str = "ph"
    longitude_column: str = "lon"
    latitude_column: str = "lat"
    speed_column: str = "mbv"
    course_column: str = "mbc"
    heading_column: str = "mbh"
    vessel_type_column: str = "type"
    nation_column: str = "gjdq"
    vessel_name_column: str = "mbmc"
    sample_rate_minutes: int = 10
    history_window: int = 12  # past timesteps (2 hours)
    forecast_horizon_minutes: int = 60
    normalize_numeric: bool = True
    use_local_projection: bool = True
    local_projection_anchor: Optional[List[float]] = None
    cluster_destinations: bool = True
    destination_cluster_count: int = 128
    minimum_track_length: int = 18
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    random_seed: int = 42


@dataclass
class ExperimentConfig:
    """Bundle of configuration sections used during experiments."""

    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

