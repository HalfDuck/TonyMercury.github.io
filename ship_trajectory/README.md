# Transformer-based Naval Trajectory Forecasting

This directory contains a full training pipeline for learning to forecast naval vessel trajectories up to one hour ahead using a Transformer-based model. The design focuses on:

- **Meter-level predictions** derived from raw latitude/longitude inputs using local tangent plane projections.
- **Land avoidance** via GeoJSON shoreline data that penalizes predicted points that intersect land.
- **Destination awareness** with an auxiliary head that classifies a vessel's likely destination cluster.
- **Physical priors** such as velocity, acceleration, and heading embeddings to stabilize roll-outs and avoid unrealistic jumps.
- **Extensibility** for future battlefield context features and military installation influence.

## Repository layout

- [`config.py`](config.py): Dataclass definitions for data, model, and training configuration.
- [`data.py`](data.py): Dataset utilities that create sliding windows from historical AIS-like tracks, normalize inputs, and attach per-sample anchors for absolute geodesic reconstruction.
- [`landmask.py`](landmask.py): Thin wrapper around the provided GeoJSON polygons using `shapely` to support land-intersection checks.
- [`model.py`](model.py): The Transformer encoder-decoder with learned future queries and auxiliary heads for speed, heading, and destination prediction.
- [`train.py`](train.py): End-to-end training/evaluation script that enforces land avoidance, computes geodesic metrics, and reports 1 NM accuracy.
- [`predict.py`](predict.py): Single-sample inference and plotting utilities that visualise history, predictions, and ground truth tracks.
- [`configs/trajectory_transformer.yaml`](../configs/trajectory_transformer.yaml): Example experiment configuration.

## Data preparation

1. **Historical tracks**: The dataset expects one CSV with the columns demonstrated in `data.csv` (`ph`, `mbmc`, `gjdq`, `type`, `time`, `mbc`, `mbv`, `mbh`, `lon`, `lat`). Observations must be at most 10 minutes apart. Tracks shorter than `minimum_track_length` are ignored.
2. **GeoJSON land mask**: Supply a shoreline polygon collection that covers the operating theater. Land penalties discourage the model from producing infeasible trajectories.
3. **Battlefield context (optional)**: Additional features (e.g., weather, threat level, ROE) can be encoded as environment tensors and passed to `TrajectoryTransformer.forward` via the `environment` argument. Update `ModelConfig.environment_dim` accordingly.
4. **Military installations**: Curate installation coordinates and categorical metadata, then encode them per sample (e.g., distance-to-installation, affiliation). These can be concatenated to the dynamic features or fed through the environment context.

## Training

```bash
python -m ship_trajectory.train --config configs/trajectory_transformer.yaml --output outputs/exp1
```

The script will:

1. Split vessels into train/validation/test sets.
2. Cluster historical endpoints (MiniBatchKMeans) to establish destination labels.
3. Normalize motion features using only the training split statistics.
4. Train with mixed precision and cosine learning rate decay.
5. Penalize predictions that intersect land polygons and report one-hour geodesic errors.

Metrics include mean per-step geodesic error, final-step error, and the percentage of trajectories finishing within **1 nautical mile** of ground truth.

## Inference & visualisation

After training completes, load the best checkpoint and generate a qualitative plot for any split/sample:

```bash
python -m ship_trajectory.predict \
  --config configs/trajectory_transformer.yaml \
  --checkpoint outputs/exp1/best.pt \
  --split test \
  --index 0 \
  --output outputs/exp1/sample0.png
```

The command also prints per-step mean error, final 60-minute error, and whether the trajectory lands within **1 NM**. The saved figure overlays:

- Blue circles: historical input trajectory.
- Green circles: ground-truth future path.
- Red circles: model prediction (with an `X` on the predicted endpoint).

For programmatic access, import `predict_sequence` and call it directly to obtain the `PredictionResult` dataclass with NumPy arrays and metrics.

## Extending to battlefield and installation effects

- **Battlefield simulation**: Generate synthetic environment tensors (e.g., vector fields for wind/current, restricted zones) and add them as additional inputs via the `environment` argument. During deployment, replace synthetic feeds with live upstream integrations.
- **Military facility influence**: Pre-compute proximity features to known installations. You can extend `TrajectoryDataset` to append distances/angles to the nearest allied/hostile facility and let the Transformer learn routing biases (e.g., loitering near friendly bases, avoiding hostile SAM coverage).
- **Terminal guidance**: The destination head can be swapped for a sequence of waypoints predicted via beam search if explicit path planning is required. The current cluster-based approach keeps the network aware of likely endpoints without hard-coding rules.

## Expected performance

With sufficient historical coverage (200k+ points as mentioned) and proper hyper-parameter tuning, the model is designed to keep the **1-hour forecast error under 1 NM** on validation data, provided the operating area is predominantly maritime. Use the validation statistics to tune the loss weights and adjust the land penalty strength for your region.

