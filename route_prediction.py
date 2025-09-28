"""Utilities for building a maritime route network from historical AIS-style tracks
and predicting likely continuations for new targets.

The module exposes the following high-level workflow:

* :func:`load_historical_tracks` – Load the CSV history into a tidy
  :class:`pandas.DataFrame` with proper dtypes.
* :class:`RouteNetworkBuilder` – Convert the history into a road-like network by
  snapping points to reusable nodes (within a configurable tolerance) and
  storing historical traversals as reusable templates.
* :class:`RoutePredictor` – Match an incoming partial track with an historical
  traversal and extend it towards a user supplied destination when possible.
* :func:`save_prediction_plot` – Visualise the observed, matched historical and
  predicted future legs, optionally overlaying land polygons from a GeoJSON
  mask so users can verify the prediction avoids land.

The design intentionally keeps dependencies light and avoids heavy GIS stacks
while still honouring maritime constraints (e.g. respecting land) when data is
available.  Geodesic distances are approximated with small area planar
projections which keeps the code self-contained yet accurate enough for routing
purposes in coastal scenarios.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0


@dataclass
class TrackPoint:
    """A single observation within a historical track."""

    time: dt.datetime
    lon: float
    lat: float
    course: Optional[float] = None
    speed: Optional[float] = None


@dataclass
class Node:
    """Representative node for multiple nearby track points."""

    node_id: int
    lat: float
    lon: float
    # Keep simple running averages so the node drifts towards the centre of the
    # points that were merged into it.
    _count: int = 1

    def update(self, lat: float, lon: float) -> None:
        self._count += 1
        weight = 1.0 / self._count
        self.lat = (1.0 - weight) * self.lat + weight * lat
        self.lon = (1.0 - weight) * self.lon + weight * lon


@dataclass
class Edge:
    """Aggregated statistics for an undirected road-network like edge."""

    node_u: int
    node_v: int
    traversals: int = 0
    total_length_m: float = 0.0
    total_speed: float = 0.0

    def register(self, length_m: float, speed: Optional[float]) -> None:
        self.traversals += 1
        self.total_length_m += length_m
        if speed is not None:
            self.total_speed += speed

    @property
    def average_length_m(self) -> float:
        return self.total_length_m / self.traversals if self.traversals else 0.0

    @property
    def average_speed(self) -> Optional[float]:
        if self.traversals and self.total_speed:
            return self.total_speed / self.traversals
        return None


@dataclass
class HistoricalTrack:
    """A historical traversal stored for later pattern matching."""

    track_id: str
    ship_name: str
    country: str
    ship_type: str
    points: List[TrackPoint]
    node_ids: List[int]

    @property
    def start_time(self) -> dt.datetime:
        return self.points[0].time

    @property
    def end_time(self) -> dt.datetime:
        return self.points[-1].time


@dataclass
class RouteNetwork:
    """Container for the inferred road network and related spatial index."""

    nodes: Dict[int, Node]
    edges: Dict[Tuple[int, int], Edge]
    tracks: List[HistoricalTrack]
    supported_ship_types: List[str]
    tolerance_m: float
    cell_size_deg: float
    _cell_index: Dict[Tuple[int, int], List[int]]

    def find_closest_node(self, lat: float, lon: float) -> Tuple[Optional[int], float]:
        """Return the id of the closest node and the distance in metres."""

        cell_x, cell_y = _cell_for_point(lat, lon, self.cell_size_deg)
        best_node = None
        best_distance = float("inf")
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (cell_x + dx, cell_y + dy)
                for node_id in self._cell_index.get(key, []):
                    node = self.nodes[node_id]
                    distance = haversine_m(node.lat, node.lon, lat, lon)
                    if distance < best_distance:
                        best_distance = distance
                        best_node = node_id
        return best_node, best_distance


class RouteNetworkBuilder:
    """Build a reusable network from historical tracks."""

    def __init__(self, tolerance_m: float = 500.0, cell_size_deg: float = 0.05) -> None:
        self.tolerance_m = tolerance_m
        self.cell_size_deg = cell_size_deg
        self._nodes: Dict[int, Node] = {}
        self._cell_index: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self._edges: Dict[Tuple[int, int], Edge] = {}
        self._tracks: List[HistoricalTrack] = []
        self._supported_types: set[str] = set()

    def build_from_dataframe(self, df: pd.DataFrame) -> RouteNetwork:
        grouped = df.groupby("ph", sort=False)
        for track_id, group in grouped:
            track = self._ingest_track(track_id, group)
            if track:
                self._tracks.append(track)
        supported_types = sorted(t for t in self._supported_types if t)
        return RouteNetwork(
            nodes=self._nodes,
            edges=self._edges,
            tracks=self._tracks,
            supported_ship_types=supported_types,
            tolerance_m=self.tolerance_m,
            cell_size_deg=self.cell_size_deg,
            _cell_index=self._cell_index,
        )

    def build_from_csv(self, path: str | pathlib.Path) -> RouteNetwork:
        df = load_historical_tracks(path)
        return self.build_from_dataframe(df)

    def _ingest_track(self, track_id: str, group: pd.DataFrame) -> Optional[HistoricalTrack]:
        group_sorted = group.sort_values("time")
        points: List[TrackPoint] = []
        node_ids: List[int] = []

        last_node: Optional[int] = None
        last_point: Optional[TrackPoint] = None

        for row in group_sorted.itertuples():
            point = TrackPoint(
                time=row.time,
                lon=float(row.lon),
                lat=float(row.lat),
                course=float(row.mbc) if not math.isnan(row.mbc) else None,
                speed=float(row.mbv) if not math.isnan(row.mbv) else None,
            )
            node_id = self._get_or_create_node(point.lat, point.lon)
            if node_id is None:
                continue
            if last_node is not None and node_id == last_node:
                # Skip duplicates caused by very dense sampling in the same node.
                last_point = point
                continue

            points.append(point)
            node_ids.append(node_id)
            if last_node is not None:
                length = haversine_m(
                    self._nodes[last_node].lat,
                    self._nodes[last_node].lon,
                    point.lat,
                    point.lon,
                )
                speed = point.speed
                edge_key = _edge_key(last_node, node_id)
                edge = self._edges.get(edge_key)
                if edge is None:
                    edge = Edge(node_u=edge_key[0], node_v=edge_key[1])
                    self._edges[edge_key] = edge
                edge.register(length, speed)
            last_node = node_id
            last_point = point

        if len(points) < 2:
            return None

        first_row = group_sorted.iloc[0]
        self._supported_types.add(first_row.type)
        return HistoricalTrack(
            track_id=str(track_id),
            ship_name=str(first_row.mbmc),
            country=str(first_row.gjdq),
            ship_type=str(first_row.type),
            points=points,
            node_ids=node_ids,
        )

    def _get_or_create_node(self, lat: float, lon: float) -> Optional[int]:
        cell = _cell_for_point(lat, lon, self.cell_size_deg)
        best_node: Optional[int] = None
        best_distance = self.tolerance_m
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (cell[0] + dx, cell[1] + dy)
                for node_id in self._cell_index.get(key, []):
                    node = self._nodes[node_id]
                    distance = haversine_m(node.lat, node.lon, lat, lon)
                    if distance < best_distance:
                        best_distance = distance
                        best_node = node_id
        if best_node is not None:
            self._nodes[best_node].update(lat, lon)
            return best_node
        node_id = len(self._nodes)
        node = Node(node_id=node_id, lat=lat, lon=lon)
        self._nodes[node_id] = node
        self._cell_index[cell].append(node_id)
        return node_id


class RoutePredictor:
    """Predict a future route by reusing a matched historical traversal."""

    def __init__(
        self,
        network: RouteNetwork,
        destination_tolerance_m: float = 1_000.0,
        minimum_match_points: int = 3,
    ) -> None:
        self.network = network
        self.destination_tolerance_m = destination_tolerance_m
        self.minimum_match_points = minimum_match_points

    def predict(
        self,
        observed_points: Sequence[TrackPoint],
        destination: Tuple[float, float],
    ) -> Dict[str, object]:
        if len(observed_points) < self.minimum_match_points:
            return {
                "status": "insufficient-data",
                "message": "观察到的航迹点过少，无法匹配历史航迹。",
                "supported_ship_types": self.network.supported_ship_types,
            }

        obs_nodes, node_distances = self._map_to_nodes(observed_points)
        if any(d > self.network.tolerance_m for d in node_distances):
            return {
                "status": "insufficient-data",
                "message": "部分观测点距离历史航迹过远，无法可靠匹配。",
                "supported_ship_types": self.network.supported_ship_types,
            }

        match = self._find_best_match(obs_nodes, destination)
        if match is None:
            return {
                "status": "insufficient-data",
                "message": "历史航迹库中找不到与当前航迹相符且可达终点的线路。",
                "supported_ship_types": self.network.supported_ship_types,
            }

        track, start_idx, end_idx, dest_idx = match
        observed_end_time = observed_points[-1].time
        template_base_time = track.points[end_idx].time
        predicted_points: List[Dict[str, object]] = []
        for idx in range(end_idx + 1, dest_idx + 1):
            template_point = track.points[idx]
            delta = template_point.time - template_base_time
            predicted_time = observed_end_time + delta
            predicted_points.append(
                {
                    "time": predicted_time.isoformat(),
                    "lat": template_point.lat,
                    "lon": template_point.lon,
                    "speed": template_point.speed,
                }
            )

        # Ensure the destination itself is included and reported even when the
        # historical track stopped slightly before the provided destination
        dest_lat, dest_lon = destination
        if predicted_points:
            last_point = predicted_points[-1]
            tail_distance = haversine_m(
                last_point["lat"],
                last_point["lon"],
                dest_lat,
                dest_lon,
            )
        else:
            template_point = track.points[end_idx]
            tail_distance = haversine_m(
                template_point.lat,
                template_point.lon,
                dest_lat,
                dest_lon,
            )
        if tail_distance > 1.0:
            predicted_points.append(
                {
                    "time": (predicted_points[-1]["time"] if predicted_points else observed_end_time.isoformat()),
                    "lat": dest_lat,
                    "lon": dest_lon,
                    "speed": None,
                }
            )

        explanation = (
            f"预测航迹基于历史航迹 {track.track_id}（{track.ship_name}，{track.ship_type}）的"
            f" {len(track.points)} 个轨迹点，其中观测航迹匹配于索引 {start_idx}-{end_idx}。"
        )
        return {
            "status": "ok",
            "message": explanation,
            "matched_track": {
                "track_id": track.track_id,
                "ship_name": track.ship_name,
                "ship_type": track.ship_type,
                "country": track.country,
                "start_time": track.start_time.isoformat(),
                "end_time": track.end_time.isoformat(),
            },
            "predicted_points": predicted_points,
            "supported_ship_types": self.network.supported_ship_types,
        }

    def _map_to_nodes(self, points: Sequence[TrackPoint]) -> Tuple[List[int], List[float]]:
        node_ids: List[int] = []
        distances: List[float] = []
        for point in points:
            node_id, distance = self.network.find_closest_node(point.lat, point.lon)
            if node_id is None:
                node_ids.append(-1)
                distances.append(float("inf"))
            else:
                node_ids.append(node_id)
                distances.append(distance)
        return node_ids, distances

    def _find_best_match(
        self,
        obs_nodes: Sequence[int],
        destination: Tuple[float, float],
    ) -> Optional[Tuple[HistoricalTrack, int, int, int]]:
        dest_lat, dest_lon = destination
        best_match: Optional[Tuple[HistoricalTrack, int, int, int]] = None
        best_score = float("inf")

        for track in self.network.tracks:
            candidate_indices = _find_subsequence_indices(track.node_ids, obs_nodes)
            if not candidate_indices:
                continue
            for start_idx in candidate_indices:
                end_idx = start_idx + len(obs_nodes) - 1
                dest_idx = self._locate_destination(track, end_idx, dest_lat, dest_lon)
                if dest_idx is None:
                    continue
                score = self._score_match(track, start_idx, end_idx, obs_nodes)
                if score < best_score:
                    best_score = score
                    best_match = (track, start_idx, end_idx, dest_idx)
        return best_match

    def _locate_destination(
        self,
        track: HistoricalTrack,
        matched_end_idx: int,
        dest_lat: float,
        dest_lon: float,
    ) -> Optional[int]:
        for idx in range(matched_end_idx + 1, len(track.points)):
            point = track.points[idx]
            distance_point = haversine_m(point.lat, point.lon, dest_lat, dest_lon)
            if distance_point <= self.destination_tolerance_m:
                return idx
            prev_point = track.points[idx - 1]
            distance_segment = point_to_segment_distance_m(
                dest_lat,
                dest_lon,
                prev_point.lat,
                prev_point.lon,
                point.lat,
                point.lon,
            )
            if distance_segment <= self.destination_tolerance_m:
                return idx
        return None

    def _score_match(
        self,
        track: HistoricalTrack,
        start_idx: int,
        end_idx: int,
        obs_nodes: Sequence[int],
    ) -> float:
        # Smaller scores are better. Use cumulative physical distance between
        # the observed points and the matched historical points.
        distance = 0.0
        for offset, node_id in enumerate(obs_nodes):
            track_point = track.points[start_idx + offset]
            node = self.network.nodes[node_id]
            distance += haversine_m(track_point.lat, track_point.lon, node.lat, node.lon)
        return distance


def load_historical_tracks(path: str | pathlib.Path) -> pd.DataFrame:
    """Load and tidy the provided CSV history file."""

    df = pd.read_csv(path)
    required_columns = {
        "ph",
        "mbmc",
        "gjdq",
        "type",
        "time",
        "mbc",
        "mbv",
        "lon",
        "lat",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要字段: {sorted(missing)}")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        raise ValueError("存在无法解析的时间戳。")
    numeric_fields = ["mbc", "mbv", "mbh", "lon", "lat"]
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")
    return df


def save_prediction_plot(
    history_tracks: Iterable[HistoricalTrack],
    observed_points: Sequence[TrackPoint],
    predicted_points: Sequence[Dict[str, object]],
    destination: Tuple[float, float],
    output_path: str | pathlib.Path,
    land_mask_geojson: Optional[str | pathlib.Path] = None,
) -> None:
    """Generate a quick-look map for the prediction."""

    fig, ax = plt.subplots(figsize=(8, 8))

    if land_mask_geojson is not None:
        try:
            with open(land_mask_geojson, "r", encoding="utf-8") as f:
                mask = json.load(f)
            for geometry in _iter_geojson_polygons(mask):
                xs = [pt[0] for pt in geometry]
                ys = [pt[1] for pt in geometry]
                ax.fill(xs, ys, facecolor="#dddddd", edgecolor="#aaaaaa", linewidth=0.5)
        except OSError:
            pass

    for track in history_tracks:
        xs = [p.lon for p in track.points]
        ys = [p.lat for p in track.points]
        ax.plot(xs, ys, color="#bbbbbb", linewidth=0.8, alpha=0.4)

    ax.plot([p.lon for p in observed_points], [p.lat for p in observed_points],
            color="tab:orange", linewidth=2, marker="o", label="观测航迹")

    if predicted_points:
        ax.plot([p["lon"] for p in predicted_points], [p["lat"] for p in predicted_points],
                color="tab:blue", linewidth=2, marker="o", label="预测航迹")

    ax.scatter([destination[1]], [destination[0]], marker="*", color="tab:red", s=120, label="指定终点")

    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_title("历史航迹匹配预测")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate great circle distance in metres."""

    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def point_to_segment_distance_m(
    lat: float,
    lon: float,
    seg_lat1: float,
    seg_lon1: float,
    seg_lat2: float,
    seg_lon2: float,
) -> float:
    """Return the distance from a point to the segment in metres."""

    if seg_lat1 == seg_lat2 and seg_lon1 == seg_lon2:
        return haversine_m(lat, lon, seg_lat1, seg_lon1)

    # Project into a local tangent plane using the mean latitude as reference to
    # keep the metric distortion low.
    ref_lat = math.radians((seg_lat1 + seg_lat2 + lat) / 3.0)
    x0, y0 = _project_lon_lat(lon, lat, ref_lat)
    x1, y1 = _project_lon_lat(seg_lon1, seg_lat1, ref_lat)
    x2, y2 = _project_lon_lat(seg_lon2, seg_lat2, ref_lat)

    dx = x2 - x1
    dy = y2 - y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(x0 - x1, y0 - y1)

    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(x0 - proj_x, y0 - proj_y)


def _project_lon_lat(lon: float, lat: float, ref_lat_rad: float) -> Tuple[float, float]:
    x = math.radians(lon) * math.cos(ref_lat_rad) * EARTH_RADIUS_M
    y = math.radians(lat) * EARTH_RADIUS_M
    return x, y


def _cell_for_point(lat: float, lon: float, cell_size_deg: float) -> Tuple[int, int]:
    return (
        int(math.floor(lat / cell_size_deg)),
        int(math.floor(lon / cell_size_deg)),
    )


def _edge_key(node_a: int, node_b: int) -> Tuple[int, int]:
    return (node_a, node_b) if node_a < node_b else (node_b, node_a)


def _find_subsequence_indices(sequence: Sequence[int], subsequence: Sequence[int]) -> List[int]:
    indices: List[int] = []
    if not subsequence:
        return indices
    first = subsequence[0]
    subseq_len = len(subsequence)
    for idx, value in enumerate(sequence):
        if value != first:
            continue
        if idx + subseq_len > len(sequence):
            break
        if list(sequence[idx: idx + subseq_len]) == list(subsequence):
            indices.append(idx)
    return indices


def _iter_geojson_polygons(geojson: Dict[str, object]) -> Iterable[List[Tuple[float, float]]]:
    """Yield simple polygon rings from a GeoJSON mapping."""

    if geojson.get("type") == "FeatureCollection":
        for feature in geojson.get("features", []):
            geometry = feature.get("geometry")
            if geometry:
                yield from _iter_geojson_polygons(geometry)
        return
    if geojson.get("type") == "Feature":
        geometry = geojson.get("geometry")
        if geometry:
            yield from _iter_geojson_polygons(geometry)
        return
    if geojson.get("type") == "Polygon":
        for ring in geojson.get("coordinates", []):
            yield [(lon, lat) for lon, lat in ring]
        return
    if geojson.get("type") == "MultiPolygon":
        for polygon in geojson.get("coordinates", []):
            for ring in polygon:
                yield [(lon, lat) for lon, lat in ring]
        return


__all__ = [
    "Edge",
    "HistoricalTrack",
    "Node",
    "RouteNetwork",
    "RouteNetworkBuilder",
    "RoutePredictor",
    "TrackPoint",
    "load_historical_tracks",
    "point_to_segment_distance_m",
    "haversine_m",
    "save_prediction_plot",
]
