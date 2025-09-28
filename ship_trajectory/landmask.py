"""GeoJSON-based land avoidance utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from shapely.geometry import Point, shape
from shapely.strtree import STRtree


class LandMask:
    """Spatial index for fast land collision checks."""

    def __init__(self, geojson_path: Path) -> None:
        geojson = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
        features = geojson.get("features", [])
        self.polygons = [shape(feature["geometry"]) for feature in features]
        self.index = STRtree(self.polygons)

    def intersects(self, lon: float, lat: float) -> bool:
        point = Point(lon, lat)
        for polygon in self.index.query(point):
            if polygon.contains(point):
                return True
        return False

    def batch_intersects(self, coordinates: Iterable[Tuple[float, float]]) -> List[bool]:
        return [self.intersects(lon, lat) for lon, lat in coordinates]

    def distance_to_land(self, lon: float, lat: float) -> float:
        point = Point(lon, lat)
        distances = [polygon.exterior.distance(point) for polygon in self.polygons]
        if not distances:
            return float("inf")
        return min(distances)

