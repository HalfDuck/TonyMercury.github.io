"""Command line entry point for maritime route prediction.

Usage example::

    python predict_route_cli.py \
        --history scene_coast_patrol.csv \
        --observed observed_track.csv \
        --destination 23.75 132.70 \
        --output prediction.png

The observed CSV must at least contain ``time``, ``lat`` and ``lon`` columns.
Additional optional columns ``mbc`` (course) and ``mbv`` (speed) are used when
available.  The script will print a JSON report to stdout describing whether a
prediction succeeded as well as the supporting ship types found in the history
library.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import List

import pandas as pd

from route_prediction import (
    RouteNetworkBuilder,
    RoutePredictor,
    TrackPoint,
    load_historical_tracks,
    save_prediction_plot,
)


def _load_observed(path: str | pathlib.Path) -> List[TrackPoint]:
    df = pd.read_csv(path)
    if not {"time", "lat", "lon"}.issubset(df.columns):
        raise ValueError("观测航迹缺少 time/lat/lon 字段。")
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        raise ValueError("存在无法解析的观测时间戳。")
    df["mbc"] = pd.to_numeric(df.get("mbc"), errors="coerce")
    df["mbv"] = pd.to_numeric(df.get("mbv"), errors="coerce")
    points: List[TrackPoint] = []
    for row in df.sort_values("time").itertuples():
        points.append(
            TrackPoint(
                time=row.time,
                lon=float(row.lon),
                lat=float(row.lat),
                course=float(row.mbc) if row.mbc == row.mbc else None,
                speed=float(row.mbv) if row.mbv == row.mbv else None,
            )
        )
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Maritime route prediction demo")
    parser.add_argument("--history", required=True, help="历史航迹库 CSV 文件")
    parser.add_argument("--observed", required=True, help="目标船当前观测航迹 CSV")
    parser.add_argument("--destination", nargs=2, type=float, metavar=("LAT", "LON"), required=True)
    parser.add_argument("--output", default="prediction.png", help="可视化输出路径")
    parser.add_argument("--land-mask", help="可选 GeoJSON 陆地区域蒙版")
    parser.add_argument("--tolerance", type=float, default=500.0, help="轨迹合并容差（米）")
    parser.add_argument("--dest-tolerance", type=float, default=1_000.0, help="终点匹配容差（米）")
    args = parser.parse_args()

    history = load_historical_tracks(args.history)
    builder = RouteNetworkBuilder(tolerance_m=args.tolerance)
    network = builder.build_from_dataframe(history)

    observed_points = _load_observed(args.observed)
    predictor = RoutePredictor(network, destination_tolerance_m=args.dest_tolerance)
    result = predictor.predict(observed_points, destination=(args.destination[0], args.destination[1]))

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result.get("status") == "ok":
        matched_track_id = result["matched_track"]["track_id"]
        history_subset = [track for track in network.tracks if track.track_id == matched_track_id]
    else:
        history_subset = []
    save_prediction_plot(
        history_tracks=history_subset,
        observed_points=observed_points,
        predicted_points=result.get("predicted_points", []),
        destination=(args.destination[0], args.destination[1]),
        output_path=args.output,
        land_mask_geojson=args.land_mask,
    )


if __name__ == "__main__":
    main()
