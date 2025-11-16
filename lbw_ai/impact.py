from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


Point = Tuple[int, int]


@dataclass
class StumpsRegion:
    x_left: int
    x_right: int


def estimate_stumps_region(frame_width: int, center_ratio: float, width_ratio: float) -> StumpsRegion:
    cx = int(center_ratio * frame_width)
    half_w = int(width_ratio * frame_width / 2)
    return StumpsRegion(x_left=cx - half_w, x_right=cx + half_w)


def predict_stump_intersection(track: List[Point], future: List[Point], stumps: StumpsRegion) -> Optional[Point]:
    path = track + future
    for p in path:
        if stumps.x_left <= p[0] <= stumps.x_right:
            return p
    return None


def pitch_zone_from_point(point: Point, frame_height: int, pitch_top_ratio: float, pitch_bottom_ratio: float) -> str:
    y_top = int(pitch_top_ratio * frame_height)
    y_bottom = int(pitch_bottom_ratio * frame_height)
    if point[1] < y_top:
        return "pre-pitch"
    if point[1] > y_bottom:
        return "post-pitch"
    return "in-line"


