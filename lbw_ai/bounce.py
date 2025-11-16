from __future__ import annotations

from typing import List, Optional, Tuple


Point = Tuple[int, int]


def detect_bounce_point(points: List[Point]) -> Optional[int]:
    if len(points) < 5:
        return None
    # Detect bounce via y-velocity sign change (downwards -> upwards)
    # y increases downward in images; falling means dy > 0, rising means dy < 0
    dy = [points[i+1][1] - points[i][1] for i in range(len(points)-1)]
    for i in range(1, len(dy)):
        if dy[i-1] > 0 and dy[i] < 0:
            return i  # index of point after bounce
    return None


