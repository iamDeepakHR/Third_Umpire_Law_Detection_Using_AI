from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple


Point = Tuple[int, int]


@dataclass
class TrackState:
    history: Deque[Point]
    max_history: int = 50

    def add(self, point: Point) -> None:
        self.history.append(point)
        while len(self.history) > self.max_history:
            self.history.popleft()

    def as_list(self) -> List[Point]:
        return list(self.history)


class SimpleBallTracker:
    def __init__(self, max_history: int = 50) -> None:
        self.state = TrackState(history=deque(), max_history=max_history)

    def update(self, detections: List[Point]) -> Optional[Point]:
        # Choose detection nearest to last point; otherwise pick first
        if not detections:
            return None
        last = self.state.history[-1] if len(self.state.history) > 0 else None
        best: Point
        if last is None:
            best = detections[0]
        else:
            best = min(detections, key=lambda p: (p[0]-last[0])**2 + (p[1]-last[1])**2)
        self.state.add(best)
        return best

    def get_track(self) -> List[Point]:
        return self.state.as_list()


