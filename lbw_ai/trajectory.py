from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

Point = Tuple[int, int]


@dataclass
class PolynomialTrajectory:
    coeffs_x: np.ndarray  # time -> x
    coeffs_y: np.ndarray  # time -> y

    def predict(self, num_future: int, dt: float = 1.0) -> List[Point]:
        n_hist = max(len(self.coeffs_x) - 1, 1)
        t0 = 0
        t_future = np.arange(1, num_future + 1) * dt
        x_future = np.polyval(self.coeffs_x, t_future + t0)
        y_future = np.polyval(self.coeffs_y, t_future + t0)
        return [(int(x), int(y)) for x, y in zip(x_future, y_future)]


def fit_polynomial_trajectory(points: List[Point], degree: int = 2) -> PolynomialTrajectory:
    if len(points) < degree + 1:
        degree = max(1, len(points) - 1)
    t = np.arange(len(points), dtype=float)
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    coeffs_x = np.polyfit(t, x, deg=degree)
    coeffs_y = np.polyfit(t, y, deg=degree)
    return PolynomialTrajectory(coeffs_x=coeffs_x, coeffs_y=coeffs_y)


