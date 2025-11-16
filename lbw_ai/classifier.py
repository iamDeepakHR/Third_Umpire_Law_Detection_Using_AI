from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:  # ImportError or other environment issues
    XGBClassifier = None  # type: ignore
    HAS_XGB = False


@dataclass
class LBWFeatures:
    pitched_in_line: int
    impact_in_line: int
    would_hit_stumps: int
    distance_to_stumps_px: float

    def as_array(self) -> np.ndarray:
        return np.array([[self.pitched_in_line, self.impact_in_line, self.would_hit_stumps, self.distance_to_stumps_px]], dtype=float)


class LBWClassifier:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model: Optional["XGBClassifier"] = None
        if HAS_XGB and Path(model_path).exists():
            try:
                self.model = XGBClassifier()
                self.model.load_model(model_path)
            except Exception:
                self.model = None

    def predict(self, features: LBWFeatures) -> Dict[str, float | str]:
        # Rule-based fallback if no model
        if self.model is None:
            score = 0
            score += 0.4 if features.pitched_in_line else -0.2
            score += 0.6 if features.impact_in_line else -0.4
            score += 0.8 if features.would_hit_stumps else -0.6
            score -= min(features.distance_to_stumps_px / 200.0, 1.0) * 0.3
            label = "OUT" if score >= 0.5 else "NOT OUT"
            prob = max(min((score + 1) / 2, 0.99), 0.01)
            return {"label": label, "probability": float(prob)}

        proba = float(self.model.predict_proba(features.as_array())[0, 1])  # type: ignore[union-attr]
        label = "OUT" if proba >= 0.5 else "NOT OUT"
        return {"label": label, "probability": proba}


