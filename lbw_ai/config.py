from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    yolo_weights_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_frames: int | None = None  # set to limit for faster runs

    # Output
    output_dir: str = "outputs"
    output_video_fps: int = 25

    # Geometry (heuristic; real projects should calibrate camera -> field)
    pitch_y_top_ratio: float = 0.35  # rough pitch top in image (0..1)
    pitch_y_bottom_ratio: float = 0.85  # rough pitch bottom in image (0..1)
    stumps_x_center_ratio: float = 0.5  # center line in image (0..1)
    stumps_width_ratio: float = 0.08  # relative width of stump region

    # Classifier
    xgb_model_path: str = "models/lbw_xgb.json"
    
    # Trajectory Prediction
    lstm_model_path: str = "models/lstm_trajectory.pth"
    use_lstm_trajectory: bool = True  # Use LSTM if available, fallback to polynomial

    # Explainer
    use_ai_explainer: bool = True

    def ensure_output(self) -> Path:
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out


