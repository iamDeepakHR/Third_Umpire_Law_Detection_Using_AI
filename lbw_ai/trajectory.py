from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

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


class LSTMTrajectoryPredictor(nn.Module if HAS_TORCH else object):
    """
    LSTM-based trajectory predictor for ball path prediction.
    Uses sequence-to-sequence architecture to predict future positions.
    """
    def __init__(self, input_size: int = 2, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for LSTM trajectory prediction")
        super(LSTMTrajectoryPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor of shape (batch, seq_len, 2) - (x, y) coordinates
            
        Returns:
            Predicted next position of shape (batch, 2)
        """
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction
    
    def predict_sequence(self, sequence: torch.Tensor, num_future: int) -> torch.Tensor:
        """
        Predict a sequence of future positions.
        
        Args:
            sequence: Input sequence of shape (1, seq_len, 2)
            num_future: Number of future positions to predict
            
        Returns:
            Predicted sequence of shape (num_future, 2)
        """
        self.eval()
        predictions = []
        current_seq = sequence.clone()
        
        with torch.no_grad():
            for _ in range(num_future):
                # Predict next position
                next_pos = self.forward(current_seq)
                predictions.append(next_pos)
                
                # Update sequence: remove first, add predicted
                current_seq = torch.cat([current_seq[:, 1:, :], next_pos.unsqueeze(1)], dim=1)
        
        return torch.cat(predictions, dim=0)


class LSTMTrajectory:
    """
    LSTM-based trajectory predictor wrapper.
    """
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, points: Optional[List[Point]] = None):
        self.model_path = model_path
        self.model: Optional[LSTMTrajectoryPredictor] = None
        self.device = device or ("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu")
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.points: Optional[List[Point]] = points  # Store points for compatibility
        
        if HAS_TORCH and model_path and Path(model_path).exists():
            try:
                self.model = LSTMTrajectoryPredictor()
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        self.scaler_mean = checkpoint.get('scaler_mean', None)
                        self.scaler_std = checkpoint.get('scaler_std', None)
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"Warning: Could not load LSTM model from {model_path}: {e}")
                self.model = None
    
    def _normalize(self, points: List[Point]) -> np.ndarray:
        """Normalize points for LSTM input."""
        arr = np.array(points, dtype=np.float32)
        if self.scaler_mean is not None and self.scaler_std is not None:
            arr = (arr - self.scaler_mean) / (self.scaler_std + 1e-8)
        return arr
    
    def _denormalize(self, arr: np.ndarray) -> np.ndarray:
        """Denormalize LSTM output."""
        if self.scaler_mean is not None and self.scaler_std is not None:
            arr = arr * self.scaler_std + self.scaler_mean
        return arr
    
    def predict(self, num_future: int, dt: float = 1.0, points: Optional[List[Point]] = None) -> List[Point]:
        """
        Predict future trajectory points using LSTM.
        
        Args:
            num_future: Number of future points to predict
            dt: Time step (not used in LSTM, kept for compatibility)
            points: Historical trajectory points (optional, uses stored points if not provided)
            
        Returns:
            List of predicted future points
        """
        # Use provided points or stored points
        if points is None:
            if self.points is None:
                raise ValueError("No points provided. Either pass points to predict() or initialize LSTMTrajectory with points.")
            points = self.points
        
        if self.model is None or len(points) < 3:
            # Fallback to polynomial if no model or insufficient data
            return fit_polynomial_trajectory(points).predict(num_future, dt)
        
        try:
            # Normalize input
            normalized = self._normalize(points)
            
            # Convert to tensor: (1, seq_len, 2)
            input_tensor = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)
            
            # Predict sequence
            predictions = self.model.predict_sequence(input_tensor, num_future)
            
            # Convert to numpy and denormalize
            pred_array = predictions.cpu().numpy()
            pred_array = self._denormalize(pred_array)
            
            # Convert to list of points
            return [(int(x), int(y)) for x, y in pred_array]
        except Exception as e:
            print(f"Warning: LSTM prediction failed: {e}, falling back to polynomial")
            return fit_polynomial_trajectory(points).predict(num_future, dt)


def fit_polynomial_trajectory(points: List[Point], degree: int = 2) -> PolynomialTrajectory:
    """Fit polynomial trajectory (fallback method)."""
    if len(points) < degree + 1:
        degree = max(1, len(points) - 1)
    t = np.arange(len(points), dtype=float)
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    coeffs_x = np.polyfit(t, x, deg=degree)
    coeffs_y = np.polyfit(t, y, deg=degree)
    return PolynomialTrajectory(coeffs_x=coeffs_x, coeffs_y=coeffs_y)


def fit_trajectory(points: List[Point], model_path: Optional[str] = None, use_lstm: bool = True) -> LSTMTrajectory | PolynomialTrajectory:
    """
    Fit trajectory using LSTM if available, otherwise use polynomial.
    
    Args:
        points: Historical trajectory points
        model_path: Path to LSTM model file
        use_lstm: Whether to attempt using LSTM
        
    Returns:
        LSTMTrajectory or PolynomialTrajectory object
    """
    if use_lstm and HAS_TORCH and model_path:
        lstm = LSTMTrajectory(model_path=model_path, points=points)
        if lstm.model is not None:
            return lstm
    
    # Fallback to polynomial
    return fit_polynomial_trajectory(points)


