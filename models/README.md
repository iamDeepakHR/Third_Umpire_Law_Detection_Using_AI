# Models Directory

This directory contains machine learning model files used by the LBW Detection System.

## Missing Model Files

### 1. `lbw_xgb.json`
- **Type**: XGBoost Classifier Model
- **Purpose**: LBW decision classification
- **Status**: **Not included in repository** (optional)
- **Current Behavior**: System uses rule-based fallback classifier

### 2. LSTM Model (Trajectory Prediction)
- **Type**: LSTM Neural Network Model (PyTorch)
- **Purpose**: Advanced trajectory prediction for ball path
- **Status**: **Implemented** - Model file not included (optional)
- **File**: `models/lstm_trajectory.pth`
- **Current Behavior**: System uses LSTM if model file exists, falls back to polynomial fitting

## Current Implementation Status

### XGBoost Classifier Fallback

If `lbw_xgb.json` is not present, the system will automatically use a **rule-based fallback classifier** that works based on heuristics:

- Pitched in-line: +0.4 / -0.2
- Impact in-line: +0.6 / -0.4  
- Would hit stumps: +0.8 / -0.6
- Distance penalty: -0.3 (scaled by distance/200)

The system will function normally without this file, but using a trained XGBoost model will provide more accurate predictions.

### Trajectory Prediction (LSTM)

The system now supports **LSTM-based trajectory prediction** for improved accuracy on complex ball trajectories. If the LSTM model file is not present, it automatically falls back to polynomial curve fitting.

**LSTM Method** (if model available):
- Uses sequence-to-sequence LSTM architecture
- Learns complex trajectory patterns from training data
- Better handling of non-linear ball movements
- Handles bounce effects and complex physics
- Input: Historical ball positions (x, y coordinates)
- Output: Predicted future positions

**Fallback Method**: Polynomial trajectory fitting (degree 2)
- Fits polynomial curves to ball's x and y coordinates over time
- Predicts future trajectory based on fitted coefficients
- Works well for simple parabolic trajectories
- Always available as backup

**Training the LSTM Model**:
```bash
# Train with synthetic data (quick start)
python train_lstm_trajectory.py --synthetic --num-synthetic 1000 --epochs 50

# Train with your own data
python train_lstm_trajectory.py --data-file path/to/trajectories.json --epochs 100

# Custom training parameters
python train_lstm_trajectory.py --synthetic --hidden-size 128 --num-layers 3 --epochs 100
```

See `train_lstm_trajectory.py` for more options.

## Training Your Own Model

To train an XGBoost model for LBW classification:

1. Collect training data with features:
   - `pitched_in_line` (0 or 1)
   - `impact_in_line` (0 or 1)
   - `would_hit_stumps` (0 or 1)
   - `distance_to_stumps_px` (float)

2. Train using XGBoost:
```python
from xgboost import XGBClassifier
import numpy as np

# Load your training data
X_train = np.array([...])  # Features
y_train = np.array([...])  # Labels (0 = NOT OUT, 1 = OUT)

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# Save model
model.save_model('models/lbw_xgb.json')
```

3. Place the saved model file in this directory.

## Model Formats

### XGBoost Model Format
The model should be saved in XGBoost's JSON format (not binary) to be compatible with the classifier.

### LSTM Model Format
The LSTM model is saved in PyTorch format (`.pth`) and includes:
- Model state dictionary
- Normalization parameters (scaler mean and std)
- Model hyperparameters (hidden size, layers, etc.)

The checkpoint format:
```python
{
    'model_state_dict': {...},
    'scaler_mean': np.array([mean_x, mean_y]),
    'scaler_std': np.array([std_x, std_y]),
    'hidden_size': 64,
    'num_layers': 2,
    'sequence_length': 10
}
```

## Summary

| Model | Status | Current Alternative |
|-------|--------|---------------------|
| `lbw_xgb.json` | Missing | Rule-based classifier |
| `lstm_trajectory.pth` | Missing | Polynomial fitting |

**Note**: The system is fully functional with the current fallback methods. Adding trained models will improve accuracy but is not required for basic operation.

