# 🏏 AI Third Umpire: LBW Detection using Artificial Intelligence

**Tagline**: Automated LBW decisions using YOLOv8, trajectory prediction, and XGBoost—complete with replay visualization and AI explanations.

## Overview

This project builds an AI-powered Third Umpire system for cricket LBW (Leg Before Wicket) decisions. It detects and tracks the cricket ball from video, identifies bounce, predicts trajectory, analyzes pitching/impact zones, estimates if the ball would hit the stumps, and renders a third-umpire style replay with an OUT/NOT OUT decision and an optional natural-language explanation.

## ✨ Features

### Core Functionality
- **Ball Detection & Tracking**: YOLOv8 + centroid tracker for accurate ball detection
- **Bounce Point Detection**: Velocity inflection analysis to identify bounce points
- **Trajectory Prediction**: Polynomial regression for ball path prediction
- **Pitching & Impact Zone Analysis**: Automated zone detection and analysis
- **Stump Hit Prediction**: Estimates if the ball would hit the stumps
- **LBW Decision**: XGBoost model with rule-based fallback
- **Visualization Overlays**: Professional video overlays with trajectory paths
- **AI Explanations**: Optional AI-generated explanations (Gemini API)

### Enhanced Features
- **🎨 Dark Theme UI**: Professional Third Umpire Room aesthetics
- **📊 3D Trajectory Visualization**: Interactive Plotly-based 3D ball path visualization
- **📈 Analytics Dashboard**: Real-time statistics and review history
- **🎬 Video Timeline Scrubber**: Frame-by-frame navigation with key event markers
- **📄 PDF Report Generation**: Comprehensive analysis reports
- **💬 AI Commentary**: Analyst and Commentator modes
- **💾 Session Storage**: Persistent review history and analytics

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Third Umpire for LBW Detection using AI"
```

2. **Create and activate a virtual environment (recommended)**
```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download YOLOv8 model (optional)**
   - The app will automatically download `yolov8n.pt` if not present
   - Or place a custom YOLOv8 weights file in the project root

5. **Run the application**
```bash
streamlit run app.py
```

6. **Open your browser**
   - The app will automatically open in your default browser
   - Or navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## 📖 Usage

### Basic Workflow

1. **Upload a Video**
   - Click "Upload a cricket video" and select a video file (mp4, mov, avi, mkv)
   - Or select a demo video from the samples folder

2. **Configure Settings** (optional)
   - Adjust YOLO weights path if needed
   - Set confidence and IoU thresholds
   - Limit max frames for faster processing
   - Enter Gemini API key for AI explanations (optional)

3. **Run Analysis**
   - Click "🚀 Run LBW Analysis"
   - Wait for the analysis to complete (progress bar will show status)

4. **View Results**
   - **Decision**: OUT or NOT OUT with confidence score
   - **Video Overlay**: Trajectory visualization on video frames
   - **3D Trajectory**: Interactive 3D plot of ball path
   - **Explanations**: Technical and simple explanations
   - **Analytics**: Statistics and review history

### Advanced Features

#### Video Timeline Navigation
- Use the frame slider to scrub through video frames
- View trajectory overlay at any point in the video
- Key events (bounce, impact) are marked on the timeline

#### Analytics Dashboard
- View statistics: total reviews, OUT/NOT OUT percentage, average confidence
- Interactive charts: decision distribution, confidence histogram
- Review history table with all key metrics

#### Generate Reports
- **PDF Report**: Comprehensive multi-page analysis document
- **JSON Report**: Machine-readable analysis data
- **Video Replay**: Download annotated video with trajectory overlay

#### AI Commentary
- **Analyst Mode**: Technical, detailed explanations
- **Commentator Mode**: Engaging, natural language commentary
- Requires Gemini API key (optional)

## 📁 Project Structure

```
.
├── app.py                      # Main Streamlit application
├── setup.py                    # Setup script for dependencies
├── test_integration.py         # Integration tests
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── yolov8n.pt                  # YOLOv8 model weights (auto-downloaded)
├── samples/                    # Sample video files
│   └── *.mp4
├── lbw_ai/                     # Core LBW detection module
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── detector.py            # Ball detection (YOLOv8)
│   ├── tracker.py             # Ball tracking
│   ├── bounce.py              # Bounce detection
│   ├── trajectory.py          # Trajectory prediction
│   ├── impact.py              # Impact zone analysis
│   ├── classifier.py          # LBW decision classifier
│   ├── visualize.py           # 2D visualization
│   ├── visualize_3d.py        # 3D visualization
│   ├── explainer.py           # AI explanation generation
│   ├── analytics.py           # Analytics dashboard
│   └── report_generator.py    # PDF report generation
├── uploads/                    # User uploaded videos (auto-created)
├── outputs/                    # Generated outputs (auto-created)
│   ├── *_overlay.mp4          # Annotated videos
│   └── *_report.pdf           # PDF reports
└── models/                     # ML models (auto-created)
    └── lbw_xgb.json           # XGBoost model (optional)
```

## ⚙️ Configuration

### Environment Variables
- `GEMINI_API_KEY`: Optional Gemini API key for AI explanations (can also be entered in UI)

### Model Paths
- YOLOv8 weights: Default `yolov8n.pt` (auto-downloaded)
- XGBoost model: Default `models/lbw_xgb.json` (rule-based fallback if not found)

### Settings in UI
- **Confidence Threshold**: Detection confidence (default: 0.25)
- **IoU Threshold**: Intersection over Union for NMS (default: 0.45)
- **Max Frames**: Limit frames for faster processing (0 = all frames)

## 🔧 Development

### Running Tests
```bash
python test_integration.py
```

### Setup Script
```bash
python setup.py
```
This will install dependencies and create necessary directories.

## 📝 Notes

- **Accuracy**: Real-world accuracy depends on video quality, camera angle, calibration, and model training
- **Model Training**: For best results, fine-tune YOLOv8 on your cricket dataset and train the XGBoost model with labeled OUT/NOT OUT samples
- **API Keys**: Gemini API key is optional. The app works without it but won't provide AI-generated explanations
- **Video Format**: Supports mp4, mov, avi, mkv formats
- **Performance**: Processing time depends on video length and resolution

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Streamlit for the web framework
- Plotly for 3D visualizations
- Google Gemini for AI explanations

## 📧 Support

For issues, questions, or contributions, please open an issue on the repository.

---

**Enjoy your AI Third Umpire system! 🏏**
