#!/usr/bin/env python3
"""
Extract ball trajectories from cricket videos for LSTM training.

This script processes videos to extract ball trajectories (x, y coordinates)
that can be used to train the LSTM trajectory prediction model.
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple
import argparse

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from lbw_ai.detector import BallDetector
from lbw_ai.tracker import SimpleBallTracker

Point = Tuple[int, int]


def extract_trajectory_from_video(
    video_path: str,
    detector: BallDetector,
    tracker: SimpleBallTracker,
    min_trajectory_length: int = 10
) -> List[Point] | None:
    """
    Extract ball trajectory from a single video.
    
    Args:
        video_path: Path to video file
        detector: Ball detector instance
        tracker: Ball tracker instance
        min_trajectory_length: Minimum number of points for valid trajectory
        
    Returns:
        List of (x, y) points or None if trajectory too short
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return None
    
    track_points: List[Point] = []
    frame_count = 0
    
    print(f"Processing {Path(video_path).name}...", end=" ", flush=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect ball in frame
        dets = detector.detect(frame)
        centers = [d.center for d in dets]
        
        # Update tracker
        pt = tracker.update(centers)
        
        if pt is not None:
            track_points.append(pt)
        
        frame_count += 1
    
    cap.release()
    
    # Filter out short trajectories
    if len(track_points) < min_trajectory_length:
        print(f"Skipped (only {len(track_points)} points)")
        return None
    
    print(f"Extracted {len(track_points)} points")
    return track_points


def extract_trajectories_from_directory(
    video_dir: str,
    output_file: str,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    min_trajectory_length: int = 10
) -> None:
    """
    Extract trajectories from all videos in a directory.
    
    Args:
        video_dir: Directory containing video files
        output_file: Output JSON file path
        confidence_threshold: YOLO confidence threshold
        iou_threshold: YOLO IoU threshold
        min_trajectory_length: Minimum trajectory length
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        print(f"Error: Directory {video_dir} does not exist")
        return
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f'*{ext}'))
        video_files.extend(video_dir.glob(f'*{ext.upper()}'))
    
    if not video_files:
        print(f"Error: No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    print(f"Initializing detector and tracker...")
    
    # Initialize detector and tracker
    detector = BallDetector(
        weights_path="yolov8n.pt",
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold
    )
    
    all_trajectories: List[List[Point]] = []
    
    # Process each video
    for video_path in video_files:
        # Create new tracker for each video
        tracker = SimpleBallTracker(max_history=100)
        
        trajectory = extract_trajectory_from_video(
            str(video_path),
            detector,
            tracker,
            min_trajectory_length
        )
        
        if trajectory is not None:
            all_trajectories.append(trajectory)
    
    # Save trajectories to JSON
    if all_trajectories:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        json_data = [[[int(x), int(y)] for x, y in traj] for traj in all_trajectories]
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\n✅ Successfully extracted {len(all_trajectories)} trajectories")
        print(f"📁 Saved to: {output_path}")
        print(f"\nNext step: Train LSTM model with:")
        print(f"  python train_lstm_trajectory.py --data-file {output_file} --epochs 100")
    else:
        print("\n❌ No valid trajectories extracted. Check video quality and detection settings.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ball trajectories from cricket videos for LSTM training"
    )
    parser.add_argument(
        "video_dir",
        type=str,
        help="Directory containing video files (or path to single video file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trajectories.json",
        help="Output JSON file path (default: trajectories.json)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="YOLO IoU threshold (default: 0.45)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum trajectory length (default: 10)"
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video_dir)
    
    # Check if it's a file or directory
    if video_path.is_file():
        # Single video file
        print(f"Processing single video: {video_path.name}")
        detector = BallDetector(
            weights_path="yolov8n.pt",
            confidence_threshold=args.confidence,
            iou_threshold=args.iou
        )
        tracker = SimpleBallTracker(max_history=100)
        
        trajectory = extract_trajectory_from_video(
            str(video_path),
            detector,
            tracker,
            args.min_length
        )
        
        if trajectory is not None:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            json_data = [[[int(x), int(y)] for x, y in trajectory]]
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"\n✅ Trajectory extracted and saved to: {output_path}")
        else:
            print("\n❌ Could not extract valid trajectory")
    elif video_path.is_dir():
        # Directory of videos
        extract_trajectories_from_directory(
            str(video_path),
            args.output,
            args.confidence,
            args.iou,
            args.min_length
        )
    else:
        print(f"Error: {video_path} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()

