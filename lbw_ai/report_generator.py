"""
PDF Report Generator for LBW Reviews
"""
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

Point = Tuple[int, int]


def generate_pdf_report(
    output_path: str,
    video_name: str,
    decision: str,
    confidence: float,
    track_points: List[Point],
    predicted_points: List[Point],
    bounce_index: Optional[int],
    would_hit_stumps: bool,
    distance_to_stumps_px: float,
    explanation: str,
    frame_width: int = 800,
    frame_height: int = 600,
    stumps_x_left: Optional[int] = None,
    stumps_x_right: Optional[int] = None
):
    """
    Generate a comprehensive PDF report for an LBW review.
    
    Args:
        output_path: Path to save the PDF
        video_name: Name of the video file
        decision: OUT or NOT OUT
        confidence: Model confidence score
        track_points: Actual tracked ball positions
        predicted_points: Predicted future positions
        bounce_index: Index where bounce occurred
        would_hit_stumps: Whether ball would hit stumps
        distance_to_stumps_px: Distance to stumps in pixels
        explanation: Text explanation of the decision
        frame_width: Video frame width
        frame_height: Video frame height
        stumps_x_left: Left boundary of stumps region
        stumps_x_right: Right boundary of stumps region
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with PdfPages(output_path) as pdf:
        # Page 1: Title and Summary
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, "AI Third Umpire - LBW Decision Report", 
                ha='center', va='top', fontsize=24, fontweight='bold',
                transform=ax.transAxes)
        
        # Date and time
        ax.text(0.5, 0.88, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ha='center', va='top', fontsize=12, style='italic',
                transform=ax.transAxes)
        
        # Video info
        ax.text(0.1, 0.75, f"Video: {video_name}", 
                ha='left', va='top', fontsize=14, transform=ax.transAxes)
        
        # Decision box
        decision_color = 'red' if decision == "OUT" else 'green'
        decision_box = patches.FancyBboxPatch(
            (0.3, 0.55), 0.4, 0.15,
            boxstyle="round,pad=0.02", 
            edgecolor=decision_color, 
            facecolor=decision_color, 
            alpha=0.2,
            linewidth=3,
            transform=ax.transAxes
        )
        ax.add_patch(decision_box)
        ax.text(0.5, 0.65, f"DECISION: {decision}", 
                ha='center', va='center', fontsize=28, fontweight='bold',
                color=decision_color, transform=ax.transAxes)
        ax.text(0.5, 0.58, f"Confidence: {confidence:.1%}", 
                ha='center', va='center', fontsize=16,
                transform=ax.transAxes)
        
        # Key metrics
        metrics_y = 0.40
        metrics = [
            f"Would Hit Stumps: {'Yes' if would_hit_stumps else 'No'}",
            f"Distance to Stumps: {distance_to_stumps_px:.1f} pixels",
            f"Track Points: {len(track_points)}",
            f"Predicted Points: {len(predicted_points)}",
            f"Bounce Frame: {bounce_index if bounce_index is not None else 'N/A'}"
        ]
        
        for i, metric in enumerate(metrics):
            ax.text(0.1, metrics_y - i * 0.06, metric, 
                   ha='left', va='top', fontsize=12, transform=ax.transAxes)
        
        # Explanation
        ax.text(0.1, 0.15, "Explanation:", 
               ha='left', va='top', fontsize=14, fontweight='bold',
               transform=ax.transAxes)
        ax.text(0.1, 0.10, explanation, 
               ha='left', va='top', fontsize=10, wrap=True,
               transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.3))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Trajectory Visualization
        if track_points:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.set_facecolor('white')
            
            # Extract coordinates
            track_x = [p[0] for p in track_points]
            track_y = [p[1] for p in track_points]
            pred_x = [p[0] for p in predicted_points] if predicted_points else []
            pred_y = [p[1] for p in predicted_points] if predicted_points else []
            
            # Draw pitch outline
            pitch_width = frame_width * 0.15
            pitch_height = frame_height * 0.6
            pitch_x = (frame_width - pitch_width) / 2
            pitch_y = frame_height * 0.2
            
            pitch_rect = patches.FancyBboxPatch(
                (pitch_x, pitch_y), pitch_width, pitch_height,
                boxstyle="round,pad=2", linewidth=3,
                edgecolor='green', facecolor='lightgreen', alpha=0.3
            )
            ax.add_patch(pitch_rect)
            
            # Stumps region
            if stumps_x_left is not None and stumps_x_right is not None:
                stumps_width = stumps_x_right - stumps_x_left
                stumps_rect = patches.FancyBboxPatch(
                    (stumps_x_left, pitch_y), stumps_width, pitch_height,
                    boxstyle="round,pad=1", linewidth=2,
                    edgecolor='yellow', facecolor='yellow', alpha=0.5
                )
                ax.add_patch(stumps_rect)
            
            # Plot trajectory
            ax.plot(track_x, track_y, 'o-', color='blue', linewidth=3, 
                   markersize=6, label='Actual Ball Path', alpha=0.8)
            
            if bounce_index is not None and 0 <= bounce_index < len(track_points):
                ax.plot(track_x[bounce_index], track_y[bounce_index], 's',
                       color='red', markersize=12, label='Bounce Point',
                       markeredgecolor='darkred', markeredgewidth=2)
            
            if pred_x and pred_y:
                ax.plot(pred_x, pred_y, '--', color='orange', linewidth=2,
                       label='Predicted Path', alpha=0.7)
                ax.plot(pred_x[-1], pred_y[-1], '^',
                       color='red', markersize=10, label='Predicted Impact')
            
            ax.set_xlim(0, frame_width)
            ax.set_ylim(0, frame_height)
            ax.invert_yaxis()
            ax.set_xlabel('X Position (pixels)', fontsize=12)
            ax.set_ylabel('Y Position (pixels)', fontsize=12)
            ax.set_title('Ball Trajectory Analysis', fontsize=16, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    return output_path

