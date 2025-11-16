"""
3D Ball Trajectory Visualization using Plotly
"""
from typing import List, Tuple, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

Point = Tuple[int, int]


def create_3d_trajectory_plot(
    track: List[Point],
    predicted: List[Point],
    bounce_index: Optional[int] = None,
    frame_height: int = 600,
    frame_width: int = 800,
    decision: Optional[str] = None,
    confidence: Optional[float] = None
) -> go.Figure:
    """
    Create an interactive 3D trajectory visualization of the ball path.
    
    Args:
        track: List of actual tracked ball positions (x, y)
        predicted: List of predicted future positions (x, y)
        bounce_index: Index in track where bounce occurred
        frame_height: Video frame height
        frame_width: Video frame width
        decision: OUT or NOT OUT
        confidence: Model confidence score
        
    Returns:
        Plotly figure object
    """
    # Convert 2D points to 3D (add time as Z-axis)
    track_3d = []
    pred_3d = []
    
    # Actual track with time dimension
    for i, (x, y) in enumerate(track):
        # Normalize coordinates to 0-1 range for better visualization
        x_norm = x / frame_width
        y_norm = y / frame_height
        track_3d.append([x_norm, y_norm, i])
    
    # Predicted path with time dimension (continuing from track)
    start_time = len(track)
    for i, (x, y) in enumerate(predicted):
        x_norm = x / frame_width
        y_norm = y / frame_height
        pred_3d.append([x_norm, y_norm, start_time + i])
    
    if not track_3d:
        # Return empty figure if no data
        fig = go.Figure()
        fig.add_annotation(text="No trajectory data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Plot actual trajectory
    track_array = np.array(track_3d)
    fig.add_trace(go.Scatter3d(
        x=track_array[:, 0],
        y=track_array[:, 1],
        z=track_array[:, 2],
        mode='lines+markers',
        name='Actual Ball Path',
        line=dict(color='blue', width=6),
        marker=dict(size=4, color='blue'),
        hovertemplate='<b>Actual Path</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Frame: %{z}<extra></extra>'
    ))
    
    # Mark bounce point
    if bounce_index is not None and 0 <= bounce_index < len(track_3d):
        bounce_pt = track_3d[bounce_index]
        fig.add_trace(go.Scatter3d(
            x=[bounce_pt[0]],
            y=[bounce_pt[1]],
            z=[bounce_pt[2]],
            mode='markers',
            name='Bounce Point',
            marker=dict(size=12, color='red', symbol='diamond'),
            hovertemplate='<b>Bounce Point</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Frame: %{z}<extra></extra>'
        ))
    
    # Plot predicted trajectory
    if pred_3d:
        pred_array = np.array(pred_3d)
        fig.add_trace(go.Scatter3d(
            x=pred_array[:, 0],
            y=pred_array[:, 1],
            z=pred_array[:, 2],
            mode='lines+markers',
            name='Predicted Path',
            line=dict(color='orange', width=4, dash='dash'),
            marker=dict(size=3, color='orange'),
            hovertemplate='<b>Predicted Path</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Frame: %{z}<extra></extra>'
        ))
        
        # Mark predicted impact point
        if pred_3d:
            impact_pt = pred_3d[-1]
            fig.add_trace(go.Scatter3d(
                x=[impact_pt[0]],
                y=[impact_pt[1]],
                z=[impact_pt[2]],
                mode='markers',
                name='Predicted Impact',
                marker=dict(size=10, color='darkred', symbol='x'),
                hovertemplate='<b>Predicted Impact</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Frame: %{z}<extra></extra>'
            ))
    
    # Add stumps region (vertical line at center)
    stumps_x = 0.5  # Normalized center
    stumps_z = np.linspace(0, len(track_3d) + len(pred_3d), 50)
    stumps_y = np.linspace(0.2, 0.8, 50)
    fig.add_trace(go.Scatter3d(
        x=[stumps_x] * len(stumps_z),
        y=[0.5] * len(stumps_z),
        z=stumps_z,
        mode='lines',
        name='Stumps Line',
        line=dict(color='yellow', width=8),
        showlegend=True
    ))
    
    # Update layout
    title_text = "3D Ball Trajectory Analysis"
    if decision:
        decision_color = "red" if decision == "OUT" else "green"
        title_text += f"<br><span style='color:{decision_color}; font-size:16px'>Decision: {decision}"
        if confidence is not None:
            title_text += f" (Confidence: {confidence:.1%})</span>"
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=18)),
        scene=dict(
            xaxis_title="X Position (normalized)",
            yaxis_title="Y Position (normalized)",
            zaxis_title="Frame Number (Time)",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0)
            ),
            bgcolor='rgb(20, 20, 30)',
            aspectmode='cube'
        ),
        width=900,
        height=700,
        template='plotly_dark',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)')
    )
    
    return fig


def create_trajectory_comparison_plot(
    track: List[Point],
    predicted: List[Point],
    frame_height: int = 600,
    frame_width: int = 800,
    decision: Optional[str] = None
) -> go.Figure:
    """
    Create a side-by-side comparison of actual vs predicted trajectory.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Actual Trajectory', 'Predicted Trajectory'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Actual trajectory
    if track:
        track_x = [p[0] for p in track]
        track_y = [p[1] for p in track]
        fig.add_trace(
            go.Scatter(x=track_x, y=track_y, mode='lines+markers', name='Actual Path',
                      line=dict(color='blue', width=3), marker=dict(size=5)),
            row=1, col=1
        )
    
    # Predicted trajectory
    if predicted:
        pred_x = [p[0] for p in predicted]
        pred_y = [p[1] for p in predicted]
        fig.add_trace(
            go.Scatter(x=pred_x, y=pred_y, mode='lines+markers', name='Predicted Path',
                      line=dict(color='orange', width=3, dash='dash'), marker=dict(size=5)),
            row=1, col=2
        )
    
    # Update axes
    for col in [1, 2]:
        fig.update_xaxes(title_text="X Position (pixels)", range=[0, frame_width], row=1, col=col)
        fig.update_yaxes(title_text="Y Position (pixels)", range=[0, frame_height], row=1, col=col, autorange='reversed')
    
    title = "Trajectory Comparison"
    if decision:
        title += f" - Decision: {decision}"
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=500,
        showlegend=True
    )
    
    return fig

