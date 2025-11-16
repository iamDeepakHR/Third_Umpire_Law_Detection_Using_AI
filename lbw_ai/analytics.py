"""
Analytics and Statistics Module for LBW Decision System
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path
import numpy as np


@dataclass
class ReviewRecord:
    """Single review record"""
    timestamp: str
    video_name: str
    decision: str
    confidence: float
    would_hit_stumps: bool
    distance_to_stumps_px: float
    num_track_points: int
    bounce_index: Optional[int] = None


class AnalyticsDashboard:
    """Manages analytics and statistics for LBW reviews"""
    
    def __init__(self, storage_path: str = "analytics/reviews.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.reviews: List[ReviewRecord] = []
        self.load_reviews()
    
    def add_review(self, record: ReviewRecord):
        """Add a new review record"""
        self.reviews.append(record)
        self.save_reviews()
    
    def load_reviews(self):
        """Load review history from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.reviews = [ReviewRecord(**r) for r in data]
            except Exception:
                self.reviews = []
        else:
            self.reviews = []
    
    def save_reviews(self):
        """Save review history to storage"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump([asdict(r) for r in self.reviews], f, indent=2)
        except Exception:
            pass
    
    def get_statistics(self) -> Dict:
        """Calculate comprehensive statistics"""
        if not self.reviews:
            return {
                "total_reviews": 0,
                "out_count": 0,
                "not_out_count": 0,
                "out_percentage": 0.0,
                "avg_confidence": 0.0,
                "avg_confidence_out": 0.0,
                "avg_confidence_not_out": 0.0,
                "hitting_stumps_percentage": 0.0,
                "avg_distance_to_stumps": 0.0,
                "avg_track_points": 0.0
            }
        
        total = len(self.reviews)
        out_count = sum(1 for r in self.reviews if r.decision == "OUT")
        not_out_count = total - out_count
        
        confidences = [r.confidence for r in self.reviews]
        confidences_out = [r.confidence for r in self.reviews if r.decision == "OUT"]
        confidences_not_out = [r.confidence for r in self.reviews if r.decision == "NOT OUT"]
        
        hitting_stumps = sum(1 for r in self.reviews if r.would_hit_stumps)
        distances = [r.distance_to_stumps_px for r in self.reviews]
        track_points = [r.num_track_points for r in self.reviews]
        
        return {
            "total_reviews": total,
            "out_count": out_count,
            "not_out_count": not_out_count,
            "out_percentage": (out_count / total * 100) if total > 0 else 0.0,
            "avg_confidence": np.mean(confidences) if confidences else 0.0,
            "avg_confidence_out": np.mean(confidences_out) if confidences_out else 0.0,
            "avg_confidence_not_out": np.mean(confidences_not_out) if confidences_not_out else 0.0,
            "hitting_stumps_percentage": (hitting_stumps / total * 100) if total > 0 else 0.0,
            "avg_distance_to_stumps": np.mean(distances) if distances else 0.0,
            "avg_track_points": np.mean(track_points) if track_points else 0.0
        }
    
    def get_recent_reviews(self, limit: int = 10) -> List[ReviewRecord]:
        """Get most recent reviews"""
        return self.reviews[-limit:] if self.reviews else []
    
    def clear_history(self):
        """Clear all review history"""
        self.reviews = []
        self.save_reviews()


def create_review_record(
    video_name: str,
    decision: str,
    confidence: float,
    would_hit_stumps: bool,
    distance_to_stumps_px: float,
    num_track_points: int,
    bounce_index: Optional[int] = None
) -> ReviewRecord:
    """Helper function to create a review record"""
    return ReviewRecord(
        timestamp=datetime.now().isoformat(),
        video_name=video_name,
        decision=decision,
        confidence=confidence,
        would_hit_stumps=would_hit_stumps,
        distance_to_stumps_px=distance_to_stumps_px,
        num_track_points=num_track_points,
        bounce_index=bounce_index
    )

