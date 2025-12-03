"""
Session Tracker - Manages session analytics and data tracking.
"""

import time
import numpy as np
from collections import Counter


class SessionTracker:
    """Tracks exercise session analytics and metrics."""
    
    def __init__(self, exercise_name):
        """
        Initialize session tracker.
        
        Args:
            exercise_name (str): Name of the exercise being tracked
        """
        self.exercise_name = exercise_name
        self.reset()
    
    def reset(self):
        """Reset all session data."""
        self.data = {
            'total_reps': 0,
            'form_scores': [],
            'common_errors': Counter(),
            'start_time': time.time(),
            'exercise_name': self.exercise_name
        }
    
    def add_rep(self, form_score=None):
        """
        Increment rep count.
        
        Args:
            form_score (float, optional): Form score for this rep (0-100)
        """
        self.data['total_reps'] += 1
        if form_score is not None:
            self.data['form_scores'].append(form_score)
    
    def add_error(self, error_type):
        """
        Record a form error.
        
        Args:
            error_type (str): Type of error that occurred
        """
        self.data['common_errors'][error_type] += 1
    
    def get_total_reps(self):
        """Get total rep count."""
        return self.data['total_reps']
    
    def get_average_form_score(self):
        """
        Calculate average form score.
        
        Returns:
            float: Average form score (0-100) or 0 if no scores
        """
        if self.data['form_scores']:
            return np.mean(self.data['form_scores'])
        return 0.0
    
    def get_duration(self):
        """
        Get session duration in seconds.
        
        Returns:
            float: Duration in seconds
        """
        return time.time() - self.data['start_time']
    
    def get_summary(self, exercise_specific_data=None):
        """
        Generate session summary.
        
        Args:
            exercise_specific_data (dict, optional): Additional exercise-specific metrics
            
        Returns:
            dict: Complete session summary
        """
        summary = {
            'exercise': self.exercise_name,
            'total_reps': self.data['total_reps'],
            'duration_seconds': self.get_duration(),
            'average_form_score': self.get_average_form_score(),
            'common_errors': dict(self.data['common_errors']),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Merge exercise-specific data
        if exercise_specific_data:
            summary.update(exercise_specific_data)
        
        return summary
    
    def print_summary(self, summary=None):
        """
        Print session summary to console.
        
        Args:
            summary (dict, optional): Pre-generated summary. If None, generates new one.
        """
        if summary is None:
            summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print(f"{self.exercise_name.upper()} SESSION SUMMARY")
        print("=" * 60)
        print(f"Total Reps: {summary['total_reps']}")
        print(f"Duration: {summary['duration_seconds']:.0f} seconds")
        print(f"Average Form Score: {summary['average_form_score']:.1f}%")
        print("\nMost Common Errors:")
        for error, count in Counter(summary['common_errors']).most_common(3):
            print(f"  • {error}: {count} times")
        print("=" * 60)
    
    def has_data(self):
        """
        Check if session has any recorded data.
        
        Returns:
            bool: True if session has reps recorded
        """
        return self.data['total_reps'] > 0
