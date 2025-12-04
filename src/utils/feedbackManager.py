"""
Feedback Manager - Manages exercise feedback and error tracking.
Exercise-agnostic feedback system for any workout type.
"""

from collections import deque
from datetime import datetime


class FeedbackManager:
    """Manages feedback messages and form error tracking for any exercise."""
    
    def __init__(self, max_feedback_items=5):
        """
        Initialize feedback manager.
        
        Args:
            max_feedback_items (int): Maximum feedback items to display
        """
        self.max_items = max_feedback_items
        self.feedback_queue = deque(maxlen=max_feedback_items)
        self.current_issues = set()
        self.issue_history = []
        self.issue_counts = {}
    
    def add_feedback(self, message, severity='info'):
        """
        Add feedback message with timestamp.
        
        Args:
            message (str): Feedback message
            severity (str): 'info', 'warning', 'error', 'success'
        """
        timestamp = datetime.now()
        self.feedback_queue.append({
            'message': message,
            'severity': severity,
            'timestamp': timestamp
        })
    
    def add_issue(self, issue_name):
        """
        Track a form issue (e.g., 'elbow_flare', 'knee_valgus').
        
        Args:
            issue_name (str): Name of the form issue
        """
        if issue_name not in self.current_issues:
            self.current_issues.add(issue_name)
            self.issue_history.append({
                'issue': issue_name,
                'timestamp': datetime.now()
            })
            
            # Update counts
            self.issue_counts[issue_name] = self.issue_counts.get(issue_name, 0) + 1
    
    def clear_issue(self, issue_name):
        """
        Clear a specific issue when form is corrected.
        
        Args:
            issue_name (str): Issue to clear
        """
        self.current_issues.discard(issue_name)
    
    def clear_all_issues(self):
        """Clear all current issues."""
        self.current_issues.clear()
    
    def has_issues(self):
        """
        Check if there are any current form issues.
        
        Returns:
            bool: True if issues exist
        """
        return len(self.current_issues) > 0
    
    def get_feedback_list(self):
        """
        Get list of recent feedback messages.
        
        Returns:
            list: Recent feedback items
        """
        return list(self.feedback_queue)
    
    def get_current_issues(self):
        """
        Get set of current active issues.
        
        Returns:
            set: Current issues
        """
        return self.current_issues.copy()
    
    def get_issue_summary(self):
        """
        Get comprehensive summary of all issues encountered.
        
        Returns:
            dict: Issue analytics
        """
        return {
            'total_unique_issues': len(self.issue_counts),
            'most_common_issue': max(self.issue_counts.items(), key=lambda x: x[1])[0] if self.issue_counts else None,
            'issue_counts': self.issue_counts.copy(),
            'total_issues': sum(self.issue_counts.values())
        }
    
    def clear_feedback(self):
        """Clear all feedback messages."""
        self.feedback_queue.clear()
    
    def reset(self):
        """Reset all feedback and issue tracking."""
        self.feedback_queue.clear()
        self.current_issues.clear()
        self.issue_history.clear()
        self.issue_counts.clear()
    
    def get_issue_count(self, issue_name):
        """
        Get count of how many times a specific issue occurred.
        
        Args:
            issue_name (str): Issue to query
            
        Returns:
            int: Count of occurrences
        """
        return self.issue_counts.get(issue_name, 0)
