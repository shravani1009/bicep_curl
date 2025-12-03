"""
Smoothing utilities for noisy sensor data.
"""

class ExponentialMovingAverage:
    """Smooth out noisy angle measurements using exponential moving average."""
    
    def __init__(self, alpha=0.3):
        """
        Initialize EMA filter.
        
        Args:
            alpha (float): Smoothing factor (0-1). Higher = more responsive, lower = smoother.
        """
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value):
        """
        Update the EMA with a new value.
        
        Args:
            new_value (float): New measurement to incorporate.
            
        Returns:
            float: Smoothed value.
        """
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self):
        """Reset the filter."""
        self.value = None


class ExponentialSmoothing:
    """
    Exponential smoothing filter for angle measurements.
    Alias for ExponentialMovingAverage for compatibility.
    """
    
    def __init__(self, alpha=0.3):
        """
        Initialize exponential smoothing filter.
        
        Args:
            alpha (float): Smoothing factor (0-1). Higher = more responsive, lower = smoother.
        """
        self.alpha = alpha
        self.value = None
    
    def smooth(self, new_value):
        """
        Smooth a new value.
        
        Args:
            new_value (float): New measurement to smooth.
            
        Returns:
            float: Smoothed value.
        """
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self):
        """Reset the filter."""
        self.value = None
