"""
Angle calculation utilities for pose analysis.
"""

import math

class AngleCalculator:
    """Static methods for calculating angles from landmarks."""
    
    @staticmethod
    def calculate_angle(a, b, c):
        """
        Calculate angle between three points (a-b-c).
        
        Args:
            a (tuple): First point (x, y)
            b (tuple): Vertex point (x, y)
            c (tuple): Third point (x, y)
            
        Returns:
            float: Angle in degrees (0-180)
        """
        ax, ay = a[0], a[1]
        bx, by = b[0], b[1]
        cx, cy = c[0], c[1]
        
        radians = math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
        angle = abs(radians * 180.0 / math.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    @staticmethod
    def calculate_vertical_angle(point1, point2):
        """
        Calculate angle from vertical (for torso lean).
        
        Args:
            point1 (tuple): Lower point (x, y)
            point2 (tuple): Upper point (x, y)
            
        Returns:
            float: Angle from vertical in degrees
        """
        dx = abs(point2[0] - point1[0])
        dy = abs(point2[1] - point1[1])
        
        if dy == 0: 
            return 90.0
        
        angle = math.degrees(math.atan(dx/dy))
        return angle
    
    @staticmethod
    def calculate_distance(point1, point2):
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1 (tuple): First point (x, y)
            point2 (tuple): Second point (x, y)
            
        Returns:
            float: Distance between points
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return math.sqrt(dx * dx + dy * dy)
