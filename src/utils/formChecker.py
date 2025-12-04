"""
Form Checker - Reusable form validation utilities for all exercises.
Exercise-agnostic validation methods that can be used across any exercise type.
"""

import numpy as np
from src.utils.angleCalculator import AngleCalculator


class FormChecker:
    """Provides reusable form checking methods for any exercise."""
    
    def __init__(self):
        self.angle_calc = AngleCalculator()
    
    def check_angle_range(self, angle, min_angle, max_angle, tolerance=5):
        """
        Check if angle is within acceptable range.
        
        Args:
            angle (float): Current angle in degrees
            min_angle (float): Minimum acceptable angle
            max_angle (float): Maximum acceptable angle
            tolerance (float): Tolerance buffer in degrees
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if angle < min_angle - tolerance:
            return False, f"Angle too small ({angle:.1f}° < {min_angle}°)"
        elif angle > max_angle + tolerance:
            return False, f"Angle too large ({angle:.1f}° > {max_angle}°)"
        return True, None
    
    def check_joint_displacement(self, current_pos, start_pos, threshold, axis='x'):
        """
        Check if joint has moved too much from starting position.
        Useful for checking elbow drift, knee valgus, etc.
        
        Args:
            current_pos (tuple): Current (x, y, z) position
            start_pos (tuple): Starting (x, y, z) position
            threshold (float): Maximum allowed displacement (normalized 0-1)
            axis (str): Axis to check ('x', 'y', 'z', or 'all')
            
        Returns:
            tuple: (is_valid, displacement_value)
        """
        if axis == 'all':
            displacement = np.linalg.norm(
                np.array(current_pos) - np.array(start_pos)
            )
        else:
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            displacement = abs(current_pos[axis_map[axis]] - start_pos[axis_map[axis]])
        
        is_valid = displacement <= threshold
        return is_valid, displacement
    
    def check_symmetry(self, left_value, right_value, max_difference=10):
        """
        Check if left and right side values are symmetric.
        Useful for bilateral exercises (squats, deadlifts, etc.)
        
        Args:
            left_value (float): Left side measurement (angle, distance, etc.)
            right_value (float): Right side measurement
            max_difference (float): Maximum allowed difference
            
        Returns:
            tuple: (is_symmetric, difference)
        """
        difference = abs(left_value - right_value)
        is_symmetric = difference <= max_difference
        return is_symmetric, difference
    
    def check_velocity(self, current_value, previous_value, max_change):
        """
        Check if rate of change is within acceptable limits.
        Prevents jerky/explosive movements.
        
        Args:
            current_value (float): Current measurement
            previous_value (float): Previous measurement
            max_change (float): Maximum allowed change per frame
            
        Returns:
            tuple: (is_valid, velocity)
        """
        velocity = abs(current_value - previous_value)
        is_valid = velocity <= max_change
        return is_valid, velocity
    
    def check_alignment(self, point1, point2, point3, max_deviation=15):
        """
        Check if three points are aligned (straight line).
        Useful for checking posture (shoulder-hip-knee alignment).
        
        Args:
            point1, point2, point3 (tuple): (x, y) coordinates
            max_deviation (float): Maximum angle deviation from 180°
            
        Returns:
            tuple: (is_aligned, angle)
        """
        angle = self.angle_calc.calculate_angle_2d(point1, point2, point3)
        deviation = abs(180 - angle)
        is_aligned = deviation <= max_deviation
        return is_aligned, angle
    
    def check_distance_threshold(self, point1, point2, min_distance=None, max_distance=None):
        """
        Check if distance between two points is within range.
        Useful for stance width, hand position, etc.
        
        Args:
            point1, point2 (tuple): (x, y, z) coordinates
            min_distance (float): Minimum allowed distance (optional)
            max_distance (float): Maximum allowed distance (optional)
            
        Returns:
            tuple: (is_valid, actual_distance)
        """
        distance = np.linalg.norm(np.array(point1) - np.array(point2))
        
        is_valid = True
        if min_distance is not None and distance < min_distance:
            is_valid = False
        if max_distance is not None and distance > max_distance:
            is_valid = False
            
        return is_valid, distance
    
    def check_vertical_alignment(self, point1, point2, max_horizontal_deviation=0.05):
        """
        Check if two points are vertically aligned.
        Useful for checking if joints are stacked (e.g., wrist over elbow).
        
        Args:
            point1, point2 (tuple): (x, y) coordinates
            max_horizontal_deviation (float): Max horizontal distance allowed
            
        Returns:
            tuple: (is_aligned, horizontal_distance)
        """
        horizontal_distance = abs(point1[0] - point2[0])
        is_aligned = horizontal_distance <= max_horizontal_deviation
        return is_aligned, horizontal_distance
    
    def check_horizontal_alignment(self, point1, point2, max_vertical_deviation=0.05):
        """
        Check if two points are horizontally aligned.
        Useful for checking level shoulders, hips, etc.
        
        Args:
            point1, point2 (tuple): (x, y) coordinates
            max_vertical_deviation (float): Max vertical distance allowed
            
        Returns:
            tuple: (is_aligned, vertical_distance)
        """
        vertical_distance = abs(point1[1] - point2[1])
        is_aligned = vertical_distance <= max_vertical_deviation
        return is_aligned, vertical_distance
    
    def check_range_of_motion(self, current_angle, full_rom_min, full_rom_max):
        """
        Calculate percentage of full range of motion achieved.
        
        Args:
            current_angle (float): Current joint angle
            full_rom_min (float): Minimum angle of full ROM
            full_rom_max (float): Maximum angle of full ROM
            
        Returns:
            float: ROM percentage (0-100)
        """
        rom_range = full_rom_max - full_rom_min
        current_rom = current_angle - full_rom_min
        percentage = (current_rom / rom_range) * 100
        return max(0, min(100, percentage))  # Clamp between 0-100
