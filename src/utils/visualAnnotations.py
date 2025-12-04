"""
Visual Annotations - Reusable visual feedback components for exercises.
"""

import cv2
import numpy as np
from config.exerciseConfig import COLORS


class VisualAnnotations:
    """Reusable visual annotation components for exercise feedback."""
    
    @staticmethod
    def draw_angle(image, point1, point2, point3, angle_value, is_valid, 
                   show_value=True, line_thickness=3):
        """
        Draw angle visualization between three points.
        
        Args:
            image: OpenCV image
            point1, point2, point3 (tuple): (x, y) pixel coordinates
            angle_value (float): Angle in degrees
            is_valid (bool): Is angle valid/form good?
            show_value (bool): Show angle value text
            line_thickness (int): Line thickness
        """
        # Draw lines
        cv2.line(image, point1, point2, COLORS['text'], line_thickness)
        cv2.line(image, point2, point3, COLORS['text'], line_thickness)
        
        # Draw joint points
        cv2.circle(image, point1, 8, COLORS['warning'], -1)
        cv2.circle(image, point2, 10, 
                  COLORS['good'] if is_valid else COLORS['bad'], -1)
        cv2.circle(image, point2, 10, (255, 255, 255), 2)
        cv2.circle(image, point3, 8, COLORS['warning'], -1)
        
        if show_value:
            # Draw angle text
            angle_color = COLORS['good'] if is_valid else COLORS['bad']
            angle_text = f"{int(angle_value)}°"
            
            # Position text near middle point
            text_x = point2[0] + 20 if point2[0] < image.shape[1] // 2 else point2[0] - 100
            cv2.putText(image, angle_text, (text_x, point2[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, angle_color, 3)
    
    @staticmethod
    def draw_target_zone(image, target_point, current_point, tolerance_pixels, 
                         is_in_zone, label="TARGET"):
        """
        Draw a target zone box showing where a joint should be.
        
        Args:
            image: OpenCV image
            target_point (tuple): (x, y) target coordinates in pixels
            current_point (tuple): (x, y) current coordinates in pixels
            tolerance_pixels (int): Radius of target zone
            is_in_zone (bool): Is current point in the zone?
            label (str): Label text
        """
        tx, ty = target_point
        cx, cy = current_point
        
        color = COLORS['good'] if is_in_zone else COLORS['bad']
        
        # Draw target circle
        cv2.circle(image, (tx, ty), tolerance_pixels, color, 2)
        
        # Semi-transparent fill
        overlay = image.copy()
        cv2.circle(overlay, (tx, ty), tolerance_pixels, color, -1)
        cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
        
        # Target center
        cv2.circle(image, (tx, ty), 5, color, -1)
        cv2.circle(image, (tx, ty), 5, (255, 255, 255), 1)
        
        # Draw arrow if not in zone
        if not is_in_zone:
            cv2.arrowedLine(image, (cx, cy), (tx, ty), 
                           COLORS['bad'], 2, tipLength=0.2)
        
        # Label
        label_text = "✓ " + label if is_in_zone else "→ " + label
        label_y = ty - tolerance_pixels - 10
        cv2.putText(image, label_text, (tx - 60, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    @staticmethod
    def draw_limb_panel(image, limb_name, panel_data, x, y, is_visible):
        """
        Draw status panel for a single limb.
        
        Args:
            image: OpenCV image
            limb_name (str): "LEFT ARM", "RIGHT ARM", etc.
            panel_data (dict): {
                'rep_count': int,
                'state': RepState or str,
                'angle': float (optional),
                'form_valid': bool,
                'feedback': list of str
            }
            x, y (int): Panel top-left position
            is_visible (bool): Is limb currently visible?
            
        Returns:
            int: New y position (for stacking panels)
        """
        y_offset = y
        
        # Header
        cv2.putText(image, f"=== {limb_name} ===", (x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['warning'], 2)
        y_offset += 35
        
        if is_visible:
            # Rep count
            cv2.putText(image, f"REPS: {panel_data.get('rep_count', 0)}", 
                       (x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['good'], 2)
            y_offset += 35
            
            # State
            state = panel_data.get('state', 'WAITING')
            if hasattr(state, 'value'):
                state = state.value
            state_text = state.replace('_', ' ')
            state_color = COLORS['warning'] if state != 'WAITING' else COLORS['text']
            cv2.putText(image, state_text, (x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)
            y_offset += 25
            
            # Angle (if available)
            if 'angle' in panel_data:
                cv2.putText(image, f"Angle: {int(panel_data['angle'])}°",
                           (x + 10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
                y_offset += 25
            
            # Form status
            form_valid = panel_data.get('form_valid', True)
            form_text = "✓ Good Form" if form_valid else "✗ Check Form"
            form_color = COLORS['good'] if form_valid else COLORS['bad']
            cv2.putText(image, form_text, (x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, form_color, 1)
            y_offset += 30
            
            # Feedback messages
            feedback = panel_data.get('feedback', [])
            if isinstance(feedback, str):
                feedback = [feedback] if feedback else []
            
            for msg in feedback[:2]:
                if msg:
                    msg_color = COLORS['bad'] if any(word in msg.lower() 
                                for word in ['not', 'abort', 'poor', 'fix']) \
                                                    else COLORS['good']
                    if len(msg) > 30:
                        msg = msg[:27] + "..."
                    cv2.putText(image, msg, (x + 15, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, msg_color, 1)
                    y_offset += 20
        
        else:
            # Not visible
            cv2.putText(image, f"REPS: {panel_data.get('rep_count', 0)}",
                       (x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['text'], 2)
            y_offset += 35
            
            cv2.putText(image, "NOT VISIBLE", (x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['bad'], 2)
            y_offset += 30
            
            cv2.putText(image, "Show limb in frame", (x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1)
            y_offset += 25
        
        return y_offset + 20
