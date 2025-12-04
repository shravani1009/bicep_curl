"""
UI Renderer - Handles countdown, instructions, and UI rendering utilities.
"""

import cv2
import time
from config.exerciseConfig import COLORS


class UIRenderer:
    """Renders UI elements for exercise applications."""
    
    def __init__(self, exercise_name, colors=None):
        """
        Initialize UI renderer.
        
        Args:
            exercise_name (str): Name of the exercise
            colors (dict, optional): Color configuration
        """
        self.exercise_name = exercise_name
        self.colors = colors or COLORS
    
    def show_countdown_with_preview(self, cap, countdown_duration=5):
        """
        Show camera preview with positioning instructions and countdown.
        
        Args:
            cap: OpenCV VideoCapture object
            countdown_duration (int): Duration of countdown in seconds
            
        Returns:
            bool: True if countdown completed, False if user quit
        """
        print("\n" + "=" * 60)
        print("GET READY - POSITIONING INSTRUCTIONS")
        print("=" * 60)
        print("1. Stand 3-4 feet away from the camera")
        print("2. Ensure your full body is visible in the frame")
        print("3. Look straight ahead")
        print("4. Position yourself in good lighting")
        print("\nCountdown starting in camera preview...")
        print("=" * 60)
        
        window_name = f"AI {self.exercise_name} Trainer - Get Ready"
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame during countdown")
                break
            
            # Calculate remaining time
            elapsed = time.time() - start_time
            remaining = countdown_duration - int(elapsed)
            
            if remaining < 0:
                break
            
            # Draw countdown overlay
            self._draw_countdown_overlay(frame, remaining)
            
            # Show frame
            cv2.imshow(window_name, frame)
            
            # Check for quit
            key = cv2.waitKey(1000) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        # Show "GO!" message
        self._show_go_message(cap, window_name)
        
        cv2.destroyWindow(window_name)
        print("\n✓ Starting exercise tracking now!\n")
        return True
    
    def _draw_countdown_overlay(self, frame, remaining):
        """
        Draw countdown overlay on frame.
        
        Args:
            frame: OpenCV image
            remaining (int): Remaining seconds
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Dark background for text
        cv2.rectangle(overlay, (0, 0), (w, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, "GET READY!", (w//2 - 150, 50),
                   cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
        
        # Instructions
        instructions = [
            "1. Stand 3-4 feet from camera",
            "2. Full body visible",
            "3. Look straight ahead",
            "4. Good lighting",
            "5. Show your PALM to reset (hold 0.5s)"
        ]
        y_pos = 100
        for instruction in instructions:
            cv2.putText(frame, instruction, (50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 40
        
        # Countdown number (large and centered)
        if remaining > 0:
            countdown_text = str(remaining)
            text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_DUPLEX, 5, 10)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 100
            
            # Countdown with color change
            color = (0, 255, 0) if remaining > 2 else (0, 165, 255)
            cv2.putText(frame, countdown_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX, 5, color, 10)
            cv2.putText(frame, "Starting in...", (w//2 - 120, text_y - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def _show_go_message(self, cap, window_name):
        """
        Show "GO!" message on screen.
        
        Args:
            cap: OpenCV VideoCapture object
            window_name (str): Window name
        """
        for _ in range(2):  # Show for ~2 frames
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                cv2.putText(frame, "GO!", (w//2 - 80, h//2),
                           cv2.FONT_HERSHEY_DUPLEX, 4, (0, 255, 0), 10)
                cv2.imshow(window_name, frame)
                cv2.waitKey(500)
    
    def print_instructions(self):
        """Print exercise instructions to console."""
        print("=" * 60)
        print(f"AI {self.exercise_name.upper()} FORM CHECKER")
        print("=" * 60)
        print("Instructions:")
        print("  • Press 'Q' to quit")
        print("  • Press 'R' to reset rep counter")
        print("  • Show your PALM to camera to reset (hold for 0.5s)")
        print("=" * 60)
    
    @staticmethod
    def draw_text_with_background(image, text, position, font_scale=0.7, 
                                   thickness=2, text_color=(255, 255, 255),
                                   bg_color=(0, 0, 0), padding=10):
        """
        Draw text with background rectangle.
        
        Args:
            image: OpenCV image
            text (str): Text to draw
            position (tuple): (x, y) position
            font_scale (float): Font scale
            thickness (int): Text thickness
            text_color (tuple): RGB color for text
            bg_color (tuple): RGB color for background
            padding (int): Padding around text
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        x, y = position
        
        # Draw background rectangle
        cv2.rectangle(image, 
                     (x - padding, y - text_size[1] - padding),
                     (x + text_size[0] + padding, y + padding),
                     bg_color, -1)
        
        # Draw text
        cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)
    
    @staticmethod
    def draw_progress_bar(image, position, size, progress, 
                         fg_color=(0, 255, 0), bg_color=(50, 50, 50),
                         border_color=(255, 255, 255)):
        """
        Draw a progress bar.
        
        Args:
            image: OpenCV image
            position (tuple): (x, y) top-left corner
            size (tuple): (width, height)
            progress (float): Progress value between 0 and 1
            fg_color (tuple): Foreground color
            bg_color (tuple): Background color
            border_color (tuple): Border color
        """
        x, y = position
        width, height = size
        
        # Background
        cv2.rectangle(image, (x, y), (x + width, y + height), bg_color, -1)
        
        # Progress
        filled_width = int(width * min(max(progress, 0), 1))
        if filled_width > 0:
            cv2.rectangle(image, (x, y), (x + filled_width, y + height), fg_color, -1)
        
        # Border
        cv2.rectangle(image, (x, y), (x + width, y + height), border_color, 2)
