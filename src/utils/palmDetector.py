"""
Palm Detector - Handles palm gesture detection for session reset.
"""

import cv2
import mediapipe as mp
from config.exerciseConfig import COLORS


class PalmDetector:
    """Detects palm gestures for resetting sessions."""
    
    def __init__(self, detection_threshold=15, cooldown_frames=30):
        """
        Initialize palm detector.
        
        Args:
            detection_threshold (int): Number of frames palm must be shown
            cooldown_frames (int): Cooldown period after reset
        """
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        
        # Detection state
        self.palm_detected_frames = 0
        self.detection_threshold = detection_threshold
        self.reset_cooldown = 0
        self.cooldown_frames = cooldown_frames
        self.colors = COLORS
    
    def process_frame(self, image_rgb):
        """
        Process frame for palm detection.
        
        Args:
            image_rgb: RGB image from camera
            
        Returns:
            bool: True if palm is detected
        """
        hand_results = self.hands.process(image_rgb)
        return self._detect_palm_gesture(hand_results)
    
    def _detect_palm_gesture(self, hand_results):
        """
        Detect if user is showing their palm (open hand facing camera).
        
        Args:
            hand_results: MediaPipe hand detection results
            
        Returns:
            bool: True if palm is detected
        """
        if not hand_results.multi_hand_landmarks:
            return False
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            # Get wrist position
            wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
            
            # Fingertips and their base joints
            finger_tips = [
                landmarks[self.mp_hands.HandLandmark.THUMB_TIP],
                landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
                landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP],
                landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
            ]
            
            finger_bases = [
                landmarks[self.mp_hands.HandLandmark.THUMB_IP],
                landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP],
                landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP],
                landmarks[self.mp_hands.HandLandmark.PINKY_PIP]
            ]
            
            # Check if all fingers are extended (palm open)
            extended_fingers = 0
            for tip, base in zip(finger_tips, finger_bases):
                # Finger is extended if tip is further from wrist than base
                tip_distance = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)**0.5
                base_distance = ((base.x - wrist.x)**2 + (base.y - wrist.y)**2)**0.5
                
                if tip_distance > base_distance:
                    extended_fingers += 1
            
            # Palm is detected if at least 4 fingers are extended
            if extended_fingers >= 4:
                return True
        
        return False
    
    def update_state(self, palm_detected):
        """
        Update palm detection state and check if reset should trigger.
        
        Args:
            palm_detected (bool): Whether palm is currently detected
            
        Returns:
            bool: True if reset should be triggered
        """
        should_reset = False
        
        if palm_detected and self.reset_cooldown == 0:
            self.palm_detected_frames += 1
            if self.palm_detected_frames >= self.detection_threshold:
                should_reset = True
                self.palm_detected_frames = 0
                self.reset_cooldown = self.cooldown_frames
        else:
            self.palm_detected_frames = 0
        
        # Decrease cooldown
        if self.reset_cooldown > 0:
            self.reset_cooldown -= 1
        
        return should_reset
    
    def draw_feedback(self, image, palm_detected):
        """
        Draw palm detection feedback on the image.
        
        Args:
            image: OpenCV image
            palm_detected (bool): Whether palm is currently detected
        """
        h, w = image.shape[:2]
        
        if palm_detected and self.reset_cooldown == 0:
            # Show progress bar for palm detection
            progress = self.palm_detected_frames / self.detection_threshold
            bar_width = 300
            bar_height = 30
            bar_x = (w - bar_width) // 2
            bar_y = h - 80
            
            # Background
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            
            # Progress
            filled_width = int(bar_width * progress)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                         self.colors['warning'], -1)
            
            # Border
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         self.colors['text'], 2)
            
            # Text
            text = "HOLD PALM TO RESET"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(image, text, (text_x, bar_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['warning'], 2)
        
        elif self.reset_cooldown > 0:
            # Show reset confirmation
            text = "SESSION RESET!"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(image, text, (text_x, h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['good'], 2)
    
    def reset(self):
        """Reset palm detector state."""
        self.palm_detected_frames = 0
    
    def cleanup(self):
        """Release MediaPipe Hands resources."""
        if self.hands:
            self.hands.close()
