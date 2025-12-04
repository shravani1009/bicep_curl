"""
Gesture Detector - Handles peace sign gesture detection for session reset.
"""

import cv2
import mediapipe as mp
import numpy as np
from config.exerciseConfig import COLORS, GESTURE_CONFIG


class GestureDetector:
    """Detects peace sign gesture for resetting sessions."""
    
    def __init__(self):
        """
        Initialize gesture detector with peace sign detection.
        """
        # Initialize MediaPipe Hands with optimized settings for long distance detection
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,  # Lower for better long-distance detection
            min_tracking_confidence=0.4,   # Lower for tracking at distance
            model_complexity=1  # Use higher complexity model for better accuracy
        )
        
        # Detection state
        self.gesture_detected_frames = 0
        self.hold_frames_required = int(GESTURE_CONFIG['hold_duration'] * GESTURE_CONFIG['frames_per_second'])
        self.reset_cooldown = 0
        self.cooldown_frames = int(GESTURE_CONFIG['cooldown_duration'] * GESTURE_CONFIG['frames_per_second'])
        self.colors = COLORS
        self.show_feedback = GESTURE_CONFIG['show_gesture_feedback']
        
        # Store last detection result for debugging
        self.last_hand_landmarks = None
        self.debug_info = {}
    
    def _get_finger_distance(self, landmark1, landmark2):
        """Calculate Euclidean distance between two landmarks."""
        return np.sqrt((landmark1.x - landmark2.x)**2 + 
                      (landmark1.y - landmark2.y)**2 + 
                      (landmark1.z - landmark2.z)**2)
    
    def is_peace_sign(self, landmarks):
        """
        Detect peace sign gesture (✌️) with STRICT criteria.
        ONLY index and middle fingers extended, ALL others MUST be curled.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            bool: True if peace sign gesture is detected (exactly 2 fingers up)
        """
        if not landmarks:
            return False
        
        try:
            # Get landmark points for all fingers
            thumb_tip = landmarks[4]
            thumb_ip = landmarks[3]
            thumb_mcp = landmarks[2]
            
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            index_mcp = landmarks[5]
            
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            middle_mcp = landmarks[9]
            
            ring_tip = landmarks[16]
            ring_pip = landmarks[14]
            ring_mcp = landmarks[13]
            
            pinky_tip = landmarks[20]
            pinky_pip = landmarks[18]
            pinky_mcp = landmarks[17]
            
            wrist = landmarks[0]
            
            # Calculate distances for better detection (scale-invariant)
            index_length = self._get_finger_distance(index_tip, index_mcp)
            middle_length = self._get_finger_distance(middle_tip, middle_mcp)
            ring_length = self._get_finger_distance(ring_tip, ring_mcp)
            pinky_length = self._get_finger_distance(pinky_tip, pinky_mcp)
            thumb_length = self._get_finger_distance(thumb_tip, thumb_mcp)
            
            # Calculate hand size for distance-adaptive thresholds
            palm_size = self._get_finger_distance(wrist, landmarks[9])  # wrist to middle MCP
            
            # Adaptive margins based on hand size (works at any distance)
            extension_margin = max(0.015, palm_size * 0.1)  # Scales with distance
            curl_margin = max(0.005, palm_size * 0.05)
            separation_threshold = max(0.015, palm_size * 0.08)
            
            # STRICT Check 1: Index finger MUST be extended (tip higher than PIP with adaptive margin)
            index_extended = (index_tip.y < index_pip.y - extension_margin) and (index_tip.y < wrist.y)
            
            # STRICT Check 2: Middle finger MUST be extended (tip higher than PIP with adaptive margin)
            middle_extended = (middle_tip.y < middle_pip.y - extension_margin) and (middle_tip.y < wrist.y)
            
            # STRICT Check 3: Ring finger MUST be curled (significantly shorter AND tip below PIP)
            ring_curled = (ring_tip.y > ring_pip.y + curl_margin) and (ring_length < index_length * 0.75)
            
            # STRICT Check 4: Pinky MUST be curled (significantly shorter AND tip below PIP)
            pinky_curled = (pinky_tip.y > pinky_pip.y + curl_margin) and (pinky_length < middle_length * 0.75)
            
            # STRICT Check 5: Thumb MUST be curled or neutral (NOT extended)
            thumb_curled = (thumb_tip.y > thumb_ip.y) or (thumb_length < index_length * 0.7)
            
            # STRICT Check 6: Index and middle MUST be separated (clear peace sign with adaptive threshold)
            finger_separation = abs(index_tip.x - middle_tip.x) > separation_threshold
            
            # STRICT Check 7: Ring finger must NOT be extended like index/middle
            ring_not_extended = ring_tip.y > ring_mcp.y
            
            # STRICT Check 8: Pinky must NOT be extended like index/middle  
            pinky_not_extended = pinky_tip.y > pinky_mcp.y
            
            # Store debug info
            self.debug_info = {
                'index_extended': index_extended,
                'middle_extended': middle_extended,
                'ring_curled': ring_curled,
                'pinky_curled': pinky_curled,
                'thumb_curled': thumb_curled,
                'finger_separation': finger_separation,
                'ring_not_extended': ring_not_extended,
                'pinky_not_extended': pinky_not_extended
            }
            
            # ALL checks MUST pass - no tolerance
            all_checks_passed = all([
                index_extended,
                middle_extended,
                ring_curled,
                pinky_curled,
                thumb_curled,
                finger_separation,
                ring_not_extended,
                pinky_not_extended
            ])
            
            return all_checks_passed
                    
        except (IndexError, AttributeError) as e:
            return False
    
    def process_frame(self, image_rgb):
        """
        Process frame for peace sign gesture detection.
        
        Args:
            image_rgb: RGB image from camera
            
        Returns:
            bool: True if peace sign gesture is detected
        """
        hand_results = self.hands.process(image_rgb)
        self.last_hand_landmarks = hand_results  # Store for drawing
        
        if not hand_results.multi_hand_landmarks:
            return False
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            if self.is_peace_sign(hand_landmarks.landmark):
                return True
        
        return False
    
    def update_state(self, gesture_detected):
        """
        Update gesture detection state and check if reset should trigger.
        
        Args:
            gesture_detected (bool): Whether peace sign is currently detected
            
        Returns:
            bool: True if reset should be triggered
        """
        should_reset = False
        
        if gesture_detected and self.reset_cooldown == 0:
            self.gesture_detected_frames += 1
            if self.gesture_detected_frames >= self.hold_frames_required:
                should_reset = True
                self.gesture_detected_frames = 0
                self.reset_cooldown = self.cooldown_frames
        else:
            self.gesture_detected_frames = 0
        
        # Decrease cooldown
        if self.reset_cooldown > 0:
            self.reset_cooldown -= 1
        
        return should_reset
    
    def draw_feedback(self, image, gesture_detected):
        """
        Draw gesture detection feedback on the image with visual debugging.
        
        Args:
            image: OpenCV image
            gesture_detected (bool): Whether peace sign is currently detected
        """
        if not self.show_feedback:
            return
        
        h, w = image.shape[:2]
        
        # Draw hand landmarks if available
        if self.last_hand_landmarks and self.last_hand_landmarks.multi_hand_landmarks:
            for hand_landmarks in self.last_hand_landmarks.multi_hand_landmarks:
                # Draw hand skeleton in top-right corner (small debug view)
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                )
        
        # Show debug info for finger checks
        if self.debug_info and self.last_hand_landmarks and self.last_hand_landmarks.multi_hand_landmarks:
            debug_y = 30
            for check_name, passed in self.debug_info.items():
                color = self.colors['good'] if passed else self.colors['bad']
                status = "✓" if passed else "✗"
                text = f"{status} {check_name.replace('_', ' ').title()}"
                cv2.putText(image, text, (10, debug_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                debug_y += 25
        
        if gesture_detected and self.reset_cooldown == 0:
            # Show progress bar for gesture hold
            progress = self.gesture_detected_frames / self.hold_frames_required
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
            text = "HOLD PEACE SIGN (✌️) TO RESET"
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
        """Reset gesture detector state."""
        self.gesture_detected_frames = 0
    
    def cleanup(self):
        """Release MediaPipe Hands resources."""
        if self.hands:
            self.hands.close()
