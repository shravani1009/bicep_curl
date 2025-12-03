"""
Base Exercise Class - Abstract class for all exercise checkers.
Future exercises (squats, push-ups, etc.) should inherit from this class.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from abc import ABC, abstractmethod
from collections import Counter

from config.exercise_config import CAMERA_CONFIG, COLORS


class BaseExercise(ABC):
    """
    Abstract base class for exercise form checkers.
    
    All exercise implementations should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self):
        """Initialize common components for all exercises."""
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=CAMERA_CONFIG['min_detection_confidence'],
            min_tracking_confidence=CAMERA_CONFIG['min_tracking_confidence'],
            model_complexity=CAMERA_CONFIG['model_complexity']
        )
        
        # Initialize MediaPipe Hands for palm detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        
        # Common state tracking (used by base class methods)
        self.feedback = []
        self.current_issues = set()
        
        # Palm detection for reset
        self.palm_detected_frames = 0
        self.palm_detection_threshold = 15  # Need to show palm for ~15 frames (0.5 seconds)
        self.reset_cooldown = 0
        self.reset_cooldown_frames = 30  # Cooldown period after reset
        
        # Session analytics
        self.session_data = {
            'total_reps': 0,
            'form_scores': [],
            'common_errors': Counter(),
            'start_time': time.time(),
            'exercise_name': self.get_exercise_name()
        }
        
        # Colors from config
        self.colors = COLORS

        # Preferred frame size
        self.frame_width = CAMERA_CONFIG.get('width', 1280)
        self.frame_height = CAMERA_CONFIG.get('height', 720)
        
    @abstractmethod
    def get_exercise_name(self):
        """Return the name of the exercise (e.g., 'Bicep Curl')."""
        pass
    
    @abstractmethod
    def analyze_pose(self, landmarks):
        """
        Analyze pose and return metrics.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            tuple: Exercise-specific metrics or None if invalid
        """
        pass
    
    @abstractmethod
    def draw_feedback_panel(self, image):
        """
        Draw UI feedback panel on the image.
        
        Args:
            image: OpenCV image to draw on
        """
        pass
    
    @abstractmethod
    def select_difficulty_level(self):
        """Allow user to select difficulty level."""
        pass
    
    def get_landmark_features(self, landmarks, landmark_name):
        """
        Extract landmark coordinates and visibility.
        
        Args:
            landmarks: MediaPipe pose landmarks
            landmark_name (str): Name of the landmark
            
        Returns:
            dict: Landmark features (x, y, z, visibility)
        """
        feature = landmarks[self.mp_pose.PoseLandmark[landmark_name].value]
        return {
            'x': feature.x, 
            'y': feature.y, 
            'z': feature.z, 
            'visibility': feature.visibility
        }
    
    def process_frame(self, frame):
        """
        Process each video frame (common logic).
        
        Args:
            frame: OpenCV video frame
            
        Returns:
            Annotated frame
        """
        # Convert to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process pose
        results = self.pose.process(image)
        
        # Process hands for palm detection
        hand_results = self.hands.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Check for palm gesture to reset
        palm_detected = self._detect_palm_gesture(hand_results)
        if palm_detected and self.reset_cooldown == 0:
            self.palm_detected_frames += 1
            if self.palm_detected_frames >= self.palm_detection_threshold:
                self.reset_session()
                self.palm_detected_frames = 0
                self.reset_cooldown = self.reset_cooldown_frames
        else:
            self.palm_detected_frames = 0
        
        # Decrease cooldown
        if self.reset_cooldown > 0:
            self.reset_cooldown -= 1

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Analyze pose (exercise-specific)
            self.analyze_pose(landmarks)
            
            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                    color=self.colors['BLUE'], thickness=2, circle_radius=3
                ),
                self.mp_drawing.DrawingSpec(
                    color=self.colors['GREEN'], thickness=2
                )
            )
            
            # Exercise-specific annotations
            self.draw_exercise_annotations(image, landmarks)
        
        # Draw palm detection feedback
        self._draw_palm_feedback(image, palm_detected)

        # Draw UI Panel (exercise-specific)
        self.draw_feedback_panel(image)
        
        return image
    
    def draw_exercise_annotations(self, image, landmarks):
        """
        Draw exercise-specific annotations (override in subclass if needed).
        
        Args:
            image: OpenCV image
            landmarks: MediaPipe pose landmarks
        """
        pass
    
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
            # Get fingertips and finger bases
            landmarks = hand_landmarks.landmark
            
            # Wrist
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
    
    def _draw_palm_feedback(self, image, palm_detected):
        """
        Draw palm detection feedback on the image.
        
        Args:
            image: OpenCV image
            palm_detected: Boolean indicating if palm is detected
        """
        h, w = image.shape[:2]
        
        if palm_detected and self.reset_cooldown == 0:
            # Show progress bar for palm detection
            progress = self.palm_detected_frames / self.palm_detection_threshold
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
    
    def get_session_summary(self):
        """
        Generate session summary.
        
        Returns:
            dict: Session summary
        """
        duration = time.time() - self.session_data['start_time']
        
        summary = {
            'exercise': self.get_exercise_name(),
            'total_reps': self.session_data['total_reps'],
            'duration_seconds': duration,
            'average_form_score': (
                np.mean(self.session_data['form_scores']) 
                if self.session_data['form_scores'] else 0
            ),
            'common_errors': dict(self.session_data['common_errors']),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add exercise-specific data
        summary.update(self.get_session_specific_data())
        
        return summary
    
    def get_session_specific_data(self):
        """
        Return exercise-specific session data (override in subclass).
        
        Returns:
            dict: Exercise-specific metrics
        """
        return {}
    
    def print_session_summary(self, summary):
        """
        Print session summary to console.
        
        Args:
            summary (dict): Session summary data
        """
        print("\n" + "=" * 60)
        print(f"{self.get_exercise_name().upper()} SESSION SUMMARY")
        print("=" * 60)
        print(f"Total Reps: {summary['total_reps']}")
        print(f"Duration: {summary['duration_seconds']:.0f} seconds")
        print(f"Average Form Score: {summary['average_form_score']:.1f}%")
        print("\nMost Common Errors:")
        for error, count in Counter(summary['common_errors']).most_common(3):
            print(f"  • {error}: {count} times")
        print("=" * 60)
    
    def run(self):
        """
        Main application loop.
        
        This method can be overridden if exercise needs custom behavior.
        """
        # Select difficulty level
        self.select_difficulty_level()
        
        # Initialize camera (supports DroidCam and regular webcams)
        cap = self.initialize_camera()
        
        # Check if camera opened successfully
        if cap is None or not cap.isOpened():
            print("ERROR: Cannot open camera")
            return
        
        # Print instructions
        self.print_instructions()
        
        # Show countdown with positioning instructions
        self.show_countdown_with_preview(cap)
        
        # Main loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame
            annotated_frame = self.process_frame(frame)
            
            # Display
            window_name = f"AI {self.get_exercise_name()} Trainer"
            cv2.imshow(window_name, annotated_frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_session()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Display summary
        if self.session_data['total_reps'] > 0:
            summary = self.get_session_summary()
            self.print_session_summary(summary)
    
    def initialize_camera(self):
        """
        Initialize camera with support for DroidCam and regular webcams.
        Returns:
            cv2.VideoCapture or None
        """
        def _prepare(cap):
            if not cap or not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            return cap

        camera_source = CAMERA_CONFIG.get('source')

        if camera_source is not None:
            print(f"Using camera source: {camera_source}")
            cap = cv2.VideoCapture(camera_source) if isinstance(camera_source, str) else cv2.VideoCapture(int(camera_source))
            prepared = _prepare(cap)
            if prepared:
                return prepared
            print("Unable to open specified camera source.")

        for idx in [1, 2, 3, 0]:
            cap = cv2.VideoCapture(idx)
            prepared = _prepare(cap)
            if prepared:
                print(f"Auto-selected camera index {idx}")
                return prepared

        print("No camera available. Update CAMERA_CONFIG['source'] with a valid index or DroidCam URL.")
        return None
    
    def print_instructions(self):
        """Print exercise instructions (can be overridden)."""
        print("=" * 60)
        print(f"AI {self.get_exercise_name().upper()} FORM CHECKER")
        print("=" * 60)
        print("Instructions:")
        print("  • Press 'Q' to quit")
        print("  • Press 'R' to reset rep counter")
        print("  • Show your PALM to camera to reset (hold for 0.5s)")
        print("=" * 60)
    
    def reset_session(self):
        """Reset session (override in subclass for specific behavior)."""
        self.session_data['total_reps'] = 0
        self.session_data['form_scores'] = []
        self.session_data['common_errors'] = Counter()
        self.session_data['start_time'] = time.time()
        self.palm_detected_frames = 0
        print("\n🔄 SESSION RESET! All reps cleared.\n")
    
    def show_countdown_with_preview(self, cap):
        """
        Show camera preview with positioning instructions and countdown.
        
        Args:
            cap: OpenCV VideoCapture object
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
        
        window_name = f"AI {self.get_exercise_name()} Trainer - Get Ready"
        countdown_duration = 5  # 5 seconds
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
            
            # Create overlay for instructions
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
            
            # Show frame
            cv2.imshow(window_name, frame)
            
            # Wait for 1 second or until key press
            key = cv2.waitKey(1000) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        # Show "GO!" message
        for _ in range(2):  # Show for ~2 frames
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                cv2.putText(frame, "GO!", (w//2 - 80, h//2),
                           cv2.FONT_HERSHEY_DUPLEX, 4, (0, 255, 0), 10)
                cv2.imshow(window_name, frame)
                cv2.waitKey(500)
        
        cv2.destroyWindow(window_name)
        print("\n✓ Starting exercise tracking now!\n")
