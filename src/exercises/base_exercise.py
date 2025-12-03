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
        
        # Common state tracking (used by base class methods)
        self.feedback = []
        self.current_issues = set()
        
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
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
        print("=" * 60)
    
    def reset_session(self):
        """Reset session (override in subclass for specific behavior)."""
        self.session_data['total_reps'] = 0
        self.session_data['form_scores'] = []
        self.session_data['common_errors'] = Counter()
        self.session_data['start_time'] = time.time()
        print("Session reset!")
