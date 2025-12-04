"""
Base Exercise Class - Abstract class for all exercise checkers.
Future exercises (squats, push-ups, etc.) should inherit from this class.
"""

import cv2
from abc import ABC, abstractmethod

from config.exerciseConfig import COLORS
from src.utils.cameraManager import CameraManager
from src.utils.palmDetector import PalmDetector
from src.utils.sessionTracker import SessionTracker
from src.utils.poseProcessor import PoseProcessor
from src.utils.uiRenderer import UIRenderer
from src.utils.feedbackManager import FeedbackManager
from src.utils.formChecker import FormChecker
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class BaseExercise(ABC):
    """
    Abstract base class for exercise form checkers.
    
    All exercise implementations should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self):
        """Initialize common components for all exercises."""
        # Initialize modular components
        self.pose_processor = PoseProcessor()
        self.palm_detector = PalmDetector()
        self.session_tracker = SessionTracker(self.get_exercise_name())
        self.camera_manager = CameraManager()
        self.ui_renderer = UIRenderer(self.get_exercise_name())
        self.feedback_manager = FeedbackManager()
        self.form_checker = FormChecker()
        
        # Colors from config
        self.colors = COLORS
        
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
        return self.pose_processor.get_landmark(landmarks, landmark_name)
    
    def process_frame(self, frame):
        """
        Process each video frame (common logic).
        
        Args:
            frame: OpenCV video frame
            
        Returns:
            Annotated frame
        """
        try:
            # Process pose using pose processor
            image, pose_results = self.pose_processor.process_frame(frame)
            
            # Process hands for palm detection
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            palm_detected = self.palm_detector.process_frame(image_rgb)
            
            # Update palm detection state and check for reset
            if self.palm_detector.update_state(palm_detected):
                self.reset_session()

            if pose_results and pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                
                # Analyze pose (exercise-specific)
                self.analyze_pose(landmarks)
                
                # Draw skeleton
                self.pose_processor.draw_skeleton(image, pose_results)
                
                # Exercise-specific annotations
                self.draw_exercise_annotations(image, landmarks)
            else:
                # Draw "no pose detected" message
                h, w = image.shape[:2]
                cv2.putText(image, "NO POSE DETECTED", (w//2 - 180, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(image, "Ensure full body is visible", (w//2 - 200, h//2 + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw palm detection feedback
            self.palm_detector.draw_feedback(image, palm_detected)

            # Draw UI Panel (exercise-specific)
            self.draw_feedback_panel(image)
            
            return image
            
        except cv2.error as e:
            logger.error(f"OpenCV error in process_frame: {e}")
            return frame  # Return original frame on error
        except Exception as e:
            logger.error(f"Error in process_frame: {e}", exc_info=True)
            return frame  # Return original frame on error
    
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
        # Get exercise-specific data and merge with session tracker summary
        exercise_specific = self.get_session_specific_data()
        return self.session_tracker.get_summary(exercise_specific)
    
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
        self.session_tracker.print_summary(summary)
    
    def run(self):
        """
        Main application loop.
        
        This method can be overridden if exercise needs custom behavior.
        """
        cap = None
        try:
            # Select difficulty level
            self.select_difficulty_level()
            
            # Initialize camera (supports DroidCam and regular webcams)
            cap = self.initialize_camera()
            
            # Check if camera opened successfully
            if cap is None or not cap.isOpened():
                logger.error("Failed to open camera")
                print("ERROR: Cannot open camera")
                print("Please check:")
                print("  1. Camera is connected")
                print("  2. No other application is using the camera")
                print("  3. Camera permissions are granted")
                return
            
            logger.info(f"Camera initialized successfully for {self.get_exercise_name()}")
            
            # Print instructions
            self.print_instructions()
            
            # Show countdown with positioning instructions
            self.show_countdown_with_preview(cap)
            
            # Main loop
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to grab frame from camera")
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
                    logger.info("User quit application")
                    break
                elif key == ord('r'):
                    self.reset_session()
            
            # Display summary
            if self.session_tracker.has_data():
                summary = self.get_session_summary()
                self.print_session_summary(summary)
                logger.info(f"Session completed: {summary['total_reps']} reps")
                
        except Exception as e:
            logger.error(f"Error in exercise run loop: {e}", exc_info=True)
            print(f"\nError during exercise: {e}")
            print("Check logs/gym_*.log for details.")
            
        finally:
            # Always cleanup resources
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            self.pose_processor.cleanup()
            self.palm_detector.cleanup()
            logger.info("Resources cleaned up")
    
    def initialize_camera(self):
        """
        Initialize camera with support for DroidCam and regular webcams.
        
        Returns:
            cv2.VideoCapture or None
        """
        return self.camera_manager.initialize_camera()
    
    def print_instructions(self):
        """Print exercise instructions (can be overridden)."""
        self.ui_renderer.print_instructions()
    
    def reset_session(self):
        """Reset session (override in subclass for specific behavior)."""
        self.session_tracker.reset()
        self.palm_detector.reset()
        self.feedback_manager.reset()
        print("\n🔄 SESSION RESET! All reps and feedback cleared.\n")
    
    def show_countdown_with_preview(self, cap):
        """
        Show camera preview with positioning instructions and countdown.
        
        Args:
            cap: OpenCV VideoCapture object
        """
        return self.ui_renderer.show_countdown_with_preview(cap)
