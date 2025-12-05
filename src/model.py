"""
Model - Handles MediaPipe Pose model initialization and management.
This module encapsulates the AI model used for pose detection.
"""

import mediapipe as mp
from config.exerciseConfig import CAMERA_CONFIG
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class PoseModel:
    """
    Encapsulates the MediaPipe Pose detection model.
    This class manages the AI model lifecycle and configuration.
    """
    
    def __init__(self, 
                 min_detection_confidence=None,
                 min_tracking_confidence=None,
                 model_complexity=None):
        """
        Initialize the MediaPipe Pose model.
        
        Args:
            min_detection_confidence (float): Minimum confidence for person detection (0-1)
            min_tracking_confidence (float): Minimum confidence for pose tracking (0-1)
            model_complexity (int): Model complexity level (0=Lite, 1=Full, 2=Heavy)
        """
        # Use config values if not provided
        self.min_detection_confidence = (
            min_detection_confidence or CAMERA_CONFIG['min_detection_confidence']
        )
        self.min_tracking_confidence = (
            min_tracking_confidence or CAMERA_CONFIG['min_tracking_confidence']
        )
        self.model_complexity = (
            model_complexity or CAMERA_CONFIG['model_complexity']
        )
        
        # Initialize MediaPipe solutions
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize the pose model
        self.pose = self._initialize_model()
        
        print(f"✓ MediaPipe Pose Model Initialized:")
        print(f"  - Detection Confidence: {self.min_detection_confidence}")
        print(f"  - Tracking Confidence: {self.min_tracking_confidence}")
        print(f"  - Model Complexity: {self._get_complexity_name()}")
    
    def _initialize_model(self):
        """
        Initialize the MediaPipe Pose model with configuration.
        
        Returns:
            MediaPipe Pose object
            
        Raises:
            RuntimeError: If model initialization fails
        """
        try:
            logger.info("Initializing MediaPipe Pose model")
            return self.mp_pose.Pose(
                static_image_mode=False,  # Optimized for video streams
                model_complexity=self.model_complexity,
                smooth_landmarks=False,  # Disable for faster processing
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Pose model: {e}")
            raise RuntimeError(f"Failed to initialize MediaPipe Pose model: {e}")
    
    def _get_complexity_name(self):
        """Get human-readable model complexity name."""
        complexity_names = {0: 'Lite', 1: 'Full', 2: 'Heavy'}
        return complexity_names.get(self.model_complexity, 'Unknown')
    
    def process(self, image_rgb):
        """
        Process an RGB image through the pose model.
        
        Args:
            image_rgb: RGB image (numpy array)
            
        Returns:
            MediaPipe pose results or None if processing fails
        """
        try:
            return self.pose.process(image_rgb)
        except Exception as e:
            logger.error(f"Error processing image through pose model: {e}")
            return None
    
    def get_pose_connections(self):
        """
        Get pose landmark connections for drawing skeleton.
        
        Returns:
            MediaPipe POSE_CONNECTIONS
        """
        return self.mp_pose.POSE_CONNECTIONS
    
    def get_pose_landmark_enum(self):
        """
        Get pose landmark enumeration.
        
        Returns:
            MediaPipe PoseLandmark enum
        """
        return self.mp_pose.PoseLandmark
    
    def get_drawing_utils(self):
        """
        Get MediaPipe drawing utilities.
        
        Returns:
            MediaPipe drawing_utils
        """
        return self.mp_drawing
    
    def update_config(self, min_detection_confidence=None, 
                     min_tracking_confidence=None, 
                     model_complexity=None):
        """
        Update model configuration and reinitialize.
        
        Args:
            min_detection_confidence (float): New detection confidence
            min_tracking_confidence (float): New tracking confidence
            model_complexity (int): New model complexity
        """
        # Update values if provided
        if min_detection_confidence is not None:
            self.min_detection_confidence = min_detection_confidence
        if min_tracking_confidence is not None:
            self.min_tracking_confidence = min_tracking_confidence
        if model_complexity is not None:
            self.model_complexity = model_complexity
        
        # Close old model
        self.cleanup()
        
        # Reinitialize with new config
        self.pose = self._initialize_model()
        
        print(f"✓ Model configuration updated and reinitialized")
    
    def cleanup(self):
        """Release model resources."""
        if self.pose:
            try:
                self.pose.close()
                print("✓ Pose model resources released")
            except (ValueError, AttributeError):
                # Already closed or None
                pass
            finally:
                self.pose = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
    
    def __del__(self):
        """Destructor - ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            # Ignore errors during garbage collection
            pass
