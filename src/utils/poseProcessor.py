"""
Pose Processor - Handles MediaPipe pose detection and processing.
"""

import cv2
from config.exerciseConfig import COLORS
from src.model import PoseModel
from src.utils.landmarkExtractor import LandmarkExtractor


class PoseProcessor:
    """Processes video frames for pose detection using MediaPipe."""
    
    def __init__(self):
        """Initialize MediaPipe Pose through PoseModel."""
        # Initialize the model
        self.model = PoseModel()
        
        # Get references for convenience
        self.mp_pose = self.model.get_pose_landmark_enum()
        self.mp_drawing = self.model.get_drawing_utils()
        
        # Landmark extractor
        self.landmark_extractor = LandmarkExtractor()
        
        self.colors = COLORS
    
    def process_frame(self, frame):
        """
        Process frame for pose detection.
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            tuple: (processed_image, pose_results)
        """
        # Convert to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process pose using the model
        results = self.model.process(image)
        
        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image, results
    
    def draw_skeleton(self, image, pose_results):
        """
        Draw pose skeleton on the image.
        
        Args:
            image: OpenCV image
            pose_results: MediaPipe pose results
        """
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                pose_results.pose_landmarks, 
                self.model.get_pose_connections(),
                self.mp_drawing.DrawingSpec(
                    color=self.colors['BLUE'], thickness=2, circle_radius=3
                ),
                self.mp_drawing.DrawingSpec(
                    color=self.colors['GREEN'], thickness=2
                )
            )
    
    def get_landmark(self, landmarks, landmark_name):
        """Extract landmark coordinates and visibility."""
        return self.landmark_extractor.get_landmark_dict(
            landmarks, self.mp_pose, landmark_name
        )
    
    def get_landmark_coords(self, landmarks, landmark_name):
        """Get landmark coordinates as tuple."""
        return self.landmark_extractor.get_coords_3d(
            landmarks, self.mp_pose, landmark_name
        )
    
    def get_landmark_2d(self, landmarks, landmark_name):
        """Get landmark 2D coordinates as tuple."""
        return self.landmark_extractor.get_coords_2d(
            landmarks, self.mp_pose, landmark_name
        )
    
    def cleanup(self):
        """Release MediaPipe Pose resources."""
        self.model.cleanup()
