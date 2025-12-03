"""
Pose Processor - Handles MediaPipe pose detection and processing.
"""

import cv2
import mediapipe as mp
from config.exercise_config import CAMERA_CONFIG, COLORS


class PoseProcessor:
    """Processes video frames for pose detection using MediaPipe."""
    
    def __init__(self):
        """Initialize MediaPipe Pose."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=CAMERA_CONFIG['min_detection_confidence'],
            min_tracking_confidence=CAMERA_CONFIG['min_tracking_confidence'],
            model_complexity=CAMERA_CONFIG['model_complexity']
        )
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
        
        # Process pose
        results = self.pose.process(image)
        
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
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                    color=self.colors['BLUE'], thickness=2, circle_radius=3
                ),
                self.mp_drawing.DrawingSpec(
                    color=self.colors['GREEN'], thickness=2
                )
            )
    
    def get_landmark(self, landmarks, landmark_name):
        """
        Extract landmark coordinates and visibility.
        
        Args:
            landmarks: MediaPipe pose landmarks
            landmark_name (str): Name of the landmark (e.g., 'LEFT_SHOULDER')
            
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
    
    def get_landmark_coords(self, landmarks, landmark_name):
        """
        Get landmark coordinates as tuple.
        
        Args:
            landmarks: MediaPipe pose landmarks
            landmark_name (str): Name of the landmark
            
        Returns:
            tuple: (x, y, z) coordinates
        """
        landmark = self.get_landmark(landmarks, landmark_name)
        return (landmark['x'], landmark['y'], landmark['z'])
    
    def get_landmark_2d(self, landmarks, landmark_name):
        """
        Get landmark 2D coordinates as tuple.
        
        Args:
            landmarks: MediaPipe pose landmarks
            landmark_name (str): Name of the landmark
            
        Returns:
            tuple: (x, y) coordinates
        """
        landmark = self.get_landmark(landmarks, landmark_name)
        return (landmark['x'], landmark['y'])
    
    def cleanup(self):
        """Release MediaPipe Pose resources."""
        if self.pose:
            self.pose.close()
