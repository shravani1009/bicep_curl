"""
Landmark Extractor - Utilities for extracting and processing pose landmarks.
Exercise-agnostic landmark utilities that work with any MediaPipe pose detection.
"""


class LandmarkExtractor:
    """Extracts and processes MediaPipe pose landmarks in a reusable way."""
    
    @staticmethod
    def get_landmark_dict(landmarks, landmark_enum, landmark_name):
        """
        Extract landmark as dictionary with all attributes.
        
        Args:
            landmarks: MediaPipe pose landmarks
            landmark_enum: MediaPipe PoseLandmark enum
            landmark_name (str): Name of landmark (e.g., 'LEFT_SHOULDER')
            
        Returns:
            dict: {x, y, z, visibility}
        """
        feature = landmarks[landmark_enum[landmark_name].value]
        return {
            'x': feature.x,
            'y': feature.y,
            'z': feature.z,
            'visibility': feature.visibility
        }
    
    @staticmethod
    def get_coords_3d(landmarks, landmark_enum, landmark_name):
        """
        Get 3D coordinates as tuple.
        
        Args:
            landmarks: MediaPipe landmarks
            landmark_enum: MediaPipe enum
            landmark_name (str): Landmark name
            
        Returns:
            tuple: (x, y, z)
        """
        lm = LandmarkExtractor.get_landmark_dict(landmarks, landmark_enum, landmark_name)
        return (lm['x'], lm['y'], lm['z'])
    
    @staticmethod
    def get_coords_2d(landmarks, landmark_enum, landmark_name):
        """
        Get 2D coordinates as tuple.
        
        Args:
            landmarks: MediaPipe landmarks
            landmark_enum: MediaPipe enum
            landmark_name (str): Landmark name
            
        Returns:
            tuple: (x, y)
        """
        lm = LandmarkExtractor.get_landmark_dict(landmarks, landmark_enum, landmark_name)
        return (lm['x'], lm['y'])
    
    @staticmethod
    def get_visibility(landmarks, landmark_enum, landmark_name):
        """
        Get landmark visibility score.
        
        Args:
            landmarks: MediaPipe landmarks
            landmark_enum: MediaPipe enum
            landmark_name (str): Landmark name
            
        Returns:
            float: Visibility score (0-1)
        """
        lm = LandmarkExtractor.get_landmark_dict(landmarks, landmark_enum, landmark_name)
        return lm['visibility']
    
    @staticmethod
    def are_landmarks_visible(landmarks, landmark_enum, landmark_names, threshold=0.5):
        """
        Check if all specified landmarks are visible above threshold.
        
        Args:
            landmarks: MediaPipe landmarks
            landmark_enum: MediaPipe enum
            landmark_names (list): List of landmark names to check
            threshold (float): Minimum visibility score (0-1)
            
        Returns:
            bool: True if all landmarks are visible
        """
        for name in landmark_names:
            visibility = LandmarkExtractor.get_visibility(landmarks, landmark_enum, name)
            if visibility < threshold:
                return False
        return True
    
    @staticmethod
    def get_multiple_coords_2d(landmarks, landmark_enum, landmark_names):
        """
        Get 2D coordinates for multiple landmarks at once.
        
        Args:
            landmarks: MediaPipe landmarks
            landmark_enum: MediaPipe enum
            landmark_names (list): List of landmark names
            
        Returns:
            dict: {landmark_name: (x, y)}
        """
        coords = {}
        for name in landmark_names:
            coords[name] = LandmarkExtractor.get_coords_2d(landmarks, landmark_enum, name)
        return coords
    
    @staticmethod
    def get_multiple_coords_3d(landmarks, landmark_enum, landmark_names):
        """
        Get 3D coordinates for multiple landmarks at once.
        
        Args:
            landmarks: MediaPipe landmarks
            landmark_enum: MediaPipe enum
            landmark_names (list): List of landmark names
            
        Returns:
            dict: {landmark_name: (x, y, z)}
        """
        coords = {}
        for name in landmark_names:
            coords[name] = LandmarkExtractor.get_coords_3d(landmarks, landmark_enum, name)
        return coords
