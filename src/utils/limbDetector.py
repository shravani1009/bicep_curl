"""
Limb Detector - Detects visible body parts for any exercise.
"""

from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class LimbDetector:
    """Detects which body parts are visible and usable."""
    
    @staticmethod
    def detect_arms(landmarks, landmark_extractor, mp_pose, visibility_threshold=0.5):
        """
        Detect which arms are visible.
        
        Args:
            landmarks: MediaPipe pose landmarks
            landmark_extractor: LandmarkExtractor instance
            mp_pose: MediaPipe PoseLandmark enum
            visibility_threshold: Minimum visibility (0-1)
            
        Returns:
            list: ['LEFT', 'RIGHT'] or subset
        """
        try:
            active_arms = []
            
            for arm in ['LEFT', 'RIGHT']:
                required_landmarks = [
                    f'{arm}_SHOULDER',
                    f'{arm}_ELBOW',
                    f'{arm}_WRIST'
                ]
                
                visibilities = [
                    landmark_extractor.get_visibility(landmarks, mp_pose, lm)
                    for lm in required_landmarks
                ]
                
                avg_visibility = sum(visibilities) / len(visibilities)
                
                if avg_visibility >= visibility_threshold:
                    active_arms.append(arm)
                    logger.debug(f"{arm} arm detected (visibility: {avg_visibility:.2f})")
            
            return active_arms
            
        except Exception as e:
            logger.error(f"Error detecting arms: {e}")
            return []
    
    @staticmethod
    def detect_legs(landmarks, landmark_extractor, mp_pose, visibility_threshold=0.5):
        """
        Detect which legs are visible (for squats, lunges).
        
        Args:
            landmarks: MediaPipe pose landmarks
            landmark_extractor: LandmarkExtractor instance
            mp_pose: MediaPipe PoseLandmark enum
            visibility_threshold: Minimum visibility (0-1)
            
        Returns:
            list: ['LEFT', 'RIGHT'] or subset
        """
        try:
            active_legs = []
            
            for leg in ['LEFT', 'RIGHT']:
                required_landmarks = [
                    f'{leg}_HIP',
                    f'{leg}_KNEE',
                    f'{leg}_ANKLE'
                ]
                
                visibilities = [
                    landmark_extractor.get_visibility(landmarks, mp_pose, lm)
                    for lm in required_landmarks
                ]
                
                avg_visibility = sum(visibilities) / len(visibilities)
                
                if avg_visibility >= visibility_threshold:
                    active_legs.append(leg)
                    logger.debug(f"{leg} leg detected (visibility: {avg_visibility:.2f})")
            
            return active_legs
            
        except Exception as e:
            logger.error(f"Error detecting legs: {e}")
            return []
