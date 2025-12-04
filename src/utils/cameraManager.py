"""
Camera Manager - Handles camera initialization and configuration.
"""

import cv2
from config.exerciseConfig import CAMERA_CONFIG
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class CameraManager:
    """Manages camera initialization and configuration."""
    
    def __init__(self, config=None):
        """
        Initialize camera manager.
        
        Args:
            config (dict, optional): Camera configuration. Defaults to CAMERA_CONFIG.
        """
        self.config = config or CAMERA_CONFIG
        self.frame_width = self.config.get('width', 1280)
        self.frame_height = self.config.get('height', 720)
    
    def initialize_camera(self):
        """
        Initialize camera with support for DroidCam and regular webcams.
        
        Returns:
            cv2.VideoCapture or None: Camera capture object
        """
        camera_source = self.config.get('source')

        if camera_source is not None:
            print(f"Using camera source: {camera_source}")
            cap = self._try_camera_source(camera_source)
            if cap:
                return cap
            print("Unable to open specified camera source.")

        # Auto-detect available cameras
        for idx in [1, 2, 3, 0]:
            cap = self._try_camera_source(idx)
            if cap:
                print(f"Auto-selected camera index {idx}")
                return cap

        print("No camera available. Update CAMERA_CONFIG['source'] with a valid index or DroidCam URL.")
        return None
    
    def _try_camera_source(self, source):
        """
        Try to open and configure a camera source.
        
        Args:
            source: Camera index (int) or URL (str)
            
        Returns:
            cv2.VideoCapture or None: Configured camera or None if failed
        """
        cap = None
        try:
            if isinstance(source, str):
                logger.info(f"Attempting to open camera from URL: {source}")
                cap = cv2.VideoCapture(source)
            else:
                logger.info(f"Attempting to open camera at index: {source}")
                cap = cv2.VideoCapture(int(source))
            
            if not cap or not cap.isOpened():
                logger.warning(f"Camera source {source} failed to open")
                if cap:
                    cap.release()
                return None
            
            # Configure camera BEFORE reading to prevent buffer issues
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            # TEST READ to ensure camera actually works
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                logger.warning(f"Camera source {source} opened but cannot read frames")
                cap.release()
                return None
            
            logger.info(f"Successfully initialized camera: {source}")
            return cap
            
        except Exception as e:
            logger.error(f"Error opening camera source {source}: {e}")
            if cap:
                cap.release()
            return None
    
    @staticmethod
    def clear_buffer(cap, frames=5):
        """
        Clear camera buffer by reading and discarding frames.
        
        Args:
            cap: Camera capture object
            frames (int): Number of frames to discard
        """
        if cap and cap.isOpened():
            for _ in range(frames):
                cap.grab()  # Fast read without decoding
    
    @staticmethod
    def release_camera(cap):
        """
        Safely release camera resources.
        
        Args:
            cap: Camera capture object
        """
        if cap:
            cap.release()
