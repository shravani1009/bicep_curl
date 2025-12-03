"""
Configuration file for exercise parameters and thresholds.
Modify these values to adjust difficulty levels and form requirements.
"""

# Camera and Detection Settings
CAMERA_CONFIG = {
    # Camera source: 
    #   - Integer (0, 1, 2, etc.) for USB/webcam device index
    #     * 0 = laptop/webcam
    #     * 1, 2, 3 = USB DroidCam (usually 1 or 2)
    #   - String URL for DroidCam WiFi (e.g., 'http://192.168.1.100:4747/video')
    #   - Leave as None to auto-detect (will try 0, 1, 2, then prompt)
    'source': 'http://192.168.1.58:4747/video',  # Try 1 for USB DroidCam, or use WiFi URL like 'http://192.168.1.100:4747/video'
    'width': 1280,
    'height': 720,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'model_complexity': 1
}

# Bicep Curl Configuration (Consolidated)
BICEP_CURL_CONFIG = {
    'difficulty_levels': {
        'BEGINNER': {
            'start_angle': 160,
            'top_angle': 60,
            'elbow_down_min': 150,
            'elbow_up_max': 40,
            'elbow_displacement_threshold': 0.12,
            'elbow_distance_tolerance': 0.20,
            'elbow_drift_angle': 35,
            'max_elbow_velocity': 0.025,
            'shoulder_shrug_threshold': 0.06,
            'torso_lean_threshold': 15,
            'max_velocity_change': 20,
            'description': 'Building foundation - focus on control'
        },
        'INTERMEDIATE': {
            'start_angle': 165,
            'top_angle': 55,
            'elbow_down_min': 155,
            'elbow_up_max': 50,
            'elbow_displacement_threshold': 0.08,
            'elbow_distance_tolerance': 0.18,
            'elbow_drift_angle': 32,
            'max_elbow_velocity': 0.018,
            'shoulder_shrug_threshold': 0.04,
            'torso_lean_threshold': 12,
            'max_velocity_change': 15,
            'description': 'Refining technique - stricter form'
        },
        'ADVANCED': {
            'start_angle': 170,
            'top_angle': 50,
            'elbow_down_min': 165,
            'elbow_up_max': 50,
            'elbow_displacement_threshold': 0.05,
            'elbow_distance_tolerance': 0.15,
            'elbow_drift_angle': 30,
            'max_elbow_velocity': 0.015,
            'shoulder_shrug_threshold': 0.03,
            'torso_lean_threshold': 10,
            'max_velocity_change': 12,
            'description': 'Elite form - maximum muscle engagement'
        }
    }
}

# UI Colors (BGR format for OpenCV)
COLORS = {
    'GREEN': (0, 255, 0),
    'RED': (0, 0, 255),
    'YELLOW': (0, 255, 255),
    'BLUE': (255, 0, 0),
    'WHITE': (255, 255, 255),
    'ORANGE': (0, 165, 255),
    'PURPLE': (255, 0, 255),
    # Friendly names for easier use
    'good': (0, 255, 0),
    'bad': (0, 0, 255),
    'warning': (0, 255, 255),
    'text': (255, 255, 255)
}
