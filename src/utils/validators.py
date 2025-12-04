"""
Validators - Input and configuration validation utilities.
"""


class InputValidator:
    """Validates user inputs."""
    
    @staticmethod
    def get_exercise_choice(max_attempts=3):
        """
        Get and validate exercise choice from user.
        
        Args:
            max_attempts (int): Maximum number of attempts allowed
            
        Returns:
            str: Exercise choice ('bicep_curl') or None if cancelled
        """
        valid_choices = {
            '1': 'bicep_curl',
            'bicep curl': 'bicep_curl',
            'bicep': 'bicep_curl',
            '0': None,
            'q': None,
            'quit': None
        }
        
        for attempt in range(max_attempts):
            try:
                choice = input("\nSelect exercise (1 for Bicep Curl, 0 to quit): ").strip().lower()
                
                if choice in valid_choices:
                    return valid_choices[choice]
                
                remaining = max_attempts - attempt - 1
                if remaining > 0:
                    print(f"Invalid choice. {remaining} attempt(s) remaining.")
                    
            except (EOFError, KeyboardInterrupt):
                print("\n\nOperation cancelled by user.")
                return None
            except Exception as e:
                print(f"Error reading input: {e}")
                return None
        
        print("Too many invalid attempts. Exiting.")
        return None
    
    @staticmethod
    def get_difficulty_choice(difficulty_levels, max_attempts=3):
        """
        Get and validate difficulty level choice.
        
        Args:
            difficulty_levels (dict): Dictionary of available difficulty levels
            max_attempts (int): Maximum number of attempts
            
        Returns:
            str: Difficulty level name or None if cancelled
        """
        level_map = {
            '1': 'BEGINNER',
            '2': 'INTERMEDIATE', 
            '3': 'ADVANCED'
        }
        
        for attempt in range(max_attempts):
            try:
                choice = input("\nSelect difficulty (1-3, 0 to cancel): ").strip()
                
                if choice == '0':
                    return None
                
                if choice in level_map:
                    level_name = level_map[choice]
                    if level_name in difficulty_levels:
                        return level_name
                
                remaining = max_attempts - attempt - 1
                if remaining > 0:
                    print(f"Invalid choice. Please enter 1, 2, or 3. {remaining} attempt(s) remaining.")
                    
            except (EOFError, KeyboardInterrupt):
                print("\n\nOperation cancelled by user.")
                return None
            except Exception as e:
                print(f"Error reading input: {e}")
                return None
        
        print("Too many invalid attempts. Using BEGINNER level as default.")
        return 'BEGINNER'


class ConfigValidator:
    """Validates configuration files."""
    
    @staticmethod
    def validate_camera_config(config):
        """
        Validate camera configuration.
        
        Args:
            config (dict): Camera configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = [
            'source', 'width', 'height', 
            'min_detection_confidence', 'min_tracking_confidence',
            'model_complexity'
        ]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required camera config key: {key}")
        
        # Validate ranges
        if not (0 <= config['min_detection_confidence'] <= 1):
            raise ValueError("min_detection_confidence must be between 0 and 1")
        
        if not (0 <= config['min_tracking_confidence'] <= 1):
            raise ValueError("min_tracking_confidence must be between 0 and 1")
        
        if config['model_complexity'] not in [0, 1, 2]:
            raise ValueError("model_complexity must be 0 (Lite), 1 (Full), or 2 (Heavy)")
        
        if config['width'] <= 0 or config['height'] <= 0:
            raise ValueError("Camera width and height must be positive")
    
    @staticmethod
    def validate_bicep_curl_config(config):
        """
        Validate bicep curl exercise configuration.
        
        Args:
            config (dict): Bicep curl configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        if 'difficulty_levels' not in config:
            raise ValueError("Bicep curl config missing 'difficulty_levels'")
        
        for level_name, params in config['difficulty_levels'].items():
            # Validate required parameters
            required_params = [
                'start_angle', 'top_angle', 'elbow_down_min', 'elbow_up_max',
                'elbow_displacement_threshold', 'elbow_distance_tolerance',
                'elbow_drift_angle', 'max_elbow_velocity',
                'shoulder_shrug_threshold', 'torso_lean_threshold',
                'max_velocity_change', 'description'
            ]
            
            for param in required_params:
                if param not in params:
                    raise ValueError(f"{level_name}: missing required parameter '{param}'")
            
            # Validate angle ranges
            if not (0 <= params['start_angle'] <= 180):
                raise ValueError(f"{level_name}: start_angle must be 0-180°")
            
            if not (0 <= params['top_angle'] <= 180):
                raise ValueError(f"{level_name}: top_angle must be 0-180°")
            
            if params['start_angle'] <= params['top_angle']:
                raise ValueError(f"{level_name}: start_angle must be > top_angle")
            
            if not (0 <= params['elbow_down_min'] <= 180):
                raise ValueError(f"{level_name}: elbow_down_min must be 0-180°")
            
            if not (0 <= params['elbow_up_max'] <= 180):
                raise ValueError(f"{level_name}: elbow_up_max must be 0-180°")
            
            # Validate thresholds are positive
            positive_params = [
                'elbow_displacement_threshold', 'elbow_distance_tolerance',
                'elbow_drift_angle', 'max_elbow_velocity',
                'shoulder_shrug_threshold', 'torso_lean_threshold',
                'max_velocity_change'
            ]
            
            for param in positive_params:
                if params[param] < 0:
                    raise ValueError(f"{level_name}: {param} cannot be negative")
