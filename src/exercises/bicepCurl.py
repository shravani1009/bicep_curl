"""
Bicep Curl Exercise Checker - Component-Based Version
Clean, maintainable code using reusable components.
"""

import cv2
import time
from src.exercises.baseExercise import BaseExercise
from src.utils.angleCalculator import AngleCalculator
from src.utils.limbDetector import LimbDetector
from src.utils.repStateMachine import RepStateMachine
from src.utils.visualAnnotations import VisualAnnotations
from src.utils.smoothing import ExponentialMovingAverage
from config.exerciseConfig import BICEP_CURL_CONFIG, COLORS
from src.utils.validators import InputValidator
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class BicepCurlChecker(BaseExercise):
    """
    Bicep curl form checker using reusable components.
    
    Monitors:
    - Elbow angle for rep counting
    - Elbow stability (prevents swinging)
    - Shoulder stability (prevents shrugging)
    """
    
    def __init__(self):
        super().__init__()
        
        # Exercise state
        self.active_arms = []
        self.difficulty_level = None
        self.thresholds = None
        
        # Arm tracking
        self.arm_data = {
            'LEFT': self._create_arm_state('LEFT ARM'),
            'RIGHT': self._create_arm_state('RIGHT ARM')
        }
    
    def _create_arm_state(self, limb_name):
        """Create state tracking for a single arm."""
        return {
            'state_machine': RepStateMachine(limb_name),
            'angle_smoother': ExponentialMovingAverage(alpha=0.3),
            'distance_smoother': ExponentialMovingAverage(alpha=0.3),
            'current_angle': 180,
            'current_distance': 0,
            'baseline_distance': None,
            'baseline_shoulder_y': None,
            'baseline_elbow_pos': None,
            'baseline_shoulder_pos': None,
            'form_valid': True,
            'form_issues': []
        }
    
    def get_exercise_name(self):
        """Return exercise name."""
        return "Bicep Curl"
    
    def select_difficulty_level(self):
        """Let user select difficulty level."""
        print("\n" + "="*50)
        print("BICEP CURL DIFFICULTY LEVELS")
        print("="*50)
        
        for i, (level, config) in enumerate(BICEP_CURL_CONFIG['difficulty_levels'].items(), 1):
            print(f"\n{i}. {level}")
            print(f"   {config['description']}")
        
        print("\n" + "="*50)
        
        # Get validated choice
        level_name = InputValidator.get_difficulty_choice(
            BICEP_CURL_CONFIG['difficulty_levels']
        )
        
        if level_name is None:
            level_name = 'BEGINNER'
            print("⚠ Defaulting to BEGINNER level")
        
        self.difficulty_level = level_name
        self.thresholds = BICEP_CURL_CONFIG['difficulty_levels'][level_name]
        
        logger.info(f"Selected difficulty: {level_name}")
        print(f"\n✓ Selected: {level_name} level\n")
    
    def analyze_pose(self, landmarks):
        """
        Analyze bicep curl pose using components.
        
        Returns:
            dict: Analysis results or None
        """
        if not landmarks:
            return None
        
        try:
            # Detect which arms are visible using LimbDetector
            self.active_arms = LimbDetector.detect_arms(
                landmarks,
                self.pose_processor.landmark_extractor,
                self.pose_processor.mp_pose,
                visibility_threshold=0.5
            )
            
            if not self.active_arms:
                return None
            
            # Analyze each visible arm
            results = {}
            for arm in self.active_arms:
                arm_result = self._analyze_arm(landmarks, arm)
                if arm_result:
                    results[arm] = arm_result
            
            return results if results else None
            
        except Exception as e:
            logger.error(f"Error analyzing pose: {e}")
            return None
    
    def _analyze_arm(self, landmarks, arm):
        """Analyze single arm using components."""
        try:
            data = self.arm_data[arm]
            
            # Get landmarks
            shoulder = self.get_landmark_features(landmarks, f'{arm}_SHOULDER')
            elbow = self.get_landmark_features(landmarks, f'{arm}_ELBOW')
            wrist = self.get_landmark_features(landmarks, f'{arm}_WRIST')
            
            if not all([shoulder, elbow, wrist]):
                return None
            
            # Calculate angle
            angle = AngleCalculator.calculate_angle(
                (shoulder['x'], shoulder['y']),
                (elbow['x'], elbow['y']),
                (wrist['x'], wrist['y'])
            )
            
            # Smooth angle
            data['current_angle'] = data['angle_smoother'].update(angle)
            
            # Calculate elbow-shoulder distance
            distance = AngleCalculator.calculate_distance(
                (shoulder['x'], shoulder['y']),
                (elbow['x'], elbow['y'])
            )
            data['current_distance'] = data['distance_smoother'].update(distance)
            
            # Set baseline when arm is down
            if data['baseline_distance'] is None and data['current_angle'] > 160:
                data['baseline_distance'] = data['current_distance']
                data['baseline_shoulder_y'] = shoulder['y']
                data['baseline_elbow_pos'] = (elbow['x'], elbow['y'])
                data['baseline_shoulder_pos'] = (shoulder['x'], shoulder['y'])
                logger.info(f"{arm} arm baseline set: distance={distance:.3f}, angle={angle:.1f}°")
                print(f"✓ {arm} arm baseline set - Ready to curl!")
            
            # Validate form ONLY during movement (not when arm is down)
            if data['baseline_distance'] is not None and data['current_angle'] < 160:
                data['form_valid'], data['form_issues'] = self._validate_form(arm, shoulder, elbow)
            else:
                # Arm is down, form is automatically valid
                data['form_valid'] = True
                data['form_issues'] = []
            
            # Update rep state machine
            state_result = data['state_machine'].update(
                current_value=data['current_angle'],
                start_threshold=self.thresholds['elbow_down_min'],
                peak_threshold=self.thresholds['elbow_up_max'],
                form_valid=data['form_valid'],
                tolerance=10
            )
            
            # Track rep completion
            if state_result['rep_completed']:
                self.session_tracker.add_rep()
                logger.info(f"{arm} arm rep completed: {state_result['rep_count']}")
            
            # Log feedback
            if state_result['feedback']:
                print(f"\n{arm}: {state_result['feedback']}")
            
            # Log form issues with detailed correction guidance
            if data['form_issues']:
                print(f"\n⚠️  {arm} FORM ERRORS:")
                for issue in data['form_issues']:
                    print(f"   ❌ {issue}")
                # Provide correction tips
                if any('ELBOW' in issue for issue in data['form_issues']):
                    print(f"   💡 TIP: Keep your elbow pinned at your side - imagine it's glued there!")
                if any('SHOULDER' in issue for issue in data['form_issues']):
                    print(f"   💡 TIP: Keep shoulders down - only move your forearm, not your whole arm!")
            
            return {
                'angle': data['current_angle'],
                'distance': data['current_distance'],
                'form_valid': data['form_valid'],
                'state': state_result['state'],
                'rep_count': state_result['rep_count'],
                'feedback': state_result['feedback']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {arm} arm: {e}")
            return None
    
    def _validate_form(self, arm, shoulder, elbow):
        """
        Validate bicep curl form using FormChecker.
        
        Returns:
            tuple: (is_valid, list of issues)
        """
        data = self.arm_data[arm]
        issues = []
        
        # This should only be called when arm is NOT down (handled in _analyze_arm)
        if data['baseline_distance'] is None:
            return True, []
        
        # Check elbow stability (no swinging forward/back)
        is_stable, displacement = self.form_checker.check_joint_displacement(
            current_pos=(elbow['x'], elbow['y'], 0),
            start_pos=(data['baseline_elbow_pos'][0], data['baseline_elbow_pos'][1], 0),
            threshold=self.thresholds.get('elbow_distance_tolerance', 0.15),
            axis='all'
        )
        
        if not is_stable:
            # Detailed feedback about elbow movement
            displacement_percent = int(displacement * 100)
            threshold_percent = int(self.thresholds.get('elbow_distance_tolerance', 0.15) * 100)
            issues.append(f"ELBOW MOVING! Keep elbow locked in place (moved {displacement_percent}%, max {threshold_percent}%)")
            logger.debug(f"{arm} elbow displacement: {displacement:.3f}")
        
        # Check shoulder stability (no shrugging)
        shoulder_lift = data['baseline_shoulder_y'] - shoulder['y']
        if shoulder_lift > self.thresholds.get('shoulder_shrug_threshold', 0.05):
            # Detailed feedback about shoulder shrugging
            lift_percent = int(shoulder_lift * 100)
            issues.append(f"SHOULDER SHRUGGING! Keep shoulders down and relaxed (lifted {lift_percent}%)")
            logger.debug(f"{arm} shoulder lift: {shoulder_lift:.3f}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def draw_feedback_panel(self, image):
        """Draw feedback panel using VisualAnnotations."""
        h, w = image.shape[:2]
        panel_width = 400
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (w - panel_width, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Title
        y = 40
        cv2.putText(image, "BICEP CURL TRACKER", (w - panel_width + 20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)
        y += 50
        
        # Total stats
        total_reps = sum(self.arm_data[arm]['state_machine'].rep_count 
                        for arm in ['LEFT', 'RIGHT'])
        cv2.putText(image, f"TOTAL REPS: {total_reps}", (w - panel_width + 20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['warning'], 2)
        y += 40
        
        # Draw each arm panel using VisualAnnotations
        for arm in ['LEFT', 'RIGHT']:
            data = self.arm_data[arm]
            is_visible = arm in self.active_arms
            
            panel_data = {
                'rep_count': data['state_machine'].rep_count,
                'state': data['state_machine'].state,
                'angle': data['current_angle'],
                'form_valid': data['form_valid'],
                'feedback': data['form_issues'] + ([data['state_machine'].current_feedback] 
                           if data['state_machine'].current_feedback else [])
            }
            
            y = VisualAnnotations.draw_limb_panel(
                image, f"{arm} ARM", panel_data,
                w - panel_width + 20, y, is_visible
            )
    
    def draw_exercise_annotations(self, image, landmarks):
        """Draw visual annotations using VisualAnnotations."""
        if not landmarks or not self.active_arms:
            return
        
        h, w, _ = image.shape
        
        for arm in self.active_arms:
            try:
                data = self.arm_data[arm]
                
                # Get landmarks
                shoulder = self.get_landmark_features(landmarks, f'{arm}_SHOULDER')
                elbow = self.get_landmark_features(landmarks, f'{arm}_ELBOW')
                wrist = self.get_landmark_features(landmarks, f'{arm}_WRIST')
                
                if not all([shoulder, elbow, wrist]):
                    continue
                
                # Convert to pixel coordinates
                sp = (int(shoulder['x'] * w), int(shoulder['y'] * h))
                ep = (int(elbow['x'] * w), int(elbow['y'] * h))
                wp = (int(wrist['x'] * w), int(wrist['y'] * h))
                
                # Draw elbow angle using VisualAnnotations
                VisualAnnotations.draw_angle(
                    image, sp, ep, wp,
                    data['current_angle'],
                    data['form_valid'],
                    show_value=True
                )
                
                # Draw elbow target zone if baseline set
                if data['baseline_elbow_pos'] is not None:
                    # Calculate target position in PIXEL coordinates
                    # Baseline positions are in normalized coords (0-1)
                    baseline_elbow_px = (
                        int(data['baseline_elbow_pos'][0] * w),
                        int(data['baseline_elbow_pos'][1] * h)
                    )
                    baseline_shoulder_px = (
                        int(data['baseline_shoulder_pos'][0] * w),
                        int(data['baseline_shoulder_pos'][1] * h)
                    )
                    
                    # Calculate offset in pixels
                    offset_x = baseline_elbow_px[0] - baseline_shoulder_px[0]
                    offset_y = baseline_elbow_px[1] - baseline_shoulder_px[1]
                    
                    # Apply offset to CURRENT shoulder position
                    target_x = sp[0] + offset_x
                    target_y = sp[1] + offset_y
                    
                    # Calculate tolerance in pixels
                    baseline_distance_px = int(data['baseline_distance'] * w)
                    tolerance_pixels = int(baseline_distance_px * 
                                          self.thresholds.get('elbow_distance_tolerance', 0.15))
                    
                    # Check if in zone
                    distance = ((ep[0] - target_x)**2 + (ep[1] - target_y)**2)**0.5
                    in_zone = distance <= tolerance_pixels
                    
                    # Draw target zone
                    VisualAnnotations.draw_target_zone(
                        image, (target_x, target_y), ep,
                        tolerance_pixels, in_zone,
                        label=f"{arm[0]}ELBOW"  # "LELBOW" or "RELBOW"
                    )
                
            except Exception as e:
                logger.error(f"Error drawing annotations for {arm}: {e}")
    
    def reset_session(self):
        """Reset session data."""
        super().reset_session()
        self.active_arms = []
        
        for arm in ['LEFT', 'RIGHT']:
            self.arm_data[arm] = self._create_arm_state(f"{arm} ARM")
        
        logger.info("Bicep curl session reset")
        print("\n🔄 SESSION RESET - All reps cleared!\n")
    
    def get_session_specific_data(self):
        """Return exercise-specific session data."""
        return {
            'left_arm_reps': self.arm_data['LEFT']['state_machine'].rep_count,
            'right_arm_reps': self.arm_data['RIGHT']['state_machine'].rep_count,
            'difficulty': self.difficulty_level
        }
    
    def show_countdown_with_preview(self, cap):
        """
        Show camera preview with bicep curl-specific positioning instructions and countdown.
        
        Args:
            cap: OpenCV VideoCapture object
        """
        print("\n" + "=" * 60)
        print("GET READY - BICEP CURL POSITIONING INSTRUCTIONS")
        print("=" * 60)
        print("1. Stand 3-4 feet away from the camera")
        print("2. Ensure your full body is visible in the frame")
        print("3. Look straight ahead")
        print("4. Position yourself in good lighting")
        print("5. Keep elbows fixed and close to body - DO NOT swing!")
        print("\nCountdown starting in camera preview...")
        print("=" * 60)
        
        # Clear buffer to prevent reading stale frames
        from src.utils.cameraManager import CameraManager
        CameraManager.clear_buffer(cap)
        
        window_name = f"AI {self.get_exercise_name()} Trainer - Get Ready"
        countdown_duration = 5  # 5 seconds
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to grab frame during countdown")
                break
            
            # Calculate remaining time
            elapsed = time.time() - start_time
            remaining = countdown_duration - int(elapsed)
            
            if remaining < 0:
                break
            
            # Create overlay for instructions
            h, w = frame.shape[:2]
            overlay = frame.copy()
            
            # Dark background for text (larger area for more instructions)
            cv2.rectangle(overlay, (0, 0), (w, 300), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Title
            cv2.putText(frame, "GET READY - BICEP CURLS!", (w//2 - 250, 40),
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)
            
            # Instructions
            instructions = [
                "1. Stand 3-4 feet from camera",
                "2. Full body visible",
                "3. Look straight ahead",
                "4. Good lighting",
                "5. Keep elbows FIXED & close to body",
                "6. Show PALM to reset (hold 0.5s)"
            ]
            y_pos = 85
            for instruction in instructions:
                # Highlight elbow instruction
                color = (0, 255, 0) if "elbows" in instruction.lower() else (255, 255, 255)
                thickness = 2 if "elbows" in instruction.lower() else 1
                cv2.putText(frame, instruction, (40, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
                y_pos += 35
            
            # Countdown number (large and centered)
            if remaining > 0:
                countdown_text = str(remaining)
                text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_DUPLEX, 5, 10)[0]
                text_x = (w - text_size[0]) // 2
                text_y = h - 100
                
                # Countdown with color change
                color = (0, 255, 0) if remaining > 2 else (0, 165, 255)
                cv2.putText(frame, countdown_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_DUPLEX, 5, color, 10)
                cv2.putText(frame, "Starting in...", (w//2 - 120, text_y - 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow(window_name, frame)
            
            # Wait for small delay for smooth video
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        # Show "GO!" message
        for _ in range(2):  # Show for ~2 frames
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                cv2.putText(frame, "GO!", (w//2 - 80, h//2),
                           cv2.FONT_HERSHEY_DUPLEX, 4, (0, 255, 0), 10)
                cv2.imshow(window_name, frame)
                cv2.waitKey(500)
        
        cv2.destroyWindow(window_name)
