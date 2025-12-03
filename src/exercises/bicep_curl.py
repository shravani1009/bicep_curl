"""
Bicep Curl Exercise Checker with Strict Form Validation
Updated to use Angle-Based Elbow Stability
"""

import cv2
import numpy as np
from collections import deque
from src.exercises.base_exercise import BaseExercise
from src.utils.angle_calculator import AngleCalculator
from src.utils.smoothing import ExponentialSmoothing
from config.exercise_config import BICEP_CURL_CONFIG, COLORS

class BicepCurlChecker(BaseExercise):
    """
    Bicep curl form checker with strict rep counting algorithm.
    Supports both left and right arm tracking.
    
    Only counts a rep when ALL form requirements are met:
    1. Elbow angle: >160° (start) to <60° (top) with controlled movement
    2. Elbow Stability: Checked via Hip-Shoulder-Elbow angle (prevents swinging)
    3. Shoulder stability: No shrugging
    4. Torso stability: No leaning back
    """
    
    def __init__(self):
        super().__init__()
        
        # Active arms tracking (can be both)
        self.active_arms = []  # Can contain 'LEFT', 'RIGHT', or both
        
        # Dual-arm tracking data - separate state for each arm
        self.arm_data = {
            'LEFT': self._create_arm_state(),
            'RIGHT': self._create_arm_state()
        }
        
        # Thresholds
        self.difficulty_level = None
        self.thresholds = None
        self.difficulty_profile = None
    
    def _create_arm_state(self):
        """Create independent state tracking for each arm."""
        return {
            # Calibration state
            'calibration_state': 'CALIBRATING',
            'calibration_frames': [],
            'calibration_frame_count': 0,
            'calibration_target_frames': 30,
            
            # Rep state machine
            'rep_state': 'WAITING',
            'rep_count': 0,
            
            # Angle tracking
            'elbow_angle': 180,
            'elbow_angle_smoother': ExponentialSmoothing(alpha=0.5),
            'angle_history': deque(maxlen=10),
            
            # Elbow Stability Tracking
            'shoulder_tuck_angle': 0,
            'elbow_shoulder_distance': 0,
            'elbow_shoulder_distance_baseline': None,
            
            # Velocity-Based Elbow Tracking
            'elbow_position_history': deque(maxlen=10),
            'elbow_velocity_history': deque(maxlen=10),
            'current_elbow_velocity': 0,
            
            # Shoulder tracking
            'shoulder_y_baseline': None,
            
            # Torso tracking
            'torso_baseline_angle': None,
            
            # Form validation
            'form_valid': True,
            'form_violations': [],
            'consecutive_bad_form_frames': 0,
            'bad_form_threshold': 8,
            
            # Rep quality tracking
            'current_rep_quality': {
                'elbow_stable': True,
                'no_shrugging': True,
                'no_leaning': True
            },
            
            # Feedback
            'feedback': []
        }
    
    def _detect_active_arms(self, landmarks):
        """Detect which arms are visible and active."""
        try:
            left_shoulder = self.get_landmark_features(landmarks, 'LEFT_SHOULDER')
            left_elbow = self.get_landmark_features(landmarks, 'LEFT_ELBOW')
            left_wrist = self.get_landmark_features(landmarks, 'LEFT_WRIST')
            
            right_shoulder = self.get_landmark_features(landmarks, 'RIGHT_SHOULDER')
            right_elbow = self.get_landmark_features(landmarks, 'RIGHT_ELBOW')
            right_wrist = self.get_landmark_features(landmarks, 'RIGHT_WRIST')
            
            left_visibility = (left_shoulder['visibility'] + left_elbow['visibility'] + left_wrist['visibility']) / 3
            right_visibility = (right_shoulder['visibility'] + right_elbow['visibility'] + right_wrist['visibility']) / 3
            
            active_arms = []
            if left_visibility > 0.5:
                active_arms.append('LEFT')
            if right_visibility > 0.5:
                active_arms.append('RIGHT')
            
            return active_arms
        except Exception as e:
            print(f"Error detecting active arms: {e}")
            return []
        
    def get_exercise_name(self):
        return "Bicep Curl"
    
    def select_difficulty_level(self):
        print("\n=== BICEP CURL DIFFICULTY LEVELS ===")
        for i, (level, profile) in enumerate(BICEP_CURL_CONFIG['difficulty_levels'].items(), 1):
            print(f"{i}. {level}")
            print(f"   {profile['description']}")
        
        while True:
            try:
                choice = input("\nSelect difficulty (1-3): ").strip()
                if choice in ['1', '2', '3']:
                    levels = {'1': 'BEGINNER', '2': 'INTERMEDIATE', '3': 'ADVANCED'}
                    self.difficulty_level = levels[choice]
                    self.thresholds = BICEP_CURL_CONFIG['difficulty_levels'][self.difficulty_level]
                    self.difficulty_profile = BICEP_CURL_CONFIG['difficulty_levels'][self.difficulty_level]
                    
                    print(f"\n✓ Selected: {self.difficulty_level} level")
                    return
                else:
                    print("Invalid choice.")
            except Exception as e:
                print(f"Error: {e}")
    
    def analyze_pose(self, landmarks):
        """
        Analyze bicep curl form for both arms simultaneously.
        """
        if not landmarks:
            return None
        
        try:
            # Detect which arms are active
            detected_arms = self._detect_active_arms(landmarks)
            if not detected_arms:
                return None
            
            # Update active arms list
            self.active_arms = detected_arms
            
            # Analyze each active arm
            results = {}
            for arm in self.active_arms:
                arm_result = self._analyze_single_arm(landmarks, arm)
                if arm_result:
                    results[arm] = arm_result
            
            return results if results else None
            
        except Exception as e:
            print(f"Error in pose analysis: {e}")
            return None
    
    def _analyze_single_arm(self, landmarks, arm):
        """Analyze a single arm's bicep curl form."""
        try:
            data = self.arm_data[arm]
            
            # Get landmarks
            shoulder = self.get_landmark_features(landmarks, f'{arm}_SHOULDER')
            elbow = self.get_landmark_features(landmarks, f'{arm}_ELBOW')
            wrist = self.get_landmark_features(landmarks, f'{arm}_WRIST')
            hip = self.get_landmark_features(landmarks, f'{arm}_HIP')
            
            if not all([shoulder, elbow, wrist, hip]):
                return None
            
            MIN_VISIBILITY = 0.3  # More lenient
            if min(shoulder['visibility'], elbow['visibility'], wrist['visibility']) < MIN_VISIBILITY:
                return None
            
            # Points for Angle Calculation
            shoulder_point = (shoulder['x'], shoulder['y'])
            elbow_point = (elbow['x'], elbow['y'])
            wrist_point = (wrist['x'], wrist['y'])
            hip_point = (hip['x'], hip['y'])
            
            # 1. Main Curl Angle (Shoulder-Elbow-Wrist)
            elbow_angle = AngleCalculator.calculate_angle(shoulder_point, elbow_point, wrist_point)
            data['elbow_angle'] = data['elbow_angle_smoother'].smooth(elbow_angle)
            data['angle_history'].append(data['elbow_angle'])
            
            # 2. Torso Angle (Vertical alignment)
            torso_angle = AngleCalculator.calculate_vertical_angle(hip_point, shoulder_point)

            # 3. Shoulder Tuck Angle (Hip-Shoulder-Elbow)
            data['shoulder_tuck_angle'] = AngleCalculator.calculate_angle(hip_point, shoulder_point, elbow_point)
            
            # 4. Elbow-Shoulder Distance
            data['elbow_shoulder_distance'] = AngleCalculator.calculate_distance(shoulder_point, elbow_point)
            
            # 5. Velocity-Based Elbow Tracking
            data['elbow_position_history'].append(elbow_point)
            
            if len(data['elbow_position_history']) >= 2:
                prev_pos = data['elbow_position_history'][-2]
                curr_pos = data['elbow_position_history'][-1]
                
                velocity = np.sqrt(
                    (curr_pos[0] - prev_pos[0])**2 + 
                    (curr_pos[1] - prev_pos[1])**2
                )
                data['current_elbow_velocity'] = velocity
                data['elbow_velocity_history'].append(velocity)
            
            # Handle calibration
            if data['calibration_state'] == "CALIBRATING":
                if 160 < data['elbow_angle'] < 180:
                    data['calibration_frames'].append({
                        'shoulder_y': shoulder['y'],
                        'torso_angle': torso_angle,
                        'elbow_shoulder_distance': data['elbow_shoulder_distance']
                    })
                    data['calibration_frame_count'] += 1
                    if data['calibration_frame_count'] >= data['calibration_target_frames']:
                        self._calculate_baselines(arm)
                        data['calibration_state'] = "READY"
                        print(f"\n✓ {arm} ARM CALIBRATION COMPLETE!")
                else:
                    if len(data['calibration_frames']) > 0:
                        data['calibration_frames'] = []
                        data['calibration_frame_count'] = 0
            
            # Validate form
            self._validate_form(arm, elbow, shoulder, torso_angle, data['shoulder_tuck_angle'], data['elbow_shoulder_distance'])
            
            # Bad form counter
            if not data['form_valid']:
                data['consecutive_bad_form_frames'] += 1
            else:
                data['consecutive_bad_form_frames'] = 0
            
            return {
                'elbow_angle': data['elbow_angle'],
                'torso_angle': torso_angle,
                'shoulder_tuck_angle': data['shoulder_tuck_angle'],
                'elbow_shoulder_distance': data['elbow_shoulder_distance'],
                'shoulder_y': shoulder['y'],
                'form_valid': data['form_valid'],
                'form_violations': data['form_violations'].copy(),
                'calibration_state': data['calibration_state'],
                'rep_state': data['rep_state'],
                'rep_count': data['rep_count']
            }
            
        except Exception as e:
            print(f"Error analyzing {arm} arm: {e}")
            return None
    
    def _validate_form(self, arm, elbow, shoulder, torso_angle, shoulder_tuck_angle, elbow_shoulder_distance):
        """
        Validate all form requirements for a specific arm.
        """
        data = self.arm_data[arm]
        data['form_violations'] = []
        data['form_valid'] = True
        
        # Get threshold from config
        drift_threshold = self.thresholds.get('elbow_drift_angle', 30)
        
        # 1. Check Elbow Stability (Angle Based)
        if data['elbow_angle'] < 160:
            if shoulder_tuck_angle > drift_threshold:
                data['form_violations'].append(f"Elbow swinging! ({int(shoulder_tuck_angle)}°)")
                data['current_rep_quality']['elbow_stable'] = False
                data['form_valid'] = False
        
        # 2. Check Elbow Stability (Distance Based)
        if data['calibration_state'] == "READY" and data['elbow_shoulder_distance_baseline'] is not None:
            distance_deviation = abs(elbow_shoulder_distance - data['elbow_shoulder_distance_baseline'])
            tolerance_percentage = self.thresholds.get('elbow_distance_tolerance', 0.15)
            max_allowed_deviation = data['elbow_shoulder_distance_baseline'] * tolerance_percentage
            
            if data['elbow_angle'] < 160:
                if distance_deviation > max_allowed_deviation:
                    deviation_percent = (distance_deviation / data['elbow_shoulder_distance_baseline']) * 100
                    data['form_violations'].append(f"Elbow moving ({int(deviation_percent)}% drift)")
                    data['current_rep_quality']['elbow_stable'] = False
                    data['form_valid'] = False
        
        # 3. Check Elbow Velocity (Movement Speed)
        if len(data['elbow_velocity_history']) >= 5 and data['calibration_state'] == "READY":
            recent_velocities = list(data['elbow_velocity_history'])[-5:]
            avg_velocity = np.mean(recent_velocities)
            max_allowed_velocity = self.thresholds.get('max_elbow_velocity', 0.015)
            
            if data['elbow_angle'] < 160:
                if avg_velocity > max_allowed_velocity:
                    velocity_percent = int((avg_velocity / max_allowed_velocity) * 100)
                    data['form_violations'].append(f"Elbow moving too fast ({velocity_percent}%)")
                    data['current_rep_quality']['elbow_stable'] = False
                    data['form_valid'] = False
        
        # 4. Check shoulder stability (No Shrugging)
        if data['calibration_state'] == "READY" and data['shoulder_y_baseline'] is not None:
            shoulder_lift = data['shoulder_y_baseline'] - shoulder['y']
            max_allowed_lift = self.thresholds['shoulder_shrug_threshold']
            
            if shoulder_lift > max_allowed_lift:
                data['form_violations'].append("Shoulders shrugging")
                data['current_rep_quality']['no_shrugging'] = False
                data['form_valid'] = False
        
        # 5. Check torso stability (No Leaning)
        if data['calibration_state'] == "READY" and data['torso_baseline_angle'] is not None:
            torso_lean = abs(torso_angle - data['torso_baseline_angle'])
            max_allowed_lean = self.thresholds['torso_lean_threshold']
            
            if torso_lean > max_allowed_lean:
                data['form_violations'].append("Torso leaning")
                data['current_rep_quality']['no_leaning'] = False
                data['form_valid'] = False
    
    def _is_form_critically_bad(self, arm):
        """Check if form has been bad for enough consecutive frames to fail the rep."""
        data = self.arm_data[arm]
        return data['consecutive_bad_form_frames'] >= data['bad_form_threshold']
    
    def _calculate_baselines(self, arm):
        """Calculate baseline measurements from calibration frames."""
        data = self.arm_data[arm]
        if len(data['calibration_frames']) == 0:
            return
        
        shoulder_y_values = [frame['shoulder_y'] for frame in data['calibration_frames']]
        torso_angle_values = [frame['torso_angle'] for frame in data['calibration_frames']]
        elbow_shoulder_distance_values = [frame['elbow_shoulder_distance'] for frame in data['calibration_frames']]
        
        data['shoulder_y_baseline'] = sum(shoulder_y_values) / len(shoulder_y_values)
        data['torso_baseline_angle'] = sum(torso_angle_values) / len(torso_angle_values)
        data['elbow_shoulder_distance_baseline'] = sum(elbow_shoulder_distance_values) / len(elbow_shoulder_distance_values)
        
        print(f"   Elbow-Shoulder distance baseline: {data['elbow_shoulder_distance_baseline']:.3f}")
        print(f"   Baseline shoulder Y: {data['shoulder_y_baseline']:.3f}")
        print(f"   Baseline torso angle: {data['torso_baseline_angle']:.1f}°")
    
    def _check_smooth_movement(self, arm):
        """Check if movement is smooth and controlled for a specific arm."""
        data = self.arm_data[arm]
        if len(data['angle_history']) < 5:
            return True
        
        angles = list(data['angle_history'])[-5:]
        velocities = [abs(angles[i] - angles[i-1]) for i in range(1, len(angles))]
        avg_velocity = sum(velocities) / len(velocities)
        
        max_velocity_change = self.thresholds.get('max_velocity_change', 15)
        if avg_velocity > max_velocity_change:
            return False
        return True
    
    def update_rep_count(self, all_metrics):
        """State machine for rep counting using difficulty profile thresholds."""
        if not all_metrics or not self.difficulty_profile:
            return
        
        # Update rep count for each active arm
        for arm, metrics in all_metrics.items():
            if not metrics or metrics['calibration_state'] == "CALIBRATING":
                continue
            
            data = self.arm_data[arm]
            angle = metrics['elbow_angle']
            
            # Use profile-specific thresholds
            elbow_down_min = self.difficulty_profile['elbow_down_min']
            elbow_up_max = self.difficulty_profile['elbow_up_max']
            
            # State transitions
            if data['rep_state'] == "WAITING":
                if angle >= elbow_down_min:
                    if data['form_valid']:
                        data['rep_state'] = "CURLING_UP"
                        self._reset_rep_quality(arm)
                        data['feedback'] = [f"{arm}: Starting rep"]
            
            elif data['rep_state'] == "CURLING_UP":
                if angle <= elbow_up_max:
                    if data['form_valid'] and self._check_smooth_movement(arm):
                        data['rep_state'] = "AT_TOP"
                        data['feedback'] = [f"{arm}: Good curl"]
                    elif self._is_form_critically_bad(arm):
                        self._abort_rep(arm)
                elif self._is_form_critically_bad(arm):
                    self._abort_rep(arm)
            
            elif data['rep_state'] == "AT_TOP":
                if angle > elbow_up_max + 10:
                    data['rep_state'] = "LOWERING_DOWN"
            
            elif data['rep_state'] == "LOWERING_DOWN":
                if angle >= elbow_down_min:
                    if data['form_valid'] and self._check_smooth_movement(arm) and self._rep_quality_met(arm):
                        self._complete_rep(arm)
                    elif self._is_form_critically_bad(arm):
                        self._abort_rep(arm)
                elif self._is_form_critically_bad(arm):
                    self._abort_rep(arm)
    
    def _reset_rep_quality(self, arm):
        data = self.arm_data[arm]
        data['current_rep_quality'] = {k: True for k in data['current_rep_quality']}
    
    def _rep_quality_met(self, arm):
        data = self.arm_data[arm]
        return all(data['current_rep_quality'].values())
    
    def _complete_rep(self, arm):
        data = self.arm_data[arm]
        data['rep_count'] += 1
        data['rep_state'] = "WAITING"
        data['consecutive_bad_form_frames'] = 0
        data['feedback'] = [f"✓ {arm} REP {data['rep_count']} COUNTED!"]
        self.session_data['total_reps'] += 1
        print(f"✓ {arm} ARM REP {data['rep_count']} - Perfect form!")
    
    def _abort_rep(self, arm):
        data = self.arm_data[arm]
        data['rep_state'] = "WAITING"
        data['consecutive_bad_form_frames'] = 0
        if data['form_violations']:
            data['feedback'] = [f"{arm}: NOT counted:"] + data['form_violations']
            for violation in data['form_violations']:
                self.session_data['common_errors'][violation] += 1
        else:
            data['feedback'] = [f"{arm}: NOT counted"]
        print(f"✗ {arm} ARM - Partial rep: {', '.join(data['form_violations'])}")

    def draw_feedback_panel(self, image):
        """Draw comprehensive feedback panel for both arms."""
        h, w = image.shape[:2]
        panel_width = 450
        
        # Overlay
        overlay = image.copy()
        cv2.rectangle(overlay, (w - panel_width, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # 1. Main Info
        y_offset = 40
        cv2.putText(image, "BICEP CURL TRACKER", (w - panel_width + 20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        y_offset += 40
        
        # Display info for each active arm
        for arm in self.active_arms:
            data = self.arm_data[arm]
            
            # Arm label
            cv2.putText(image, f"=== {arm} ARM ===", (w - panel_width + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['warning'], 2)
            y_offset += 30
            
            # Calibration status
            if data['calibration_state'] == "CALIBRATING":
                cv2.putText(image, f"CAL: {data['calibration_frame_count']}/{data['calibration_target_frames']}", 
                           (w - panel_width + 30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['warning'], 1)
            else:
                cv2.putText(image, "READY", (w - panel_width + 30, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['good'], 1)
            y_offset += 25
            
            # Reps
            cv2.putText(image, f"REPS: {data['rep_count']}", (w - panel_width + 30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['good'], 2)
            y_offset += 30
            
            # Phase
            phase_text = data['rep_state'].replace('_', ' ')
            phase_color = self.colors['warning'] if data['rep_state'] == "CURLING_UP" else self.colors['good']
            cv2.putText(image, phase_text, (w - panel_width + 30, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, phase_color, 1)
            y_offset += 25
            
            # Joint angles
            cv2.putText(image, f"Elbow: {int(data['elbow_angle'])}°", 
                       (w - panel_width + 30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            y_offset += 18
            cv2.putText(image, f"Shoulder: {int(data['shoulder_tuck_angle'])}°", 
                       (w - panel_width + 30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            y_offset += 20
            
            # Form checks (compact)
            checks = [
                ("Stable", data['current_rep_quality']['elbow_stable']),
                ("No Shrug", data['current_rep_quality']['no_shrugging']),
                ("No Lean", data['current_rep_quality']['no_leaning'])
            ]
            
            for check_name, check_status in checks:
                symbol = "✓" if check_status else "✗"
                color = self.colors['good'] if check_status else self.colors['bad']
                cv2.putText(image, f"{symbol} {check_name}", (w - panel_width + 35, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                y_offset += 16
            
            # Feedback
            y_offset += 10
            for msg in data['feedback'][:2]:
                msg_color = self.colors['bad'] if "NOT" in msg else self.colors['good']
                if len(msg) > 35:
                    msg = msg[:32] + "..."
                cv2.putText(image, msg, (w - panel_width + 35, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, msg_color, 1)
                y_offset += 16
            
            y_offset += 20  # Space between arms
                        
    def draw_angle_on_body(self, image, landmarks):
        """Draw the elbow angle and tuck angle on both arms."""
        if not landmarks or not self.active_arms:
            return
        
        h, w, _ = image.shape
        
        # Draw for each active arm
        for arm in self.active_arms:
            try:
                data = self.arm_data[arm]
                
                # Get landmarks
                shoulder = self.get_landmark_features(landmarks, f'{arm}_SHOULDER')
                elbow = self.get_landmark_features(landmarks, f'{arm}_ELBOW')
                wrist = self.get_landmark_features(landmarks, f'{arm}_WRIST')
                hip = self.get_landmark_features(landmarks, f'{arm}_HIP')
                
                if not all([shoulder, elbow, wrist, hip]):
                    continue
                if min(shoulder['visibility'], elbow['visibility'], wrist['visibility']) < 0.5:
                    continue
                
                # Pixels
                sx, sy = int(shoulder['x'] * w), int(shoulder['y'] * h)
                ex, ey = int(elbow['x'] * w), int(elbow['y'] * h)
                wx, wy = int(wrist['x'] * w), int(wrist['y'] * h)
                hx, hy = int(hip['x'] * w), int(hip['y'] * h)
                
                # Draw Skeleton
                cv2.line(image, (sx, sy), (ex, ey), self.colors['text'], 3)
                cv2.line(image, (ex, ey), (wx, wy), self.colors['text'], 3)
                cv2.line(image, (hx, hy), (sx, sy), (100, 100, 100), 2)
                
                # Draw joint circles
                cv2.circle(image, (sx, sy), 8, self.colors['warning'], -1)
                cv2.circle(image, (ex, ey), 8, self.colors['good'], -1)
                cv2.circle(image, (wx, wy), 8, self.colors['warning'], -1)
                cv2.circle(image, (hx, hy), 8, (128, 128, 128), -1)
                
                # Draw elbow angle
                angle_color = self._get_angle_color(data['elbow_angle'])
                angle_text = f"{int(data['elbow_angle'])}°"
                
                # Position text based on arm side
                if arm == 'LEFT':
                    cv2.putText(image, angle_text, (ex + 20, ey),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, angle_color, 3)
                else:  # RIGHT
                    cv2.putText(image, angle_text, (ex - 100, ey),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, angle_color, 3)
                
                # Draw shoulder tuck angle
                tuck_valid = data['shoulder_tuck_angle'] < self.thresholds.get('elbow_drift_angle', 30)
                tuck_color = self.colors['good'] if tuck_valid else self.colors['bad']
                
                if arm == 'LEFT':
                    cv2.putText(image, f"S:{int(data['shoulder_tuck_angle'])}°", (sx + 10, sy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuck_color, 2)
                else:
                    cv2.putText(image, f"S:{int(data['shoulder_tuck_angle'])}°", (sx - 80, sy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuck_color, 2)
                
                # Visualize error
                if not tuck_valid:
                    cv2.circle(image, (sx, sy), 10, self.colors['bad'], -1)

            except Exception as e:
                print(f"Error drawing angle for {arm}: {e}")

    def _get_angle_color(self, angle):
        if angle > self.thresholds['start_angle']: return self.colors['good']
        elif angle < self.thresholds['top_angle']: return self.colors['good']
        else: return self.colors['warning']

    def run(self):
        """Main loop."""
        if self.difficulty_level is None: self.select_difficulty_level()
        
        cap = cv2.VideoCapture(0)
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n=== STARTING BICEP CURL TRACKER ===")
        print("Press 'Q' to Quit, 'R' to Reset")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Analyze
                metrics = self.analyze_pose(results.pose_landmarks.landmark)
                if metrics: self.update_rep_count(metrics)
                
                # Draw
                self.draw_angle_on_body(frame, results.pose_landmarks.landmark)
            
            self.draw_feedback_panel(frame)
            cv2.imshow('Bicep Curl Form Checker', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'): self.reset_session()
        
        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
        self._print_session_summary()

    def reset_session(self):
        """Reset session data for both arms."""
        super().reset_session() if hasattr(super(), 'reset_session') else None
        self.active_arms = []
        
        # Reset both arms
        for arm in ['LEFT', 'RIGHT']:
            self.arm_data[arm] = self._create_arm_state()
        
        print("\n✓ Session reset - Both arms")

    def _print_session_summary(self):
        """Print session summary for both arms."""
        total_reps_left = self.arm_data['LEFT']['rep_count']
        total_reps_right = self.arm_data['RIGHT']['rep_count']
        
        print(f"\n=== SESSION COMPLETE ===")
        print(f"LEFT ARM:  {total_reps_left} reps")
        print(f"RIGHT ARM: {total_reps_right} reps")
        print(f"TOTAL:     {total_reps_left + total_reps_right} reps")