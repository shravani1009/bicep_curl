"""
Bicep Curl Exercise Checker - Simplified Version
Monitors proper bicep curl form with essential checks only.
"""

import cv2
import numpy as np
from src.exercises.base_exercise import BaseExercise
from src.utils.angle_calculator import AngleCalculator
from config.exercise_config import BICEP_CURL_CONFIG, COLORS


class BicepCurlChecker(BaseExercise):
    """
    Simplified bicep curl form checker.
    
    Monitors:
    1. Elbow angle (shoulder-elbow-wrist) - main rep tracking
    2. Elbow stability (distance from shoulder) - prevents swinging
    3. Shoulder shrug check
    """
    
    def __init__(self):
        super().__init__()
        self.active_arms = []
        self.arm_data = {
            'LEFT': self._create_arm_state(),
            'RIGHT': self._create_arm_state()
        }
        self.difficulty_level = None
        self.thresholds = None
    
    def _create_arm_state(self):
        """Create state tracking for each arm."""
        return {
            'rep_state': 'WAITING',
            'rep_count': 0,
            'elbow_angle': 180,
            'elbow_shoulder_distance': 0,
            'elbow_shoulder_baseline': None,
            'shoulder_y_baseline': None,
            'baseline_elbow_pos': None,
            'baseline_shoulder_pos': None,
            'form_valid': True,
            'feedback': [],
            'last_console_feedback': None
        }
    
    def get_exercise_name(self):
        return "Bicep Curl"
    
    def select_difficulty_level(self):
        """Select difficulty level."""
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
                    print(f"\n✓ Selected: {self.difficulty_level} level")
                    return
                else:
                    print("Invalid choice.")
            except Exception as e:
                print(f"Error: {e}")
    
    def _detect_active_arms(self, landmarks):
        """Detect which arms are visible."""
        try:
            left_shoulder = self.get_landmark_features(landmarks, 'LEFT_SHOULDER')
            left_elbow = self.get_landmark_features(landmarks, 'LEFT_ELBOW')
            left_wrist = self.get_landmark_features(landmarks, 'LEFT_WRIST')
            
            right_shoulder = self.get_landmark_features(landmarks, 'RIGHT_SHOULDER')
            right_elbow = self.get_landmark_features(landmarks, 'RIGHT_ELBOW')
            right_wrist = self.get_landmark_features(landmarks, 'RIGHT_WRIST')
            
            active_arms = []
            left_visibility = (left_shoulder['visibility'] + left_elbow['visibility'] + left_wrist['visibility']) / 3
            right_visibility = (right_shoulder['visibility'] + right_elbow['visibility'] + right_wrist['visibility']) / 3
            
            if left_visibility > 0.5:
                active_arms.append('LEFT')
            if right_visibility > 0.5:
                active_arms.append('RIGHT')
            
            return active_arms
        except Exception:
            return []
    
    def analyze_pose(self, landmarks):
        """Analyze bicep curl form for active arms."""
        if not landmarks:
            return None
        
        try:
            detected_arms = self._detect_active_arms(landmarks)
            if not detected_arms:
                return None
            
            self.active_arms = detected_arms
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
        """Analyze a single arm's form."""
        try:
            data = self.arm_data[arm]
            
            shoulder = self.get_landmark_features(landmarks, f'{arm}_SHOULDER')
            elbow = self.get_landmark_features(landmarks, f'{arm}_ELBOW')
            wrist = self.get_landmark_features(landmarks, f'{arm}_WRIST')
            
            if not all([shoulder, elbow, wrist]):
                return None
            
            if min(shoulder['visibility'], elbow['visibility'], wrist['visibility']) < 0.3:
                return None
            
            shoulder_point = (shoulder['x'], shoulder['y'])
            elbow_point = (elbow['x'], elbow['y'])
            wrist_point = (wrist['x'], wrist['y'])
            
            elbow_angle = AngleCalculator.calculate_angle(shoulder_point, elbow_point, wrist_point)
            data['elbow_angle'] = elbow_angle
            data['elbow_shoulder_distance'] = AngleCalculator.calculate_distance(shoulder_point, elbow_point)
            
            if data['elbow_shoulder_baseline'] is None and elbow_angle > 160:
                data['elbow_shoulder_baseline'] = data['elbow_shoulder_distance']
                data['shoulder_y_baseline'] = shoulder['y']
                data['baseline_elbow_pos'] = elbow_point
                data['baseline_shoulder_pos'] = shoulder_point
                print(f"✓ {arm} arm baseline set - Keep your elbow in the target zone!")
            
            self._validate_form(arm, shoulder)
            self._update_rep_count(arm)
            
            return {
                'elbow_angle': data['elbow_angle'],
                'form_valid': data['form_valid'],
                'rep_state': data['rep_state'],
                'rep_count': data['rep_count']
            }
            
        except Exception as e:
            print(f"Error analyzing {arm} arm: {e}")
            return None
    
    def _validate_form(self, arm, shoulder):
        """Validate form - check elbow stability and shoulder shrug."""
        data = self.arm_data[arm]
        data['form_valid'] = True
        data['feedback'] = []
        console_msg = None
        
        if data['elbow_shoulder_baseline'] is None:
            return
        
        if data['elbow_angle'] < 160:
            distance_deviation = abs(data['elbow_shoulder_distance'] - data['elbow_shoulder_baseline'])
            tolerance = self.thresholds.get('elbow_distance_tolerance', 0.15)
            max_deviation = data['elbow_shoulder_baseline'] * tolerance
            
            if distance_deviation > max_deviation:
                data['form_valid'] = False
                deviation_pct = int((distance_deviation / data['elbow_shoulder_baseline']) * 100)
                data['feedback'].append(f"Elbow swinging ({deviation_pct}%)")
                console_msg = f"⚠️  {arm} ARM: Elbow swinging {deviation_pct}% - Keep elbow fixed!"
        
        if data['shoulder_y_baseline'] is not None:
            shoulder_lift = data['shoulder_y_baseline'] - shoulder['y']
            max_lift = self.thresholds.get('shoulder_shrug_threshold', 0.05)
            
            if shoulder_lift > max_lift:
                data['form_valid'] = False
                data['feedback'].append("Shoulders shrugging")
                if not console_msg:
                    console_msg = f"⚠️  {arm} ARM: Shoulders shrugging - Relax shoulders!"
        
        if console_msg and console_msg != data['last_console_feedback']:
            print(console_msg)
            data['last_console_feedback'] = console_msg
        elif data['form_valid'] and data['last_console_feedback']:
            data['last_console_feedback'] = None
    
    def _update_rep_count(self, arm):
        """Rep counting state machine."""
        data = self.arm_data[arm]
        angle = data['elbow_angle']
        elbow_down_min = self.thresholds['elbow_down_min']
        elbow_up_max = self.thresholds['elbow_up_max']
        
        if data['rep_state'] == 'WAITING':
            if angle >= elbow_down_min and data['form_valid']:
                data['rep_state'] = 'CURLING_UP'
                data['feedback'] = [f"{arm}: Starting rep"]
        
        elif data['rep_state'] == 'CURLING_UP':
            if angle <= elbow_up_max:
                if data['form_valid']:
                    data['rep_state'] = 'AT_TOP'
                    data['feedback'] = [f"{arm}: Good curl"]
                    print(f"✓ {arm} ARM: Reached top position - Good form!")
                else:
                    data['rep_state'] = 'WAITING'
                    data['feedback'] = [f"{arm}: Rep aborted"] + data['feedback']
                    print(f"✗ {arm} ARM: Rep aborted - {'; '.join(data['feedback'][1:])}")
            elif angle > elbow_down_min + 20:
                data['rep_state'] = 'WAITING'
        
        elif data['rep_state'] == 'AT_TOP':
            if angle > elbow_up_max + 10:
                data['rep_state'] = 'LOWERING_DOWN'
        
        elif data['rep_state'] == 'LOWERING_DOWN':
            if angle >= elbow_down_min:
                if data['form_valid']:
                    data['rep_count'] += 1
                    data['rep_state'] = 'WAITING'
                    data['feedback'] = [f"✓ {arm} REP {data['rep_count']}!"]
                    self.session_data['total_reps'] += 1
                    print(f"🎉 {arm} ARM: REP {data['rep_count']} COMPLETE - Perfect form!")
                else:
                    data['rep_state'] = 'WAITING'
                    data['feedback'] = [f"{arm}: NOT counted"] + data['feedback']
                    print(f"✗ {arm} ARM: Rep NOT counted - {'; '.join(data['feedback'][1:])}")
            elif angle < elbow_up_max:
                data['rep_state'] = 'AT_TOP'
    
    def draw_feedback_panel(self, image):
        """Draw feedback panel on the image."""
        h, w = image.shape[:2]
        panel_width = 400
        
        overlay = image.copy()
        cv2.rectangle(overlay, (w - panel_width, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        y_offset = 40
        cv2.putText(image, "BICEP CURL TRACKER", (w - panel_width + 20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        y_offset += 40
        
        for arm in self.active_arms:
            data = self.arm_data[arm]
            
            cv2.putText(image, f"=== {arm} ARM ===", (w - panel_width + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['warning'], 2)
            y_offset += 30
            
            cv2.putText(image, f"REPS: {data['rep_count']}", (w - panel_width + 30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['good'], 2)
            y_offset += 35
            
            state_text = data['rep_state'].replace('_', ' ')
            state_color = self.colors['warning'] if data['rep_state'] != 'WAITING' else self.colors['text']
            cv2.putText(image, state_text, (w - panel_width + 30, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)
            y_offset += 25
            
            cv2.putText(image, f"Angle: {int(data['elbow_angle'])}°", 
                       (w - panel_width + 30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            y_offset += 25
            
            form_text = "✓ Good Form" if data['form_valid'] else "✗ Check Form"
            form_color = self.colors['good'] if data['form_valid'] else self.colors['bad']
            cv2.putText(image, form_text, (w - panel_width + 30, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, form_color, 1)
            y_offset += 25
            
            for msg in data['feedback'][:2]:
                msg_color = self.colors['bad'] if "NOT" in msg or "aborted" in msg.lower() else self.colors['good']
                if len(msg) > 35:
                    msg = msg[:32] + "..."
                cv2.putText(image, msg, (w - panel_width + 35, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, msg_color, 1)
                y_offset += 18
            
            y_offset += 20
    
    def draw_exercise_annotations(self, image, landmarks):
        """Draw angle, key points, and visual guide box."""
        if not landmarks or not self.active_arms:
            return
        
        h, w, _ = image.shape
        
        for arm in self.active_arms:
            try:
                data = self.arm_data[arm]
                
                shoulder = self.get_landmark_features(landmarks, f'{arm}_SHOULDER')
                elbow = self.get_landmark_features(landmarks, f'{arm}_ELBOW')
                wrist = self.get_landmark_features(landmarks, f'{arm}_WRIST')
                
                if not all([shoulder, elbow, wrist]):
                    continue
                if min(shoulder['visibility'], elbow['visibility'], wrist['visibility']) < 0.5:
                    continue
                
                sx, sy = int(shoulder['x'] * w), int(shoulder['y'] * h)
                ex, ey = int(elbow['x'] * w), int(elbow['y'] * h)
                wx, wy = int(wrist['x'] * w), int(wrist['y'] * h)
                
                if data['elbow_shoulder_baseline'] is not None and data['baseline_elbow_pos'] is not None:
                    current_shoulder_norm = (shoulder['x'], shoulder['y'])
                    baseline_offset_x = data['baseline_elbow_pos'][0] - data['baseline_shoulder_pos'][0]
                    baseline_offset_y = data['baseline_elbow_pos'][1] - data['baseline_shoulder_pos'][1]
                    
                    target_elbow_norm_x = current_shoulder_norm[0] + baseline_offset_x
                    target_elbow_norm_y = current_shoulder_norm[1] + baseline_offset_y
                    
                    target_x = int(target_elbow_norm_x * w)
                    target_y = int(target_elbow_norm_y * h)
                    
                    tolerance = self.thresholds.get('elbow_distance_tolerance', 0.15)
                    baseline_distance_pixels = np.sqrt((target_x - sx)**2 + (target_y - sy)**2)
                    box_size = int(baseline_distance_pixels * tolerance * 2)
                    
                    current_distance = np.sqrt((ex - target_x)**2 + (ey - target_y)**2)
                    in_zone = current_distance < box_size / 2
                    
                    box_color = self.colors['good'] if in_zone else self.colors['bad']
                    box_half = box_size // 2
                    top_left = (target_x - box_half, target_y - box_half)
                    bottom_right = (target_x + box_half, target_y + box_half)
                    
                    overlay = image.copy()
                    cv2.rectangle(overlay, top_left, bottom_right, box_color, -1)
                    cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
                    cv2.rectangle(image, top_left, bottom_right, box_color, 2)
                    
                    cv2.circle(image, (target_x, target_y), 5, box_color, -1)
                    cv2.circle(image, (target_x, target_y), 5, (255, 255, 255), 1)
                    
                    if not in_zone:
                        cv2.line(image, (ex, ey), (target_x, target_y), self.colors['bad'], 2)
                        dx = target_x - ex
                        dy = target_y - ey
                        angle = np.arctan2(dy, dx)
                        arrow_len = 15
                        cv2.arrowedLine(image, (ex, ey), 
                                       (int(ex + arrow_len * np.cos(angle)), 
                                        int(ey + arrow_len * np.sin(angle))),
                                       self.colors['bad'], 2, tipLength=0.3)
                    
                    label_text = "TARGET ZONE" if in_zone else "MOVE ELBOW HERE"
                    label_color = self.colors['good'] if in_zone else self.colors['bad']
                    label_y = target_y - box_half - 10
                    if label_y < 20:
                        label_y = target_y + box_half + 20
                    
                    cv2.putText(image, label_text, (target_x - 60, label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
                
                cv2.line(image, (sx, sy), (ex, ey), self.colors['text'], 3)
                cv2.line(image, (ex, ey), (wx, wy), self.colors['text'], 3)
                
                cv2.circle(image, (sx, sy), 8, self.colors['warning'], -1)
                cv2.circle(image, (ex, ey), 10, self.colors['good'] if data['form_valid'] else self.colors['bad'], -1)
                cv2.circle(image, (ex, ey), 10, (255, 255, 255), 2)
                cv2.circle(image, (wx, wy), 8, self.colors['warning'], -1)
                
                angle_color = self.colors['good'] if data['form_valid'] else self.colors['bad']
                angle_text = f"{int(data['elbow_angle'])}°"
                
                if arm == 'LEFT':
                    cv2.putText(image, angle_text, (ex + 20, ey),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, angle_color, 3)
                else:
                    cv2.putText(image, angle_text, (ex - 100, ey),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, angle_color, 3)
                
            except Exception as e:
                print(f"Error drawing {arm}: {e}")
    
    def reset_session(self):
        """Reset session data."""
        super().reset_session()
        self.active_arms = []
        for arm in ['LEFT', 'RIGHT']:
            self.arm_data[arm] = self._create_arm_state()
        print("\n✓ Session reset - Baselines will be recalibrated")
    
    def get_session_specific_data(self):
        """Return arm-specific rep counts."""
        return {
            'left_arm_reps': self.arm_data['LEFT']['rep_count'],
            'right_arm_reps': self.arm_data['RIGHT']['rep_count']
        }
