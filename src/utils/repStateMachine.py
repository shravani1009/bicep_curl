"""
Rep State Machine - Generic rep counting logic for any exercise.
"""

from enum import Enum
from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class RepState(Enum):
    """Generic rep states for any exercise."""
    WAITING = "WAITING"
    MOVING_TO_PEAK = "MOVING_TO_PEAK"
    AT_PEAK = "AT_PEAK"
    RETURNING = "RETURNING"


class RepStateMachine:
    """
    Generic rep counting state machine.
    Works for any exercise with: start position → peak position → return cycle.
    """
    
    def __init__(self, limb_name=""):
        """
        Initialize rep state machine.
        
        Args:
            limb_name (str): Name of limb (e.g., 'LEFT ARM', 'RIGHT LEG')
        """
        self.limb_name = limb_name
        self.state = RepState.WAITING
        self.rep_count = 0
        self.form_valid = True
        self.current_feedback = None
        self.previous_angle = None
    
    def update(self, current_value, start_threshold, peak_threshold, 
               form_valid=True, tolerance=10):
        """
        Update state machine with current measurement.
        
        Args:
            current_value (float): Current angle/distance measurement
            start_threshold (float): Value at start position (e.g., 160° for bicep curl down)
            peak_threshold (float): Value at peak position (e.g., 50° for bicep curl up)
            form_valid (bool): Is form currently valid?
            tolerance (float): Tolerance for thresholds
            
        Returns:
            dict: {
                'rep_completed': bool,
                'state': RepState,
                'feedback': str,
                'rep_count': int
            }
        """
        self.form_valid = form_valid
        rep_completed = False
        feedback = None
        
        # Track angle changes
        if self.previous_angle is not None:
            angle_change = self.previous_angle - current_value
        else:
            angle_change = 0
        self.previous_angle = current_value
        
        # === STATE MACHINE ===
        
        # WAITING → MOVING_TO_PEAK: At start position, begin movement
        if self.state == RepState.WAITING:
            if self._is_at_start(current_value, start_threshold, tolerance):
                # Check if starting to move towards peak
                if angle_change > 5:  # Moving from 160° towards 50° (decreasing)
                    if form_valid:
                        self.state = RepState.MOVING_TO_PEAK
                        feedback = "Starting rep..."
                        logger.debug(f"{self.limb_name} - Starting rep (angle: {current_value:.1f}°)")
        
        # MOVING_TO_PEAK → AT_PEAK: Reached peak position
        elif self.state == RepState.MOVING_TO_PEAK:
            if self._is_at_peak(current_value, peak_threshold, tolerance):
                if form_valid:
                    self.state = RepState.AT_PEAK
                    feedback = "Peak reached!"
                    logger.debug(f"{self.limb_name} - At peak (angle: {current_value:.1f}°)")
                else:
                    # Bad form during curl up
                    self.state = RepState.WAITING
                    feedback = "Rep aborted - Fix form"
                    logger.warning(f"{self.limb_name} - Rep aborted (poor form going up)")
            
            elif not form_valid:
                # Form broke while moving to peak
                self.state = RepState.WAITING
                feedback = "Rep aborted - Fix form"
                logger.warning(f"{self.limb_name} - Rep aborted (form broke)")
            
            elif angle_change < -5:  # Moving back towards start (angle increasing)
                # Didn't reach peak, moving back
                self.state = RepState.WAITING
                feedback = "Incomplete curl"
                logger.debug(f"{self.limb_name} - Incomplete curl (didn't reach peak)")
        
        # AT_PEAK → RETURNING: Started returning to start
        elif self.state == RepState.AT_PEAK:
            # Check if angle is increasing (moving back towards 160°)
            if angle_change < -3:  # Angle increasing = returning
                self.state = RepState.RETURNING
                feedback = "Returning to start..."
                logger.debug(f"{self.limb_name} - Returning to start")
        
        # RETURNING → WAITING: Completed full rep!
        elif self.state == RepState.RETURNING:
            if self._is_at_start(current_value, start_threshold, tolerance):
                if form_valid:
                    self.rep_count += 1
                    rep_completed = True
                    feedback = f"✓ REP {self.rep_count} COMPLETE!"
                    logger.info(f"{self.limb_name} - Rep {self.rep_count} completed")
                else:
                    feedback = "Rep NOT counted - Poor form on return"
                    logger.warning(f"{self.limb_name} - Rep not counted (bad form on return)")
                
                self.state = RepState.WAITING
            
            elif not form_valid:
                # Form broke while returning
                self.state = RepState.WAITING
                feedback = "Rep NOT counted - Form broke"
                logger.warning(f"{self.limb_name} - Rep not counted (form broke)")
            
            elif self._is_at_peak(current_value, peak_threshold, tolerance + 10):
                # Went back to peak instead of returning to start
                self.state = RepState.AT_PEAK
                logger.debug(f"{self.limb_name} - Back at peak")
        
        self.current_feedback = feedback
        
        return {
            'rep_completed': rep_completed,
            'state': self.state,
            'feedback': feedback,
            'rep_count': self.rep_count
        }
    
    def _is_at_start(self, value, start_threshold, tolerance):
        """Check if at start position."""
        return value >= start_threshold - tolerance
    
    def _is_at_peak(self, value, peak_threshold, tolerance):
        """Check if at peak position (arm curled for bicep curl)."""
        return value <= peak_threshold + tolerance
    
    def reset(self):
        """Reset state machine."""
        self.state = RepState.WAITING
        self.rep_count = 0
        self.form_valid = True
        self.current_feedback = None
        self.previous_angle = None
        logger.info(f"{self.limb_name} - State machine reset")
    
    def get_state_name(self):
        """Get current state as readable string."""
        state_names = {
            RepState.WAITING: "READY",
            RepState.MOVING_TO_PEAK: "CURLING UP",
            RepState.AT_PEAK: "AT TOP",
            RepState.RETURNING: "LOWERING"
        }
        return state_names.get(self.state, self.state.value)
