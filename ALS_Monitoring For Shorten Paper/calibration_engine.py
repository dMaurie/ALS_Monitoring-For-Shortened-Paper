# calibration_engine.py
import numpy as np

class PersonalizedCalibrationEngine:
    """
    Manages the 30-day silent baseline calibration and transitions 
    the system to active real-time Z-score normalization on Day 31.
    """
    def __init__(self, calibration_days=30, epsilon=1e-6):
        self.calibration_days = calibration_days
        self.epsilon = epsilon
        self.raw_history_buffer = []  # Passive buffer for baseline collection
        self.is_calibrated = False     # Calibration state flag
        self.user_means = None         # Baseline mean vector (mu)
        self.user_stds = None          # Baseline standard deviation vector (sigma)

    def process_day(self, raw_telemetry_vector):
        """
        Routes incoming raw daily data.
        Returns None during calibration, and normalized Z-scores on Day 31+.
        """
        if not self.is_calibrated:
            self.raw_history_buffer.append(raw_telemetry_vector)
            
            # If the calibration window has been met, finalize parameters
            if len(self.raw_history_buffer) == self.calibration_days:
                self._compute_baseline_parameters()
            return None # Inference remains deactivated to preserve behavioral purity
        else:
            return self._normalize(raw_telemetry_vector)

    def _compute_baseline_parameters(self):
        matrix = np.array(self.raw_history_buffer)
        
        # Calculate static baseline parameters across the 30-day window
        self.user_means = np.mean(matrix, axis=0)
        self.user_stds = np.std(matrix, axis=0) + self.epsilon
        
        self.is_calibrated = True
        
        # Memory Optimization: Immediate RAM purge of the raw history array
        self.raw_history_buffer = []

    def _normalize(self, raw_telemetry_vector):
        raw_array = np.array(raw_telemetry_vector)
        # Static Z-Score Transformation: (Current Raw - Locked Baseline Mean) / Locked Baseline Std
        return (raw_array - self.user_means) / self.user_stds