import time
import numpy as np

class FeatureExtractor:
    def __init__(self, window_size=60):
        self.window_size = window_size  # seconds

        self.ear_values = []
        self.blink_count_start = 0
        self.start_time = time.time()
        self.last_feature_time = time.time()

    def update(self, ear, total_blinks, detected_frames, total_frames):
        self.ear_values.append(ear)

        current_time = time.time()

        # Check if window complete
        if current_time - self.last_feature_time >= self.window_size:
            duration = current_time - self.last_feature_time

            # Features
            blink_rate = (total_blinks - self.blink_count_start) / duration * 60
            ear_mean = np.mean(self.ear_values)
            ear_std = np.std(self.ear_values)
            session_time = current_time - self.start_time

            # Detection rate
            detection_rate = detected_frames / total_frames if total_frames > 0 else 0

            # Reset window
            self.ear_values = []
            self.blink_count_start = total_blinks
            self.last_feature_time = current_time

            return {
                "blink_rate": float(blink_rate),
                "ear_mean": float(ear_mean),
                "ear_std": float(ear_std),
                "session_time": float(session_time),
                "detection_rate": float(detection_rate)
            }

        return None