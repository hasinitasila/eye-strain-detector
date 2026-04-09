import csv
import os

class DataLogger:
    def __init__(self, filename="data.csv"):
        self.filename = filename

        # Create file with header if not exists
        if not os.path.exists(self.filename):
            with open(self.filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "blink_rate",
                    "ear_mean",
                    "ear_std",
                    "session_time",
                    "label"
                ])

    def log(self, features, label):
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                features["blink_rate"],
                features["ear_mean"],
                features["ear_std"],
                features["session_time"],
                label
            ])