import numpy as np


class StabilityDetector:
    def __init__(self, history_size=100, threshold=0.05, has_accel=True, has_gyro=True):
        self.history_size = history_size
        self.threshold = threshold
        self.has_accel = has_accel
        self.has_gyro = has_gyro
        self.accel_history = []
        self.gyro_history = []

    def add_accel(self, entry):
        self.accel_history.append(entry)
        if len(self.accel_history) > self.history_size:
            self.accel_history.pop(0)

    def add_gyro(self, entry):
        self.gyro_history.append(entry)
        if len(self.gyro_history) > self.history_size:
            self.gyro_history.pop(0)

    def is_stable(self):
        checks = []
        if self.has_accel:
            if len(self.accel_history) < self.history_size:
                return False
            vals = np.array([[e["x"], e["y"], e["z"]] for e in self.accel_history])
            checks.append(np.all(np.std(vals, axis=0) < self.threshold))

        if self.has_gyro:
            if len(self.gyro_history) < self.history_size:
                return False
            vals = np.array([[e["x"], e["y"], e["z"]] for e in self.gyro_history])
            checks.append(np.all(np.std(vals, axis=0) < self.threshold))

        return all(checks) if checks else False

    def get_stability_score(self):
        stds = []
        if self.has_accel and len(self.accel_history) >= self.history_size:
            vals = np.array([[e["x"], e["y"], e["z"]] for e in self.accel_history])
            stds.append(np.max(np.std(vals, axis=0)))

        if self.has_gyro and len(self.gyro_history) >= self.history_size:
            vals = np.array([[e["x"], e["y"], e["z"]] for e in self.gyro_history])
            stds.append(np.max(np.std(vals, axis=0)))

        return max(stds) if stds else 1.0  # Default to high score if not enough data
