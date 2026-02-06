import pytest
import numpy as np
from realsense_frame.stability import StabilityDetector

def generate_imu_entry(val, ts, type_str="accel"):
    return {"ts": ts, "type": type_str, "x": val[0], "y": val[1], "z": val[2]}

def test_insufficient_history():
    detector = StabilityDetector(history_size=10, has_accel=True, has_gyro=True)
    for i in range(5):
        entry = generate_imu_entry([0, 0, 9.8], i)
        detector.add_accel(entry)
        detector.add_gyro(entry)
    
    assert detector.is_stable() is False

def test_stable_state():
    detector = StabilityDetector(history_size=10, threshold=0.05, has_accel=True, has_gyro=True)
    for i in range(10):
        noise = np.random.normal(0, 0.01, 3)
        detector.add_accel(generate_imu_entry([0, 0, 9.8] + noise, i))
        detector.add_gyro(generate_imu_entry([0, 0, 0] + noise, i))
    
    assert detector.is_stable() is True

def test_unstable_accel():
    detector = StabilityDetector(history_size=10, threshold=0.05, has_accel=True, has_gyro=False)
    for i in range(10):
        accel = [0, 0, 9.8] + np.random.normal(0, 0.2, 3) # High noise
        detector.add_accel(generate_imu_entry(accel, i))
    
    assert detector.is_stable() is False

def test_unstable_gyro():
    detector = StabilityDetector(history_size=10, threshold=0.05, has_accel=False, has_gyro=True)
    for i in range(10):
        gyro = [0, 0, 0] + np.random.normal(0, 0.2, 3) # High noise
        detector.add_gyro(generate_imu_entry(gyro, i))
    
    assert detector.is_stable() is False

def test_only_one_sensor_needed():
    """Verify it works if only one sensor is enabled."""
    detector = StabilityDetector(history_size=10, threshold=0.05, has_accel=True, has_gyro=False)
    for i in range(10):
        detector.add_accel(generate_imu_entry([0, 0, 9.8], i))
    
    # Should be stable because accel is stable and gyro is ignored
    assert detector.is_stable() is True
