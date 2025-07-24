"""
Sensor Simulation Module

This module converts ground truth trajectory data into synthetic IMU and GNSS measurements.
It simulates realistic sensor behavior including noise, bias, and measurement characteristics.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, Tuple, Optional
import warnings


class SensorSimulator:
    """
    Simulates IMU and GNSS measurements from ground truth trajectory data.
    
    This class implements realistic sensor simulation including:
    - Accelerometer measurements with gravity compensation
    - Gyroscope measurements from angular velocity
    - GNSS position measurements with noise and dropout
    """
    
    def __init__(
        self,
        accel_noise_std: float = 0.1,
        gyro_noise_std: float = 0.01,
        accel_bias_std: float = 0.01,
        gyro_bias_std: float = 0.001,
        gnss_noise_std: float = 1.0,
        gnss_rate: float = 1.0,
        gravity: np.ndarray = np.array([0, 0, -9.81])
    ):
        """
        Initialize the sensor simulator.
        
        Args:
            accel_noise_std: Standard deviation of accelerometer noise (m/s²)
            gyro_noise_std: Standard deviation of gyroscope noise (rad/s)
            accel_bias_std: Standard deviation of accelerometer bias (m/s²)
            gyro_bias_std: Standard deviation of gyroscope bias (rad/s)
            gnss_noise_std: Standard deviation of GNSS position noise (m)
            gnss_rate: GNSS update frequency (Hz)
            gravity: Gravity vector in world frame (m/s²)
        """
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.accel_bias_std = accel_bias_std
        self.gyro_bias_std = gyro_bias_std
        self.gnss_noise_std = gnss_noise_std
        self.gnss_rate = gnss_rate
        self.gravity = gravity
        
        # Initialize sensor biases (constant for the simulation)
        self.accel_bias = np.random.randn(3) * accel_bias_std
        self.gyro_bias = np.random.randn(3) * gyro_bias_std
    
    def compute_acceleration(
        self,
        velocities: np.ndarray,
        timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Compute acceleration from velocity data using finite differences.
        
        Args:
            velocities: Velocity vectors (N, 3)
            timestamps: Time vector (N,)
            
        Returns:
            Acceleration vectors (N, 3)
        """
        N = len(velocities)
        accelerations = np.zeros_like(velocities)
        
        if N < 2:
            return accelerations
        
        # Compute time steps
        dt = np.diff(timestamps)
        
        # Central differences for interior points
        if N > 2:
            # Reshape denominator to (N-2, 1) for proper broadcasting with 3D vectors
            dt_central = (dt[:-1] + dt[1:]).reshape(-1, 1)
            accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / dt_central
        
        # Forward/backward differences for boundary points
        if N > 1:
            accelerations[0] = (velocities[1] - velocities[0]) / dt[0]
            accelerations[-1] = (velocities[-1] - velocities[-2]) / dt[-1]
        
        return accelerations
    
    def quaternion_to_angular_velocity(
        self,
        orientations: np.ndarray,
        timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Compute angular velocity from quaternion orientation data.
        
        Args:
            orientations: Quaternion orientations (N, 4) [w, x, y, z]
            timestamps: Time vector (N,)
            
        Returns:
            Angular velocity vectors (N, 3) in rad/s
        """
        N = len(orientations)
        angular_velocities = np.zeros((N, 3))
        
        if N < 2:
            return angular_velocities
        
        # Compute time steps
        dt = np.diff(timestamps)
        
        # Convert quaternions to rotation objects
        rotations = R.from_quat(orientations)
        
        # Compute angular velocities using quaternion differences
        for i in range(1, N):
            # Get consecutive quaternions
            q1 = orientations[i-1]
            q2 = orientations[i]
            
            # Compute quaternion difference
            q_diff = self._quaternion_difference(q1, q2)
            
            # Convert to angular velocity (simplified approximation)
            # For small rotations: ω ≈ 2 * q_vec / dt
            angular_velocities[i] = 2.0 * q_diff[1:4] / dt[i-1]
        
        # Set first value to second value to avoid discontinuity
        if N > 1:
            angular_velocities[0] = angular_velocities[1]
        
        return angular_velocities
    
    def _quaternion_difference(
        self,
        q1: np.ndarray,
        q2: np.ndarray
    ) -> np.ndarray:
        """
        Compute the quaternion difference q2 * q1^(-1).
        
        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]
            
        Returns:
            Quaternion difference [w, x, y, z]
        """
        # Normalize quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Compute quaternion difference: q_diff = q2 * q1^(-1)
        # For unit quaternions: q^(-1) = [w, -x, -y, -z]
        q1_conj = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
        
        # Quaternion multiplication
        w = q2[0] * q1_conj[0] - q2[1] * q1_conj[1] - q2[2] * q1_conj[2] - q2[3] * q1_conj[3]
        x = q2[0] * q1_conj[1] + q2[1] * q1_conj[0] + q2[2] * q1_conj[3] - q2[3] * q1_conj[2]
        y = q2[0] * q1_conj[2] - q2[1] * q1_conj[3] + q2[2] * q1_conj[0] + q2[3] * q1_conj[1]
        z = q2[0] * q1_conj[3] + q2[1] * q1_conj[2] - q2[2] * q1_conj[1] + q2[3] * q1_conj[0]
        
        return np.array([w, x, y, z])
    
    def simulate_imu_measurements(
        self,
        timestamps: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        orientations: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Simulate IMU measurements from ground truth trajectory data.
        
        Args:
            timestamps: Time vector (N,)
            positions: Position vectors (N, 3)
            velocities: Velocity vectors (N, 3)
            orientations: Quaternion orientations (N, 4) [w, x, y, z]
            
        Returns:
            Dictionary containing IMU data:
            {
                "timestamps": (N,),
                "accel": (N, 3),
                "gyro": (N, 3)
            }
        """
        N = len(timestamps)
        
        # Compute world-frame acceleration
        world_accel = self.compute_acceleration(velocities, timestamps)
        
        # Compute angular velocity
        angular_vel = self.quaternion_to_angular_velocity(orientations, timestamps)
        
        # Convert quaternions to rotation matrices
        rotations = R.from_quat(orientations)
        
        # Initialize sensor measurements
        accel_measurements = np.zeros((N, 3))
        gyro_measurements = np.zeros((N, 3))
        
        # Simulate measurements for each time step
        for i in range(N):
            # Transform world acceleration to body frame
            body_accel = rotations[i].inv().apply(world_accel[i])
            
            # Add gravity in body frame
            gravity_body = rotations[i].inv().apply(self.gravity)
            
            # Accelerometer measurement: acceleration + gravity in body frame
            accel_measurements[i] = body_accel + gravity_body
            
            # Gyroscope measurement: angular velocity in body frame
            gyro_measurements[i] = angular_vel[i]
        
        # Add sensor noise and bias
        accel_noise = np.random.randn(N, 3) * self.accel_noise_std
        gyro_noise = np.random.randn(N, 3) * self.gyro_noise_std
        
        accel_measurements += accel_noise + self.accel_bias
        gyro_measurements += gyro_noise + self.gyro_bias
        
        return {
            "timestamps": timestamps,
            "accel": accel_measurements,
            "gyro": gyro_measurements
        }
    
    def simulate_gnss_measurements(
        self,
        timestamps: np.ndarray,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate GNSS measurements from ground truth trajectory data.
        
        Args:
            timestamps: Time vector (N,)
            positions: Position vectors (N, 3)
            velocities: Velocity vectors (N, 3) - optional, for future use
            
        Returns:
            Dictionary containing GNSS data:
            {
                "timestamps": (M,),
                "positions": (M, 3)
            }
        """
        N = len(timestamps)
        
        # Determine GNSS sampling rate
        dt = timestamps[1] - timestamps[0] if N > 1 else 0.01
        gnss_dt = 1.0 / self.gnss_rate
        
        # Find GNSS measurement indices
        gnss_indices = []
        current_time = 0.0
        
        for i, t in enumerate(timestamps):
            if t >= current_time:
                gnss_indices.append(i)
                current_time += gnss_dt
        
        # If no GNSS measurements found, use first and last
        if len(gnss_indices) == 0:
            gnss_indices = [0, N-1] if N > 1 else [0]
        
        # Extract GNSS measurements
        gnss_timestamps = timestamps[gnss_indices]
        gnss_positions = positions[gnss_indices]
        
        # Add GNSS noise
        gnss_noise = np.random.randn(len(gnss_indices), 3) * self.gnss_noise_std
        gnss_positions += gnss_noise
        
        return {
            "timestamps": gnss_timestamps,
            "positions": gnss_positions
        }
    
    def simulate_sensors(
        self,
        timestamps: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        orientations: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Simulate both IMU and GNSS measurements from ground truth trajectory.
        
        Args:
            timestamps: Time vector (N,)
            positions: Position vectors (N, 3)
            velocities: Velocity vectors (N, 3)
            orientations: Quaternion orientations (N, 4) [w, x, y, z]
            
        Returns:
            Tuple of (imu_data, gnss_data) dictionaries
        """
        # Validate inputs
        if len(timestamps) != len(positions) or len(timestamps) != len(velocities) or len(timestamps) != len(orientations):
            raise ValueError("All input arrays must have the same length")
        
        if len(timestamps) < 2:
            raise ValueError("At least 2 time points are required for sensor simulation")
        
        # Simulate IMU measurements
        imu_data = self.simulate_imu_measurements(
            timestamps, positions, velocities, orientations
        )
        
        # Simulate GNSS measurements
        gnss_data = self.simulate_gnss_measurements(
            timestamps, positions, velocities
        )
        
        return imu_data, gnss_data


def create_sensor_simulator(
    sensor_type: str = "standard",
    **kwargs
) -> SensorSimulator:
    """
    Factory function to create sensor simulators with predefined configurations.
    
    Args:
        sensor_type: Type of sensor configuration ("standard", "high_quality", "low_quality")
        **kwargs: Additional parameters to override defaults
        
    Returns:
        Configured SensorSimulator instance
    """
    if sensor_type == "standard":
        # Standard consumer-grade sensors
        default_params = {
            "accel_noise_std": 0.1,
            "gyro_noise_std": 0.01,
            "accel_bias_std": 0.01,
            "gyro_bias_std": 0.001,
            "gnss_noise_std": 1.0,
            "gnss_rate": 1.0
        }
    elif sensor_type == "high_quality":
        # High-quality industrial sensors
        default_params = {
            "accel_noise_std": 0.01,
            "gyro_noise_std": 0.001,
            "accel_bias_std": 0.001,
            "gyro_bias_std": 0.0001,
            "gnss_noise_std": 0.1,
            "gnss_rate": 10.0
        }
    elif sensor_type == "low_quality":
        # Low-quality sensors
        default_params = {
            "accel_noise_std": 0.5,
            "gyro_noise_std": 0.05,
            "accel_bias_std": 0.05,
            "gyro_bias_std": 0.005,
            "gnss_noise_std": 5.0,
            "gnss_rate": 0.5
        }
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")
    
    # Override with provided parameters
    default_params.update(kwargs)
    
    return SensorSimulator(**default_params) 