#!/usr/bin/env python3
"""
Extended Kalman Filter for 6-DOF Sensor Fusion

This module implements an EKF to estimate 6-DOF pose (position, velocity, orientation)
using IMU (accelerometer + gyroscope) and GNSS measurements.

State Vector: [x, y, z, vx, vy, vz, roll, pitch, yaw]
"""

import numpy as np
from typing import Tuple, Optional
import warnings


class SensorFusionEKF:
    """
    Extended Kalman Filter for sensor fusion of IMU and GNSS data.
    
    State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw]
    - Position: (x, y, z) in meters
    - Velocity: (vx, vy, vz) in m/s
    - Orientation: (roll, pitch, yaw) in radians
    """
    
    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        initial_covariance: Optional[np.ndarray] = None,
        process_noise_std: Optional[np.ndarray] = None,
        measurement_noise_std: Optional[np.ndarray] = None,
        gravity: float = 9.81
    ):
        """
        Initialize the EKF.
        
        Args:
            initial_state: Initial state vector (9,) or None for zeros
            initial_covariance: Initial state covariance (9, 9) or None for identity
            process_noise_std: Process noise standard deviations (9,) or None for defaults
            measurement_noise_std: Measurement noise standard deviations (3,) or None for defaults
            gravity: Gravitational acceleration (m/s²)
        """
        # State dimension
        self.state_dim = 9
        self.measurement_dim = 3  # GNSS position only
        
        # Initialize state
        if initial_state is None:
            self.state = np.zeros(self.state_dim)
        else:
            self.state = np.array(initial_state, dtype=np.float64)
            if self.state.shape != (self.state_dim,):
                raise ValueError(f"Initial state must have shape ({self.state_dim},), got {self.state.shape}")
        
        # Initialize covariance
        if initial_covariance is None:
            self.covariance = np.eye(self.state_dim)
        else:
            self.covariance = np.array(initial_covariance, dtype=np.float64)
            if self.covariance.shape != (self.state_dim, self.state_dim):
                raise ValueError(f"Initial covariance must have shape ({self.state_dim}, {self.state_dim})")
        
        # Process noise (Q matrix)
        if process_noise_std is None:
            # Default process noise standard deviations
            self.process_noise_std = np.array([
                0.1, 0.1, 0.1,      # Position noise (m)
                0.5, 0.5, 0.5,      # Velocity noise (m/s)
                0.01, 0.01, 0.01    # Orientation noise (rad)
            ])
        else:
            self.process_noise_std = np.array(process_noise_std, dtype=np.float64)
            if self.process_noise_std.shape != (self.state_dim,):
                raise ValueError(f"Process noise must have shape ({self.state_dim},)")
        
        # Measurement noise (R matrix)
        if measurement_noise_std is None:
            # Default GNSS measurement noise standard deviations
            self.measurement_noise_std = np.array([1.0, 1.0, 1.0])  # Position noise (m)
        else:
            self.measurement_noise_std = np.array(measurement_noise_std, dtype=np.float64)
            if self.measurement_noise_std.shape != (self.measurement_dim,):
                raise ValueError(f"Measurement noise must have shape ({self.measurement_dim},)")
        
        # Physical constants
        self.gravity = gravity
        
        # Pre-compute noise matrices
        self.Q = np.diag(self.process_noise_std ** 2)
        self.R = np.diag(self.measurement_noise_std ** 2)
        
        # State indices for readability
        self.idx_pos = slice(0, 3)    # x, y, z
        self.idx_vel = slice(3, 6)    # vx, vy, vz
        self.idx_att = slice(6, 9)    # roll, pitch, yaw
        
        print(f"✅ EKF initialized with state dimension {self.state_dim}")
        print(f"   Process noise std: {self.process_noise_std}")
        print(f"   Measurement noise std: {self.measurement_noise_std}")
    
    def predict(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        dt: float
    ) -> None:
        """
        Predict step using IMU measurements.
        
        Args:
            accel: Accelerometer reading (3,) in body frame (m/s²)
            gyro: Gyroscope reading (3,) in body frame (rad/s)
            dt: Time step (s)
        """
        # Validate inputs
        accel = np.array(accel, dtype=np.float64).flatten()
        gyro = np.array(gyro, dtype=np.float64).flatten()
        
        if accel.shape != (3,):
            raise ValueError(f"Accelerometer must have shape (3,), got {accel.shape}")
        if gyro.shape != (3,):
            raise ValueError(f"Gyroscope must have shape (3,), got {gyro.shape}")
        if dt <= 0:
            raise ValueError(f"Time step must be positive, got {dt}")
        
        # Extract current state
        pos = self.state[self.idx_pos]
        vel = self.state[self.idx_vel]
        roll, pitch, yaw = self.state[self.idx_att]
        
        # Small-angle approximation for orientation
        # This assumes small orientation changes during the time step
        cos_roll, sin_roll = np.cos(roll), np.sin(roll)
        cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        
        # Rotation matrix from body to world frame (simplified)
        # Using small-angle approximation for the Jacobian
        R_body_to_world = np.array([
            [cos_yaw * cos_pitch, -sin_yaw * cos_roll + cos_yaw * sin_pitch * sin_roll, sin_yaw * sin_roll + cos_yaw * sin_pitch * cos_roll],
            [sin_yaw * cos_pitch, cos_yaw * cos_roll + sin_yaw * sin_pitch * sin_roll, -cos_yaw * sin_roll + sin_yaw * sin_pitch * cos_roll],
            [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll]
        ])
        
        # State prediction
        # Position: p_k+1 = p_k + v_k * dt
        pos_pred = pos + vel * dt
        
        # Velocity: v_k+1 = v_k + R * a_k * dt + g * dt
        # Note: accel includes gravity in body frame, so we need to handle it properly
        accel_world = R_body_to_world @ accel
        # Add gravity correction (assuming z is up)
        gravity_world = np.array([0, 0, -self.gravity])
        vel_pred = vel + accel_world * dt + gravity_world * dt
        
        # Orientation: attitude_k+1 = attitude_k + ω_k * dt
        # Using small-angle approximation
        att_pred = np.array([roll, pitch, yaw]) + gyro * dt
        
        # Normalize angles to [-π, π]
        att_pred = np.arctan2(np.sin(att_pred), np.cos(att_pred))
        
        # Update state
        self.state[self.idx_pos] = pos_pred
        self.state[self.idx_vel] = vel_pred
        self.state[self.idx_att] = att_pred
        
        # Covariance prediction: P_k+1 = F_k * P_k * F_k^T + Q
        F = self._state_transition_jacobian(accel, gyro, dt)
        self.covariance = F @ self.covariance @ F.T + self.Q
    
    def update(self, gnss_position: np.ndarray) -> None:
        """
        Update step using GNSS position measurement.
        
        Args:
            gnss_position: GNSS position measurement (3,) in world frame (m)
        """
        # Validate input
        gnss_position = np.array(gnss_position, dtype=np.float64).flatten()
        if gnss_position.shape != (3,):
            raise ValueError(f"GNSS position must have shape (3,), got {gnss_position.shape}")
        
        # Measurement model: h(x) = [x, y, z]
        predicted_measurement = self.state[self.idx_pos]
        
        # Innovation: y = z - h(x)
        innovation = gnss_position - predicted_measurement
        
        # Measurement Jacobian: H = ∂h/∂x
        H = self._measurement_jacobian()
        
        # Innovation covariance: S = H * P * H^T + R
        S = H @ self.covariance @ H.T + self.R
        
        # Kalman gain: K = P * H^T * S^(-1)
        try:
            K = self.covariance @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            warnings.warn("Singular innovation covariance matrix. Skipping update.")
            return
        
        # State update: x = x + K * y
        self.state = self.state + K @ innovation
        
        # Covariance update: P = (I - K * H) * P
        I = np.eye(self.state_dim)
        self.covariance = (I - K @ H) @ self.covariance
    
    def _state_transition_jacobian(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Compute the state transition Jacobian matrix F.
        
        Args:
            accel: Accelerometer reading (3,)
            gyro: Gyroscope reading (3,)
            dt: Time step (s)
            
        Returns:
            State transition Jacobian matrix (9, 9)
        """
        # Initialize Jacobian matrix
        F = np.eye(self.state_dim)
        
        # Extract current orientation
        roll, pitch, yaw = self.state[self.idx_att]
        
        # Position derivatives: ∂p/∂p = I, ∂p/∂v = I*dt, ∂p/∂att = 0
        F[self.idx_pos, self.idx_vel] = np.eye(3) * dt
        
        # Velocity derivatives: ∂v/∂p = 0, ∂v/∂v = I, ∂v/∂att = ∂(R*a)/∂att * dt
        # For small angles, we can approximate the derivative
        F[self.idx_vel, self.idx_att] = self._velocity_orientation_jacobian(accel, dt)
        
        # Orientation derivatives: ∂att/∂p = 0, ∂att/∂v = 0, ∂att/∂att = I
        # (assuming gyro measurements are independent of position/velocity)
        
        return F
    
    def _velocity_orientation_jacobian(
        self,
        accel: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Compute the Jacobian of velocity with respect to orientation.
        
        Args:
            accel: Accelerometer reading (3,)
            dt: Time step (s)
            
        Returns:
            Jacobian matrix (3, 3)
        """
        # Extract current orientation
        roll, pitch, yaw = self.state[self.idx_att]
        
        # Simplified Jacobian for small angles
        # This is an approximation - for more accuracy, compute the full derivative
        ax, ay, az = accel
        
        # ∂v/∂roll
        dvdroll = np.array([
            -ay * np.cos(roll) - az * np.sin(roll),
            ax * np.cos(roll) - az * np.cos(roll),
            ax * np.sin(roll) + ay * np.cos(roll)
        ])
        
        # ∂v/∂pitch
        dvdpitch = np.array([
            -ax * np.sin(pitch) + az * np.cos(pitch),
            -ay * np.sin(pitch),
            -ax * np.cos(pitch) - az * np.sin(pitch)
        ])
        
        # ∂v/∂yaw
        dvdyaw = np.array([
            -ax * np.sin(yaw) - ay * np.cos(yaw),
            ax * np.cos(yaw) - ay * np.sin(yaw),
            0
        ])
        
        return np.column_stack([dvdroll, dvdpitch, dvdyaw]) * dt
    
    def _measurement_jacobian(self) -> np.ndarray:
        """
        Compute the measurement Jacobian matrix H.
        
        Returns:
            Measurement Jacobian matrix (3, 9)
        """
        # Measurement model: h(x) = [x, y, z]
        # So H = [∂h/∂x] = [I_3x3, 0_3x3, 0_3x3]
        H = np.zeros((self.measurement_dim, self.state_dim))
        H[:3, :3] = np.eye(3)  # ∂h/∂pos = I
        return H
    
    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.state.copy()
    
    def get_covariance(self) -> np.ndarray:
        """Get current state covariance."""
        return self.covariance.copy()
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.state[self.idx_pos].copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.state[self.idx_vel].copy()
    
    def get_orientation(self) -> np.ndarray:
        """Get current orientation estimate (roll, pitch, yaw)."""
        return self.state[self.idx_att].copy()
    
    def get_position_std(self) -> np.ndarray:
        """Get position standard deviations."""
        return np.sqrt(np.diag(self.covariance[self.idx_pos, self.idx_pos]))
    
    def get_velocity_std(self) -> np.ndarray:
        """Get velocity standard deviations."""
        return np.sqrt(np.diag(self.covariance[self.idx_vel, self.idx_vel]))
    
    def get_orientation_std(self) -> np.ndarray:
        """Get orientation standard deviations."""
        return np.sqrt(np.diag(self.covariance[self.idx_att, self.idx_att]))
    
    def reset(
        self,
        initial_state: Optional[np.ndarray] = None,
        initial_covariance: Optional[np.ndarray] = None
    ) -> None:
        """
        Reset the EKF to initial conditions.
        
        Args:
            initial_state: New initial state or None to keep current
            initial_covariance: New initial covariance or None to keep current
        """
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=np.float64)
        if initial_covariance is not None:
            self.covariance = np.array(initial_covariance, dtype=np.float64)
        
        print("🔄 EKF reset to initial conditions")


def create_ekf(
    initial_position: Optional[np.ndarray] = None,
    initial_velocity: Optional[np.ndarray] = None,
    initial_orientation: Optional[np.ndarray] = None,
    position_std: float = 1.0,
    velocity_std: float = 0.5,
    orientation_std: float = 0.1,
    process_noise_multiplier: float = 1.0,
    measurement_noise_multiplier: float = 1.0
) -> SensorFusionEKF:
    """
    Factory function to create a SensorFusionEKF with reasonable defaults.
    
    Args:
        initial_position: Initial position [x, y, z] or None for zeros
        initial_velocity: Initial velocity [vx, vy, vz] or None for zeros
        initial_orientation: Initial orientation [roll, pitch, yaw] or None for zeros
        position_std: Initial position uncertainty (m)
        velocity_std: Initial velocity uncertainty (m/s)
        orientation_std: Initial orientation uncertainty (rad)
        process_noise_multiplier: Multiplier for process noise
        measurement_noise_multiplier: Multiplier for measurement noise
        
    Returns:
        Configured SensorFusionEKF instance
    """
    # Build initial state
    initial_state = np.zeros(9)
    if initial_position is not None:
        initial_state[0:3] = initial_position
    if initial_velocity is not None:
        initial_state[3:6] = initial_velocity
    if initial_orientation is not None:
        initial_state[6:9] = initial_orientation
    
    # Build initial covariance
    initial_covariance = np.diag([
        position_std**2, position_std**2, position_std**2,      # Position
        velocity_std**2, velocity_std**2, velocity_std**2,      # Velocity
        orientation_std**2, orientation_std**2, orientation_std**2  # Orientation
    ])
    
    # Process noise (scaled by multiplier)
    process_noise_std = np.array([
        0.1, 0.1, 0.1,      # Position noise (m)
        0.5, 0.5, 0.5,      # Velocity noise (m/s)
        0.01, 0.01, 0.01    # Orientation noise (rad)
    ]) * process_noise_multiplier
    
    # Measurement noise (scaled by multiplier)
    measurement_noise_std = np.array([1.0, 1.0, 1.0]) * measurement_noise_multiplier
    
    return SensorFusionEKF(
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        process_noise_std=process_noise_std,
        measurement_noise_std=measurement_noise_std
    )


def main():
    """Example usage of the EKF."""
    print("🎯 Sensor Fusion EKF Example")
    print("=" * 40)
    
    # Create EKF
    ekf = create_ekf(
        initial_position=np.array([0, 0, 0]),
        initial_velocity=np.array([1, 0, 0]),
        initial_orientation=np.array([0, 0, 0])
    )
    
    print(f"Initial state: {ekf.get_state()}")
    print(f"Initial position std: {ekf.get_position_std()}")
    
    # Simulate some measurements
    dt = 0.01  # 100 Hz
    
    # Example IMU measurements (accelerometer and gyroscope)
    accel = np.array([0.1, 0.0, 9.81])  # Small acceleration + gravity
    gyro = np.array([0.0, 0.0, 0.1])    # Small yaw rate
    
    # Predict step
    ekf.predict(accel, gyro, dt)
    print(f"\nAfter predict:")
    print(f"  Position: {ekf.get_position()}")
    print(f"  Velocity: {ekf.get_velocity()}")
    print(f"  Orientation: {ekf.get_orientation()}")
    
    # Example GNSS measurement
    gnss_position = np.array([0.01, 0.0, 0.0])  # Slight position measurement
    
    # Update step
    ekf.update(gnss_position)
    print(f"\nAfter update:")
    print(f"  Position: {ekf.get_position()}")
    print(f"  Position std: {ekf.get_position_std()}")
    
    print("\n✅ EKF example completed!")


if __name__ == "__main__":
    main() 