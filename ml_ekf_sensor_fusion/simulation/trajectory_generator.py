"""
6-DOF Ground Truth Trajectory Generator

This module provides synthetic trajectory generation for testing and training
ML models and EKF pipelines. Supports multiple trajectory types with configurable
parameters for position, velocity, and orientation.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_trajectory(
    duration: float = 10.0,
    dt: float = 0.01,
    mode: str = "circular",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic 6-DOF trajectory.

    Args:
        duration (float): Total simulation time in seconds
        dt (float): Timestep in seconds
        mode (str): Trajectory mode: 'circular', 'sinusoidal', 'random_walk', 'helix', 'figure8'
        **kwargs: Additional parameters for specific trajectory modes

    Returns:
        timestamps (np.ndarray): Time vector of shape (N,)
        positions (np.ndarray): Position vectors (N, 3)
        velocities (np.ndarray): Velocity vectors (N, 3)
        orientations (np.ndarray): Quaternion orientation (N, 4)

    Raises:
        ValueError: If mode is not supported
    """
    timestamps = np.arange(0, duration, dt)
    N = len(timestamps)

    positions = np.zeros((N, 3))
    velocities = np.zeros((N, 3))
    orientations = np.zeros((N, 4))

    if mode == "circular":
        radius = kwargs.get('radius', 5.0)
        angular_speed = kwargs.get('angular_speed', 2 * np.pi / duration)
        z_amplitude = kwargs.get('z_amplitude', 1.0)
        
        positions[:, 0] = radius * np.cos(angular_speed * timestamps)
        positions[:, 1] = radius * np.sin(angular_speed * timestamps)
        positions[:, 2] = z_amplitude * np.sin(0.5 * angular_speed * timestamps)

        velocities[:, 0] = -radius * angular_speed * np.sin(angular_speed * timestamps)
        velocities[:, 1] = radius * angular_speed * np.cos(angular_speed * timestamps)
        velocities[:, 2] = 0.5 * z_amplitude * angular_speed * np.cos(0.5 * angular_speed * timestamps)

        yaws = angular_speed * timestamps
        rotations = R.from_euler("zyx", np.vstack([yaws, np.zeros(N), np.zeros(N)]).T)
        orientations = rotations.as_quat()

    elif mode == "sinusoidal":
        amplitude = kwargs.get('amplitude', 2.0)
        freq_x = kwargs.get('freq_x', 0.5)
        freq_y = kwargs.get('freq_y', 0.7)
        freq_z = kwargs.get('freq_z', 0.9)
        
        positions[:, 0] = amplitude * np.sin(freq_x * timestamps)
        positions[:, 1] = amplitude * np.sin(freq_y * timestamps)
        positions[:, 2] = amplitude * 0.5 * np.sin(freq_z * timestamps)

        velocities[:, 0] = amplitude * freq_x * np.cos(freq_x * timestamps)
        velocities[:, 1] = amplitude * freq_y * np.cos(freq_y * timestamps)
        velocities[:, 2] = amplitude * 0.5 * freq_z * np.cos(freq_z * timestamps)

        pitch = 0.2 * np.sin(0.3 * timestamps)
        roll = 0.2 * np.sin(0.5 * timestamps)
        yaw = 0.5 * timestamps
        rotations = R.from_euler("zyx", np.vstack([yaw, pitch, roll]).T)
        orientations = rotations.as_quat()

    elif mode == "random_walk":
        step_std = kwargs.get('step_std', 0.1)
        orientation_std = kwargs.get('orientation_std', 0.01)
        
        steps = np.random.randn(N, 3) * step_std
        positions = np.cumsum(steps, axis=0)
        
        velocities[1:] = np.diff(positions, axis=0) / dt
        velocities[0] = velocities[1]
        
        yaw = np.cumsum(np.random.randn(N) * orientation_std)
        rotations = R.from_euler("zyx", np.vstack([yaw, np.zeros(N), np.zeros(N)]).T)
        orientations = rotations.as_quat()

    elif mode == "helix":
        radius = kwargs.get('radius', 3.0)
        pitch = kwargs.get('pitch', 1.0)
        angular_speed = kwargs.get('angular_speed', 2 * np.pi / duration)
        
        positions[:, 0] = radius * np.cos(angular_speed * timestamps)
        positions[:, 1] = radius * np.sin(angular_speed * timestamps)
        positions[:, 2] = pitch * timestamps / duration

        velocities[:, 0] = -radius * angular_speed * np.sin(angular_speed * timestamps)
        velocities[:, 1] = radius * angular_speed * np.cos(angular_speed * timestamps)
        velocities[:, 2] = pitch / duration

        yaws = angular_speed * timestamps
        rotations = R.from_euler("zyx", np.vstack([yaws, np.zeros(N), np.zeros(N)]).T)
        orientations = rotations.as_quat()

    elif mode == "figure8":
        radius = kwargs.get('radius', 4.0)
        angular_speed = kwargs.get('angular_speed', 2 * np.pi / duration)
        
        positions[:, 0] = radius * np.sin(angular_speed * timestamps)
        positions[:, 1] = radius * np.sin(angular_speed * timestamps) * np.cos(angular_speed * timestamps)
        positions[:, 2] = 0.5 * np.sin(0.3 * angular_speed * timestamps)

        velocities[:, 0] = radius * angular_speed * np.cos(angular_speed * timestamps)
        velocities[:, 1] = radius * angular_speed * (np.cos(angular_speed * timestamps)**2 - np.sin(angular_speed * timestamps)**2)
        velocities[:, 2] = 0.5 * 0.3 * angular_speed * np.cos(0.3 * angular_speed * timestamps)

        yaws = angular_speed * timestamps
        rotations = R.from_euler("zyx", np.vstack([yaws, np.zeros(N), np.zeros(N)]).T)
        orientations = rotations.as_quat()

    else:
        raise ValueError(f"Unsupported trajectory mode: {mode}. "
                        f"Supported modes: circular, sinusoidal, random_walk, helix, figure8")

    return timestamps, positions, velocities, orientations


def add_noise(
    positions: np.ndarray,
    velocities: np.ndarray,
    orientations: np.ndarray,
    pos_std: float = 0.01,
    vel_std: float = 0.01,
    orient_std: float = 0.001
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add Gaussian noise to trajectory data.

    Args:
        positions (np.ndarray): Ground truth positions
        velocities (np.ndarray): Ground truth velocities
        orientations (np.ndarray): Ground truth orientations (quaternions)
        pos_std (float): Standard deviation for position noise
        vel_std (float): Standard deviation for velocity noise
        orient_std (float): Standard deviation for orientation noise

    Returns:
        Tuple of noisy positions, velocities, and orientations
    """
    noisy_positions = positions + np.random.randn(*positions.shape) * pos_std
    noisy_velocities = velocities + np.random.randn(*velocities.shape) * vel_std
    
    # Add noise to quaternions and renormalize
    noisy_orientations = orientations + np.random.randn(*orientations.shape) * orient_std
    noisy_orientations = noisy_orientations / np.linalg.norm(noisy_orientations, axis=1, keepdims=True)
    
    return noisy_positions, noisy_velocities, noisy_orientations


def compute_metrics(
    positions: np.ndarray,
    velocities: np.ndarray,
    orientations: np.ndarray
) -> Dict[str, float]:
    """
    Compute trajectory metrics.

    Args:
        positions (np.ndarray): Position vectors
        velocities (np.ndarray): Velocity vectors
        orientations (np.ndarray): Orientation quaternions

    Returns:
        Dictionary containing trajectory metrics
    """
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    max_velocity = np.max(np.linalg.norm(velocities, axis=1))
    avg_velocity = np.mean(np.linalg.norm(velocities, axis=1))
    
    # Compute angular velocity from quaternions
    angular_velocities = []
    for i in range(1, len(orientations)):
        q1 = orientations[i-1]
        q2 = orientations[i]
        # Simple quaternion difference (not exact angular velocity)
        q_diff = q2 - q1
        angular_velocities.append(np.linalg.norm(q_diff))
    
    max_angular_velocity = np.max(angular_velocities) if angular_velocities else 0.0
    
    return {
        'total_distance': total_distance,
        'max_velocity': max_velocity,
        'avg_velocity': avg_velocity,
        'max_angular_velocity': max_angular_velocity,
        'trajectory_duration': len(positions) * 0.01  # Assuming dt=0.01
    }


def plot_trajectory(
    timestamps: np.ndarray,
    positions: np.ndarray,
    velocities: Optional[np.ndarray] = None,
    orientations: Optional[np.ndarray] = None,
    title: str = "6-DOF Trajectory",
    save_path: Optional[str] = None
) -> None:
    """
    Plot the generated trajectory.

    Args:
        timestamps (np.ndarray): Time vector
        positions (np.ndarray): Position vectors
        velocities (np.ndarray, optional): Velocity vectors
        orientations (np.ndarray, optional): Orientation quaternions
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # Position components over time
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(timestamps, positions[:, 0], 'r-', label='X')
    ax2.plot(timestamps, positions[:, 1], 'g-', label='Y')
    ax2.plot(timestamps, positions[:, 2], 'b-', label='Z')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position')
    ax2.set_title('Position vs Time')
    ax2.legend()
    
    # Velocity components over time
    if velocities is not None:
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(timestamps, velocities[:, 0], 'r-', label='Vx')
        ax3.plot(timestamps, velocities[:, 1], 'g-', label='Vy')
        ax3.plot(timestamps, velocities[:, 2], 'b-', label='Vz')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity')
        ax3.set_title('Velocity vs Time')
        ax3.legend()
    
    # Orientation components over time
    if orientations is not None:
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(timestamps, orientations[:, 0], 'r-', label='qw')
        ax4.plot(timestamps, orientations[:, 1], 'g-', label='qx')
        ax4.plot(timestamps, orientations[:, 2], 'b-', label='qy')
        ax4.plot(timestamps, orientations[:, 3], 'm-', label='qz')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Quaternion')
        ax4.set_title('Orientation vs Time')
        ax4.legend()
    
    # Velocity magnitude
    if velocities is not None:
        ax5 = fig.add_subplot(2, 3, 5)
        vel_magnitude = np.linalg.norm(velocities, axis=1)
        ax5.plot(timestamps, vel_magnitude, 'k-', linewidth=2)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Velocity Magnitude')
        ax5.set_title('Velocity Magnitude vs Time')
    
    # XY projection
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
    ax6.scatter(positions[0, 0], positions[0, 1], c='g', s=100, label='Start')
    ax6.scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, label='End')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_title('XY Projection')
    ax6.legend()
    ax6.axis('equal')
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_trajectory_data(
    timestamps: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    orientations: np.ndarray,
    filepath: str
) -> None:
    """
    Save trajectory data to a NumPy file.

    Args:
        timestamps (np.ndarray): Time vector
        positions (np.ndarray): Position vectors
        velocities (np.ndarray): Velocity vectors
        orientations (np.ndarray): Orientation quaternions
        filepath (str): Path to save the data
    """
    trajectory_data = {
        'timestamps': timestamps,
        'positions': positions,
        'velocities': velocities,
        'orientations': orientations
    }
    np.save(filepath, trajectory_data)
    print(f"Trajectory data saved to {filepath}")


def load_trajectory_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load trajectory data from a NumPy file.

    Args:
        filepath (str): Path to the data file

    Returns:
        Tuple of timestamps, positions, velocities, and orientations
    """
    trajectory_data = np.load(filepath, allow_pickle=True).item()
    return (
        trajectory_data['timestamps'],
        trajectory_data['positions'],
        trajectory_data['velocities'],
        trajectory_data['orientations']
    )


if __name__ == "__main__":
    # Example usage and demonstration
    print("6-DOF Trajectory Generator Demo")
    print("=" * 40)
    
    # Generate different trajectory types
    modes = ["circular", "sinusoidal", "helix", "figure8"]
    
    for mode in modes:
        print(f"\nGenerating {mode} trajectory...")
        
        # Generate trajectory
        timestamps, positions, velocities, orientations = generate_trajectory(
            duration=10.0,
            dt=0.01,
            mode=mode
        )
        
        # Compute metrics
        metrics = compute_metrics(positions, velocities, orientations)
        print(f"Metrics for {mode} trajectory:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.3f}")
        
        # Plot trajectory
        plot_trajectory(
            timestamps, positions, velocities, orientations,
            title=f"{mode.capitalize()} Trajectory"
        )
        
        # Save data
        save_trajectory_data(
            timestamps, positions, velocities, orientations,
            f"trajectory_{mode}.npy"
        )
    
    # Demonstrate noise addition
    print("\nDemonstrating noise addition...")
    timestamps, positions, velocities, orientations = generate_trajectory(
        duration=5.0, dt=0.01, mode="circular"
    )
    
    noisy_pos, noisy_vel, noisy_orient = add_noise(
        positions, velocities, orientations,
        pos_std=0.05, vel_std=0.05, orient_std=0.01
    )
    
    plot_trajectory(
        timestamps, noisy_pos, noisy_vel, noisy_orient,
        title="Noisy Circular Trajectory"
    ) 