"""
Visualization Module for Sensor Data

This module provides plotting functions to visualize and verify synthetic IMU and GNSS data
generated from the sensor simulator. It includes time series plots and 3D trajectory comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Optional, Tuple
import matplotlib.patches as mpatches


def plot_imu_timeseries(
    imu_data: Dict[str, np.ndarray],
    title: str = "IMU Measurements",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot accelerometer and gyroscope time series data.
    
    Args:
        imu_data: Dictionary containing IMU data with keys "timestamps", "accel", "gyro"
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot
    """
    timestamps = imu_data["timestamps"]
    accel_data = imu_data["accel"]
    gyro_data = imu_data["gyro"]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot accelerometer data
    ax1.plot(timestamps, accel_data[:, 0], 'r-', label='X-axis', linewidth=1.5)
    ax1.plot(timestamps, accel_data[:, 1], 'g-', label='Y-axis', linewidth=1.5)
    ax1.plot(timestamps, accel_data[:, 2], 'b-', label='Z-axis', linewidth=1.5)
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('Accelerometer Measurements')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot gyroscope data
    ax2.plot(timestamps, gyro_data[:, 0], 'r-', label='X-axis', linewidth=1.5)
    ax2.plot(timestamps, gyro_data[:, 1], 'g-', label='Y-axis', linewidth=1.5)
    ax2.plot(timestamps, gyro_data[:, 2], 'b-', label='Z-axis', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_title('Gyroscope Measurements')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"IMU plot saved to {save_path}")
    
    plt.show()


def plot_gnss_vs_ground_truth(
    gnss_data: Dict[str, np.ndarray],
    ground_truth_positions: np.ndarray,
    ground_truth_timestamps: np.ndarray,
    title: str = "GNSS vs Ground Truth Trajectory",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Compare GNSS noisy positions with ground truth trajectory.
    
    Args:
        gnss_data: Dictionary containing GNSS data with keys "timestamps", "positions"
        ground_truth_positions: Ground truth position array (N, 3)
        ground_truth_timestamps: Ground truth timestamps (N,)
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot
    """
    gnss_timestamps = gnss_data["timestamps"]
    gnss_positions = gnss_data["positions"]
    
    # Create figure with 3D subplot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ground truth trajectory
    ax.plot(
        ground_truth_positions[:, 0],
        ground_truth_positions[:, 1],
        ground_truth_positions[:, 2],
        'b-', linewidth=2, label='Ground Truth Trajectory'
    )
    
    # Plot GNSS measurements
    ax.scatter(
        gnss_positions[:, 0],
        gnss_positions[:, 1],
        gnss_positions[:, 2],
        c='red', s=50, alpha=0.7, label='GNSS Measurements'
    )
    
    # Mark start and end points
    ax.scatter(
        ground_truth_positions[0, 0], ground_truth_positions[0, 1], ground_truth_positions[0, 2],
        c='green', s=100, marker='o', label='Start Point'
    )
    ax.scatter(
        ground_truth_positions[-1, 0], ground_truth_positions[-1, 1], ground_truth_positions[-1, 2],
        c='blue', s=100, marker='s', label='End Point'
    )
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    # Make axes equal aspect ratio
    max_range = np.array([
        ground_truth_positions[:, 0].max() - ground_truth_positions[:, 0].min(),
        ground_truth_positions[:, 1].max() - ground_truth_positions[:, 1].min(),
        ground_truth_positions[:, 2].max() - ground_truth_positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (ground_truth_positions[:, 0].max() + ground_truth_positions[:, 0].min()) * 0.5
    mid_y = (ground_truth_positions[:, 1].max() + ground_truth_positions[:, 1].min()) * 0.5
    mid_z = (ground_truth_positions[:, 2].max() + ground_truth_positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"GNSS vs Ground Truth plot saved to {save_path}")
    
    plt.show()


def plot_sensor_comparison(
    imu_data: Dict[str, np.ndarray],
    gnss_data: Dict[str, np.ndarray],
    ground_truth_positions: np.ndarray,
    ground_truth_timestamps: np.ndarray,
    title: str = "Sensor Data Overview",
    figsize: Tuple[int, int] = (20, 12),
    save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive overview plot of all sensor data.
    
    Args:
        imu_data: Dictionary containing IMU data
        gnss_data: Dictionary containing GNSS data
        ground_truth_positions: Ground truth position array (N, 3)
        ground_truth_timestamps: Ground truth timestamps (N,)
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot
    """
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 3D trajectory comparison (top left, spans 2x2)
    ax_3d = fig.add_subplot(gs[0:2, 0:2], projection='3d')
    
    # Plot ground truth trajectory
    ax_3d.plot(
        ground_truth_positions[:, 0],
        ground_truth_positions[:, 1],
        ground_truth_positions[:, 2],
        'b-', linewidth=2, label='Ground Truth'
    )
    
    # Plot GNSS measurements
    gnss_positions = gnss_data["positions"]
    ax_3d.scatter(
        gnss_positions[:, 0],
        gnss_positions[:, 1],
        gnss_positions[:, 2],
        c='red', s=30, alpha=0.7, label='GNSS'
    )
    
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Trajectory Comparison')
    ax_3d.legend()
    
    # 2. Accelerometer magnitude (top right)
    ax_accel_mag = fig.add_subplot(gs[0, 2])
    accel_data = imu_data["accel"]
    accel_magnitude = np.linalg.norm(accel_data, axis=1)
    ax_accel_mag.plot(imu_data["timestamps"], accel_magnitude, 'g-', linewidth=1.5)
    ax_accel_mag.set_ylabel('Accel Magnitude (m/s²)')
    ax_accel_mag.set_title('Accelerometer Magnitude')
    ax_accel_mag.grid(True, alpha=0.3)
    
    # 3. Gyroscope magnitude (middle right)
    ax_gyro_mag = fig.add_subplot(gs[1, 2])
    gyro_data = imu_data["gyro"]
    gyro_magnitude = np.linalg.norm(gyro_data, axis=1)
    ax_gyro_mag.plot(imu_data["timestamps"], gyro_magnitude, 'm-', linewidth=1.5)
    ax_gyro_mag.set_ylabel('Gyro Magnitude (rad/s)')
    ax_gyro_mag.set_title('Gyroscope Magnitude')
    ax_gyro_mag.grid(True, alpha=0.3)
    
    # 4. GNSS error over time (bottom left)
    ax_gnss_error = fig.add_subplot(gs[2, 0])
    gnss_timestamps = gnss_data["timestamps"]
    
    # Find closest ground truth points to GNSS measurements
    gnss_errors = []
    for i, gnss_time in enumerate(gnss_timestamps):
        # Find closest ground truth timestamp
        gt_idx = np.argmin(np.abs(ground_truth_timestamps - gnss_time))
        error = np.linalg.norm(gnss_positions[i] - ground_truth_positions[gt_idx])
        gnss_errors.append(error)
    
    ax_gnss_error.plot(gnss_timestamps, gnss_errors, 'r-', linewidth=1.5, marker='o')
    ax_gnss_error.set_xlabel('Time (s)')
    ax_gnss_error.set_ylabel('Position Error (m)')
    ax_gnss_error.set_title('GNSS Position Error')
    ax_gnss_error.grid(True, alpha=0.3)
    
    # 5. Sensor statistics (bottom middle)
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_stats.axis('off')
    
    # Calculate statistics
    accel_std = np.std(accel_data, axis=0)
    gyro_std = np.std(gyro_data, axis=0)
    gnss_error_mean = np.mean(gnss_errors)
    gnss_error_std = np.std(gnss_errors)
    
    stats_text = f"""
    Sensor Statistics:
    
    Accelerometer Std Dev (m/s²):
    X: {accel_std[0]:.3f}
    Y: {accel_std[1]:.3f}
    Z: {accel_std[2]:.3f}
    
    Gyroscope Std Dev (rad/s):
    X: {gyro_std[0]:.4f}
    Y: {gyro_std[1]:.4f}
    Z: {gyro_std[2]:.4f}
    
    GNSS Error:
    Mean: {gnss_error_mean:.3f} m
    Std: {gnss_error_std:.3f} m
    """
    
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # 6. XY projection (bottom right)
    ax_xy = fig.add_subplot(gs[2, 2])
    ax_xy.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], 
               'b-', linewidth=2, label='Ground Truth')
    ax_xy.scatter(gnss_positions[:, 0], gnss_positions[:, 1], 
                  c='red', s=30, alpha=0.7, label='GNSS')
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('XY Projection')
    ax_xy.legend()
    ax_xy.axis('equal')
    ax_xy.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sensor comparison plot saved to {save_path}")
    
    plt.show()


def plot_imu_histograms(
    imu_data: Dict[str, np.ndarray],
    title: str = "IMU Data Distributions",
    figsize: Tuple[int, int] = (15, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot histograms of IMU data to analyze noise distributions.
    
    Args:
        imu_data: Dictionary containing IMU data
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot
    """
    accel_data = imu_data["accel"]
    gyro_data = imu_data["gyro"]
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=figsize)
    
    # Accelerometer histograms
    ax1.hist(accel_data[:, 0], bins=50, alpha=0.7, color='red', label='X-axis')
    ax1.set_title('Accelerometer X-axis')
    ax1.set_xlabel('Acceleration (m/s²)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(accel_data[:, 1], bins=50, alpha=0.7, color='green', label='Y-axis')
    ax2.set_title('Accelerometer Y-axis')
    ax2.set_xlabel('Acceleration (m/s²)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.hist(accel_data[:, 2], bins=50, alpha=0.7, color='blue', label='Z-axis')
    ax3.set_title('Accelerometer Z-axis')
    ax3.set_xlabel('Acceleration (m/s²)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gyroscope histograms
    ax4.hist(gyro_data[:, 0], bins=50, alpha=0.7, color='red', label='X-axis')
    ax4.set_title('Gyroscope X-axis')
    ax4.set_xlabel('Angular Velocity (rad/s)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5.hist(gyro_data[:, 1], bins=50, alpha=0.7, color='green', label='Y-axis')
    ax5.set_title('Gyroscope Y-axis')
    ax5.set_xlabel('Angular Velocity (rad/s)')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6.hist(gyro_data[:, 2], bins=50, alpha=0.7, color='blue', label='Z-axis')
    ax6.set_title('Gyroscope Z-axis')
    ax6.set_xlabel('Angular Velocity (rad/s)')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"IMU histograms plot saved to {save_path}")
    
    plt.show()


def plot_gnss_timing(
    gnss_data: Dict[str, np.ndarray],
    ground_truth_timestamps: np.ndarray,
    title: str = "GNSS Measurement Timing",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot GNSS measurement timing and intervals.
    
    Args:
        gnss_data: Dictionary containing GNSS data
        ground_truth_timestamps: Ground truth timestamps for reference
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot
    """
    gnss_timestamps = gnss_data["timestamps"]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot GNSS measurement times
    ax1.scatter(gnss_timestamps, np.ones_like(gnss_timestamps), 
                c='red', s=50, alpha=0.7, label='GNSS Measurements')
    ax1.set_ylabel('GNSS Measurements')
    ax1.set_title('GNSS Measurement Timing')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot measurement intervals
    if len(gnss_timestamps) > 1:
        intervals = np.diff(gnss_timestamps)
        ax2.plot(gnss_timestamps[1:], intervals, 'b-o', linewidth=1.5, markersize=4)
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Expected 1s interval')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Interval (s)')
        ax2.set_title('GNSS Measurement Intervals')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"GNSS timing plot saved to {save_path}")
    
    plt.show() 