#!/usr/bin/env python3
"""
Inference Module for RNN-based Sensor Simulation

This module provides functions to use trained RNN models for generating
simulated IMU and GNSS measurements from ground truth trajectory data.
"""

import sys
import os

# Fix OpenMP warning on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append('..')

import numpy as np
import torch
from typing import Tuple, Optional
import warnings

# Import the model class from training script
from .train_sensor_model import SensorSimulationRNN


def load_trained_model(
    model_path: str,
    input_dim: int = 10,
    hidden_dim: int = 128,
    output_dim: int = 9,
    num_layers: int = 2,
    device: str = 'cpu'
) -> SensorSimulationRNN:
    """
    Load a trained SensorSimulationRNN model from a saved checkpoint.
    
    Args:
        model_path: Path to the saved model checkpoint (.pth file)
        input_dim: Input dimension (default: 10 for pos+vel+orientation)
        hidden_dim: Hidden dimension of the LSTM
        output_dim: Output dimension (default: 9 for accel+gyro+gnss_pos)
        num_layers: Number of LSTM layers
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        Loaded and configured SensorSimulationRNN model
    """
    # Create model instance
    model = SensorSimulationRNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers
    )
    
    # Load trained weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded trained model from {model_path}")
    else:
        warnings.warn(f"Model file {model_path} not found. Using untrained model.")
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model


def simulate_with_model(
    model: SensorSimulationRNN,
    input_sequence: np.ndarray,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate sensor measurements using a trained RNN model.
    
    Args:
        model: Trained SensorSimulationRNN model
        input_sequence: Ground truth trajectory data of shape (T, 10)
                       where 10 = [position(3), velocity(3), orientation(4)]
        device: Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        Tuple of (imu_data, gnss_data) as NumPy arrays:
        - imu_data: shape (T, 6) → [accelerometer(3), gyroscope(3)]
        - gnss_data: shape (T, 3) → [noisy_position(3)]
    """
    # Validate input
    if input_sequence.ndim != 2:
        raise ValueError(f"Expected 2D input array, got {input_sequence.ndim}D")
    
    if input_sequence.shape[1] != 10:
        raise ValueError(f"Expected input dimension 10, got {input_sequence.shape[1]}")
    
    T = input_sequence.shape[0]
    
    # Convert to torch tensor and add batch dimension
    input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0)  # Shape: (1, T, 10)
    
    # Move to device
    input_tensor = input_tensor.to(device)
    model = model.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)  # Shape: (1, T, 9)
        output = output.squeeze(0)    # Shape: (T, 9)
    
    # Convert back to numpy
    output_np = output.cpu().numpy()
    
    # Split output into IMU and GNSS components
    imu_data = output_np[:, :6]   # First 6 dimensions: accel(3) + gyro(3)
    gnss_data = output_np[:, 6:]  # Last 3 dimensions: gnss_pos(3)
    
    return imu_data, gnss_data


def simulate_with_model_batch(
    model: SensorSimulationRNN,
    input_sequences: np.ndarray,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate sensor measurements for multiple trajectories using a trained RNN model.
    
    Args:
        model: Trained SensorSimulationRNN model
        input_sequences: Ground truth trajectory data of shape (batch_size, T, 10)
        device: Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        Tuple of (imu_data, gnss_data) as NumPy arrays:
        - imu_data: shape (batch_size, T, 6) → [accelerometer(3), gyroscope(3)]
        - gnss_data: shape (batch_size, T, 3) → [noisy_position(3)]
    """
    # Validate input
    if input_sequences.ndim != 3:
        raise ValueError(f"Expected 3D input array, got {input_sequences.ndim}D")
    
    if input_sequences.shape[2] != 10:
        raise ValueError(f"Expected input dimension 10, got {input_sequences.shape[2]}")
    
    batch_size, T, _ = input_sequences.shape
    
    # Convert to torch tensor
    input_tensor = torch.FloatTensor(input_sequences)  # Shape: (batch_size, T, 10)
    
    # Move to device
    input_tensor = input_tensor.to(device)
    model = model.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)  # Shape: (batch_size, T, 9)
    
    # Convert back to numpy
    output_np = output.cpu().numpy()
    
    # Split output into IMU and GNSS components
    imu_data = output_np[:, :, :6]   # First 6 dimensions: accel(3) + gyro(3)
    gnss_data = output_np[:, :, 6:]  # Last 3 dimensions: gnss_pos(3)
    
    return imu_data, gnss_data


def create_sensor_data_dict(
    imu_data: np.ndarray,
    gnss_data: np.ndarray,
    timestamps: np.ndarray
) -> Tuple[dict, dict]:
    """
    Convert raw sensor arrays into dictionary format matching the sensor simulator output.
    
    Args:
        imu_data: IMU measurements of shape (T, 6) or (batch_size, T, 6)
        gnss_data: GNSS measurements of shape (T, 3) or (batch_size, T, 3)
        timestamps: Time vector of shape (T,)
        
    Returns:
        Tuple of (imu_dict, gnss_dict) in the same format as SensorSimulator output
    """
    # Handle batch dimension
    if imu_data.ndim == 3:
        # Batch mode - take first batch for now
        imu_data = imu_data[0]
        gnss_data = gnss_data[0]
    
    # Split IMU data into accelerometer and gyroscope
    accel_data = imu_data[:, :3]  # First 3 dimensions
    gyro_data = imu_data[:, 3:]   # Last 3 dimensions
    
    # Create IMU dictionary
    imu_dict = {
        "timestamps": timestamps,
        "accel": accel_data,
        "gyro": gyro_data
    }
    
    # For GNSS, we'll subsample to simulate realistic GNSS rates
    # This is a simplified approach - in practice, you might want more sophisticated subsampling
    gnss_rate = 1.0  # Hz
    dt = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.01
    gnss_interval = int(1.0 / (gnss_rate * dt))
    
    if gnss_interval > 1:
        gnss_indices = np.arange(0, len(timestamps), gnss_interval)
        gnss_timestamps = timestamps[gnss_indices]
        gnss_positions = gnss_data[gnss_indices]
    else:
        gnss_timestamps = timestamps
        gnss_positions = gnss_data
    
    # Create GNSS dictionary
    gnss_dict = {
        "timestamps": gnss_timestamps,
        "positions": gnss_positions
    }
    
    return imu_dict, gnss_dict


def compare_ml_vs_traditional(
    ml_imu_data: np.ndarray,
    ml_gnss_data: np.ndarray,
    traditional_imu_data: dict,
    traditional_gnss_data: dict,
    timestamps: np.ndarray,
    ml_gnss_timestamps: np.ndarray = None
) -> dict:
    """
    Compare ML-generated sensor data with traditional sensor simulation.
    
    Args:
        ml_imu_data: ML-generated IMU data (T, 6)
        ml_gnss_data: ML-generated GNSS data (T, 3)
        traditional_imu_data: Traditional sensor simulator IMU output
        traditional_gnss_data: Traditional sensor simulator GNSS output
        timestamps: Time vector
        
    Returns:
        Dictionary containing comparison metrics
    """
    # Extract traditional sensor data
    trad_accel = traditional_imu_data['accel']
    trad_gyro = traditional_imu_data['gyro']
    trad_gnss_pos = traditional_gnss_data['positions']
    
    # Extract ML sensor data
    ml_accel = ml_imu_data[:, :3]
    ml_gyro = ml_imu_data[:, 3:]
    
    # Calculate comparison metrics
    metrics = {}
    
    # IMU comparison
    accel_mse = np.mean((ml_accel - trad_accel) ** 2)
    gyro_mse = np.mean((ml_gyro - trad_gyro) ** 2)
    
    metrics['accel_mse'] = accel_mse
    metrics['gyro_mse'] = gyro_mse
    metrics['imu_mse'] = (accel_mse + gyro_mse) / 2
    
    # GNSS comparison (need to interpolate ML data to match traditional timestamps)
    ml_gnss_interpolated = np.zeros_like(trad_gnss_pos)
    
    # Use ML GNSS timestamps if provided, otherwise use full timestamps
    if ml_gnss_timestamps is None:
        ml_gnss_timestamps = timestamps
    
    for i, trad_time in enumerate(traditional_gnss_data['timestamps']):
        # Find closest ML GNSS timestamp
        ml_idx = np.argmin(np.abs(ml_gnss_timestamps - trad_time))
        # Add bounds check to prevent IndexError
        if ml_idx >= len(ml_gnss_data):
            ml_idx = -1  # fallback to last available ML GNSS
        ml_gnss_interpolated[i] = ml_gnss_data[ml_idx]
    
    gnss_mse = np.mean((ml_gnss_interpolated - trad_gnss_pos) ** 2)
    metrics['gnss_mse'] = gnss_mse
    
    # Statistical comparisons
    metrics['accel_correlation'] = np.corrcoef(ml_accel.flatten(), trad_accel.flatten())[0, 1]
    metrics['gyro_correlation'] = np.corrcoef(ml_gyro.flatten(), trad_gyro.flatten())[0, 1]
    metrics['gnss_correlation'] = np.corrcoef(ml_gnss_interpolated.flatten(), trad_gnss_pos.flatten())[0, 1]
    
    return metrics


def simulate_trajectory_with_ml(
    model: SensorSimulationRNN,
    timestamps: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    orientations: np.ndarray,
    device: str = 'cpu'
) -> Tuple[dict, dict]:
    """
    Complete pipeline to simulate sensor data for a trajectory using ML model.
    
    Args:
        model: Trained SensorSimulationRNN model
        timestamps: Time vector (T,)
        positions: Position vectors (T, 3)
        velocities: Velocity vectors (T, 3)
        orientations: Quaternion orientations (T, 4)
        device: Device to run inference on
        
    Returns:
        Tuple of (imu_dict, gnss_dict) in sensor simulator format
    """
    # Prepare input sequence
    input_sequence = np.concatenate([
        positions, velocities, orientations
    ], axis=1)  # Shape: (T, 10)
    
    # Run ML inference
    imu_data, gnss_data = simulate_with_model(
        model=model,
        input_sequence=input_sequence,
        device=device
    )
    
    # Convert to sensor simulator format
    imu_dict, gnss_dict = create_sensor_data_dict(
        imu_data=imu_data,
        gnss_data=gnss_data,
        timestamps=timestamps
    )
    
    return imu_dict, gnss_dict


def main():
    """Example usage of the inference module."""
    print("🤖 ML Sensor Simulation Inference Module")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example: Load a trained model (if available)
    model_path = "trained_sensor_model.pth"
    
    if os.path.exists(model_path):
        model = load_trained_model(
            model_path=model_path,
            device=device
        )
        print("✅ Model loaded successfully!")
    else:
        print("⚠️  No trained model found. Creating untrained model for demonstration.")
        model = SensorSimulationRNN()
        model = model.to(device)
        model.eval()
    
    # Example: Generate sample trajectory data
    T = 100  # 100 time steps
    input_sequence = np.random.randn(T, 10)  # Random trajectory data
    
    print(f"\n🧪 Running inference on {T} time steps...")
    
    # Run inference
    imu_data, gnss_data = simulate_with_model(
        model=model,
        input_sequence=input_sequence,
        device=device
    )
    
    print(f"✅ Inference completed!")
    print(f"   IMU data shape: {imu_data.shape}")
    print(f"   GNSS data shape: {gnss_data.shape}")
    print(f"   Accelerometer range: [{imu_data[:, :3].min():.3f}, {imu_data[:, :3].max():.3f}]")
    print(f"   Gyroscope range: [{imu_data[:, 3:].min():.3f}, {imu_data[:, 3:].max():.3f}]")
    print(f"   GNSS position range: [{gnss_data.min():.3f}, {gnss_data.max():.3f}]")
    
    print("\n🎯 Inference module ready for use!")
    print("   Use simulate_with_model() for single trajectories")
    print("   Use simulate_with_model_batch() for multiple trajectories")
    print("   Use simulate_trajectory_with_ml() for complete pipeline")


if __name__ == "__main__":
    main() 