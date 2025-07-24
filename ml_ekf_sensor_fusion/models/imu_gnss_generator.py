"""
ML-based IMU and GNSS Sensor Generator

This module implements machine learning models to generate realistic sensor
measurements from ground truth trajectories. The models learn the complex
relationships between motion and sensor outputs, including noise patterns
and sensor-specific characteristics.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any, List
from scipy.spatial.transform import Rotation as R
import pickle
import os


class IMUGenerator(nn.Module):
    """Neural network for generating IMU measurements from trajectory data."""
    
    def __init__(
        self,
        input_dim: int = 13,  # position(3) + velocity(3) + orientation(4) + acceleration(3)
        hidden_dim: int = 128,
        output_dim: int = 6,  # accelerometer(3) + gyroscope(3)
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super(IMUGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the IMU generator.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Project to output
        output = self.output_projection(lstm_out)
        
        return output


class GNSSGenerator(nn.Module):
    """Neural network for generating GNSS measurements from trajectory data."""
    
    def __init__(
        self,
        input_dim: int = 6,  # position(3) + velocity(3)
        hidden_dim: int = 64,
        output_dim: int = 6,  # noisy position(3) + velocity(3)
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(GNSSGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNSS generator.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        lstm_out, _ = self.lstm(x)
        output = self.output_projection(lstm_out)
        return output


class TrajectoryDataset(Dataset):
    """Dataset for training sensor generators."""
    
    def __init__(
        self,
        ground_truth_data: List[Dict],
        sensor_data: List[Dict],
        sequence_length: int = 100,
        overlap: int = 50
    ):
        """
        Initialize dataset.
        
        Args:
            ground_truth_data: List of ground truth trajectory dictionaries
            sensor_data: List of corresponding sensor measurement dictionaries
            sequence_length: Length of training sequences
            overlap: Overlap between consecutive sequences
        """
        self.sequence_length = sequence_length
        self.overlap = overlap
        
        # Prepare sequences
        self.sequences = self._prepare_sequences(ground_truth_data, sensor_data)
    
    def _prepare_sequences(
        self,
        ground_truth_data: List[Dict],
        sensor_data: List[Dict]
    ) -> List[Dict]:
        """Prepare training sequences from trajectory data."""
        sequences = []
        
        for gt_traj, sensor_traj in zip(ground_truth_data, sensor_data):
            timestamps = gt_traj['timestamps']
            positions = gt_traj['positions']
            velocities = gt_traj['velocities']
            orientations = gt_traj['orientations']
            
            # Compute accelerations
            dt = timestamps[1] - timestamps[0]
            accelerations = np.zeros_like(velocities)
            accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2 * dt)
            accelerations[0] = (velocities[1] - velocities[0]) / dt
            accelerations[-1] = (velocities[-1] - velocities[-2]) / dt
            
            # Prepare input features
            input_features = np.concatenate([
                positions, velocities, orientations, accelerations
            ], axis=1)
            
            # Extract sensor measurements
            imu_accel = sensor_traj['imu_accel']
            imu_gyro = sensor_traj['imu_gyro']
            gnss_pos = sensor_traj['gnss_pos']
            gnss_vel = sensor_traj['gnss_vel']
            
            # Create overlapping sequences
            for start_idx in range(0, len(timestamps) - self.sequence_length + 1, 
                                 self.sequence_length - self.overlap):
                end_idx = start_idx + self.sequence_length
                
                sequence = {
                    'input_features': input_features[start_idx:end_idx],
                    'imu_accel': imu_accel[start_idx:end_idx],
                    'imu_gyro': imu_gyro[start_idx:end_idx],
                    'gnss_pos': gnss_pos[start_idx:end_idx],
                    'gnss_vel': gnss_vel[start_idx:end_idx]
                }
                sequences.append(sequence)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        return {
            'input_features': torch.FloatTensor(sequence['input_features']),
            'imu_accel': torch.FloatTensor(sequence['imu_accel']),
            'imu_gyro': torch.FloatTensor(sequence['imu_gyro']),
            'gnss_pos': torch.FloatTensor(sequence['gnss_pos']),
            'gnss_vel': torch.FloatTensor(sequence['gnss_vel'])
        }


class SensorGeneratorTrainer:
    """Trainer for sensor generator models."""
    
    def __init__(
        self,
        imu_model: IMUGenerator,
        gnss_model: GNSSGenerator,
        device: str = 'cpu'
    ):
        self.imu_model = imu_model.to(device)
        self.gnss_model = gnss_model.to(device)
        self.device = device
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        
        # Optimizers
        self.imu_optimizer = optim.Adam(imu_model.parameters(), lr=1e-3)
        self.gnss_optimizer = optim.Adam(gnss_model.parameters(), lr=1e-3)
        
        # Learning rate schedulers
        self.imu_scheduler = optim.lr_scheduler.StepLR(self.imu_optimizer, step_size=50, gamma=0.5)
        self.gnss_scheduler = optim.lr_scheduler.StepLR(self.gnss_optimizer, step_size=50, gamma=0.5)
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.imu_model.train()
        self.gnss_model.train()
        
        total_imu_loss = 0.0
        total_gnss_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Move to device
            input_features = batch['input_features'].to(self.device)
            imu_accel = batch['imu_accel'].to(self.device)
            imu_gyro = batch['imu_gyro'].to(self.device)
            gnss_pos = batch['gnss_pos'].to(self.device)
            gnss_vel = batch['gnss_vel'].to(self.device)
            
            # Train IMU model
            self.imu_optimizer.zero_grad()
            imu_output = self.imu_model(input_features)
            imu_target = torch.cat([imu_accel, imu_gyro], dim=-1)
            imu_loss = self.mse_loss(imu_output, imu_target)
            imu_loss.backward()
            self.imu_optimizer.step()
            
            # Train GNSS model
            self.gnss_optimizer.zero_grad()
            gnss_input = torch.cat([
                input_features[:, :, :3],  # positions
                input_features[:, :, 3:6]  # velocities
            ], dim=-1)
            gnss_output = self.gnss_model(gnss_input)
            gnss_target = torch.cat([gnss_pos, gnss_vel], dim=-1)
            gnss_loss = self.mse_loss(gnss_output, gnss_target)
            gnss_loss.backward()
            self.gnss_optimizer.step()
            
            total_imu_loss += imu_loss.item()
            total_gnss_loss += gnss_loss.item()
            num_batches += 1
        
        # Update learning rates
        self.imu_scheduler.step()
        self.gnss_scheduler.step()
        
        return {
            'imu_loss': total_imu_loss / num_batches,
            'gnss_loss': total_gnss_loss / num_batches
        }
    
    def save_models(self, save_dir: str):
        """Save trained models."""
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.imu_model.state_dict(), os.path.join(save_dir, 'imu_generator.pth'))
        torch.save(self.gnss_model.state_dict(), os.path.join(save_dir, 'gnss_generator.pth'))
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir: str):
        """Load trained models."""
        imu_path = os.path.join(save_dir, 'imu_generator.pth')
        gnss_path = os.path.join(save_dir, 'gnss_generator.pth')
        
        if os.path.exists(imu_path):
            self.imu_model.load_state_dict(torch.load(imu_path, map_location=self.device))
            print(f"Loaded IMU model from {imu_path}")
        
        if os.path.exists(gnss_path):
            self.gnss_model.load_state_dict(torch.load(gnss_path, map_location=self.device))
            print(f"Loaded GNSS model from {gnss_path}")


class MLSensorGenerator:
    """Main class for ML-based sensor generation."""
    
    def __init__(
        self,
        model_dir: str = "models",
        device: str = 'cpu'
    ):
        self.device = device
        self.model_dir = model_dir
        
        # Initialize models
        self.imu_model = IMUGenerator()
        self.gnss_model = GNSSGenerator()
        
        # Initialize trainer
        self.trainer = SensorGeneratorTrainer(self.imu_model, self.gnss_model, device)
        
        # Load pre-trained models if available
        if os.path.exists(model_dir):
            self.trainer.load_models(model_dir)
    
    def train(
        self,
        ground_truth_data: List[Dict],
        sensor_data: List[Dict],
        num_epochs: int = 100,
        batch_size: int = 32,
        sequence_length: int = 100
    ):
        """Train the sensor generator models."""
        print("Preparing training data...")
        
        # Create dataset
        dataset = TrajectoryDataset(ground_truth_data, sensor_data, sequence_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Training on {len(dataset)} sequences...")
        
        # Training loop
        for epoch in range(num_epochs):
            losses = self.trainer.train_epoch(dataloader, epoch)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"IMU Loss: {losses['imu_loss']:.6f}, "
                      f"GNSS Loss: {losses['gnss_loss']:.6f}")
        
        # Save models
        self.trainer.save_models(self.model_dir)
        print("Training completed!")
    
    def generate_imu_measurements(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        orientations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate IMU measurements using trained model."""
        self.imu_model.eval()
        
        # Compute accelerations
        dt = 0.01  # Assuming fixed timestep
        accelerations = np.zeros_like(velocities)
        accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2 * dt)
        accelerations[0] = (velocities[1] - velocities[0]) / dt
        accelerations[-1] = (velocities[-1] - velocities[-2]) / dt
        
        # Prepare input features
        input_features = np.concatenate([
            positions, velocities, orientations, accelerations
        ], axis=1)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(self.device)
        
        # Generate predictions
        with torch.no_grad():
            output = self.imu_model(input_tensor)
            output = output.squeeze(0).cpu().numpy()
        
        # Split into accelerometer and gyroscope
        accel_measurements = output[:, :3]
        gyro_measurements = output[:, 3:]
        
        return accel_measurements, gyro_measurements
    
    def generate_gnss_measurements(
        self,
        positions: np.ndarray,
        velocities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate GNSS measurements using trained model."""
        self.gnss_model.eval()
        
        # Prepare input features
        input_features = np.concatenate([positions, velocities], axis=1)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(self.device)
        
        # Generate predictions
        with torch.no_grad():
            output = self.gnss_model(input_tensor)
            output = output.squeeze(0).cpu().numpy()
        
        # Split into position and velocity
        pos_measurements = output[:, :3]
        vel_measurements = output[:, 3:]
        
        return pos_measurements, vel_measurements 