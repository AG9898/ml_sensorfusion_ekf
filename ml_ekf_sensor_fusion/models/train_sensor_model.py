#!/usr/bin/env python3
"""
Training Script for RNN-based Sensor Simulation Model

This script trains a neural network to learn the mapping from ground truth
trajectory data to realistic IMU and GNSS measurements.
"""

import sys
import os

# Fix OpenMP warning on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our custom modules
from simulation.trajectory_generator import generate_trajectory
from simulation.sensor_simulator import SensorSimulator
from models.imu_gnss_generator import IMUGenerator, GNSSGenerator


class SensorSimulationDataset(Dataset):
    """
    Dataset for training sensor simulation models.
    
    Creates input-output pairs from ground truth trajectory data:
    - Input: [position(3), velocity(3), orientation(4)] → shape (seq_len, 10)
    - Output: [accel(3), gyro(3), gnss_pos(3)] → shape (seq_len, 9)
    """
    
    def __init__(
        self,
        sequence_length: int = 50,
        overlap: int = 25,
        num_trajectories: int = 100,
        trajectory_duration: float = 10.0,
        dt: float = 0.01,
        modes: List[str] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            sequence_length: Length of training sequences
            overlap: Overlap between consecutive sequences
            num_trajectories: Number of trajectories to generate
            trajectory_duration: Duration of each trajectory (seconds)
            dt: Time step (seconds)
            modes: List of trajectory modes to use
        """
        self.sequence_length = sequence_length
        self.overlap = overlap
        
        if modes is None:
            modes = ["circular", "sinusoidal", "helix", "figure8"]
        
        # Generate training data
        self.sequences = self._generate_training_data(
            num_trajectories, trajectory_duration, dt, modes
        )
        
        print(f"✅ Generated {len(self.sequences)} training sequences")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Overlap: {overlap}")
        print(f"   Input shape: (seq_len, 10)")
        print(f"   Output shape: (seq_len, 9)")
    
    def _generate_trajectory_data(
        self,
        mode: str,
        duration: float,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Generate trajectory and sensor data for a single trajectory."""
        # Generate ground truth trajectory
        timestamps, positions, velocities, orientations = generate_trajectory(
            duration=duration,
            dt=dt,
            mode=mode,
            radius=np.random.uniform(3.0, 8.0),
            angular_speed=np.random.uniform(0.5, 2.0),
            z_amplitude=np.random.uniform(0.5, 2.0)
        )
        
        # Create sensor simulator with random noise parameters
        simulator = SensorSimulator(
            accel_noise_std=np.random.uniform(0.05, 0.2),
            gyro_noise_std=np.random.uniform(0.005, 0.02),
            accel_bias_std=np.random.uniform(0.005, 0.02),
            gyro_bias_std=np.random.uniform(0.0005, 0.002),
            gnss_noise_std=np.random.uniform(0.3, 2.0),
            gnss_rate=np.random.choice([0.5, 1.0, 2.0])
        )
        
        # Simulate sensor measurements
        imu_data, gnss_data = simulator.simulate_sensors(
            timestamps=timestamps,
            positions=positions,
            velocities=velocities,
            orientations=orientations
        )
        
        return timestamps, positions, velocities, orientations, imu_data, gnss_data
    
    def _generate_training_data(
        self,
        num_trajectories: int,
        duration: float,
        dt: float,
        modes: List[str]
    ) -> List[Dict]:
        """Generate all training sequences."""
        sequences = []
        
        print(f"🔄 Generating {num_trajectories} trajectories...")
        
        for i in tqdm(range(num_trajectories), desc="Generating trajectories"):
            # Randomly select trajectory mode
            mode = np.random.choice(modes)
            
            try:
                # Generate trajectory and sensor data
                timestamps, positions, velocities, orientations, imu_data, gnss_data = \
                    self._generate_trajectory_data(mode, duration, dt)
                
                # Create sequences from this trajectory
                trajectory_sequences = self._create_sequences(
                    timestamps, positions, velocities, orientations,
                    imu_data, gnss_data
                )
                
                sequences.extend(trajectory_sequences)
                
            except Exception as e:
                print(f"⚠️  Skipping trajectory {i} due to error: {e}")
                continue
        
        return sequences
    
    def _create_sequences(
        self,
        timestamps: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        orientations: np.ndarray,
        imu_data: Dict,
        gnss_data: Dict
    ) -> List[Dict]:
        """Create overlapping sequences from trajectory data."""
        sequences = []
        
        # Extract sensor measurements
        imu_accel = imu_data['accel']
        imu_gyro = imu_data['gyro']
        gnss_positions = gnss_data['positions']
        gnss_timestamps = gnss_data['timestamps']
        
        # Interpolate GNSS data to match IMU timestamps
        gnss_interpolated = self._interpolate_gnss_data(
            gnss_timestamps, gnss_positions, timestamps
        )
        
        # Prepare input features: [pos(3), vel(3), orientation(4)]
        input_features = np.concatenate([
            positions, velocities, orientations
        ], axis=1)
        
        # Prepare output features: [accel(3), gyro(3), gnss_pos(3)]
        output_features = np.concatenate([
            imu_accel, imu_gyro, gnss_interpolated
        ], axis=1)
        
        # Create overlapping sequences
        step_size = self.sequence_length - self.overlap
        
        for start_idx in range(0, len(timestamps) - self.sequence_length + 1, step_size):
            end_idx = start_idx + self.sequence_length
            
            sequence = {
                'input': input_features[start_idx:end_idx],
                'output': output_features[start_idx:end_idx],
                'timestamps': timestamps[start_idx:end_idx]
            }
            sequences.append(sequence)
        
        return sequences
    
    def _interpolate_gnss_data(
        self,
        gnss_timestamps: np.ndarray,
        gnss_positions: np.ndarray,
        target_timestamps: np.ndarray
    ) -> np.ndarray:
        """Interpolate GNSS data to match IMU timestamps."""
        if len(gnss_timestamps) == 0:
            # If no GNSS data, use zeros
            return np.zeros((len(target_timestamps), 3))
        
        if len(gnss_timestamps) == 1:
            # If only one GNSS measurement, repeat it
            return np.tile(gnss_positions[0], (len(target_timestamps), 1))
        
        # Linear interpolation
        interpolated = np.zeros((len(target_timestamps), 3))
        
        for i, t in enumerate(target_timestamps):
            # Find the two GNSS measurements that bracket this time
            if t <= gnss_timestamps[0]:
                interpolated[i] = gnss_positions[0]
            elif t >= gnss_timestamps[-1]:
                interpolated[i] = gnss_positions[-1]
            else:
                # Find the index where gnss_timestamps[j] <= t < gnss_timestamps[j+1]
                j = np.searchsorted(gnss_timestamps, t) - 1
                if j < 0:
                    j = 0
                
                # Linear interpolation
                t1, t2 = gnss_timestamps[j], gnss_timestamps[j+1]
                pos1, pos2 = gnss_positions[j], gnss_positions[j+1]
                
                alpha = (t - t1) / (t2 - t1)
                interpolated[i] = (1 - alpha) * pos1 + alpha * pos2
        
        return interpolated
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        return {
            'input': torch.FloatTensor(sequence['input']),
            'output': torch.FloatTensor(sequence['output']),
            'timestamps': torch.FloatTensor(sequence['timestamps'])
        }


class SensorSimulationRNN(nn.Module):
    """
    RNN-based model for sensor simulation.
    
    Learns to map ground truth trajectory data to realistic sensor measurements.
    """
    
    def __init__(
        self,
        input_dim: int = 10,  # pos(3) + vel(3) + orientation(4)
        hidden_dim: int = 128,
        output_dim: int = 9,  # accel(3) + gyro(3) + gnss_pos(3)
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(SensorSimulationRNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection layers
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
        Forward pass through the network.
        
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


class SensorSimulationTrainer:
    """Trainer for the sensor simulation model."""
    
    def __init__(
        self,
        model: SensorSimulationRNN,
        device: str = 'cpu',
        learning_rate: float = 1e-3
    ):
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training", leave=False):
            # Move data to device
            inputs = batch['input'].to(self.device)
            targets = batch['output'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Update learning rate
        self.scheduler.step()
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating", leave=False):
                inputs = batch['input'].to(self.device)
                targets = batch['output'].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 10
    ):
        """Train the model for multiple epochs."""
        print(f"🚀 Starting training for {num_epochs} epochs...")
        print(f"   Device: {self.device}")
        print(f"   Learning rate: {self.optimizer.param_groups[0]['lr']}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_dataloader)
            self.val_losses.append(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}")
            
            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"   🎯 New best validation loss: {best_val_loss:.6f}")
        
        print(f"✅ Training completed! Best validation loss: {best_val_loss:.6f}")
    
    def plot_training_history(self):
        """Plot training and validation loss history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def create_dataloaders(
    train_ratio: float = 0.8,
    batch_size: int = 32,
    sequence_length: int = 50,
    num_trajectories: int = 100
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    print("📊 Creating datasets and dataloaders...")
    
    # Create full dataset
    full_dataset = SensorSimulationDataset(
        sequence_length=sequence_length,
        overlap=25,
        num_trajectories=num_trajectories,
        trajectory_duration=10.0,
        dt=0.01,
        modes=["circular", "sinusoidal", "helix", "figure8"]
    )
    
    # Split into train and validation
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✅ Created dataloaders:")
    print(f"   Training: {len(train_dataset)} sequences")
    print(f"   Validation: {len(val_dataset)} sequences")
    print(f"   Batch size: {batch_size}")
    
    return train_dataloader, val_dataloader


def evaluate_model(
    model: SensorSimulationRNN,
    dataloader: DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Evaluate the trained model."""
    model.eval()
    
    total_loss = 0.0
    accel_loss = 0.0
    gyro_loss = 0.0
    gnss_loss = 0.0
    num_batches = 0
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch['input'].to(device)
            targets = batch['output'].to(device)
            
            outputs = model(inputs)
            
            # Overall loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Component-wise losses
            accel_loss += criterion(outputs[:, :, :3], targets[:, :, :3]).item()
            gyro_loss += criterion(outputs[:, :, 3:6], targets[:, :, 3:6]).item()
            gnss_loss += criterion(outputs[:, :, 6:], targets[:, :, 6:]).item()
            
            num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'accel_loss': accel_loss / num_batches,
        'gyro_loss': gyro_loss / num_batches,
        'gnss_loss': gnss_loss / num_batches
    }


def main():
    """Main training function."""
    print("🤖 RNN-based Sensor Simulation Training")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        train_ratio=0.8,
        batch_size=32,
        sequence_length=50,
        num_trajectories=100
    )
    
    # Create model
    model = SensorSimulationRNN(
        input_dim=10,      # pos(3) + vel(3) + orientation(4)
        hidden_dim=128,
        output_dim=9,      # accel(3) + gyro(3) + gnss_pos(3)
        num_layers=2,
        dropout=0.1
    )
    
    print(f"📊 Model architecture:")
    print(f"   Input dimension: {model.input_dim}")
    print(f"   Hidden dimension: {model.hidden_dim}")
    print(f"   Output dimension: {model.output_dim}")
    print(f"   Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = SensorSimulationTrainer(
        model=model,
        device=device,
        learning_rate=1e-3
    )
    
    # Train the model
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=10
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate final model
    print("\n📈 Final Model Evaluation")
    print("-" * 30)
    
    final_metrics = evaluate_model(model, val_dataloader, device)
    
    for metric, value in final_metrics.items():
        print(f"   {metric}: {value:.6f}")
    
    print("\n✅ Training completed successfully!")
    print("🎯 The model has learned to simulate sensor measurements from ground truth trajectories.")


if __name__ == "__main__":
    main() 