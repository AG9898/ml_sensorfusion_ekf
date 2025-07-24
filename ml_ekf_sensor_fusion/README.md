# 6-DOF Ground Truth Trajectory Generator

A comprehensive Python library for generating synthetic 6-DOF (6 Degrees of Freedom) trajectories for testing and training machine learning models and Extended Kalman Filter (EKF) pipelines.

## 🚀 Features

- **Multiple Trajectory Types**: Circular, sinusoidal, random walk, helix, and figure-8 patterns
- **Configurable Parameters**: Duration, timestep, and trajectory-specific parameters
- **6-DOF Output**: Position (3D), velocity (3D), and orientation (quaternions)
- **Noise Addition**: Realistic sensor noise simulation
- **Visualization**: Comprehensive plotting capabilities
- **Data I/O**: Save/load trajectory data
- **Metrics Computation**: Trajectory analysis and statistics
- **Vectorized Operations**: Optimized for computational efficiency

## 📦 Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

```python
from trajectory_generator import generate_trajectory, plot_trajectory

# Generate a simple circular trajectory
timestamps, positions, velocities, orientations = generate_trajectory(
    duration=10.0,
    dt=0.01,
    mode="circular"
)

# Plot the trajectory
plot_trajectory(timestamps, positions, velocities, orientations)
```

## 📋 API Reference

### Main Function

#### `generate_trajectory(duration, dt, mode, **kwargs)`

Generate a synthetic 6-DOF trajectory.

**Parameters:**
- `duration` (float): Total simulation time in seconds (default: 10.0)
- `dt` (float): Timestep in seconds (default: 0.01)
- `mode` (str): Trajectory type (see supported modes below)
- `**kwargs`: Additional parameters for specific trajectory modes

**Returns:**
- `timestamps` (np.ndarray): Time vector of shape (N,)
- `positions` (np.ndarray): Position vectors (N, 3)
- `velocities` (np.ndarray): Velocity vectors (N, 3)
- `orientations` (np.ndarray): Quaternion orientation (N, 4)

### Supported Trajectory Modes

#### 1. Circular (`mode="circular"`)
Generates a circular trajectory with optional vertical oscillation.

**Parameters:**
- `radius` (float): Circle radius (default: 5.0)
- `angular_speed` (float): Angular velocity (default: 2π/duration)
- `z_amplitude` (float): Z-axis oscillation amplitude (default: 1.0)

#### 2. Sinusoidal (`mode="sinusoidal"`)
Generates independent sinusoidal motion in each axis.

**Parameters:**
- `amplitude` (float): Motion amplitude (default: 2.0)
- `freq_x`, `freq_y`, `freq_z` (float): Frequencies for each axis

#### 3. Random Walk (`mode="random_walk"`)
Generates a random walk trajectory.

**Parameters:**
- `step_std` (float): Standard deviation of position steps (default: 0.1)
- `orientation_std` (float): Standard deviation of orientation changes (default: 0.01)

#### 4. Helix (`mode="helix"`)
Generates a helical trajectory.

**Parameters:**
- `radius` (float): Helix radius (default: 3.0)
- `pitch` (float): Helix pitch (default: 1.0)
- `angular_speed` (float): Angular velocity (default: 2π/duration)

#### 5. Figure-8 (`mode="figure8"`)
Generates a figure-8 trajectory.

**Parameters:**
- `radius` (float): Figure-8 size (default: 4.0)
- `angular_speed` (float): Angular velocity (default: 2π/duration)

### Utility Functions

#### `add_noise(positions, velocities, orientations, pos_std, vel_std, orient_std)`
Add Gaussian noise to trajectory data.

#### `compute_metrics(positions, velocities, orientations)`
Compute trajectory statistics and metrics.

#### `plot_trajectory(timestamps, positions, velocities, orientations, title, save_path)`
Create comprehensive trajectory visualization.

#### `save_trajectory_data(timestamps, positions, velocities, orientations, filepath)`
Save trajectory data to NumPy file.

#### `load_trajectory_data(filepath)`
Load trajectory data from NumPy file.

## 🔧 Usage Examples

### Basic Usage

```python
from trajectory_generator import generate_trajectory

# Generate a circular trajectory
timestamps, positions, velocities, orientations = generate_trajectory(
    duration=10.0,
    dt=0.01,
    mode="circular",
    radius=5.0
)
```

### Custom Parameters

```python
# Generate a helix with custom parameters
timestamps, positions, velocities, orientations = generate_trajectory(
    duration=15.0,
    dt=0.005,
    mode="helix",
    radius=2.0,
    pitch=2.0,
    angular_speed=3.0
)
```

### Adding Noise

```python
from trajectory_generator import generate_trajectory, add_noise

# Generate clean trajectory
timestamps, positions, velocities, orientations = generate_trajectory(
    duration=10.0, dt=0.01, mode="circular"
)

# Add realistic sensor noise
noisy_pos, noisy_vel, noisy_orient = add_noise(
    positions, velocities, orientations,
    pos_std=0.05,    # GPS-like noise
    vel_std=0.1,     # IMU-like noise
    orient_std=0.01  # Magnetometer-like noise
)
```

### Visualization

```python
from trajectory_generator import generate_trajectory, plot_trajectory

# Generate and plot trajectory
timestamps, positions, velocities, orientations = generate_trajectory(
    duration=10.0, dt=0.01, mode="figure8"
)

plot_trajectory(
    timestamps, positions, velocities, orientations,
    title="Figure-8 Trajectory",
    save_path="trajectory_plot.png"
)
```

### ML Training Dataset

```python
import numpy as np
from trajectory_generator import generate_trajectory, add_noise

# Generate diverse dataset
trajectories = []
modes = ["circular", "sinusoidal", "helix", "figure8"]

for mode in modes:
    timestamps, positions, velocities, orientations = generate_trajectory(
        duration=10.0, dt=0.01, mode=mode
    )
    
    # Add noise for realistic training data
    noisy_pos, noisy_vel, noisy_orient = add_noise(
        positions, velocities, orientations
    )
    
    trajectories.append({
        'ground_truth': (positions, velocities, orientations),
        'noisy': (noisy_pos, noisy_vel, noisy_orient),
        'timestamps': timestamps
    })
```

## 📊 Output Format

The trajectory generator returns data in the following format:

- **timestamps**: 1D array of shape (N,) containing time points
- **positions**: 2D array of shape (N, 3) containing [x, y, z] positions
- **velocities**: 2D array of shape (N, 3) containing [vx, vy, vz] velocities
- **orientations**: 2D array of shape (N, 4) containing quaternions [qw, qx, qy, qz]

## 🎨 Visualization

The `plot_trajectory` function creates a comprehensive 6-panel visualization:

1. **3D Trajectory**: Full 3D path with start/end markers
2. **Position vs Time**: Individual axis positions over time
3. **Velocity vs Time**: Individual axis velocities over time
4. **Orientation vs Time**: Quaternion components over time
5. **Velocity Magnitude**: Speed over time
6. **XY Projection**: Top-down view of the trajectory

## 🔬 Applications

### Machine Learning Training
- Generate synthetic datasets for trajectory prediction models
- Create diverse training data with known ground truth
- Simulate sensor noise for robust model training

### EKF Testing and Validation
- Test Extended Kalman Filter implementations
- Validate sensor fusion algorithms
- Benchmark filtering performance

### Robotics Simulation
- Generate reference trajectories for path planning
- Test control algorithms with known dynamics
- Validate state estimation methods

## 📈 Performance

- **Vectorized Operations**: All computations use NumPy vectorization for efficiency
- **Memory Efficient**: Minimal memory overhead during generation
- **Scalable**: Handles long trajectories with thousands of points
- **Fast**: Typical generation time < 1ms for 1000-point trajectories

## 🤝 Contributing

Feel free to contribute by:
- Adding new trajectory types
- Improving visualization capabilities
- Optimizing performance
- Adding more utility functions

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues, questions, or feature requests, please open an issue in the repository. 