# Genesis MARL VecEnv V4 Framework

This document provides comprehensive guidance for the Genesis Multi-Agent Reinforcement Learning (MARL) Vectorized Environment framework. This framework combines PettingZoo's Agent-Environment Cycle (AEC) API with various RL training frameworks (RSL-RL, MAPPO, HARL, OpenRL) to enable GPU-accelerated, vectorized MARL training using Genesis as the primary physics simulator backend.

## Table of Contents

1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Key Components](#key-components)
4. [Configuration System](#configuration-system)
5. [Training Integration](#training-integration)
6. [Simulation Interface](#simulation-interface)
7. [Support Systems](#support-systems)
8. [Usage Examples](#usage-examples)
9. [Development Guidelines](#development-guidelines)
10. [Testing Strategy](#testing-strategy)

## Overview

### Purpose

The Genesis MARL VecEnv framework provides a unified interface for training multiple robots with different control frequencies in a shared environment. It bridges the gap between:
- **Multi-agent environments** (using a modified PettingZoo AEC API)
- **Single-agent RL training frameworks** (RSL-RL, HARL, OpenRL, etc.)
- **Physics simulators** (Genesis, with plans for MuJoCo and Isaac Gym/Lab, pymunk, etc.)

### Key Features

- **Vectorized Environments**: Efficient parallel simulation of multiple environments on GPU
- **Multi-frequency Control**: Support for robots operating at different control frequencies
- **Flexible Training**: Integration with multiple RL frameworks through unified interfaces
- **Simulator Abstraction**: Clean abstraction layer for different physics backends
- **Soft Robot Death**: Robots can "die" within episodes without terminating the entire environment
- **Auto-reset**: Automatic environment reset when episodes complete
- **Camera System**: Built-in camera tracking and video recording capabilities
- **Hierarchical Logging**: Instance-based logging system for debugging complex multi-threaded training

## Core Architecture

### Design Principles

1. **Separation of Concerns**: Clear boundaries between environment management, agent control, and training
2. **Reference-based Data Sharing**: Use `RefDict` to avoid memory duplication in joint training scenarios
3. **GPU-first Design**: Keep computations on GPU using tensors whenever possible
4. **Thread Safety**: Careful synchronization for multi-threaded training processes
5. **Extensibility**: Easy to add new robots, training algorithms, and simulators

### Component Hierarchy

```
VectorizedAECEnv (Main Orchestrator)
├── Scene Interface (Simulator Abstraction)
│   ├── Genesis Interface
│   ├── Camera Manager
│   └── Debug Mark Manager
├── Agents (Grouped by Frequency)
│   ├── Robot Interfaces
│   └── Locomotion Models
├── SubVecEnvs (Training Interfaces)
│   ├── RSL-RL Interface
│   ├── MAPPO Interface
│   ├── HARL Interface
│   └── OpenRL Interface
├── Scheduler (Frequency Management)
└── Coordinator (Thread Synchronization)
```

### Data Flow

1. **Initialization**: Create robots, setup training configs, initialize buffers
2. **AEC Loop**: `reset()` → `last()` → `step()` → `last()` → `step()` ...
3. **Communication**: Agents ↔ SubVecEnvs via coordinator
4. **Training**: SubVecEnvs interface with external RL frameworks

## Key Components

### VectorizedAECEnv (`vectorized_aec_env.py`)

The main orchestrator that manages the entire system:

```python
class VectorizedAECEnv:
    """Core environment following modified AEC pattern"""
    
    def __init__(self, training_configs, n_envs, device, ...):
        # Initialize scene, agents, subvecenvs, scheduler
        
    def reset(self):
        # Initial reset of all environments
        
    def last(self):
        # Calculate obs/reward/termination for current agent
        # Handle auto-reset for terminated environments
        
    def step(self):
        # Execute current agent's actions
```

**Key Modifications from Standard AEC**:
- Agents never die (only robots have soft death)
- Automatic agent selection based on scheduler
- No action argument in `step()` - actions come via semaphores
- Auto-reset handled internally

### Agent (`agent.py`)

Manages robot control and communication with training subsystems:

```python
class Agent:
    """Basic unit of interaction with AEC environment"""
    
    def __init__(self, agent_name, frequency, training_configs, mother_env):
        # Setup robots, locomotion models, subvecenv references
        
    def step(self):
        # Get actions from subvecenvs
        # Apply locomotion models if needed
        # Send commands to simulator
        
    def last(self):
        # Forward obs/reward/info from AEC to subvecenvs
```

**Key Features**:
- Batch locomotion inference for efficiency
- Two-layer control support (high-level actions → joint commands)
- Handles multiple training configurations

### SubVecEnv (`trainer_interfaces/sub_vecenv.py`)

Interface between AEC environment and RL training frameworks:

```python
class SubVecEnv(ABC):
    """Bridge to RL training algorithms"""
    
    def reset(self):
        # Get initial observations
        
    def step(self, actions):
        # Send actions to AEC, get next obs/reward/done
        
    @abstractmethod
    def _process_data(self, data, is_reset=False):
        # Convert AEC format to trainer-specific format
        
    @abstractmethod
    def _process_action(self, action):
        # Convert trainer action format to AEC format
```

**Implementations**:
- `RSL_RLSubVecEnv`: For RSL-RL training
- `HARLSubVecEnv`: For HARL algorithms
- `OpenRLSubVecEnv`: For OpenRL framework

## Configuration System

### RobotConfig (`configs/RobotConfig.py`)

Defines individual robot properties and behaviors:

```python
@dataclass
class RobotConfig:
    # Basic properties
    name: str
    urdf_path: str
    frequency: float
    control_mode: str = "position"
    
    # Spaces (auto-detected if None)
    action_space: gym.spaces.Space = None
    observation_space: gym.spaces.Space = None
    
    # Hooks for customization
    pre_build_hook: Optional[Callable] = None
    post_build_hook: Optional[Callable] = None
    pre_reset_hook: Optional[Callable] = None
    post_reset_hook: Optional[Callable] = None
    
    # Core functions
    setup_function: Optional[Callable] = None
    obs_function: Optional[Callable] = None
    reward_function: Optional[Callable] = None
    truncation_function: Optional[Callable] = None
    info_function: Optional[Callable] = None
    
    # Physical parameters
    initial_position: Optional[List[float]] = None
    initial_orientation: Optional[List[float]] = None
    DP_kp: Union[float, List[float]] = 20.0
    DP_kd: Union[float, List[float]] = 0.5
```

### TrainingConfig (`configs/TrainingConfig.py`)

Groups robots for training with specific algorithms:

```python
@dataclass
class TrainingConfig(ABC):
    training_name: str
    training_type: Literal["NORMAL", "JOINT", "SHARED"]
    robot_configs: Dict[str, RobotConfig]
    
    # Override robot-level functions if needed
    obs_function: Optional[Callable] = None
    reward_function: Optional[Callable] = None
    shared_obs_function: Optional[Callable] = None
    
    # Trainer-specific properties
    @property
    @abstractmethod
    def trainer_name(self):
        pass
    
    @property
    @abstractmethod
    def trainer_env_factory(self):
        pass
```

**Training Types**:
- **NORMAL**: Single robot training
- **JOINT**: Multiple robots share observations/rewards
- **SHARED**: Multiple homogeneous robots share network weights

### Implementation Example: RSL-RL Config

```python
@dataclass
class RSL_RLTrainingConfig(TrainingConfig):
    trainer_name: str = "rsl_rl"
    
    # RSL-RL specific configs
    algorithm_cfg: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    runner_cfg: RunnerConfig = field(default_factory=RunnerConfig)
    policy_cfg: PolicyConfig = field(default_factory=PolicyConfig)
    
    @property
    def trainer_env_factory(self):
        return RSL_RLSubVecEnv
    
    @property
    def trainer_launcher(self):
        return launch_rsl_rl_training
```

## Training Integration

### RSL-RL Integration

The framework makes RSL-RL think it's training single-agent RL:

```python
# In training script
def launch_rsl_rl_training(subvecenv, training_config):
    # Reset to get initial observations
    subvecenv.reset()
    
    # Create OnPolicyRunner
    runner = OnPolicyRunner(
        env=subvecenv,
        train_cfg=training_config.runner_cfg,
        log_dir=log_dir,
        device=device
    )
    
    # Train
    runner.learn(
        num_learning_iterations=max_iterations,
        init_at_random_ep_len=True
    )
```

### MAPPO Integration

MAPPO expects multi-agent observations in specific format:

```python
class MAPPOSubVecEnv(SubVecEnv):
    def _process_data(self, data, is_reset=False):
        # Convert to MAPPO format:
        # obs: [n_envs, n_agents, obs_dim]
        # rewards: [n_envs, n_agents, 1]
        # dones: [n_envs, n_agents]
        # infos: List[List[dict]]
```

## Simulation Interface

### Abstract Interface Design

The framework abstracts simulator-specific details:

```python
class SceneInterface(ABC):
    """Manages simulation world"""
    
    @abstractmethod
    def add_robot(self, urdf_path, pos, quat, **kwargs) -> RobotInterface:
        pass
    
    @abstractmethod
    def add_sphere(self, name, radius, pos, **kwargs) -> PrimitiveInterface:
        pass
    
    @abstractmethod
    def step(self, n_steps: int = 1):
        pass

class EntityInterface(ABC):
    """Base for all simulation entities"""
    
    @abstractmethod
    def get_pos(self, env_indices=None) -> torch.Tensor:
        pass
    
    @abstractmethod
    def set_pos(self, pos, env_indices=None):
        pass

class RobotInterface(EntityInterface):
    """Robot-specific functionality"""
    
    @abstractmethod
    def get_joint_pos(self, joint_indices=None) -> torch.Tensor:
        pass
    
    @abstractmethod
    def control_joint_pos(self, target_pos, joint_indices=None):
        pass
```

### Genesis Implementation

```python
class GenesisSceneInterface(SceneInterface):
    def __init__(self, simulator_config):
        self._scene = gs.Scene(...)
        self._robots = {}
        self._primitives = {}
        
    def add_robot(self, urdf_path, pos, quat, **kwargs):
        entity = self._scene.add_entity(gs.morphs.URDF(urdf_path))
        return GenesisRobotInterface(entity, self._scene)
```

### Camera System

Built-in camera tracking with automatic recording:

```python
class CameraManager:
    """Manages camera tracking and recording"""
    
    def __init__(self, scene, config):
        self.camera = scene.add_camera(...)
        self.tracker = CameraTracker(targets=robots)
        
    def update(self):
        # Update camera position to track robots
        center = self.tracker.get_tracking_center()
        self.camera.set_pose(center + offset)
        
    def start_recording(self, filepath):
        # Begin video recording
        
    def render_frame(self):
        # Capture current frame
```

## Support Systems

### Logging System (`marl_logging.py`)

Hierarchical, instance-based logging:

```python
# Get class-specific logger
logger = get_class_logger("Agent", "agent_50Hz")

# Hierarchical structure: AEC.ClassName.instance_id
# Examples:
# - AEC.main (VectorizedAECEnv)
# - AEC.Agent.agent_50Hz
# - AEC.Training.go2_locomotion
# - AEC.Robot.go2_robot_1
```

**Features**:
- Color-coded log levels
- Instance-specific identification
- Hierarchical organization under AEC root
- Memory leak prevention with cleanup utilities

### Scheduler (`centralized_scheduler.py`)

Manages multi-frequency robot control:

```python
class CentralizedFrequencyScheduler:
    """Determines which agent acts when"""
    
    def __init__(self, agent_frequencies, genesis_frequency):
        # Calculate frame periods for each agent
        self.schedules = self._compute_schedules()
        
    def get_next_agent(self):
        # Return agent that should act next
        return self._schedule_queue.pop()
```

### Coordinator (`coordinator.py`)

Thread/process synchronization:

```python
class ThreadingCoordinator:
    """Multi-state thread coordination"""
    
    def wake(self, states):
        # Wake threads waiting for states
        
    def wait(self, states):
        # Wait for states to finish
        
    def set_finished(self, state):
        # Mark state as complete
```

### RefDict (`utils/ref_dict.py`)

Memory-efficient data sharing:

```python
class RefDict(dict):
    """Dictionary with reference support"""
    
    def add_ref(self, ref_key, target_key):
        # Make ref_key return target_key's value
        
    # Transparent access - references are invisible to users
    obs_dict = RefDict()
    obs_dict["robot1"] = tensor1
    obs_dict.add_ref("robot2", "robot1")  # robot2 shares robot1's data
```

## Usage Examples

### Basic Go2 Locomotion Training

```python
# Create robot config
go2_config = RobotConfig(
    name="go2",
    urdf_path="robots/go2/urdf/go2.urdf",
    frequency=50.0,
    obs_function=compute_go2_obs,
    reward_function=compute_go2_reward,
    DP_kp=20.0,
    DP_kd=0.5
)

# Create training config
training_config = RSL_RLTrainingConfig(
    robot_cfgs=[go2_config],
    training_name="go2_locomotion",
    algorithm_cfg=AlgorithmConfig(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2
    )
)

# Create environment
env = VectorizedAECEnv(
    training_configs=[training_config],
    n_envs=4096,
    render=True,
    enable_camera_tracking=True
)

# Launch training
env.launch_training()
```

### Multi-Robot Shared Training

```python
# Two Go2 robots sharing observations
robot_configs = {
    "go2_1": RobotConfig(name="go2_1", ...),
    "go2_2": RobotConfig(name="go2_2", ...)
}

shared_training = RSL_RLTrainingConfig(
    robot_cfgs=list(robot_configs.values()),
    training_name="go2_shared",
    training_type="SHARED",
    shared_obs_function=compute_shared_obs
)
```

### Custom Observation Function

```python
def compute_custom_obs(env) -> Dict[str, torch.Tensor]:
    """Custom observation computation"""
    obs = {}
    
    for robot_name in env.robot_names:
        robot = env.robots[robot_name]
        
        # Get robot state
        base_vel = robot.get_base_lin_vel()
        joint_pos = robot.get_joint_pos()
        
        # Compute observation
        obs[robot_name] = torch.cat([
            base_vel,
            joint_pos,
            env.commands[robot_name]
        ], dim=-1)
    
    return obs
```

## Development Guidelines

### Code Organization

1. **Use Instance-level Loggers**: Always use `get_class_logger()` for new classes
2. **Avoid Duplication**: Use `@property` to access data from parent classes
3. **Keep GPU Operations**: Use tensors and batch operations
4. **Follow Skeleton-first Approach**: Define signatures and docstrings before implementation

### Adding New Trainer Interface

```python
class NewTrainerSubVecEnv(SubVecEnv):
    # Only override these methods:
    def _process_data(self, data, is_reset=False):
        """Convert AEC format to trainer format"""
        pass
    
    def _process_action(self, action):
        """Convert trainer action to AEC format"""
        pass
    
    # DON'T override reset() or step()
```

### Memory Management

- Use `RefDict` for shared data in joint training
- Pre-allocate buffers based on detected dimensions
- Clean up loggers to prevent memory leaks

### Error Handling

- Minimal try-catch blocks
- Let errors propagate for easier debugging
- Use logging extensively for debugging

## Testing Strategy

### Test Hierarchy

1. **Unit Tests**: Core functions in configs
2. **Component Tests**: Individual agent functionality
3. **Integration Tests**: Agent + TrainingConfig interactions
4. **System Tests**: Full training pipeline

### Test Structure

```python
def test_robot_config_validation():
    """Test robot configuration validation"""
    # Arrange
    config = RobotConfig(...)
    
    # Act
    result = config.validate()
    
    # Assert
    assert result.is_valid
```

### Key Test Areas

- Observation calculation correctness
- Action dispatching accuracy
- Multi-threaded communication
- Training convergence
- Memory efficiency with RefDict

## Environment Setup

Always use the conda environment:
```bash
conda activate genesis
```

## Command Line Interface

Most training scripts support:
```bash
python go2_locomotion_train.py \
    --num_envs 4096 \
    --max_iterations 4000 \
    --seed 1 \
    --enable_camera \
    --camera_interval 100 \
    --camera_duration 30 \
    --camera_res 1280x720 \
    --record_video \
    --video_path training_video.mp4
```

## Future Extensions

### Planned Features

1. **MuJoCo Backend**: Support for MuJoCo physics
2. **Isaac Gym/Lab Backend**: Integration with NVIDIA Isaac
3. **Enhanced Debug Visualization**: Better debug mark management
4. **Primitive Object Support**: First-class support for non-robot entities
5. **Advanced Camera Tracking**: Multi-target tracking with smart framing

### Extension Points

- New simulator backends via `SceneInterface`
- Custom training algorithms via `SubVecEnv`
- Robot behaviors via hooks and functions
- Debugging tools via `DebugMarkManager`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure Genesis environment is activated
2. **GPU Memory**: Reduce `n_envs` if running out of memory
3. **Frequency Mismatch**: Check robot frequencies are compatible
4. **Logging Spam**: Adjust log levels with `set_aec_hierarchy_level()`

### Debug Techniques

1. Enable debug logging: `logger.setLevel("DEBUG")`
2. Use debug marks to visualize robot states
3. Check RefDict references for joint training issues
4. Monitor thread states with coordinator

## Performance Optimization

1. **Batch Operations**: Group similar robots for batch inference
2. **Locomotion Model Caching**: Share models across robots
3. **Buffer Pre-allocation**: Allocate once based on detected dimensions
4. **GPU Kernels**: Ensure Genesis kernel compilation is enabled

This framework provides a powerful foundation for multi-agent reinforcement learning with heterogeneous robots operating at different frequencies. The clean abstractions and extensible design make it suitable for a wide range of research applications.