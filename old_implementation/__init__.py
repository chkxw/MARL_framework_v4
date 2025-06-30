"""Genesis MARL VecEnv Framework V3.

A vectorized multi-agent reinforcement learning framework built on Genesis
physics engine and compatible with rsl_rl training library.

Key features:
- Simplified direct tensor sharing architecture
- Bundle-based agent configuration (joint/shared network training)
- Condition variable coordination (no complex communication manager)
- Centralized agent scheduling for vectorized execution
- Soft dead agent handling with NaN observations
- Auto-reset for completed environments
- Multi-frequency robot support
"""

from .robot_config import (
    RobotConfig,
    RobotConfigBundle,
    JointTrainingBundle,
    SharedNetworkBundle,
    create_go2_robot_config,
    create_drone_robot_config
)
from .vectorized_aec_env import VectorizedAECEnv
from .sub_vecenv_wrapper import SubVecEnvWrapper, create_subvecenv_v3
from .agent import Agent
from .joint_training_subvecenv import JointTrainingSubVecEnv
from .shared_network_subvecenv import SharedNetworkSubVecEnv
from .centralized_scheduler import (
    CentralizedFrequencyScheduler,
    find_optimal_genesis_frequency
)

__all__ = [
    # Core classes
    "VectorizedAECEnv",
    "Agent",
    
    # Configuration classes
    "RobotConfig",
    "RobotConfigBundle",
    "JointTrainingBundle", 
    "SharedNetworkBundle",
    
    # SubVecEnv classes
    "SubVecEnvWrapper",
    "JointTrainingSubVecEnv",
    "SharedNetworkSubVecEnv",
    
    # Factory functions
    "create_subvecenv_v3",
    "create_go2_robot_config",
    "create_drone_robot_config",
    
    # Scheduling
    "CentralizedFrequencyScheduler",
    "find_optimal_genesis_frequency",
]

__version__ = "3.0.0"