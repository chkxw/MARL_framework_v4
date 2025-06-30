#!/usr/bin/env python3
"""
Robot Configuration for Genesis MARL Framework V3

This module defines the RobotConfig dataclass, bundle classes for joint/shared training,
and factory functions for creating robot configurations for different robot types.
"""

from typing import Dict, List, Tuple, Callable, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import torch


@dataclass
class RobotConfig:
    """Configuration for a robot in the MARL environment (V3 with functional computation)"""
    name: str
    urdf_path: str
    frequency: float  # Hz
    initial_position: List[float]  # [x, y, z]
    initial_orientation: List[float]  # [x, y, z, w] quaternion
    action_dim: int
    observation_dim: int
    control_mode: str  # "position", "velocity", "torque"
    joint_names: List[str] = field(default_factory=list)  # Empty means auto-detect
    action_scale: float = 1.0
    reward_config: Dict = None
    
    # V3: Functional observation and reward computation
    obs_function: Optional[Callable] = None  # Function(mother_env, robot_idx) -> obs_tensor
    reward_function: Optional[Callable] = None  # Function(mother_env, robot_idx) -> reward_tensor


def create_go2_robot_config(name: str, position: List[float], frequency: float = 50.0) -> RobotConfig:
    """Create a Go2 robot configuration"""
    return RobotConfig(
        name=name,
        urdf_path="genesis/assets/urdf/go2/urdf/go2.urdf",
        frequency=frequency,
        initial_position=position,
        initial_orientation=[0.0, 0.0, 0.0, 1.0],  # No rotation
        action_dim=18,  # 18 DOF for Go2 (actual robot DOF)
        observation_dim=50,  # Rich observation for quadruped
        control_mode="position",
        joint_names=None,  # Auto-detect joints from URDF
        action_scale=0.1,  # Small movements for stability
        reward_config={"height_weight": 1.0, "stability_weight": 1.0}
    )


def create_drone_robot_config(name: str, position: List[float], frequency: float = 100.0) -> RobotConfig:
    """Create a drone robot configuration"""
    return RobotConfig(
        name=name,
        urdf_path="urdf/drones/cf2x.urdf",
        frequency=frequency,
        initial_position=position,
        initial_orientation=[0.0, 0.0, 0.0, 1.0],
        action_dim=4,  # [thrust, roll, pitch, yaw]
        observation_dim=12,  # [pos, quat, vel, ang_vel]
        control_mode="velocity",
        action_scale=1.0,
        reward_config={"altitude_weight": 1.0, "stability_weight": 0.5}
    )


# =============================================================================
# Bundle System
# =============================================================================

class RobotConfigBundle:
    """Base class for robot configuration bundles (V3).
    
    Bundles group robots with the same frequency for coordinated training.
    """
    
    def __init__(self, bundle_name: str, frequency: float):
        """Initialize bundle.
        
        Args:
            bundle_name: Name of the bundle
            frequency: Control frequency for all robots in bundle (must be same)
        """
        self.bundle_name = bundle_name
        self.frequency = frequency
        self.robot_configs: List[RobotConfig] = []
    
    @abstractmethod
    def get_action_dim(self) -> int:
        """Get total action dimension for this bundle."""
        pass
    
    @abstractmethod
    def get_observation_dim(self) -> int:
        """Get total observation dimension for this bundle."""
        pass
    
    @abstractmethod
    def get_n_envs_multiplier(self) -> int:
        """Get environment multiplier (1 for joint, num_robots for shared)."""
        pass
    
    def add_robot(self, config: RobotConfig):
        """Add a robot to this bundle."""
        if config.frequency != self.frequency:
            raise ValueError(f"Robot {config.name} frequency {config.frequency} != bundle frequency {self.frequency}")
        self.robot_configs.append(config)
    
    def get_robot_configs(self) -> List[RobotConfig]:
        """Get all robot configs in this bundle."""
        return self.robot_configs.copy()


class JointTrainingBundle(RobotConfigBundle):
    """Bundle for joint training - robots share concatenated observations/actions.
    
    In joint training:
    - Observations from all robots are concatenated per environment
    - Actions are concatenated and applied to all robots
    - Single policy controls all robots as one entity
    - n_envs stays the same, but obs/action dims increase
    """
    
    def get_action_dim(self) -> int:
        """Total action dimension = sum of all robot action dims."""
        return sum(config.action_dim for config in self.robot_configs)
    
    def get_observation_dim(self) -> int:
        """Total observation dimension = sum of all robot observation dims."""
        return sum(config.observation_dim for config in self.robot_configs)
    
    def get_n_envs_multiplier(self) -> int:
        """Environment multiplier = 1 (same number of environments)."""
        return 1
    
    def split_action(self, joint_action: torch.Tensor) -> List[torch.Tensor]:
        """Split joint action tensor into individual robot actions.
        
        Args:
            joint_action: Tensor of shape [n_envs, total_action_dim]
            
        Returns:
            List of action tensors for each robot
        """
        actions = []
        start_idx = 0
        for config in self.robot_configs:
            end_idx = start_idx + config.action_dim
            robot_action = joint_action[:, start_idx:end_idx]
            actions.append(robot_action)
            start_idx = end_idx
        return actions
    
    def concatenate_observations(self, obs_list: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate observations from all robots.
        
        Args:
            obs_list: List of observation tensors [n_envs, obs_dim] for each robot
            
        Returns:
            Concatenated observation tensor [n_envs, total_obs_dim]
        """
        return torch.cat(obs_list, dim=-1)


class SharedNetworkBundle(RobotConfigBundle):
    """Bundle for shared network training - same network for multiple robot instances.
    
    In shared network training:
    - Multiple robots of the same type use the same policy network
    - Each robot appears as a separate environment to the policy
    - n_envs gets multiplied by number of robots
    - obs/action dims stay the same as individual robot
    """
    
    def __init__(self, bundle_name: str, base_config: RobotConfig, 
                 positions: List[List[float]], orientations: Optional[List[List[float]]] = None):
        """Initialize shared network bundle.
        
        Args:
            bundle_name: Name of the bundle
            base_config: Base robot configuration to replicate
            positions: List of initial positions for each robot instance
            orientations: List of initial orientations (uses base_config orientation if None)
        """
        super().__init__(bundle_name, base_config.frequency)
        
        if orientations is None:
            orientations = [base_config.initial_orientation] * len(positions)
        
        if len(positions) != len(orientations):
            raise ValueError("Number of positions must match number of orientations")
        
        # Create individual robot configs with different names and positions
        for i, (pos, orn) in enumerate(zip(positions, orientations)):
            robot_config = RobotConfig(
                name=f"{base_config.name}_{i}",
                urdf_path=base_config.urdf_path,
                frequency=base_config.frequency,
                initial_position=pos,
                initial_orientation=orn,
                action_dim=base_config.action_dim,
                observation_dim=base_config.observation_dim,
                control_mode=base_config.control_mode,
                joint_names=base_config.joint_names.copy(),
                action_scale=base_config.action_scale,
                reward_config=base_config.reward_config.copy() if base_config.reward_config else None,
                obs_function=base_config.obs_function,
                reward_function=base_config.reward_function
            )
            self.add_robot(robot_config)
    
    def get_action_dim(self) -> int:
        """Action dimension = individual robot action dim (same for all)."""
        return self.robot_configs[0].action_dim if self.robot_configs else 0
    
    def get_observation_dim(self) -> int:
        """Observation dimension = individual robot obs dim (same for all)."""
        return self.robot_configs[0].observation_dim if self.robot_configs else 0
    
    def get_n_envs_multiplier(self) -> int:
        """Environment multiplier = number of robots (each robot = separate env)."""
        return len(self.robot_configs)
    
    def reshape_to_shared_format(self, robot_tensors: List[torch.Tensor]) -> torch.Tensor:
        """Reshape individual robot tensors to shared network format.
        
        Args:
            robot_tensors: List of tensors [n_envs, dim] for each robot
            
        Returns:
            Reshaped tensor [n_robots * n_envs, dim]
        """
        return torch.cat(robot_tensors, dim=0)
    
    def reshape_from_shared_format(self, shared_tensor: torch.Tensor, n_envs: int) -> List[torch.Tensor]:
        """Reshape shared network tensor back to individual robot format.
        
        Args:
            shared_tensor: Tensor [n_robots * n_envs, dim]
            n_envs: Number of environments per robot
            
        Returns:
            List of tensors [n_envs, dim] for each robot
        """
        n_robots = len(self.robot_configs)
        return shared_tensor.view(n_robots, n_envs, -1).unbind(0)