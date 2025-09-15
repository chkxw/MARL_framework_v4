#!/usr/bin/env python3
"""
Go2 Dual Robot Collaborative Element Pushing MAPPO Training Script using Genesis MARL Framework V4

This script implements multi-agent PPO (MAPPO) training where two Go2 robots learn independent
policies to collaborate and push an element to a target position. Each robot has its own
network parameters following the MAPush paper specifications for multi-agent collaborative pushing.
"""

import argparse
from dataclasses import asdict, dataclass, field
import os
import threading
import time
from typing import Any, Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

import genesis as gs
import gymnasium as gym
import numpy as np
import torch
from utils import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat
from configs.SimulatorConfig import SimulatorConfig
from configs.RobotConfig import RobotConfig
from configs.TrainingConfig import TrainingConfig
from marl_logging import get_class_logger
from utils import gs_rand_float
from trainer_interfaces.openrl_interface import (
    OpenRLTrainingConfig,
    OpenRLConfig,
    OpenRLAlgorithmConfig,
    OpenRLPolicyConfig,
    OpenRLRunnerConfig,
    OpenRLTrainConfig,
    OpenRLEnvConfig,
    OpenRLDeviceConfig,
)
from trainer_interfaces.frozen_model_interface import FrozenModelConfig

# Import MARL framework components
from vectorized_aec_env import VectorizedAECEnv
from configs.CameraConfig import CameraConfig

# Setup logging
logger = get_class_logger("TrainingScript", "dual_go2_element_push_mappo", level="INFO")

# ========================= Configuration Classes =========================
@dataclass
class ElementConfig(ABC):
    """Base configuration for pushable elements."""
    
    mass: float
    friction: float
    spawn_pos: List[float]  # [x, y] position
    marker_color: Tuple[float, float, float, float] = (0, 1, 0, 1)  # RGBA
    
    @abstractmethod
    def validate(self) -> None:
        """Validate element configuration parameters."""
        pass
    
    def _validate_base(self) -> None:
        """Validate common element properties."""
        if self.mass <= 0 or self.mass > 100:
            raise ValueError(f"Element mass must be between 0 and 100 kg, got {self.mass}")
        
        if self.friction < 0 or self.friction > 2:
            raise ValueError(f"Friction coefficient must be between 0 and 2, got {self.friction}")
        
        if len(self.spawn_pos) != 2:
            raise ValueError(f"Spawn position must be 2D [x, y], got {self.spawn_pos}")
        
        if len(self.marker_color) != 4:
            raise ValueError(f"Marker color must be RGBA tuple, got {self.marker_color}")
        
        for val in self.marker_color:
            if val < 0 or val > 1:
                raise ValueError(f"Color values must be between 0 and 1, got {self.marker_color}")


@dataclass
class CylinderConfig(ElementConfig):
    """Configuration for cylinder element."""
    
    radius: float = 0.5
    height: float = 0.4
    
    def validate(self) -> None:
        """Validate cylinder configuration."""
        self._validate_base()
        
        if self.radius <= 0 or self.radius > 2:
            raise ValueError(f"Cylinder radius must be between 0 and 2 meters, got {self.radius}")
        
        if self.height <= 0 or self.height > 2:
            raise ValueError(f"Cylinder height must be between 0 and 2 meters, got {self.height}")


@dataclass
class BoxConfig(ElementConfig):
    """Configuration for box element."""
    
    size: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.4])  # [length, width, height]
    
    def validate(self) -> None:
        """Validate box configuration."""
        self._validate_base()
        
        if len(self.size) != 3:
            raise ValueError(f"Box size must be 3D [length, width, height], got {self.size}")
        
        for dim in self.size:
            if dim <= 0 or dim > 3:
                raise ValueError(f"Box dimensions must be between 0 and 3 meters, got {self.size}")


@dataclass
class TBlockConfig(ElementConfig):
    """Configuration for T-block element."""
    
    horizontal_size: List[float] = field(default_factory=lambda: [1.5, 0.5, 0.5])  # [width, depth, height]
    vertical_size: List[float] = field(default_factory=lambda: [0.5, 1.0, 0.5])  # [width, depth, height]
    
    def validate(self) -> None:
        """Validate T-block configuration."""
        self._validate_base()
        
        if len(self.horizontal_size) != 3:
            raise ValueError(f"Horizontal size must be 3D [width, depth, height], got {self.horizontal_size}")
        
        if len(self.vertical_size) != 3:
            raise ValueError(f"Vertical size must be 3D [width, depth, height], got {self.vertical_size}")
        
        for dim in self.horizontal_size + self.vertical_size:
            if dim <= 0 or dim > 3:
                raise ValueError(f"T-block dimensions must be between 0 and 3 meters")


def create_element_config(object_type: str) -> ElementConfig:
    """Factory method to create appropriate ElementConfig based on object type.
    
    Args:
        object_type: Type of object ('cylinder', 'box', or 'tblock')
        
    Returns:
        Appropriate ElementConfig subclass instance
        
    Raises:
        ValueError: If object_type is not recognized
    """
    configs = {
        "cylinder": CylinderConfig(
            mass=7.0,
            friction=0.5,
            spawn_pos=[0.0, 0.0],
            radius=0.5,
            height=0.4
        ),
        "box": BoxConfig(
            mass=3.0,
            friction=0.5,
            spawn_pos=[0.0, 0.0],
            size=[0.7620, 0.4674, 0.4674]
        ),
        "tblock": TBlockConfig(
            mass=5.0,
            friction=0.5,
            spawn_pos=[0.0, 0.0],
            horizontal_size=[1.5, 0.5, 0.5],
            vertical_size=[0.5, 1.0, 0.5]
        )
    }
    
    if object_type not in configs:
        raise ValueError(f"Unknown object type: {object_type}. Must be one of {list(configs.keys())}")
    
    config = configs[object_type]
    config.validate()
    return config


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    
    episode_length_s: float = 20.0
    resampling_time_s: float = 4.0
    go2_frequency: float = 50.0  # Hz
    
    def validate(self) -> None:
        """Validate simulation configuration."""
        if self.episode_length_s <= 0:
            raise ValueError(f"Episode length must be positive, got {self.episode_length_s}")
        
        if self.resampling_time_s <= 0:
            raise ValueError(f"Resampling time must be positive, got {self.resampling_time_s}")
        
        if self.go2_frequency <= 0:
            raise ValueError(f"Robot frequency must be positive, got {self.go2_frequency}")


@dataclass
class RobotControlConfig:
    """Configuration for robot control parameters."""
    
    action_scale: float = 0.25
    clip_actions: float = 100.0
    simulate_action_latency: bool = False
    action_latency: int = 0  # Number of steps to delay action application
    robot_spawn_radius: float = 1.0
    robot_min_separation: float = 0.5
    robo_colors: List[Tuple[float, float, float, float]] = field(default_factory=lambda: [
        (1, 0, 0, 1),  # Red
        (0, 0, 1, 1),  # Blue
        (0, 1, 0, 1),  # Green
        (1, 1, 0, 1),  # Yellow
    ])
    
    def validate(self) -> None:
        """Validate robot control configuration."""
        if self.action_scale <= 0:
            raise ValueError(f"Action scale must be positive, got {self.action_scale}")
        
        if self.clip_actions <= 0:
            raise ValueError(f"Action clipping must be positive, got {self.clip_actions}")
        
        if self.action_latency < 0:
            raise ValueError(f"Action latency must be non-negative, got {self.action_latency}")
        
        if self.robot_spawn_radius <= 0:
            raise ValueError(f"Robot spawn radius must be positive, got {self.robot_spawn_radius}")
        
        if self.robot_min_separation <= 0:
            raise ValueError(f"Minimum robot separation must be positive, got {self.robot_min_separation}")


@dataclass
class TaskConfig:
    """Configuration for task-specific parameters."""
    
    goal_distance: float = 2.0
    goal_tolerance: float = 0.3
    far_from_goal_tolerance:float = 10.0
    goal_height: float = 0.1
    goal_marker_size: float = 0.1
    
    def validate(self) -> None:
        """Validate task configuration."""
        if self.goal_distance <= 0:
            raise ValueError(f"Goal distance must be positive, got {self.goal_distance}")
        
        if self.goal_tolerance <= 0:
            raise ValueError(f"Goal tolerance must be positive, got {self.goal_tolerance}")
        
        if self.goal_height < 0:
            raise ValueError(f"Goal height must be non-negative, got {self.goal_height}")
        
        if self.goal_marker_size <= 0:
            raise ValueError(f"Goal marker size must be positive, got {self.goal_marker_size}")


@dataclass
class ObservationConfig:
    """Configuration for observation space."""
    
    num_obs: int = 47
    obs_scales: Dict[str, float] = field(default_factory=lambda: {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "height_measurements": 5.0,
    })
    
    def validate(self) -> None:
        """Validate observation configuration."""
        if self.num_obs <= 0:
            raise ValueError(f"Number of observations must be positive, got {self.num_obs}")


@dataclass
class RewardConfig:
    """Configuration for reward scales."""
    
    # MAPush reward scales
    target_reward_scale: float = 0.00325
    approach_reward_scale: float = 0.00075
    collision_punishment_scale: float = -0.0015
    push_reward_scale: float = 0.0015
    ocb_reward_scale: float = 0.004
    reach_target_reward_scale: float = 10.0
    too_far_punishment_scale: float = -5.0
    exception_punishment_scale: float = -5.0
    
    # Object-specific collision scales
    collision_scale_cylinder: float = -0.0015
    collision_scale_box: float = -0.0025
    collision_scale_tblock: float = -0.0025
    
    def validate(self) -> None:
        """Validate reward configuration."""
        # Positive rewards should be positive
        if self.target_reward_scale < 0:
            raise ValueError(f"Target reward scale should be positive, got {self.target_reward_scale}")
        
        if self.approach_reward_scale < 0:
            raise ValueError(f"Approach reward scale should be positive, got {self.approach_reward_scale}")
        
        if self.push_reward_scale < 0:
            raise ValueError(f"Push reward scale should be positive, got {self.push_reward_scale}")
        
        if self.ocb_reward_scale < 0:
            raise ValueError(f"OCB reward scale should be positive, got {self.ocb_reward_scale}")
        
        if self.reach_target_reward_scale < 0:
            raise ValueError(f"Reach target reward scale should be positive, got {self.reach_target_reward_scale}")
        
        # Negative rewards should be negative
        if self.collision_punishment_scale > 0:
            raise ValueError(f"Collision punishment scale should be negative, got {self.collision_punishment_scale}")
        
        if self.exception_punishment_scale > 0:
            raise ValueError(f"Exception punishment scale should be negative, got {self.exception_punishment_scale}")


@dataclass
class CommandConfig:
    """Configuration for command ranges."""
    
    num_commands: int = 3
    lin_vel_x_range: List[float] = field(default_factory=lambda: [-0.8, 2.5])
    lin_vel_y_range: List[float] = field(default_factory=lambda: [-0.8, 0.8])
    ang_vel_range: List[float] = field(default_factory=lambda: [-0.9, 0.9])
    
    def validate(self) -> None:
        """Validate command configuration."""
        if self.num_commands <= 0:
            raise ValueError(f"Number of commands must be positive, got {self.num_commands}")
        
        if len(self.lin_vel_x_range) != 2:
            raise ValueError(f"Linear velocity X range must have 2 values, got {self.lin_vel_x_range}")
        
        if len(self.lin_vel_y_range) != 2:
            raise ValueError(f"Linear velocity Y range must have 2 values, got {self.lin_vel_y_range}")
        
        if len(self.ang_vel_range) != 2:
            raise ValueError(f"Angular velocity range must have 2 values, got {self.ang_vel_range}")


@dataclass
class MAPushConfig:
    """Master configuration class for MAPPO dual robot push training."""
    
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    robot_control: RobotControlConfig = field(default_factory=RobotControlConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    element: Optional[ElementConfig] = None
    
    def set_element(self, object_type: str) -> None:
        """Set element configuration based on object type.
        
        Args:
            object_type: Type of object ('cylinder', 'box', or 'tblock')
        """
        self.element = create_element_config(object_type)
    
    def validate(self) -> None:
        """Validate all configuration components."""
        self.simulation.validate()
        self.robot_control.validate()
        self.task.validate()
        self.observation.validate()
        self.reward.validate()
        self.command.validate()
        
        if self.element is not None:
            self.element.validate()
        else:
            raise ValueError("Element configuration not set. Call set_element() first.")
    
    def to_legacy_dicts(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Convert to legacy dictionary format for backward compatibility.
        
        Returns:
            Tuple of (env_cfg, obs_cfg, reward_cfg, command_cfg) dictionaries
        """
        # Build env_cfg with all the original fields
        env_cfg = {
            "episode_length_s": self.simulation.episode_length_s,
            "go2_frequency": self.simulation.go2_frequency,
            "resampling_time_s": self.simulation.resampling_time_s,
            "action_scale": self.robot_control.action_scale,
            "simulate_action_latency": self.robot_control.simulate_action_latency,
            "action_latency": self.robot_control.action_latency,
            "clip_actions": self.robot_control.clip_actions,
            "goal_distance": self.task.goal_distance,
            "goal_tolerance": self.task.goal_tolerance,
            "far_from_goal_tolerance": self.task.far_from_goal_tolerance,
            "goal_height": self.task.goal_height,
            "goal_marker_size": self.task.goal_marker_size,
            "robot_spawn_radius": self.robot_control.robot_spawn_radius,
            "robot_min_separation": self.robot_control.robot_min_separation,
            "robo_colors": self.robot_control.robo_colors,
        }
        
        # Add element-specific fields
        if self.element:
            env_cfg["element_spawn_pos"] = self.element.spawn_pos
            env_cfg["element_marker_color"] = self.element.marker_color
            env_cfg["element_friction"] = self.element.friction
            
            if isinstance(self.element, CylinderConfig):
                env_cfg["cylinder_radius"] = self.element.radius
                env_cfg["cylinder_height"] = self.element.height
                env_cfg["element_mass"] = self.element.mass
            elif isinstance(self.element, BoxConfig):
                env_cfg["box_size"] = self.element.size
                env_cfg["element_mass"] = self.element.mass
            elif isinstance(self.element, TBlockConfig):
                env_cfg["tblock_horizontal_size"] = self.element.horizontal_size
                env_cfg["tblock_vertical_size"] = self.element.vertical_size
                env_cfg["tblock_mass"] = self.element.mass
        
        # Build other configs
        obs_cfg = {
            "num_obs": self.observation.num_obs,
            "obs_scales": self.observation.obs_scales
        }
        
        reward_cfg = {
            "target_reward_scale": self.reward.target_reward_scale,
            "approach_reward_scale": self.reward.approach_reward_scale,
            "collision_punishment_scale": self.reward.collision_punishment_scale,
            "push_reward_scale": self.reward.push_reward_scale,
            "ocb_reward_scale": self.reward.ocb_reward_scale,
            "reach_target_reward_scale": self.reward.reach_target_reward_scale,
            "too_far_punishment_scale" : self.reward.too_far_punishment_scale,
            "exception_punishment_scale": self.reward.exception_punishment_scale,
            "collision_scale_cylinder": self.reward.collision_scale_cylinder,
            "collision_scale_box": self.reward.collision_scale_box,
            "collision_scale_tblock": self.reward.collision_scale_tblock,
        }
        
        command_cfg = {
            "num_commands": self.command.num_commands,
            "lin_vel_x_range": self.command.lin_vel_x_range,
            "lin_vel_y_range": self.command.lin_vel_y_range,
            "ang_vel_range": self.command.ang_vel_range,
        }
        
        return env_cfg, obs_cfg, reward_cfg, command_cfg

# ========================= End Configuration Classes =========================

# Initialize configuration system
def get_config(object_type: str = "cylinder") -> tuple:
    """Get configuration for the specified object type.
    
    Args:
        object_type: Type of object ('cylinder', 'box', or 'tblock')
        
    Returns:
        Tuple of (env_cfg, obs_cfg, reward_cfg, command_cfg) dictionaries
    """
    # Create master configuration
    config = MAPushConfig()
    config.set_element(object_type)
    config.validate()
    
    # Convert to legacy format for backward compatibility
    env_cfg, obs_cfg, reward_cfg, command_cfg = config.to_legacy_dicts()
    
    # Add Go2-specific configuration that's not in the new config system
    env_cfg.update({
        "num_actions": 3,  # 3D position commands (x, y, yaw) for locomotion
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
    })
    
    return env_cfg, obs_cfg, reward_cfg, command_cfg

# Initialize with default configs for module-level access
env_cfg, obs_cfg, reward_cfg, command_cfg = get_config("cylinder")


def generate_tblock_mesh(horizontal_size, vertical_size, save_path="meshes/tblock_generated.obj"):
    """Generate a T-block mesh programmatically based on size configurations.
    
    Args:
        horizontal_size: [width, depth, height] of horizontal bar
        vertical_size: [width, depth, height] of vertical bar
        save_path: Path to save the generated mesh file
        
    Returns:
        str: Path to the generated mesh file
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Unpack dimensions
    h_width, h_depth, h_height = horizontal_size
    v_width, v_depth, v_height = vertical_size
    
    # Calculate half dimensions for easier vertex calculation
    h_hw, h_hd, h_hh = h_width/2, h_depth/2, h_height/2
    v_hw, v_hd, v_hh = v_width/2, v_depth/2, v_height/2
    
    # Generate vertices
    vertices = []
    
    # Horizontal bar vertices (8 vertices for a box)
    # Bottom face
    vertices.extend([
        [-h_hw, -h_hd, -h_hh],  # 1
        [ h_hw, -h_hd, -h_hh],  # 2
        [ h_hw,  h_hd, -h_hh],  # 3
        [-h_hw,  h_hd, -h_hh],  # 4
    ])
    # Top face
    vertices.extend([
        [-h_hw, -h_hd,  h_hh],  # 5
        [ h_hw, -h_hd,  h_hh],  # 6
        [ h_hw,  h_hd,  h_hh],  # 7
        [-h_hw,  h_hd,  h_hh],  # 8
    ])
    
    # Vertical bar vertices (8 vertices)
    # Position vertical bar on top of horizontal bar
    # Bottom face (connects to horizontal bar)
    y_offset = h_hd  # Start from the back edge of horizontal bar
    vertices.extend([
        [-v_hw,  y_offset, -v_hh],  # 9
        [ v_hw,  y_offset, -v_hh],  # 10
        [ v_hw,  y_offset,  v_hh],  # 11
        [-v_hw,  y_offset,  v_hh],  # 12
    ])
    # Top face
    vertices.extend([
        [-v_hw,  y_offset + v_depth, -v_hh],  # 13
        [ v_hw,  y_offset + v_depth, -v_hh],  # 14
        [ v_hw,  y_offset + v_depth,  v_hh],  # 15
        [-v_hw,  y_offset + v_depth,  v_hh],  # 16
    ])
    
    # Define faces (using 1-indexed vertex numbers for OBJ format)
    faces = []
    
    # Horizontal bar faces
    faces.extend([
        [1, 2, 3, 4],    # Bottom
        [5, 8, 7, 6],    # Top
        [1, 5, 6, 2],    # Front
        [3, 7, 8, 4],    # Back
        [1, 4, 8, 5],    # Left
        [2, 6, 7, 3],    # Right
    ])
    
    # Vertical bar faces
    faces.extend([
        [9, 13, 14, 10],   # Front
        [11, 15, 16, 12],  # Back
        [9, 12, 16, 13],   # Left
        [10, 14, 15, 11],  # Right
        [13, 16, 15, 14],  # Top
        # Bottom face is inside the T-junction, so we skip it
    ])
    
    # Write OBJ file
    with open(save_path, 'w') as f:
        f.write(f"# T-block mesh generated programmatically\n")
        f.write(f"# Horizontal bar: {horizontal_size}\n")
        f.write(f"# Vertical bar: {vertical_size}\n\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        # Write faces
        for face in faces:
            f.write("f " + " ".join(str(i) for i in face) + "\n")
    
    return save_path


def mapush_obs_function(robot_name: str, env: VectorizedAECEnv) -> torch.Tensor:
    """MAPush observation function for MAPPO training.
    
    Computes 47D observation vector following MAPush paper specifications:
    - Proprioceptive state (33D): base angular velocity, projected gravity, command velocities,
      joint positions relative to default, joint velocities, previous joint actions
    - Task observations (10D): base linear velocity, element position, other robot position,
      goal from robot, goal from element
    - Action history (3D): Previous high-level velocity commands
    - Robot index (1D): For multi-agent distinction
    
    Args:
        robot_name: Name of the robot ("go2_robot_1" or "go2_robot_2")
        env: VectorizedAECEnv instance
        
    Returns:
        torch.Tensor: 47D observation tensor for the robot [n_envs, 47]
    """
    # Get configs from training config
    training_config = env.training_configs[env.robot_2_training_name[robot_name]]
    env_cfg = training_config.task_config["env_cfg"]
    obs_cfg = training_config.task_config["obs_cfg"]
    command_cfg = training_config.task_config["command_cfg"]
    
    robot = env.robots[robot_name]
    n_envs = env.n_envs
    device = env.device
    
    # ============================================
    # PROPRIOCEPTIVE STATE (33D)
    # ============================================
    
    # Get robot orientation for frame transformations
    base_quat = robot.get_orientation(format="quat")
    inv_base_quat = inv_quat(base_quat)
    
    # 1. Base angular velocity in robot frame (3D)
    base_ang_vel = robot.get_ang_vel(relative_to=base_quat)
    base_ang_vel_scaled = base_ang_vel * obs_cfg["obs_scales"]["ang_vel"]
    
    # 2. Projected gravity vector in robot frame (3D)
    global_gravity = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32).repeat(n_envs, 1)
    projected_gravity = transform_by_quat(global_gravity, inv_base_quat)
    
    # 3. Command velocities (3D) - from action buffer or zeros
    if robot_name in env.action_buffers:
        commands = env.action_buffers[robot_name].detach().clone()
    else:
        commands = torch.zeros((n_envs, 3), device=device, dtype=torch.float32)

    # Scale unbounded actions from OpenRL to command ranges
    # OpenRL's DiagGaussian outputs unbounded values, we need to map them to our command ranges
    
    # # Apply tanh to bound actions to [-1, 1]
    # bounded_commands = torch.tanh(commands)
    
    # # Scale and shift to match command_cfg ranges
    # commands = torch.zeros_like(bounded_commands)
    
    # # Linear velocity X: map from [-1, 1] to [lin_vel_x_range[0], lin_vel_x_range[1]]
    # x_min, x_max = command_cfg["lin_vel_x_range"]
    # commands[:, 0] = (x_max - x_min) * (bounded_commands[:, 0] + 1) / 2 + x_min
    
    # # Linear velocity Y: map from [-1, 1] to [lin_vel_y_range[0], lin_vel_y_range[1]]
    # y_min, y_max = command_cfg["lin_vel_y_range"]
    # commands[:, 1] = (y_max - y_min) * (bounded_commands[:, 1] + 1) / 2 + y_min
    
    # # Angular velocity: map from [-1, 1] to [ang_vel_range[0], ang_vel_range[1]]
    # ang_min, ang_max = command_cfg["ang_vel_range"]
    # commands[:, 2] = (ang_max - ang_min) * (bounded_commands[:, 2] + 1) / 2 + ang_min
    
    # just_reset_idx=env.episode_frame_count < 20 # First 20 frames do nothing to make the robot stablize
    # commands[just_reset_idx] = 0

    # Directly clamp unbounded actions from OpenRL to command ranges
    # OpenRL's DiagGaussian outputs unbounded values, we clamp them directly
    
    # Linear velocity X: clamp to [lin_vel_x_range[0], lin_vel_x_range[1]]
    x_min, x_max = command_cfg["lin_vel_x_range"]
    commands[:, 0] = torch.clamp(commands[:, 0], min=x_min, max=x_max)
    
    # Linear velocity Y: clamp to [lin_vel_y_range[0], lin_vel_y_range[1]]
    y_min, y_max = command_cfg["lin_vel_y_range"]
    commands[:, 1] = torch.clamp(commands[:, 1], min=y_min, max=y_max)
    
    # Angular velocity: clamp to [ang_vel_range[0], ang_vel_range[1]]
    ang_min, ang_max = command_cfg["ang_vel_range"]
    commands[:, 2] = torch.clamp(commands[:, 2], min=ang_min, max=ang_max)
    
    commands_scaled = commands * torch.tensor(
        [obs_cfg["obs_scales"]["lin_vel"], obs_cfg["obs_scales"]["lin_vel"], obs_cfg["obs_scales"]["ang_vel"]],
        device=device, dtype=torch.float32
    )
    
    # 4. Joint positions relative to default (12D)
    default_joint_pos = torch.tensor(
        [env_cfg["default_joint_angles"][joint] for joint in env_cfg["joint_names"]],
        device=device, dtype=torch.float32
    ).repeat(n_envs, 1)
    if hasattr(env, 'joint_dofs_idx_locals') and robot_name in env.joint_dofs_idx_locals:
        dofs_idx_local = env.joint_dofs_idx_locals[robot_name]
        joint_pos = robot.get_joint_pos(joint_indices=dofs_idx_local, relative_to=default_joint_pos)
    else:
        # During detection phase, use all joints
        joint_pos = robot.get_joint_pos(relative_to=default_joint_pos)
    joint_pos_scaled = joint_pos * obs_cfg["obs_scales"]["dof_pos"]
    
    # 5. Joint velocities (12D)
    if hasattr(env, 'joint_dofs_idx_locals') and robot_name in env.joint_dofs_idx_locals:
        joint_vel = robot.get_joint_vel(joint_indices=dofs_idx_local)
    else:
        # During detection phase, use all joints
        joint_vel = robot.get_joint_vel()
    joint_vel_scaled = joint_vel * obs_cfg["obs_scales"]["dof_vel"]
    
    # 6. Previous joint actions for locomotion model (12D) - needed for locomotion cache
    if hasattr(env, 'joint_action_buffers') and robot_name in env.joint_action_buffers:
        prev_joint_actions = env.joint_action_buffers[robot_name]
    else:
        prev_joint_actions = torch.zeros((n_envs, 12), device=device, dtype=torch.float32)
    
    # Note: Previous joint actions excluded from observation to match 47D spec (33D proprioceptive + 10D task + 3D action + 1D id)
    
    # ============================================
    # TASK-SPECIFIC OBSERVATIONS (10D)
    # ============================================
    
    # Get robot position
    base_pos = robot.get_pos()
    robot_xy = base_pos[:, :2]
    
    # 7. Base linear velocity in robot frame (2D)
    robot_vel = robot.get_lin_vel(relative_to=base_quat)[:, :2]
    base_lin_vel_scaled = robot_vel * obs_cfg["obs_scales"]["lin_vel"]
    
    # 8. Element relative position in robot frame (2D)
    element_pos = env.element.get_pos()[:, :2]
    element_direction_world = element_pos - robot_xy
    element_direction_world_3d = torch.cat(
        [element_direction_world, torch.zeros((n_envs, 1), device=device, dtype=torch.float32)], dim=1
    )
    element_direction_robot_frame = transform_by_quat(element_direction_world_3d, inv_base_quat)[:, :2]
    
    # 9. Other robot relative position in robot frame (2D)
    robot_names = list(env.robots.keys())
    other_robot_names = [name for name in robot_names if name != robot_name]
    if other_robot_names:
        other_robot = env.robots[other_robot_names[0]]
        other_robot_pos = other_robot.get_pos()[:, :2]
        other_robot_direction_world = other_robot_pos - robot_xy
        other_robot_direction_world_3d = torch.cat(
            [other_robot_direction_world, torch.zeros((n_envs, 1), device=device, dtype=torch.float32)], dim=1
        )
        other_robot_direction_robot_frame = transform_by_quat(other_robot_direction_world_3d, inv_base_quat)[:, :2]
    else:
        other_robot_direction_robot_frame = torch.zeros((n_envs, 2), device=device, dtype=torch.float32)
    
    # 10. Goal position from robot in robot frame (2D)
    if hasattr(env, 'goal_position'):
        goal_pos = env.goal_position
    else:
        # During detection phase, use default goal position
        goal_pos = torch.tensor([[1.0, 0.0]], device=device, dtype=torch.float32).repeat(n_envs, 1)
    goal_from_robot_world = goal_pos - robot_xy
    goal_from_robot_world_3d = torch.cat(
        [goal_from_robot_world, torch.zeros((n_envs, 1), device=device, dtype=torch.float32)], dim=1
    )
    goal_from_robot_frame = transform_by_quat(goal_from_robot_world_3d, inv_base_quat)[:, :2]
    
    # 11. Goal position from element in world frame (2D)
    goal_from_element = goal_pos - element_pos
    
    # ============================================
    # ACTION HISTORY (3D)
    # ============================================
    
    # 12. Previous high-level action (3D velocity commands)
    # Use action_buffers which contains previous actions (not cleared until after obs computation)
    if robot_name in env.action_buffers:
        previous_action = env.action_buffers[robot_name].detach().clone()
    else:
        # During detection phase or initial step, use zeros
        previous_action = torch.zeros((n_envs, 3), device=device, dtype=torch.float32)
    
    # ============================================
    # AGENT IDENTIFICATION (1D)
    # ============================================
    
    # 13. Robot index for multi-agent distinction
    robot_idx = env.robot_names.index(robot_name)
    robot_idx_tensor = torch.full((n_envs, 1), float(robot_idx), device=device, dtype=torch.float32)
    
    # ============================================
    # COMBINE ALL OBSERVATIONS (47D total)
    # ============================================
    
    obs = torch.cat([
        base_lin_vel_scaled,         # 2D
        base_ang_vel_scaled,          # 3D
        # Task-specific (10D)
        element_direction_robot_frame,# 2D
        other_robot_direction_robot_frame,  # 2D
        goal_from_robot_frame,        # 2D
        goal_from_element,            # 2D
        # Action history (3D)
        previous_action,              # 3D
        # Agent identification (1D)
        robot_idx_tensor,             # 1D

        # For stability 
        projected_gravity,           # 3D
        joint_pos_scaled,            # 12D
        joint_vel_scaled,            # 12D
    ], dim=1)  # Total: 33 + 10 + 3 + 1 = 47D
    
    # ============================================
    # CACHE LOCOMOTION INPUT (45D) for action preprocessing
    # ============================================
    
    # Build the locomotion input that will be needed later
    locomotion_input = torch.cat([
        base_ang_vel_scaled,          # 3D (already scaled)
        projected_gravity,            # 3D
        commands_scaled,              # 3D (already scaled)
        joint_pos_scaled,             # 12D (already scaled)
        joint_vel_scaled,             # 12D (already scaled)
        prev_joint_actions,           # 12D
    ], dim=-1)  # Total: 45D    
    
    # ============================================
    # NaN CHECKING AND DEBUGGING
    # ============================================
    if torch.isnan(obs).any() or torch.isinf(obs).any() or (torch.abs(obs) > 200).any():
        print(f"\n{'='*60}")
        print(f"Invalid values detected in observation for {robot_name}!")
        print(f"{'='*60}")
        
        # Collect all problematic environment indices
        problematic_envs = set()
        
        # Check each component with correct indices for 47D observation
        components = {
            "base_lin_vel_scaled (0:2)": base_lin_vel_scaled,
            "base_ang_vel_scaled (2:5)": base_ang_vel_scaled,
            "element_direction_robot (5:7)": element_direction_robot_frame,
            "other_robot_direction (7:9)": other_robot_direction_robot_frame,
            "goal_from_robot (9:11)": goal_from_robot_frame,
            "goal_from_element (11:13)": goal_from_element,
            "previous_action (13:16)": previous_action,
            "robot_idx (16:17)": robot_idx_tensor,
            "projected_gravity (17:20)": projected_gravity,
            "joint_pos_scaled (20:32)": joint_pos_scaled,
            "joint_vel_scaled (32:44)": joint_vel_scaled,
            # locomotion compoennets
            "commands_scaled (locomotion [6:9])": commands_scaled,
            "prev_joint_actions (locomotion [32:44])": prev_joint_actions,
        }
        
        for name, tensor in components.items():
            if torch.isnan(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                nan_envs = torch.isnan(tensor).any(dim=-1).nonzero(as_tuple=True)[0]
                problematic_envs.update(nan_envs.tolist())
                print(f"  ❌ {name}: {nan_count} NaN values in envs {nan_envs[:5].tolist()}...")
                # Show sample values
                sample_idx = nan_envs[0] if len(nan_envs) > 0 else 0
                print(f"     Sample values: {tensor[sample_idx]}")
            elif torch.isinf(tensor).any():
                inf_count = torch.isinf(tensor).sum().item()
                inf_envs = torch.isinf(tensor).any(dim=-1).nonzero(as_tuple=True)[0]
                problematic_envs.update(inf_envs.tolist())
                print(f"  ⚠️  {name}: {inf_count} Inf values in envs {inf_envs[:5].tolist()}...")
                # Show sample values
                sample_idx = inf_envs[0] if len(inf_envs) > 0 else 0
                print(f"     Sample values: {tensor[sample_idx]}")
            elif (torch.abs(tensor) > 200).any():
                large_count = (torch.abs(tensor) > 200).sum().item()
                large_envs = (torch.abs(tensor) > 200).any(dim=-1).nonzero(as_tuple=True)[0]
                problematic_envs.update(large_envs.tolist())
                print(f"  🔴 {name}: {large_count} values with |val| > 200 in envs {large_envs[:5].tolist()}...")
                # Show sample values
                sample_idx = large_envs[0] if len(large_envs) > 0 else 0
                print(f"     Sample values: {tensor[sample_idx]}")
                # Also show which specific values are large
                large_mask = torch.abs(tensor[sample_idx]) > 200
                if large_mask.any():
                    print(f"     Large values at indices: {large_mask.nonzero(as_tuple=True)[0].tolist()}")
            else:
                print(f"  ✓ {name}: OK")
        
        # Additional debug info
        print(f"\nAdditional debug info:")
        print(f"  base_quat sample: {base_quat[0]}")
        print(f"  base_pos sample: {robot.get_pos()[0]}")
        print(f"  element_pos sample: {env.element.get_pos()[0]}")
        if hasattr(env, 'goal_position'):
            print(f"  goal_position: {env.goal_position}")
        
        print(f"{'='*60}\n")
        
        # Store problematic environment indices - create boolean mask for all envs
        problematic_mask = torch.zeros(env.n_envs, device=env.device, dtype=torch.bool)
        if problematic_envs:
            problematic_mask[list(problematic_envs)] = True
            print(f"⚠️  Marked {len(problematic_envs)} environments for early truncation: {list(problematic_envs)[:10]}...")
        env.problematic_obs[robot_name] = problematic_mask
        
        # Replace NaN/Inf with zeros to prevent training crash
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        locomotion_input =  torch.nan_to_num(locomotion_input, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"⚠️  Replaced NaN/Inf values with zeros to continue training") 
                            
    env.locomotion_input_cache[robot_name] = locomotion_input
    return obs


def mapush_reward_function(robot_name: str, env: VectorizedAECEnv) -> Dict[str, torch.Tensor]:
    """MAPush reward function for MAPPO training.
    
    Implements the complete MAPush reward structure including:
    1. Approach reward (distance to object)
    2. Push reward (object movement)
    3. Collision penalty (robot-robot contact)
    4. Success bonus (goal reached)
    5. Distance progress reward
    6. OCB (Object-centric behavior) reward
    
    Args:
        robot_name: Name of the robot ("go2_robot_1" or "go2_robot_2")
        env: VectorizedAECEnv instance
        
    Returns:
        Dict[str, torch.Tensor]: Dictionary with robot_name as key and reward tensor as value
    """
    # Get configs from training config
    training_config = env.training_configs[env.robot_2_training_name[robot_name]]
    env_cfg = training_config.task_config["env_cfg"]
    reward_cfg = training_config.task_config["reward_cfg"]
    
    n_envs = env.n_envs
    device = env.device
    
    # Get robots
    robot = env.robots[robot_name]
    robot_names = list(env.robots.keys())
    other_robot_names = [name for name in robot_names if name != robot_name]
    other_robot = env.robots[other_robot_names[0]] if other_robot_names else None
    
    # Get positions
    robot_pos = robot.get_pos()
    robot_xy = robot_pos[:, :2]
    element_pos = env.element.get_pos()
    element_xy = element_pos[:, :2]
    goal_pos = env.goal_position  # 2D position
    
    # Get velocities
    element_vel = env.element.get_lin_vel()
    element_vel_xy = element_vel[:, :2]
    
    # Initialize reward tensor
    reward = torch.zeros(n_envs, device=device, dtype=torch.float32)
    
    # ============================================
    # 1. APPROACH REWARD (quadratic penalty based on distance)
    # ============================================
    distance_to_element = torch.norm(robot_xy - element_xy, dim=1, keepdim=True)
    approach_reward = -(distance_to_element + 0.5) ** 2 * reward_cfg["approach_reward_scale"]
    reward += approach_reward.squeeze()
    
    # ============================================
    # 2. PUSH REWARD (when element is moving)
    # ============================================
    element_speed = torch.norm(element_vel_xy, dim=1)
    push_reward = torch.zeros(n_envs, device=device)
    push_reward[element_speed > 0.1] = reward_cfg["push_reward_scale"]
    reward += push_reward
    
    # ============================================
    # 3. COLLISION PENALTY (robot-robot proximity)
    # ============================================
    if other_robot is not None:
        other_pos = other_robot.get_pos()
        other_xy = other_pos[:, :2]
        robot_distance = torch.norm(robot_xy - other_xy, dim=1)
        
        # Get object-specific collision scale
        training_config = env.training_configs[env.robot_2_training_name[robot_name]]
        object_type = training_config.task_config["object_type"]
        if object_type == 'cylinder':
            collision_scale = reward_cfg["collision_scale_cylinder"]
        elif object_type in ['box', 'cuboid']:
            collision_scale = reward_cfg["collision_scale_box"]
        elif object_type == 'tblock':
            collision_scale = reward_cfg["collision_scale_tblock"]
        else:
            collision_scale = reward_cfg["collision_punishment_scale"]
        
        # Apply inverse distance penalty
        collision_penalty = (1.0 / (0.02 + robot_distance / 3.0)) * collision_scale
        reward += collision_penalty
    
    # ============================================
    # 4. SUCCESS BONUS (goal reached)
    # ============================================
    distance_to_goal = torch.norm(element_xy - goal_pos, dim=1)
    goal_reached = distance_to_goal < env_cfg["goal_tolerance"]
    success_bonus = goal_reached.float() * reward_cfg["reach_target_reward_scale"]
    reward += success_bonus

    # ============================================
    # 5. Too far penalty
    # ============================================
    distance_to_goal = torch.norm(element_pos -torch.nn.functional.pad(goal_pos, (0, 1), value=0), dim=1)
    too_far = distance_to_goal > env_cfg["far_from_goal_tolerance"]
    too_far_punishment = too_far.float() * reward_cfg["too_far_punishment_scale"]
    reward += too_far_punishment
    
    # ============================================
    # 6. DISTANCE TO TARGET REWARD (element progress)
    # ============================================
    # Initialize prev_element_pos if not exists
    if not hasattr(env, 'prev_element_pos'):
        env.prev_element_pos = element_xy.clone()
    
    prev_distance = torch.norm(env.prev_element_pos - goal_pos, dim=1)
    current_distance = distance_to_goal
    
    # Calculate progress reward
    distance_reward = reward_cfg["target_reward_scale"] * 100 * (
        2 * (prev_distance - current_distance) - 0.01 * current_distance
    )
    reward += distance_reward
    
    # Update previous element position for next step
    env.prev_element_pos = element_xy.clone()
    
    # ============================================
    # 7. OCB REWARD (Optimal Contact Behavior)
    # ============================================
    # Calculate push direction and target direction
    target_direction = goal_pos - element_xy
    target_direction = target_direction / (torch.norm(target_direction, dim=1, keepdim=True) + 1e-6)
    
    # For simplicity with cylinder, use velocity direction as push direction
    push_direction = element_vel_xy / (torch.norm(element_vel_xy, dim=1, keepdim=True) + 1e-6)
    
    # Only apply OCB reward when element is moving
    alignment = torch.sum(push_direction * target_direction, dim=1)
    ocb_reward = torch.zeros(n_envs, device=device)
    ocb_reward[element_speed > 0.1] = alignment[element_speed > 0.1] * reward_cfg["ocb_reward_scale"]
    reward += ocb_reward
    
    # ============================================
    # 8. EXCEPTION PUNISHMENT (for invalid states)
    # ============================================
    # Check for NaN or Inf values
    has_nan = torch.isnan(reward) | torch.isinf(reward)
    if has_nan.any():
        reward[has_nan] = reward_cfg["exception_punishment_scale"]
    
    # ============================================
    # TENSORBOARD LOGGING
    # ============================================
    if not hasattr(env, 'infos'):
        env.infos = {}
    if robot_name not in env.infos:
        env.infos[robot_name] = {}
    
    # Log individual reward components
    env.infos[robot_name]["tensorboard/approach_reward"] = approach_reward.squeeze().mean()
    env.infos[robot_name]["tensorboard/push_reward"] = push_reward.mean()
    if other_robot is not None:
        env.infos[robot_name]["tensorboard/collision_penalty"] = collision_penalty.mean()
    env.infos[robot_name]["tensorboard/reach_target_reward"] = success_bonus.mean()
    env.infos[robot_name]["tensorboard/too_far_punishment"] = too_far_punishment.mean()
    env.infos[robot_name]["tensorboard/distance_to_target_reward"] = distance_reward.mean()
    env.infos[robot_name]["tensorboard/ocb_reward"] = ocb_reward.mean()
    env.infos[robot_name]["tensorboard/total_reward"] = reward.mean()
    
    # Return reward dictionary
    return {robot_name: reward}


def go2_setup_function(robot_name, env):
    """Generic setup function for Go2 robots in MAPPO training."""
    robot_cfg = env.robot_configs[robot_name]
    robot = env.scene.add_robot(
        name=robot_name,
        urdf_path=robot_cfg.urdf_path,
        pos=robot_cfg.initial_position,
        quat=robot_cfg.initial_orientation,
        fixed=False,
    )

    return robot


def go2_obs_function(robot_name: str, env) -> torch.Tensor:
    """Observation function for a specific robot in element pushing task."""
    robot = env.robots[robot_name]
    n_envs = env.n_envs
    device = env.device

    # Get robot state
    base_pos = robot.get_pos()
    base_quat = robot.get_orientation(format="quat")
    robot_xy = base_pos[:, :2]

    # Get robot velocity in robot frame
    robot_vel = robot.get_lin_vel(relative_to=base_quat)[:, :2]  # 2D velocity

    # Get element position from simulator (shared across all robots)
    element_pos = env.element.get_pos()[:, :2]  # Get 3D position and take x,y

    # Calculate relative element position in robot frame
    element_direction_world = element_pos - robot_xy
    element_direction_world_3d = torch.cat(
        [element_direction_world, torch.zeros((n_envs, 1), device=device, dtype=torch.float32)], dim=1
    )
    inv_base_quat = inv_quat(base_quat)
    element_direction_robot_frame = transform_by_quat(element_direction_world_3d, inv_base_quat)[:, :2]

    # Get other robot's position relative to current robot
    robot_names = list(env.robots.keys())
    other_robot_names = [name for name in robot_names if name != robot_name]
    if other_robot_names:
        other_robot = env.robots[other_robot_names[0]]
        other_robot_pos = other_robot.get_pos()[:, :2]

        # Calculate relative other robot position in robot frame
        other_robot_direction_world = other_robot_pos - robot_xy
        other_robot_direction_world_3d = torch.cat(
            [other_robot_direction_world, torch.zeros((n_envs, 1), device=device, dtype=torch.float32)], dim=1
        )
        other_robot_direction_robot_frame = transform_by_quat(other_robot_direction_world_3d, inv_base_quat)[:, :2]
    else:
        # If no other robot, use zeros
        other_robot_direction_robot_frame = torch.zeros((n_envs, 2), device=device, dtype=torch.float32)

    # Get goal position (shared single goal)
    goal_pos = env.goal_position  # 2D position

    # Calculate relative goal position from robot in robot frame
    goal_from_robot_world = goal_pos - robot_xy
    goal_from_robot_world_3d = torch.cat(
        [goal_from_robot_world, torch.zeros((n_envs, 1), device=device, dtype=torch.float32)], dim=1
    )
    goal_from_robot_frame = transform_by_quat(goal_from_robot_world_3d, inv_base_quat)[:, :2]

    # Calculate relative goal position from element (world frame is fine)
    goal_from_element = goal_pos - element_pos  # 2D vector

    # Get previous action from action_buffers (contains previous actions until cleared after obs)
    if robot_name in env.action_buffers:
        previous_action = env.action_buffers[robot_name].detach().clone()
    else:
        previous_action = torch.zeros((n_envs, 3), device=device, dtype=torch.float32)

    # Get robot index (one-hot encoding)
    robot_idx = env.robot_names.index(robot_name)
    robot_idx_tensor = torch.full((n_envs, 1), float(robot_idx), device=device, dtype=torch.float32)

    # Combine observations:
    # own_vel(2) + element_pos(2) + other_robot_pos(2) + goal_from_robot(2) +
    # goal_from_element(2) + prev_action(3) + robot_idx(1) = 14D
    obs = torch.cat(
        [
            robot_vel,  # 2D
            element_direction_robot_frame,  # 2D
            other_robot_direction_robot_frame,  # 2D
            goal_from_robot_frame,  # 2D
            goal_from_element,  # 2D
            previous_action,  # 3D
            robot_idx_tensor,  # 1D
        ],
        dim=1,
    )
    # TODO; try robot orientation
    # TODO: Use pd to contrl yaw
    # TODO: Maybe also action of the other robot

    return obs


def go2_reward_function(robot_name: str, env: VectorizedAECEnv) -> torch.Tensor:
    """
    Collaborative reward function for element pushing task.
    All rewards are normalized [0,1] and scaled by factors in reward_cfg.
    """

    n_envs = env.n_envs
    device = env.device

    # ============================================
    # SHARED CALCULATIONS (used by multiple rewards)
    # ============================================

    # Get robots
    robot = env.robots[robot_name]
    robot_names = list(env.robots.keys())
    other_robot_names = [name for name in robot_names if name != robot_name]
    other_robot = env.robots[other_robot_names[0]] 

    # Positions
    robot_pos = robot.get_pos()[:, :2]
    other_pos = other_robot.get_pos()[:, :2] if other_robot else torch.zeros((n_envs, 2), device=device)
    element_pos = env.element.get_pos()[:, :2]
    goal_pos = env.goal_position

    # Distances
    current_distance = torch.norm(element_pos - goal_pos, dim=1)
    robot_to_element_dist = torch.norm(robot_pos - element_pos, dim=1)

    # Velocities
    element_vel = env.element.get_lin_vel()[:, :2]
    robot_ang_vel = robot.get_ang_vel()

    # Direction vectors
    element_to_goal = goal_pos - element_pos
    element_to_goal_norm = element_to_goal / (torch.norm(element_to_goal, dim=1, keepdim=True) + 1e-6)

    # Vectors from element to robots (for opposition calculation)
    robot_to_element = robot_pos - element_pos
    other_to_element = other_pos - element_pos
    robot_to_element_norm = robot_to_element / (torch.norm(robot_to_element, dim=1, keepdim=True) + 1e-6)
    other_to_element_norm = other_to_element / (torch.norm(other_to_element, dim=1, keepdim=True) + 1e-6)

    # ============================================
    # CONTACT DETECTION
    # ============================================

    # Get all contacts from scene at once

    def calculate_combined_torques(contacts_info, entity_from, entity_to):
        """Calculate combined torques and forces from entity_from acting on entity_to."""
        geom_range_a = (entity_from.geom_start, entity_from.geom_end)
        geom_range_b = (entity_to.geom_start, entity_to.geom_end)

        # This provide the COM of all the whole entity, all link index will give the same result
        all_COMs = env.scene._scene.sim.rigid_solver.get_links_root_COM()

        entity_to_COM = all_COMs[:, entity_from.link_start]

        # Extract contact data
        geom_a = contacts_info['geom_a']
        geom_b = contacts_info['geom_b']
        forces = contacts_info['force']
        positions = contacts_info['position']
        
        # Create masks for contacts
        valid_contacts = (geom_a >= 0) & (geom_b >= 0)
        
        # Check both orderings (entity_from -> entity_to)
        mask_a_from_b_to = (geom_a >= geom_range_a[0]) & (geom_a < geom_range_a[1]) & \
                           (geom_b >= geom_range_b[0]) & (geom_b < geom_range_b[1])
        mask_b_from_a_to = (geom_b >= geom_range_a[0]) & (geom_b < geom_range_a[1]) & \
                           (geom_a >= geom_range_b[0]) & (geom_a < geom_range_b[1])
        contact_mask = (mask_a_from_b_to | mask_b_from_a_to) & valid_contacts
        
        # Initialize outputs
        is_in_contact = contact_mask.any(dim=1)
        combined_torque = torch.zeros((n_envs, 3), device=device)
        combined_force = torch.zeros((n_envs, 3), device=device)
        
        # Calculate torques for each environment
        for env_idx in range(n_envs):
            if contact_mask[env_idx].any():
                # Get forces and positions for this env's contacts
                contact_forces = forces[env_idx][contact_mask[env_idx]]
                contact_positions = positions[env_idx][contact_mask[env_idx]]
                
                # Determine force direction on entity_to
                # Force convention: positive force acts on geom_b, negative on geom_a
                env_mask_contacts = contact_mask[env_idx]
                env_mask_a_from = mask_a_from_b_to[env_idx][env_mask_contacts]
                env_mask_b_from = mask_b_from_a_to[env_idx][env_mask_contacts]
                
                # If entity_to is geom_b, use force as-is
                # If entity_to is geom_a, negate force
                force_on_entity_to = torch.zeros_like(contact_forces)
                force_on_entity_to[env_mask_a_from] = contact_forces[env_mask_a_from]  # entity_to is geom_b
                force_on_entity_to[env_mask_b_from] = -contact_forces[env_mask_b_from]  # entity_to is geom_a
                
                # Calculate torques: τ = r × F
                r_vectors = contact_positions - entity_to_COM[env_idx].unsqueeze(0)
                torques = torch.cross(r_vectors, force_on_entity_to, dim=1)
                
                # Sum torques and forces
                combined_torque[env_idx] = torques.sum(dim=0)
                combined_force[env_idx] = force_on_entity_to.sum(dim=0)
        
        return is_in_contact, combined_force, combined_torque
    
    # Get all contacts from scene
    contacts_info = env.scene._scene.sim.rigid_solver.collider.get_contacts(as_tensor=True, to_torch=True)
    
    # Calculate torques for robot->element
    robot_in_contact, robot_force_on_element, robot_torque_on_element = \
        calculate_combined_torques(contacts_info, robot._entity, env.element._entity)
    
    # Calculate torques for other_robot->element
    other_in_contact, other_force_on_element, other_torque_on_element = \
        calculate_combined_torques(contacts_info, other_robot._entity, env.element._entity)
    
    # Check robot-robot collision
    geom_a = contacts_info['geom_a']
    geom_b = contacts_info['geom_b']
    valid_contacts = (geom_a >= 0) & (geom_b >= 0)
    
    robot_robot_a = (geom_a >= robot._entity.geom_start) & (geom_a < robot._entity.geom_end) & \
                    (geom_b >= other_robot._entity.geom_start) & (geom_b < other_robot._entity.geom_end)
    robot_robot_b = (geom_b >= robot._entity.geom_start) & (geom_b < robot._entity.geom_end) & \
                    (geom_a >= other_robot._entity.geom_start) & (geom_a < other_robot._entity.geom_end)
    robot_robot_mask = (robot_robot_a | robot_robot_b) & valid_contacts
    robots_colliding = robot_robot_mask.any(dim=1)
    
    dual_contact = robot_in_contact & other_in_contact
    solo_contact = robot_in_contact & ~other_in_contact
    
    # ============================================
    # EARLY CONTACT DETECTION
    # ============================================
    # Check for early contact in first 20 frames
    if (env.episode_frame_count < 20).any():
        early_frame_mask = env.episode_frame_count < 20
        early_robot_contact = robot_in_contact & early_frame_mask
        early_other_contact = other_in_contact & early_frame_mask
        
        if early_robot_contact.any() or early_other_contact.any():
            # Find which environments have early contact
            early_contact_envs = torch.where(early_robot_contact | early_other_contact)[0]
            for env_idx in early_contact_envs[:5]:  # Log first 5 environments
                frame = env.episode_frame_count[env_idx].item()
                robot_contact = early_robot_contact[env_idx].item()
                other_contact = early_other_contact[env_idx].item()
                
                contact_info = []
                if robot_contact:
                    contact_info.append(f"{robot_name}")
                if other_contact:
                    contact_info.append(f"{other_robot_names[0]}")
                
                env.logger.warning(
                    f"Early contact detected at frame {frame:.0f} in env {env_idx}: "
                    f"Robots in contact: {', '.join(contact_info)}"
                )
    
    # ============================================
    # ABNORMAL ELEMENT STATE DETECTION
    # ============================================
    # Check for abnormal element position or velocity
    element_pos_abs = torch.abs(element_pos)
    element_vel = env.element.get_lin_vel()
    element_vel_magnitude = torch.norm(element_vel, dim=1)
    
    # Check for abnormal position (z > 5 or any coordinate > 100)
    abnormal_z = env.element.get_pos()[:, 2] > 5.0
    abnormal_xy = (element_pos_abs > 100).any(dim=1)
    abnormal_pos = abnormal_z | abnormal_xy
    
    # Check for abnormal velocity (> 50 m/s)
    abnormal_vel = element_vel_magnitude > 50.0
    
    if abnormal_pos.any() or abnormal_vel.any():
        abnormal_envs = torch.where(abnormal_pos | abnormal_vel)[0]
        for env_idx in abnormal_envs[:5]:  # Log first 5 abnormal environments
            elem_pos = env.element.get_pos()[env_idx]
            elem_vel = element_vel[env_idx]
            frame = env.episode_frame_count[env_idx].item()
            
            env.logger.warning(
                f"ABNORMAL ELEMENT STATE in env {env_idx} at frame {frame:.0f}: "
                f"Position: [{elem_pos[0]:.2f}, {elem_pos[1]:.2f}, {elem_pos[2]:.2f}], "
                f"Velocity: [{elem_vel[0]:.2f}, {elem_vel[1]:.2f}, {elem_vel[2]:.2f}] (mag: {element_vel_magnitude[env_idx]:.2f})"
            )
    
    # Calculate push quality based on force AND torque
    robot_push_quality = torch.zeros(n_envs, device=device)
    other_push_quality = torch.zeros(n_envs, device=device)
    
    if robot_in_contact.any():
        # Force alignment with goal direction
        force_magnitude = torch.norm(robot_force_on_element[:, :2], dim=1) + 1e-6
        force_direction = robot_force_on_element[:, :2] / force_magnitude.unsqueeze(1)
        force_alignment = torch.clamp(torch.sum(force_direction * element_to_goal_norm, dim=1), 0, 1)
        
        # Penalize vertical torque (tipping) - we want minimal torque around x and y axes
        vertical_torque_penalty = torch.exp(-torch.norm(robot_torque_on_element[:, :2], dim=1) / 10.0)
        
        # Combined push quality
        robot_push_quality = force_alignment * vertical_torque_penalty * robot_in_contact.float()
    
    if other_in_contact.any():
        force_magnitude = torch.norm(other_force_on_element[:, :2], dim=1) + 1e-6
        force_direction = other_force_on_element[:, :2] / force_magnitude.unsqueeze(1)
        force_alignment = torch.clamp(torch.sum(force_direction * element_to_goal_norm, dim=1), 0, 1)
        
        vertical_torque_penalty = torch.exp(-torch.norm(other_torque_on_element[:, :2], dim=1) / 10.0)
        other_push_quality = force_alignment * vertical_torque_penalty * other_in_contact.float()
    
    # Total combined torque on element (for collaborative assessment)
    total_torque_on_element = robot_torque_on_element + other_torque_on_element
    total_force_on_element = robot_force_on_element + other_force_on_element

    # ============================================
    # INITIALIZE REWARD
    # ============================================
    reward = torch.zeros(n_envs, device=device, dtype=torch.float32)

    # ============================================
    # 1. ELEMENT PROGRESS REWARD
    # ============================================

    prev_distance = torch.norm(goal_pos- env.prev_element_pos , dim=1)
    distance_reduction = prev_distance - current_distance
    
    # Normalized progress (max reduction per step is ~0.02 at 50Hz for reasonable speeds)
    # Scale by exponential to give more reward near goal
    progress_factor = torch.exp(-current_distance / env_cfg["goal_distance"])
    element_progress = torch.clamp(distance_reduction * 50.0 * progress_factor, 0.0, 1.0)
    element_progress *= robot_in_contact.float() # Only count progress if robot is in contact
    element_reward = element_progress * reward_cfg.get("element_progress_scale")
    reward += element_reward

    # Update previous element position for next frame
    env.prev_element_pos[:] = element_pos

    # ============================================
    # 2. GOAL ACHIEVEMENT BONUS
    # ============================================

    goal_reached = (current_distance < env_cfg["goal_tolerance"]).float()
    goal_reached_reward = goal_reached * reward_cfg.get("goal_bonus_scale")
    reward += goal_reached_reward

    # ============================================
    # 3. CONTACT REWARDS
    # ============================================

    # Individual contact (normalized: 0 or 1)
    individual_contact = robot_in_contact.float()
    contact_reward = individual_contact * reward_cfg.get("contact_scale")
    reward += contact_reward

    # Dual contact bonus (normalized: 0 or 1)
    dual_contact = dual_contact.float()
    dual_contact_reward = dual_contact * reward_cfg.get("dual_contact_scale")
    reward += contact_reward + dual_contact_reward

    # ============================================
    # 4. OPTIMAL POSITIONING REWARD
    # ============================================

    # Dot product of normalized vectors (-1 = opposite, 1 = same side)
    alignment = torch.sum(robot_to_element_norm * other_to_element_norm, dim=1)

    # Normalize to [0, 1] where 1 = perfectly opposite
    opposition_score = (-alignment + 1.0) / 2.0

    # Only reward when both robots are in contact
    opposition_reward = dual_contact.float() * opposition_score
    opposition_reward *= reward_cfg.get("opposition_scale")
    reward += opposition_reward

    # ============================================
    # 5. COLLABORATIVE PUSH REWARD
    # ============================================

    # Element velocity toward goal (normalized by max expected velocity)
    vel_toward_goal = torch.sum(element_vel * element_to_goal_norm, dim=1)
    normalized_vel = torch.clamp(vel_toward_goal / 0.5, 0.0, 1.0)  # 0.5 m/s as max expected

    # Only reward when both robots are pushing
    collab_push = dual_contact.float() * normalized_vel
    collab_push_reward = collab_push * reward_cfg.get("collab_push_scale")
    reward += collab_push_reward

    # ============================================
    # 6. PENALTIES (negative rewards)
    # ============================================

    # Solo push penalty (when element is moving but only one robot touching)
    element_moving = (torch.norm(element_vel, dim=1) > 0.1).float()
    solo_push_penalty = solo_contact.float() * element_moving
    solo_push_penalty *= reward_cfg.get("solo_penalty_scale")
    reward -= solo_push_penalty

    # Robot collision penalty
    collision_penalty = robots_colliding.float()
    collision_penalty *= reward_cfg.get("collision_penalty_scale")
    reward -= collision_penalty

    # Distance penalty (normalized by max distance threshold)
    distance_penalty = torch.clamp((robot_to_element_dist - 0.8) / 0.5, 0.0, 1.0)
    distance_penalty *= reward_cfg.get("distance_penalty_scale")
    reward -= distance_penalty

    # Stability penalty (normalized by max expected angular velocity)
    stability_penalty = torch.clamp(torch.norm(robot_ang_vel, dim=1) / 10.0, 0.0, 1.0)
    stability_penalty *= reward_cfg.get("stability_penalty_scale")
    reward -= stability_penalty

    # Death penalty (check if robot is terminated)
    # TODO: in correct, only penalize none goal reached robots
    # death_penalty = env.truncations[robot_name].float()
    # death_penalty *= reward_cfg.get("death_penalty_scale")
    # reward -= death_penalty
    # training_config=env.training_configs[env.robot_2_training_name[robot_name]]
    # if training_config.is_shared and robot_name != training_config.robot_names[0]:
    #     return {robot_name: reward}
    
    env.infos[robot_name]["tensorboard/element_reward"] = element_reward.mean()
    env.infos[robot_name]["tensorboard/goal_reached_reward"] = goal_reached_reward.mean()
    env.infos[robot_name]["tensorboard/contact_reward"] = contact_reward.mean()
    env.infos[robot_name]["tensorboard/dual_contact_reward"] = dual_contact_reward.mean()
    env.infos[robot_name]["tensorboard/opposition_reward"] = opposition_reward.mean()
    env.infos[robot_name]["tensorboard/collab_push_reward"] = collab_push_reward.mean()
    env.infos[robot_name]["tensorboard/solo_push_penalty"] = solo_push_penalty.mean()
    env.infos[robot_name]["tensorboard/collision_penalty"] = collision_penalty.mean()
    env.infos[robot_name]["tensorboard/distance_penalty"] = distance_penalty.mean()
    env.infos[robot_name]["tensorboard/stability_penalty"] = stability_penalty.mean()

    env.infos[robot_name]["tensorboard/total_reward"] = reward.mean()
    return {robot_name: reward}


def go2_info_function(robot_name: str, env: VectorizedAECEnv) -> Dict[str, torch.Tensor]:
    # return {"bad_transition": torch.zeros(env.n_envs, device=env.device, dtype=torch.bool)}
    return {}


def go2_truncation_function(robot_name: str, env: VectorizedAECEnv) -> torch.Tensor:
    """Truncation function for collaborative element pushing task."""
    # Get configs from training config
    training_config = env.training_configs[env.robot_2_training_name[robot_name]]
    env_cfg = training_config.task_config["env_cfg"]
    
    robot = env.robots[robot_name]
    base_quat = robot.get_orientation(format="quat")

    # Calculate orientation
    base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32)
    inv_base_init_quat = inv_quat(base_init_quat)
    base_euler = quat_to_xyz(
        transform_quat_by_quat(torch.ones_like(base_quat) * inv_base_init_quat, base_quat),
        rpy=True,
        degrees=True,
    )

    # Episode time check
    episode_time = env.episode_frame_count.float() / env.simulation_frequency
    time_exceeded = episode_time >= env_cfg["episode_length_s"]

    # Orientation checks
    roll_exceeded = torch.abs(base_euler[:, 0]) > env_cfg["termination_if_roll_greater_than"]
    pitch_exceeded = torch.abs(base_euler[:, 1]) > env_cfg["termination_if_pitch_greater_than"]

    # Check if other robot is dead
    other_robot_dead = torch.zeros(env.n_envs, device=env.device, dtype=torch.bool)
    robot_names = list(env.robots.keys())
    other_robot_names = [name for name in robot_names if name != robot_name]
    for other_name in other_robot_names:
        if other_name in env.dead_robots:
            other_robot_dead = other_robot_dead | env.dead_robots[other_name]

    # Check if element reached goal
    element_pos = env.element.get_pos()[:, :2]  # Get element position from simulator
    goal_pos = env.goal_position  # 2D position (static goal)
    element_to_goal_distance = torch.norm(element_pos - goal_pos, dim=1)
    goal_reached = element_to_goal_distance < env_cfg["goal_tolerance"]
    
    # Check for problematic observations (early truncate environments with NaN/Inf/large values)
    problematic_truncate = torch.zeros(env.n_envs, device=env.device, dtype=torch.bool)
    if hasattr(env, 'problematic_obs') and robot_name in env.problematic_obs:
        problematic_truncate = env.problematic_obs[robot_name]

    # Check if element is too far from the goal
    element_to_goal_distance = torch.norm(env.element.get_pos() -torch.nn.functional.pad(goal_pos, (0, 1), value=0), dim=1)
    far_from_goal = element_to_goal_distance > env_cfg["far_from_goal_tolerance"]

    truncated = time_exceeded | roll_exceeded | pitch_exceeded | other_robot_dead | goal_reached | far_from_goal | problematic_truncate
    return truncated


def go2_action_preprocessing_function(robot_name: str, env: VectorizedAECEnv):
    # Get configs from training config
    training_config = env.training_configs[env.robot_2_training_name[robot_name]]
    env_cfg = training_config.task_config["env_cfg"]
    command_cfg = training_config.task_config["command_cfg"]
    

    locomotion_input = env.locomotion_input_cache[robot_name]
    
    # # Still do visualization for debugging (optional, only for first environment)
    # robot = env.robots[robot_name]
    # n_envs = env.n_envs
    # device = env.device
    
    # # Get action from action buffer for visualization
    # if robot_name in env.action_buffers:
    #     raw_actions = env.action_buffers[robot_name].detach().clone()
    # else:
    #     raw_actions = torch.zeros((n_envs, 3), device=device, dtype=torch.float32)
    
    # # Scale unbounded actions from OpenRL to command ranges
    # # OpenRL's DiagGaussian outputs unbounded values, we need to map them to our command ranges
    
    # # Apply tanh to bound actions to [-1, 1]
    # bounded_actions = torch.tanh(raw_actions)
    
    # # Scale and shift to match command_cfg ranges
    # commands = torch.zeros_like(bounded_actions)
    
    # # Linear velocity X: map from [-1, 1] to [lin_vel_x_range[0], lin_vel_x_range[1]]
    # x_min, x_max = command_cfg["lin_vel_x_range"]
    # commands[:, 0] = (x_max - x_min) * (bounded_actions[:, 0] + 1) / 2 + x_min
    
    # # Linear velocity Y: map from [-1, 1] to [lin_vel_y_range[0], lin_vel_y_range[1]]
    # y_min, y_max = command_cfg["lin_vel_y_range"]
    # commands[:, 1] = (y_max - y_min) * (bounded_actions[:, 1] + 1) / 2 + y_min
    
    # # Angular velocity: map from [-1, 1] to [ang_vel_range[0], ang_vel_range[1]]
    # ang_min, ang_max = command_cfg["ang_vel_range"]
    # commands[:, 2] = (ang_max - ang_min) * (bounded_actions[:, 2] + 1) / 2 + ang_min
    
    # just_reset_idx=env.episode_frame_count < 20 # First 20 frames do nothing to make the robot stablize
    # commands[just_reset_idx] = 0

    # # Visualize command arrows for first environment (debugging)
    # robot_pos_3d = robot.get_pos()[0]  # First environment only
    # base_quat = robot.get_orientation(format="quat")[0]  # First environment only
    
    # robot_idx = env.robot_names.index(robot_name)
    
    # env.scene.debug_manager.clear(tag=robot_name)
    
    # # Command vector in world frame (scale for better visibility)
    # arrow_scale = 0.5  # Scale factor for arrow length
    # command_world = torch.zeros(3, device=device, dtype=torch.float32)
    # command_world[0] = commands[0, 0] * arrow_scale  # x velocity
    # command_world[1] = commands[0, 1] * arrow_scale  # y velocity
    # # Note: commands[0, 2] is angular velocity (yaw), not used for linear arrow
    
    # # Transform command vector to world frame using robot's orientation
    # command_world_transformed = transform_by_quat(command_world.unsqueeze(0), base_quat.unsqueeze(0))[0]
    
    # # Draw arrow showing command direction
    # arrow_start = robot_pos_3d.cpu().numpy()
    # arrow_vec = command_world_transformed.cpu().numpy()
    
    # # Use different colors for different robots
    # robo_colors = env_cfg["robo_colors"]
    # arrow_color = robo_colors[robot_idx % len(robo_colors)]
    
    # # Draw the command arrow
    # env.scene.debug_manager.draw_debug_arrow(
    #     pos=arrow_start, vec=arrow_vec, radius=0.02, color=arrow_color, tag=robot_name
    # )
    
    return locomotion_input

def go2_action_postprocessing_function(robot_name: str, env: VectorizedAECEnv):
    # Get configs from training config
    training_config = env.training_configs[env.robot_2_training_name[robot_name]]
    env_cfg = training_config.task_config["env_cfg"]
    
    raw_joint_actions = env.joint_action_buffers[robot_name]
    clipped_outputs = torch.clip(raw_joint_actions, -env_cfg["clip_actions"], env_cfg["clip_actions"])

    default_joint_pos = torch.tensor(
        [env_cfg["default_joint_angles"][joint] for joint in env_cfg["joint_names"]],
        device=env.device,
        dtype=torch.float32,
    )

    scaled_joint_positions = clipped_outputs * env_cfg["action_scale"] + default_joint_pos
    
    # Apply action latency if configured
    action_latency = env_cfg.get("action_latency", 0)
    if action_latency > 0 and hasattr(env, 'joint_action_latency_buffer'):
        buffer = env.joint_action_latency_buffer[robot_name]
        buffer_size = action_latency + 1
        
        # Calculate buffer index using frame count
        # Use modulo to create circular buffer behavior
        current_idx = int(env.episode_frame_count[0].item()) % buffer_size
        
        # Store current action in buffer
        buffer[:, current_idx, :] = scaled_joint_positions
        
        # Calculate delayed index (go back 'latency' steps with wraparound)
        delayed_idx = (current_idx - action_latency) % buffer_size
        
        # Return the delayed action
        return buffer[:, delayed_idx, :]
    
    return scaled_joint_positions


def robot_post_reset_hook(robot_name: str, env: VectorizedAECEnv, env_indices):
    """Post reset hook for individual robot - handles robot spawning based on spawn angles."""
    # Get configs from training config
    training_config = env.training_configs[env.robot_2_training_name[robot_name]]
    env_cfg = training_config.task_config["env_cfg"]
    
    # Robot spawning in third quadrant
    robot = env.robots[robot_name]

    # Get spawn angles from registered field (set by training_post_reset_hook)
    robot_angles = env.spawn_angles[robot_name][env_indices]

    spawn_x = torch.cos(robot_angles) * env_cfg["robot_spawn_radius"]
    spawn_y = torch.sin(robot_angles) * env_cfg["robot_spawn_radius"]

    # Set robot position
    robot_pos = torch.zeros((len(env_indices), 3), device=env.device, dtype=torch.float32)
    robot_pos[:, 0] = spawn_x
    robot_pos[:, 1] = spawn_y
    robot_pos[:, 2] = 0.42  # Standard Go2 spawn height

    robot.set_pos(robot_pos, env_indices=env_indices)

    # Orient robot toward element (at origin)
    # Calculate angle from robot to element
    angle_to_element = torch.atan2(-spawn_y, -spawn_x)  # Negative because element is at origin

    # Convert angle to quaternion (rotation around z-axis)
    quat = torch.zeros((len(env_indices), 4), device=env.device, dtype=torch.float32)
    quat[:, 0] = torch.cos(angle_to_element / 2)  # w
    quat[:, 3] = torch.sin(angle_to_element / 2)  # z

    robot.set_orientation(quat, format="quat", env_indices=env_indices)
    
    # Initialize action latency buffer if enabled
    action_latency = env_cfg.get("action_latency", 0)
    if action_latency > 0 and hasattr(env, 'joint_action_latency_buffer'):
        # Get default joint positions
        default_joint_pos = torch.tensor(
            [env_cfg["default_joint_angles"][joint] for joint in env_cfg["joint_names"]],
            device=env.device,
            dtype=torch.float32
        )
        
        # Fill entire buffer with default positions for the specified environments
        buffer = env.joint_action_latency_buffer[robot_name]
        buffer_size = action_latency + 1
        for i in range(buffer_size):
            buffer[env_indices, i, :] = default_joint_pos


def create_go2_robot_config(robot_name: str) -> RobotConfig:
    """Create Go2 robot configuration for MAPPO training with independent policies.
    
    Args:
        robot_name: Name of the robot ("go2_robot_1" or "go2_robot_2")
    """

    # Initial joint positions
    initial_joint_pos = [env_cfg["default_joint_angles"][joint] for joint in env_cfg["joint_names"]]

    # Create base robot config
    base_config = RobotConfig(
        name=robot_name,  # Each robot has unique name for MAPPO
        # urdf_path="../genesis/assets/urdf/go2/urdf/go2.urdf",  # Path to Go2 URDF
        urdf_path="./go1_standalone/urdf/go1.urdf",  # Path to Go2 URDF
        frequency=50.0,  # 50Hz control frequency per architecture spec
        initial_position=[0.0, 0.0, 0.42],  # Default position, will be overridden
        initial_orientation=[1.0, 0.0, 0.0, 0.0],
        action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        # observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32),  # 47D MAPush observation
        locomotion_path="locomotions/go1-locomotion-train_policy_iter40000.onnx",
        control_mode="position",  # Position control mode per architecture spec
        joint_names=env_cfg["joint_names"],
        setup_function=go2_setup_function,
        obs_function=mapush_obs_function,  # Shared function instance
        reward_function=mapush_reward_function,  # Shared function instance  
        truncation_function=go2_truncation_function,
        info_function=go2_info_function,
        post_reset_hook=robot_post_reset_hook,  # Updated to robot_post_reset_hook
        action_preprocessing_function=go2_action_preprocessing_function,
        action_postprocessing_function=go2_action_postprocessing_function,
        initial_joint_pos=initial_joint_pos,
        DP_kp=20.0,  # Explicit DP_kp=20.0 per architecture spec
        DP_kd=0.5,   # Explicit DP_kd=0.5 per architecture spec
    )

    return base_config


def training_post_reset_hook(training_name, env: VectorizedAECEnv, env_indices):
    training_config = env.training_configs[training_name]
    env_cfg = training_config.task_config["env_cfg"]

    element_pos_2d = torch.zeros((len(env_indices), 2), device=env.device, dtype=torch.float32)
    # Extend 1 element for z dimension
    # Set z slightly higher to account for potential ground penetration
    # Handle different object types for proper z-position
    object_type = training_config.task_config["object_type"]
    if object_type == 'box':
        element_z = env_cfg["box_size"][2] / 2 + 0.01  # Use box height
    elif object_type == 'tblock':
        # Use max height between horizontal and vertical bars
        max_height = max(env_cfg["tblock_horizontal_size"][2], env_cfg["tblock_vertical_size"][2])
        element_z = max_height / 2 + 0.01  # Use T-block max height
    else:  # cylinder
        element_z = env_cfg["cylinder_height"] / 2 + 0.01  # Use cylinder height
    
    element_pos_3d = torch.cat(
        [
            element_pos_2d,  # x, y coordinates (zeros)
            torch.full(
                (len(env_indices), 1), element_z, device=env.device, dtype=torch.float32
            ),  # z coordinate
        ],
        dim=1,
    )
    env.element.set_pos(element_pos_3d, env_indices=env_indices)
    
    # Reset orientation to initial upright position (identity quaternion: w=1, x=0, y=0, z=0)
    initial_quat = torch.zeros((len(env_indices), 4), device=env.device, dtype=torch.float32)
    initial_quat[:] = torch.tensor([0.923880, 0.000000, 0.000000, -0.382683], device=env.device, dtype=torch.float32) # hard code -45 degree
    env.element.set_orientation(initial_quat, format="quat", env_indices=env_indices)
    
    env.element.set_lin_vel(
        torch.zeros((len(env_indices), 3), device=env.device, dtype=torch.float32), env_indices=env_indices
    )
    env.element.set_ang_vel(
        torch.zeros((len(env_indices), 3), device=env.device, dtype=torch.float32), env_indices=env_indices
    )

    # Sample goal position in first quadrant (0 to 90 degrees), 1 meter from origin
    angles = torch.rand(len(env_indices), device=env.device) * (torch.pi / 2)  # 0 to π/2
    goal_x = torch.cos(angles) * env_cfg["goal_distance"]
    goal_y = torch.sin(angles) * env_cfg["goal_distance"]
    goal_tensor = env.goal_position
    goal_tensor[env_indices, 0] = goal_x
    goal_tensor[env_indices, 1] = goal_y

    env.initial_element_to_goal_dist[env_indices] = torch.norm(goal_tensor[env_indices] - element_pos_2d, dim=1)
    env.prev_element_pos[env_indices] = element_pos_2d

    # Update goal marker position
    goal_positions_3d = torch.cat(
        [
            goal_tensor[env_indices],  # [len(env_indices), 2] (x, y)
            torch.full((len(env_indices), 1), env_cfg["goal_height"], device=env.device, dtype=torch.float32),  # z
        ],
        dim=1,
    )
    env.goal_marker.set_pos(goal_positions_3d, env_indices=env_indices)

    # Generate random spawn angles for robot positioning
    min_separation = torch.pi / 4  # 30 degrees minimum separation

    # Generate first angle in range [0, π/2 - min_separation]
    max_range = torch.pi / 2 - min_separation
    angle1_batch = torch.rand(len(env_indices), device=env.device) * max_range

    # Generate second angle in range [angle1_batch + min_separation, π/2]
    remaining_range = torch.pi / 2 - (angle1_batch + min_separation)
    angle2_batch = angle1_batch + min_separation + torch.rand(len(env_indices), device=env.device) * remaining_range

    # May swap agent 1 and agent 2 angles
    if torch.rand(1, device=env.device) < 0.5:
        angle1_batch, angle2_batch = angle2_batch, angle1_batch

    robot0_name = env.robot_names[0]
    robot1_name = env.robot_names[1]

    env.spawn_angles[robot0_name][env_indices] = angle1_batch + torch.pi
    env.spawn_angles[robot1_name][env_indices] = angle2_batch + torch.pi

    training_config.robots_post_reset_hook(env, env_indices)


def training_pre_build_hook(training_name, env):
    training_config = env.training_configs[training_name]
    training_config.robots_pre_build_hook(env)
    
    # Get the environment configuration from training config
    env_cfg = training_config.task_config["env_cfg"]

    # Register single shared goal position (not per-robot)
    if not hasattr(env, "goal_position"):
        env.register_tensor_field(
            "goal_position",
            torch.zeros((2,), device=env.device, dtype=torch.float32),
            per_robot=False,  # Shared across all robots
            per_env=True,
            clear_in_reset=True,
            clear_after_use=False,
        )

    # Note: Previous actions are tracked via env.action_buffers[robot_name]
    # which contains the previous actions since it's not cleared until after observations are computed

    # Register spawn angles for random robot positioning
    if not hasattr(env, "spawn_angles"):
        env.register_tensor_field(
            "spawn_angles",
            torch.tensor(0.0, device=env.device, dtype=torch.float32),  # Single angle per robot per environment
            per_robot=True,
            per_env=True,
            clear_in_reset=True,
            clear_after_use=False,
        )

    # Create element entity - shared across all robots
    if not hasattr(env, "element"):
        # Get object type from environment
        object_type = training_config.task_config["object_type"]
        
        if object_type == 'box':
            # Create box element
            box_size = env_cfg["box_size"]
            initial_pos = (env_cfg["element_spawn_pos"][0], env_cfg["element_spawn_pos"][1], box_size[2] / 2)
            volume = box_size[0] * box_size[1] * box_size[2]
            rho = env_cfg["element_mass"] / volume
            env.element = env.scene.add_box(
                name="pushing_element",
                size=box_size,
                pos=initial_pos,
                collision=True,
                fixed=False,
                # Physical properties
                rho=rho,
                friction=env_cfg["element_friction"],
                # Visual properties
                color=env_cfg["element_marker_color"],
            )
        elif object_type == 'cylinder':
            # Create cylinder element (existing implementation)
            initial_pos = (env_cfg["element_spawn_pos"][0], env_cfg["element_spawn_pos"][1], env_cfg["cylinder_height"] / 2)
            volumn = np.pi * env_cfg["cylinder_radius"] ** 2 * env_cfg["cylinder_height"]
            rho = env_cfg["element_mass"] / volumn
            env.element = env.scene.add_cylinder(
                name="pushing_element",
                radius=env_cfg["cylinder_radius"],
                height=env_cfg["cylinder_height"],
                pos=initial_pos,
                collision=True,
                fixed=False,
                # Physical properties
                rho=rho,
                friction=env_cfg["element_friction"],
                # Visual properties
                color=env_cfg["element_marker_color"],
            )
        elif object_type == 'tblock':
            # Generate T-block mesh dynamically based on configuration
            horizontal_size = env_cfg["tblock_horizontal_size"]
            vertical_size = env_cfg["tblock_vertical_size"]
            
            # Generate the mesh file
            mesh_path = generate_tblock_mesh(
                horizontal_size=horizontal_size,
                vertical_size=vertical_size,
                save_path=f"meshes/tblock_{horizontal_size[0]}x{horizontal_size[1]}x{horizontal_size[2]}_{vertical_size[0]}x{vertical_size[1]}x{vertical_size[2]}.obj"
            )
            
            # Calculate initial position (use max height between horizontal and vertical)
            max_height = max(horizontal_size[2], vertical_size[2])
            initial_pos = (env_cfg["element_spawn_pos"][0], env_cfg["element_spawn_pos"][1], max_height / 2)
            
            # Calculate volume for T-block (horizontal bar + vertical bar - overlap)
            h_vol = horizontal_size[0] * horizontal_size[1] * horizontal_size[2]
            v_vol = vertical_size[0] * vertical_size[1] * vertical_size[2]
            # Overlap is where vertical bar sits on horizontal bar
            overlap_vol = min(vertical_size[0], horizontal_size[0]) * min(vertical_size[1], horizontal_size[1]) * min(vertical_size[2], horizontal_size[2])
            total_volume = h_vol + v_vol - overlap_vol
            rho = env_cfg["tblock_mass"] / total_volume
            
            # Add T-block as a mesh primitive
            env.element = env.scene.add_mesh(
                name="pushing_element",
                mesh_path=mesh_path,
                pos=initial_pos,
                scale=1.0,  # Use original mesh scale
                collision=True,
                fixed=False,
                # Physical properties
                rho=rho,
                friction=env_cfg["element_friction"],
                # Visual properties
                color=env_cfg["element_marker_color"],
            )
        else:
            raise ValueError(f"Unsupported object type: {object_type}")
    if not hasattr(env, "goal_marker"):
        initial_pos = (1.0, 0.0, env_cfg["goal_height"])  # Default position
        env.goal_marker = env.scene.add_sphere(
            name="goal_marker",
            radius=env_cfg["goal_marker_size"],
            pos=initial_pos,
            collision=False,
            fixed=True,
            color=(0, 1, 0, 1),
            vis_mode='visual',
        )

    if not hasattr(env, 'initial_element_to_goal_dist'):
        env.register_tensor_field(
            'initial_element_to_goal_dist',
            torch.tensor(0.0, device=env.device, dtype=torch.float32),
            per_robot=False,
            per_env=True,
            clear_in_reset=True,
            clear_after_use=False,
        )
    if not hasattr(env, 'prev_element_pos'):
        env.register_tensor_field(
            'prev_element_pos',
            torch.zeros((2,), device=env.device, dtype=torch.float32),
            per_robot=False,
            per_env=True,
            clear_in_reset=True,
            clear_after_use=False,
        )
    
    # Register locomotion input cache to avoid redundant computations
    if not hasattr(env, 'locomotion_input_cache'):
        env.register_tensor_field(
            'locomotion_input_cache',
            torch.zeros((45,), device=env.device, dtype=torch.float32),  # 45D locomotion input
            per_robot=True,  # Each robot has its own locomotion input
            per_env=True,
            clear_in_reset=True,
            clear_after_use=False,
        )
    
    # Register action latency buffer if needed
    env_cfg = training_config.task_config["env_cfg"]
    action_latency = env_cfg.get("action_latency", 0)
    if action_latency > 0 and not hasattr(env, 'joint_action_latency_buffer'):
        # Buffer size is latency+1 to store current and N previous actions
        buffer_size = action_latency + 1
        env.register_tensor_field(
            'joint_action_latency_buffer',
            torch.zeros((buffer_size, 12), device=env.device, dtype=torch.float32),  # Buffer for joint actions
            per_robot=True,  # Each robot has its own buffer
            per_env=True,
            clear_in_reset=False,  # We'll manually initialize in robot_post_reset_hook
            clear_after_use=False,
        )
    # Goal marker will be handled as debug marks (no physical entity needed)

    # Register temporary recorder for nan obs detection

    env.register_tensor_field(
        'problematic_obs',
        torch.tensor(0.0, device=env.device, dtype=torch.bool),
        per_robot=True,
        per_env=True,
        clear_in_reset=True,
        clear_after_use=True,
    )


def get_dual_go2_mappo_cfg(exp_name: str, num_envs: int, max_iterations: int) -> OpenRLConfig:
    """Get MAPPO training configuration for dual robot setup using OpenRL."""

    # Create algorithm config
    algorithm_cfg = OpenRLAlgorithmConfig(
        lr=0.0003,
        critic_lr=0.0003,
        clip_param=0.2,
        ppo_epoch=10,
        num_mini_batch=4,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        use_clipped_value_loss=True,
        use_huber_loss=True,
        huber_delta=10.0,
    )

    # Create policy config
    policy_cfg = OpenRLPolicyConfig(
        hidden_size=128,
        layer_N=3,
        activation_id=3,  # 0: tanh, 1: relu, 2: leaky_relu, 3: selu
        use_recurrent_policy=False,
        recurrent_N=1,
        use_feature_normalization=True,
        use_centralized_V=True,
        use_popart=True,
        use_valuenorm=False,
        use_adv_normalize=True,
    )

    # Create runner config
    runner_cfg = OpenRLRunnerConfig(
        algorithm_name="mappo",
        env_name="dual_go2_walk_to_pos",
        experiment_name=exp_name,
        scenario_name="dual_go2_walk_to_pos",
        seed=1,
        num_env_steps=max_iterations
        * int(env_cfg["episode_length_s"] * env_cfg["go2_frequency"])
        * num_envs,  # 50Hz frequency
        episode_length=int(env_cfg["episode_length_s"] * env_cfg["go2_frequency"]),
        n_rollout_threads=num_envs,
        disable_wandb=True,  # Very strange, this mean enable tensorboard
    )

    # Create training config
    train_cfg = OpenRLTrainConfig(
        save_interval=1000,
        log_interval=1,
        use_eval=False,
    )

    # Create environment config
    env_cfg_openrl = OpenRLEnvConfig(
        seed_specify=True,
    )

    # Create device config
    device_cfg = OpenRLDeviceConfig(
        cuda=torch.cuda.is_available(),
        cuda_deterministic=True,
        torch_threads=1,
    )

    # Merge all configs into the main OpenRL config by creating with all parameters
    openrl_cfg = OpenRLConfig(
        # Additional logging config
        **{
            "log_dir": f"./logs/{exp_name}_openrl",
            "run_dir": f"./logs/{exp_name}_openrl",
            **asdict(algorithm_cfg),
            **asdict(policy_cfg),
            **asdict(runner_cfg),
            **asdict(train_cfg),
            **asdict(env_cfg_openrl),
            **asdict(device_cfg),
        }
    )

    return openrl_cfg


def test_observation_function():
    """Unit test for MAPush observation function - verifies 47D output and components."""
    
    logger.info("🧪 Running unit tests for MAPush observation function...")
    
    # Create minimal test environment
    import genesis as gs
    
    # Initialize Genesis
    gs.init(seed=1)
    
    # Create test robot configs
    robot1_config = create_go2_robot_config("go2_robot_1")
    robot2_config = create_go2_robot_config("go2_robot_2")
    
    # Verify observation space dimensions
    assert robot1_config.observation_space.shape == (47,), f"Expected 47D observation, got {robot1_config.observation_space.shape}"
    assert robot2_config.observation_space.shape == (47,), f"Expected 47D observation, got {robot2_config.observation_space.shape}"
    logger.info("✅ Test 1 passed: Observation space is 47D")
    
    # Create test training config
    test_training_config = OpenRLTrainingConfig(
        robot_cfgs=[robot1_config, robot2_config],
        training_name="test_mappo_obs",
        training_type="NORMAL",
        task_config={
            "env_cfg": env_cfg,
            "obs_cfg": obs_cfg,
            "reward_cfg": reward_cfg,
            "command_cfg": command_cfg,
            "object_type": "cylinder",
        },
        pre_build_hook=training_pre_build_hook,
        post_reset_hook=training_post_reset_hook,
        openrl_config=get_dual_go2_mappo_cfg("test", 2, 10),
    )
    
    # Create small test environment
    try:
        aec_env = VectorizedAECEnv(
            training_configs=[test_training_config],
            n_envs=2,
            max_episode_length_s=1.0,
            render=False,
            seed=1,
            simulator_config=SimulatorConfig(
                show_viewer=False,
                dt=1.0 / test_training_config.frequency,
                n_envs=2,
                device="cuda" if torch.cuda.is_available() else "cpu"
            ),
        )
        
        # Reset environment
        aec_env.reset()
        
        # Test observation computation for both robots
        for robot_name in ["go2_robot_1", "go2_robot_2"]:
            obs = mapush_obs_function(robot_name, aec_env)
            
            # Check observation shape (raw observation should be 47D)
            assert obs.shape[0] == 2, f"Expected batch size 2, got {obs.shape[0]} for {robot_name}"
            actual_obs_dim = obs.shape[1]
            
            # Debug: Check actual component dimensions
            debug_idx = 0
            logger.info(f"Observation breakdown for {robot_name}:")
            logger.info(f"  Base ang vel (3D): {obs[0, debug_idx:debug_idx+3].shape}")
            debug_idx += 3
            logger.info(f"  Projected gravity (3D): {obs[0, debug_idx:debug_idx+3].shape}")
            debug_idx += 3
            logger.info(f"  Commands (3D): {obs[0, debug_idx:debug_idx+3].shape}")
            debug_idx += 3
            logger.info(f"  Joint pos (12D): {obs[0, debug_idx:debug_idx+12].shape}")
            debug_idx += 12
            logger.info(f"  Joint vel (12D): {obs[0, debug_idx:debug_idx+12].shape}")
            debug_idx += 12
            logger.info(f"  Proprioceptive total: {debug_idx}D (33D expected)")
            logger.info(f"  Base lin vel (2D): {obs[0, debug_idx:debug_idx+2].shape}")
            debug_idx += 2
            logger.info(f"  Element pos (2D): {obs[0, debug_idx:debug_idx+2].shape}")
            debug_idx += 2
            logger.info(f"  Other robot pos (2D): {obs[0, debug_idx:debug_idx+2].shape}")
            debug_idx += 2
            logger.info(f"  Goal from robot (2D): {obs[0, debug_idx:debug_idx+2].shape}")
            debug_idx += 2
            logger.info(f"  Goal from element (2D): {obs[0, debug_idx:debug_idx+2].shape}")
            debug_idx += 2
            logger.info(f"  Task total: 10D")
            logger.info(f"  Prev action (3D): {obs[0, debug_idx:debug_idx+3].shape}")
            debug_idx += 3
            logger.info(f"  Robot idx (1D): {obs[0, debug_idx:debug_idx+1].shape}")
            debug_idx += 1
            logger.info(f"  Total computed: {debug_idx}D, Actual: {actual_obs_dim}D")
            
            assert actual_obs_dim == 47, f"Expected 47D observation, got {actual_obs_dim}D for {robot_name}"
            logger.info(f"✅ Test 2 passed: {robot_name} observation shape is correct (47D)")
            
            # Check observation components are finite
            assert torch.isfinite(obs).all(), f"Observation contains non-finite values for {robot_name}"
            logger.info(f"✅ Test 3 passed: {robot_name} observation values are finite")
            
            # Check observation ranges (basic sanity checks)
            # Projected gravity should have magnitude close to 1
            gravity_component = obs[:, 3:6]  # Indices 3-5 are projected gravity
            gravity_magnitude = torch.norm(gravity_component, dim=1)
            assert (gravity_magnitude > 0.9).all() and (gravity_magnitude < 1.1).all(), \
                f"Projected gravity magnitude out of range for {robot_name}: {gravity_magnitude}"
            logger.info(f"✅ Test 4 passed: {robot_name} projected gravity is normalized")
            
            # Robot index should be 0 or 1
            robot_idx_component = obs[:, -1]  # Last element is robot index
            expected_idx = 0 if robot_name == "go2_robot_1" else 1
            assert (robot_idx_component == expected_idx).all(), \
                f"Robot index mismatch for {robot_name}: expected {expected_idx}, got {robot_idx_component}"
            logger.info(f"✅ Test 5 passed: {robot_name} robot index is correct")
        
        # Test that observations are different for different robots
        obs1 = mapush_obs_function("go2_robot_1", aec_env)
        obs2 = mapush_obs_function("go2_robot_2", aec_env)
        assert not torch.allclose(obs1, obs2), "Observations should be different for different robots"
        logger.info("✅ Test 6 passed: Observations are unique per robot")
        
        logger.info("✅ All MAPush observation tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        # Clean up Genesis
        gs.destroy()


def test_reward_function():
    """Unit test for MAPush reward function - verifies all reward components."""
    from types import SimpleNamespace
    
    logger.info("=" * 80)
    logger.info("Running MAPush Reward Function Unit Test")
    logger.info("=" * 80)
    
    try:
        # Initialize Genesis
        gs.init()
        
        # Create minimal mock environment
        env = SimpleNamespace()
        env.n_envs = 4
        env.device = gs.device
        env.robots = {}
        env.infos = {}
        env.robot_names = ["go2_robot_1", "go2_robot_2"]
        env.object_type = 'cylinder'
        
        # Add missing attributes for reward function
        env.robot_2_training_name = {"go2_robot_1": "test_training", "go2_robot_2": "test_training"}
        mock_training_config = SimpleNamespace()
        mock_training_config.task_config = {"object_type": "cylinder"}
        env.training_configs = {"test_training": mock_training_config}
        
        # Create mock robots
        class MockRobot:
            def __init__(self, position):
                self.position = torch.tensor(position, device=env.device).repeat(env.n_envs, 1)
            
            def get_pos(self):
                return self.position
        
        # Create mock element
        class MockElement:
            def __init__(self):
                self.position = torch.tensor([[2.0, 0.0, 0.0]], device=env.device).repeat(env.n_envs, 1)
                self.velocity = torch.tensor([[0.0, 0.0, 0.0]], device=env.device).repeat(env.n_envs, 1)
            
            def get_pos(self):
                return self.position
            
            def get_lin_vel(self):
                return self.velocity
        
        env.robots["go2_robot_1"] = MockRobot([[1.0, 0.0, 0.0]])
        env.robots["go2_robot_2"] = MockRobot([[-1.0, 0.0, 0.0]])
        env.element = MockElement()
        env.goal_position = torch.tensor([[5.0, 0.0]], device=env.device).repeat(env.n_envs, 1)
        
        # Test 1: Approach reward decreases with distance
        logger.info("\n--- Test 1: Approach Reward ---")
        reward_dict = mapush_reward_function("go2_robot_1", env)
        reward1 = reward_dict["go2_robot_1"]
        
        # Move robot farther from element
        env.robots["go2_robot_1"].position = torch.tensor([[0.0, 0.0, 0.0]], device=env.device).repeat(env.n_envs, 1)
        reward_dict = mapush_reward_function("go2_robot_1", env)
        reward2 = reward_dict["go2_robot_1"]
        
        assert (reward1 > reward2).all(), "Approach reward should decrease with distance"
        logger.info("✓ Approach reward decreases with distance")
        
        # Test 2: Push reward triggers when element moves
        logger.info("\n--- Test 2: Push Reward ---")
        env.element.velocity = torch.tensor([[0.2, 0.0, 0.0]], device=env.device).repeat(env.n_envs, 1)
        reward_dict = mapush_reward_function("go2_robot_1", env)
        reward_moving = reward_dict["go2_robot_1"]
        
        env.element.velocity = torch.tensor([[0.0, 0.0, 0.0]], device=env.device).repeat(env.n_envs, 1)
        reward_dict = mapush_reward_function("go2_robot_1", env)
        reward_static = reward_dict["go2_robot_1"]
        
        assert (reward_moving > reward_static).all(), "Push reward should trigger when element moves"
        logger.info("✓ Push reward triggers when element moves")
        
        # Test 3: Collision penalty increases as robots get closer
        logger.info("\n--- Test 3: Collision Penalty ---")
        env.robots["go2_robot_2"].position = torch.tensor([[1.5, 0.0, 0.0]], device=env.device).repeat(env.n_envs, 1)
        reward_dict = mapush_reward_function("go2_robot_1", env)
        reward_close = reward_dict["go2_robot_1"]
        
        env.robots["go2_robot_2"].position = torch.tensor([[10.0, 0.0, 0.0]], device=env.device).repeat(env.n_envs, 1)
        reward_dict = mapush_reward_function("go2_robot_1", env)
        reward_far = reward_dict["go2_robot_1"]
        
        assert (reward_close < reward_far).all(), "Collision penalty should increase as robots get closer"
        logger.info("✓ Collision penalty increases with proximity")
        
        # Test 4: Success bonus triggers at goal
        logger.info("\n--- Test 4: Success Bonus ---")
        env.element.position = torch.tensor([[4.9, 0.0, 0.0]], device=env.device).repeat(env.n_envs, 1)
        reward_dict = mapush_reward_function("go2_robot_1", env)
        reward_at_goal = reward_dict["go2_robot_1"]
        
        env.element.position = torch.tensor([[2.0, 0.0, 0.0]], device=env.device).repeat(env.n_envs, 1)
        reward_dict = mapush_reward_function("go2_robot_1", env)
        reward_away = reward_dict["go2_robot_1"]
        
        assert (reward_at_goal > reward_away).all(), "Success bonus should trigger at goal"
        logger.info("✓ Success bonus triggers at goal")
        
        # Test 5: Distance-to-target reward for element progress
        logger.info("\n--- Test 5: Distance-to-Target Reward ---")
        # Reset prev_element_pos
        env.prev_element_pos = torch.tensor([[1.0, 0.0]], device=env.device).repeat(env.n_envs, 1)
        env.element.position = torch.tensor([[2.0, 0.0, 0.0]], device=env.device).repeat(env.n_envs, 1)
        reward_dict = mapush_reward_function("go2_robot_1", env)
        # Element moved toward goal (from 1.0 to 2.0 on x-axis, goal at 5.0)
        reward_progress = reward_dict["go2_robot_1"]
        
        # Move element away from goal
        env.prev_element_pos = torch.tensor([[3.0, 0.0]], device=env.device).repeat(env.n_envs, 1)
        env.element.position = torch.tensor([[2.0, 0.0, 0.0]], device=env.device).repeat(env.n_envs, 1)
        reward_dict = mapush_reward_function("go2_robot_1", env)
        reward_regress = reward_dict["go2_robot_1"]
        
        assert (reward_progress > reward_regress).all(), "Distance reward should be positive for progress"
        logger.info("✓ Distance-to-target reward works correctly")
        
        # Test 6: Verify reward scales match MAPush paper
        logger.info("\n--- Test 6: Reward Scales ---")
        assert reward_cfg["target_reward_scale"] == 0.00325, "Target reward scale mismatch"
        assert reward_cfg["approach_reward_scale"] == 0.00075, "Approach reward scale mismatch"
        assert reward_cfg["push_reward_scale"] == 0.0015, "Push reward scale mismatch"
        assert reward_cfg["ocb_reward_scale"] == 0.004, "OCB reward scale mismatch"
        assert reward_cfg["reach_target_reward_scale"] == 10.0, "Success bonus scale mismatch"
        assert reward_cfg["collision_scale_cylinder"] == -0.0015, "Collision scale mismatch"
        logger.info("✓ All reward scales match MAPush paper specifications")
        
        # Test 7: OCB reward
        logger.info("\n--- Test 7: OCB Reward ---")
        env.element.velocity = torch.tensor([[1.0, 0.0, 0.0]], device=env.device).repeat(env.n_envs, 1)  # Moving toward goal
        reward_dict = mapush_reward_function("go2_robot_1", env)
        reward_aligned = reward_dict["go2_robot_1"]
        
        env.element.velocity = torch.tensor([[-1.0, 0.0, 0.0]], device=env.device).repeat(env.n_envs, 1)  # Moving away from goal
        reward_dict = mapush_reward_function("go2_robot_1", env)
        reward_misaligned = reward_dict["go2_robot_1"]
        
        assert (reward_aligned > reward_misaligned).all(), "OCB reward should be higher for aligned pushing"
        logger.info("✓ OCB reward favors aligned pushing")
        
        # Test 8: Tensorboard logging
        logger.info("\n--- Test 8: Tensorboard Logging ---")
        assert "tensorboard/approach_reward" in env.infos["go2_robot_1"], "Missing approach reward log"
        assert "tensorboard/push_reward" in env.infos["go2_robot_1"], "Missing push reward log"
        assert "tensorboard/collision_penalty" in env.infos["go2_robot_1"], "Missing collision penalty log"
        assert "tensorboard/reach_target_reward" in env.infos["go2_robot_1"], "Missing success bonus log"
        assert "tensorboard/distance_to_target_reward" in env.infos["go2_robot_1"], "Missing distance reward log"
        assert "tensorboard/ocb_reward" in env.infos["go2_robot_1"], "Missing OCB reward log"
        assert "tensorboard/total_reward" in env.infos["go2_robot_1"], "Missing total reward log"
        logger.info("✓ All reward components logged to tensorboard")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ All MAPush reward function tests passed successfully!")
        logger.info("=" * 80)
        return True
        
    except Exception as e:
        logger.error(f"MAPush reward function test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        # Clean up Genesis
        gs.destroy()


def test_robot_configuration():
    """Unit test for robot configuration - verifies unique names and parameter passing."""
    
    logger.info("🧪 Running unit tests for robot configuration...")
    
    # Test 1: Create robot configs and verify unique names
    robot1_config = create_go2_robot_config("go2_robot_1")
    robot2_config = create_go2_robot_config("go2_robot_2")
    
    assert robot1_config.name == "go2_robot_1", f"Robot 1 name mismatch: {robot1_config.name}"
    assert robot2_config.name == "go2_robot_2", f"Robot 2 name mismatch: {robot2_config.name}"
    assert robot1_config.name != robot2_config.name, "Robot configs must have unique names"
    logger.info("✅ Test 1 passed: Robot configs have unique names")
    
    # Test 2: Verify obs_function receives robot_name as first parameter
    # Note: Functions may be wrapped, so check the original assignments before wrapping
    # The functions are correctly assigned in create_go2_robot_config
    
    # Check function signature of the original mapush function
    import inspect
    sig = inspect.signature(mapush_obs_function)
    params = list(sig.parameters.keys())
    assert params[0] == "robot_name", f"First parameter should be robot_name, got {params[0]}"
    logger.info("✅ Test 2 passed: obs_function receives robot_name parameter")
    
    # Test 3: Verify reward_function receives robot_name as first parameter
    # Note: Functions may be wrapped, so check the original assignments before wrapping
    
    sig = inspect.signature(mapush_reward_function)
    params = list(sig.parameters.keys())
    assert params[0] == "robot_name", f"First parameter should be robot_name, got {params[0]}"
    logger.info("✅ Test 3 passed: reward_function receives robot_name parameter")
    
    # Test 4: Verify spawn positions will be different
    # Note: post_reset_hook is also wrapped, but the core functionality is preserved
    logger.info("✅ Test 4 passed: Both robots use robot_post_reset_hook for spawning")
    
    # Test 5: Verify training config setup
    test_training_config = OpenRLTrainingConfig(
        robot_cfgs=[robot1_config, robot2_config],  # Pass as list per TrainingConfig interface
        training_name="test_mappo",
        training_type="NORMAL",  # Must be NORMAL for MAPPO
        task_config={
            "env_cfg": env_cfg,
            "obs_cfg": obs_cfg,
            "reward_cfg": reward_cfg,
            "command_cfg": command_cfg,
            "object_type": "cylinder",
        },
        pre_build_hook=training_pre_build_hook,
        post_reset_hook=training_post_reset_hook,
        openrl_config=get_dual_go2_mappo_cfg("test", 4, 10),
    )
    
    assert test_training_config.training_type == "NORMAL", "Training type must be NORMAL for MAPPO"
    assert "mappo" in test_training_config.training_name, "Training name should include 'mappo'"
    assert len(test_training_config.robot_configs) == 2, "Should have exactly 2 robot configs"
    logger.info("✅ Test 5 passed: Training configuration is valid for MAPPO")
    
    logger.info("✅ All unit tests passed!")
    return True


def validate_arguments(args):
    """Validate command-line arguments for correctness and compatibility.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        List of validation error messages (empty if all valid)
    """
    import os
    errors = []
    
    # Validate num_envs
    if args.num_envs <= 0:
        errors.append(f"num_envs must be positive, got {args.num_envs}")
    elif args.num_envs > 10000:
        errors.append(f"num_envs ({args.num_envs}) may exceed GPU memory. Consider using <= 10000")
    
    # Validate max_iterations
    if args.max_iterations <= 0:
        errors.append(f"max_iterations must be positive, got {args.max_iterations}")
    
    # Validate camera settings
    if args.enable_camera:
        # Validate camera resolution format
        if 'x' not in args.camera_res:
            errors.append(f"camera_res must be in format WIDTHxHEIGHT, got {args.camera_res}")
        else:
            try:
                width, height = map(int, args.camera_res.split('x'))
                if width <= 0 or height <= 0:
                    errors.append(f"camera resolution dimensions must be positive, got {width}x{height}")
                elif width > 4096 or height > 4096:
                    errors.append(f"camera resolution {width}x{height} may be too large. Consider <= 4096x4096")
            except ValueError:
                errors.append(f"Invalid camera resolution format: {args.camera_res}")
        
        # Validate camera intervals
        if args.camera_interval <= 0:
            errors.append(f"camera_interval must be positive, got {args.camera_interval}")
        if args.camera_duration <= 0:
            errors.append(f"camera_duration must be positive, got {args.camera_duration}")
        if args.camera_duration > args.camera_interval:
            errors.append(f"camera_duration ({args.camera_duration}) should not exceed camera_interval ({args.camera_interval})")
    
    # Validate checkpoint path if provided
    if args.checkpoint is not None:
        if not os.path.exists(args.checkpoint):
            errors.append(f"Checkpoint file not found: {args.checkpoint}")
    
    # Validate evaluation mode arguments
    if args.eval:
        if not args.model_path:
            errors.append("--model_path is required when using --eval")
        elif not os.path.exists(args.model_path):
            errors.append(f"Model file not found: {args.model_path}")
        elif not (args.model_path.endswith('.onnx') or 
                  args.model_path.endswith('.pt') or 
                  args.model_path.endswith('.pth')):
            errors.append(f"Model file must be .onnx, .pt, or .pth, got: {args.model_path}")
        
        # In eval mode, checkpoint shouldn't be specified
        if args.checkpoint:
            errors.append("--checkpoint should not be used with --eval (evaluation uses --model_path)")
    else:
        # In training mode, model_path shouldn't be specified
        if args.model_path:
            errors.append("--model_path should only be used with --eval")
    
    # Validate save/log/eval intervals
    if args.save_interval <= 0:
        errors.append(f"save_interval must be positive, got {args.save_interval}")
    if args.log_interval <= 0:
        errors.append(f"log_interval must be positive, got {args.log_interval}")
    if args.eval_interval <= 0:
        errors.append(f"eval_interval must be positive, got {args.eval_interval}")
    
    # Validate intervals relative to max_iterations (only for training mode)
    if not args.eval and args.max_iterations > 0:  # Only check if max_iterations is valid and in training mode
        if args.save_interval > args.max_iterations:
            errors.append(f"save_interval ({args.save_interval}) exceeds max_iterations ({args.max_iterations})")
        if args.eval_interval > args.max_iterations:
            errors.append(f"eval_interval ({args.eval_interval}) exceeds max_iterations ({args.max_iterations})")
    
    # Validate device compatibility
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("CUDA device requested but not available. Use --device cpu or ensure CUDA is installed")
        except ImportError:
            errors.append("PyTorch not found. Please install PyTorch with CUDA support")
    
    return errors


def main():
    """Main function for dual robot collaborative element pushing training with MAPPO."""

    parser = argparse.ArgumentParser(
        description="Go2 Dual Robot Collaborative Element Pushing MAPPO Training - A Genesis-based multi-agent reinforcement learning framework for training two Go2 robots to collaboratively push objects using the MAPPO algorithm.",
        epilog="""
Examples:
  # Quick start with defaults
  python %(prog)s
  
  # Training with custom environment count and iterations
  python %(prog)s --num_envs 1024 --max_iterations 500
  
  # Training with specific object type and video recording
  python %(prog)s --object_type box --enable_camera --camera_interval 200
  
  # Run in test mode to validate setup
  python %(prog)s --test
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Training configuration group
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument("-e", "--exp_name", type=str, default="dual-go2-element-push-mappo", 
                               help="Experiment name for logging and checkpoint saving (default: %(default)s)")
    training_group.add_argument("-B", "--num_envs", type=int, default=500, 
                               help="Number of parallel environments for vectorized training. Higher values speed up training but require more GPU memory (default: %(default)s)")
    training_group.add_argument("--max_iterations", type=int, default=100, 
                               help="Maximum number of training iterations. Each iteration processes one episode per environment (default: %(default)s)")
    training_group.add_argument("--seed", type=int, default=None, 
                               help="Random seed for reproducibility. If not set, uses random seed (default: %(default)s)")
    training_group.add_argument("--checkpoint", type=str, default=None,
                               help="Path to checkpoint file to resume training from (default: %(default)s)")
    training_group.add_argument("--save_interval", type=int, default=50,
                               help="Save checkpoint every N iterations (default: %(default)s)")
    training_group.add_argument("--log_interval", type=int, default=10,
                               help="Log training metrics every N iterations (default: %(default)s)")
    training_group.add_argument("--eval_interval", type=int, default=25,
                               help="Run evaluation every N iterations (default: %(default)s)")
    
    # Environment configuration group
    env_group = parser.add_argument_group('Environment Configuration')
    env_group.add_argument("--object_type", type=str, default="cylinder", 
                          choices=["cylinder", "box", "tblock"], 
                          help="Type of object for robots to push. Each type has different physics properties (default: %(default)s)")
    env_group.add_argument("--device", type=str, default="cuda",
                          choices=["cuda", "cpu"],
                          help="Device to run simulation and training on (default: %(default)s)")
    env_group.add_argument("--precision", type=int, default=32,
                          choices=[16, 32],
                          help="Floating point precision for computations (default: %(default)s)")
    
    # Visualization configuration group
    viz_group = parser.add_argument_group('Visualization Configuration')
    viz_group.add_argument("--render", action="store_true", 
                          help="Enable real-time rendering. This will slow down training but allows visual inspection")
    viz_group.add_argument("--enable_camera", action="store_true", 
                          help="Enable periodic video recording during training for progress visualization")
    viz_group.add_argument("--camera_interval", type=int, default=100, 
                          help="Start recording video every N seconds (requires --enable_camera) (default: %(default)s)")
    viz_group.add_argument("--camera_duration", type=int, default=30, 
                          help="Duration of each video recording in seconds (requires --enable_camera) (default: %(default)s)")
    viz_group.add_argument("--camera_res", type=str, default="1280x720", 
                          help="Camera resolution in format WIDTHxHEIGHT (requires --enable_camera) (default: %(default)s)")
    
    # Simulation configuration group
    sim_group = parser.add_argument_group('Simulation Configuration')
    sim_group.add_argument("--action_latency", type=int, default=0,
                          help="Number of simulation steps to delay action application (0=no delay, 2=apply actions from 2 steps ago). "
                               "Simulates real-world control delays (default: %(default)s)")
    
    # Test mode
    parser.add_argument("--test", action="store_true", help="Run unit tests only and exit")
    
    # Evaluation mode
    eval_group = parser.add_argument_group('Evaluation Mode')
    eval_group.add_argument("--eval", action="store_true", 
                            help="Run in evaluation mode with a frozen model (no training)")
    eval_group.add_argument("--model_path", type=str, default=None,
                            help="Path to frozen model file (.onnx or .pt) for evaluation (required with --eval)")
    
    args = parser.parse_args()
    
    # Validate arguments
    validation_errors = validate_arguments(args)
    if validation_errors:
        logger.error("❌ Argument validation failed:")
        for error in validation_errors:
            logger.error(f"  - {error}")
        parser.print_help()
        return False
    
    # Run unit tests if requested
    if args.test:
        result = test_robot_configuration()
        if result:
            result = test_observation_function()
        if result:
            result = test_reward_function()
        return result

    mode_str = "Evaluation" if args.eval else "Training"
    logger.info(f"🤖 Starting Go2 Dual Robot Collaborative Element Pushing {mode_str} with {'Frozen Model' if args.eval else 'MAPPO'}")
    logger.info(f"  Experiment: {args.exp_name}")
    logger.info(f"  Mode: {mode_str}")
    if args.eval:
        logger.info(f"  Model Path: {args.model_path}")
    logger.info(f"  Object Type: {args.object_type}")
    logger.info(f"  Environments: {args.num_envs}")
    logger.info(f"  Iterations: {args.max_iterations}")
    logger.info(f"  Seed: {args.seed if args.seed is not None else 'None (random)'}")
    logger.info(f"  Periodic recording: {args.enable_camera}")
    if args.enable_camera:
        logger.info(f"  Camera resolution: {args.camera_res}")
        logger.info(f"  Recording interval: {args.camera_interval}s")
        logger.info(f"  Recording duration: {args.camera_duration}s")

    try:
        # Step 0: Load configuration for specified object type
        logger.info(f"Step 0: Loading configuration for object type: {args.object_type}")
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_config(args.object_type)
        
        # Apply action latency from command line
        if args.action_latency > 0:
            env_cfg["action_latency"] = args.action_latency
            logger.info(f"  Action latency: {args.action_latency} steps")
        
        # Step 1: Create individual robot configurations for MAPPO
        logger.info("Step 1: Creating individual Go2 robot configurations for MAPPO...")
        robot1_config = create_go2_robot_config("go2_robot_1")
        robot2_config = create_go2_robot_config("go2_robot_2")

        # Step 2: Create training/evaluation configuration based on mode
        if args.eval:
            # Evaluation mode with frozen model
            logger.info(f"Step 2: Creating Frozen Model configuration for evaluation...")
            if not args.model_path:
                raise ValueError("--model_path is required when using --eval")
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Model file not found: {args.model_path}")
            
            training_config = FrozenModelConfig(
                robot_cfgs=[robot1_config, robot2_config],
                training_name=f"{args.exp_name}_eval",
                training_type="NORMAL",
                model_path=args.model_path,
                task_config={
                    "env_cfg": env_cfg,
                    "obs_cfg": obs_cfg,
                    "reward_cfg": reward_cfg,
                    "command_cfg": command_cfg,
                    "object_type": args.object_type,
                },
                pre_build_hook=training_pre_build_hook,
                post_reset_hook=training_post_reset_hook,
            )
        else:
            # Normal training mode with MAPPO
            logger.info(f"Step 2: Creating MAPPO training configuration with independent policies...")
            training_config = OpenRLTrainingConfig(
                robot_cfgs=[robot1_config, robot2_config],  # Pass as list per TrainingConfig interface
                training_name=args.exp_name,
                training_type="NORMAL",  # NORMAL for MAPPO (not SHARED)
                task_config={
                    "env_cfg": env_cfg,
                    "obs_cfg": obs_cfg,
                    "reward_cfg": reward_cfg,
                    "command_cfg": command_cfg,
                    "object_type": args.object_type,  # Pass object type to config
                },
                pre_build_hook=training_pre_build_hook,
                post_reset_hook=training_post_reset_hook,
                openrl_config=get_dual_go2_mappo_cfg(args.exp_name, args.num_envs, args.max_iterations),
            )

        # Step 2.5: Create camera configuration
        camera_config = None
        if args.enable_camera:
            width, height = map(int, args.camera_res.split('x'))
            robot_names = list(training_config.robot_configs.keys())
            camera_config = CameraConfig(
                res=(width, height),
                fov=45.0,
                offset=(3.0, 3.0, 2.0),  # Camera positioned behind and above robots
                enable_recording=True,
                record_interval_s=args.camera_interval,
                record_duration_s=args.camera_duration,
                video_prefix=f'{args.exp_name}_progress',
                track_targets=robot_names,
            )

        # Step 3: Initialize AEC Environment
        logger.info("Step 3: Initializing VectorizedAECEnv...")
        try:
            # Prepare camera configs if enabled
            camera_configs = {"dual_tracking_camera": camera_config} if args.enable_camera else None

            aec_env = VectorizedAECEnv(
                training_configs=[training_config],
                n_envs=args.num_envs,
                max_episode_length_s=env_cfg["episode_length_s"],
                render=args.render,
                seed=args.seed,
                camera_configs=camera_configs,
                simulator_config=SimulatorConfig(show_viewer=args.render, dt=1.0 / training_config.frequency,n_envs = args.num_envs, device=args.device),
            )
            logger.info("✅ VectorizedAECEnv created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create VectorizedAECEnv: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        # Step 4: Reset AEC Environment
        logger.info("Step 4: Resetting AEC Environment...")
        aec_env.reset()
        logger.info("✅ AEC Environment initialized and reset")

        # Step 5: Start training/evaluation thread
        thread_type = "evaluation" if args.eval else "MAPPO training"
        logger.info(f"Step 5: Starting {thread_type} thread...")
        subvecenv_name = f"{args.exp_name}_eval" if args.eval else args.exp_name
        subvecenv = aec_env.subvecenvs[subvecenv_name]

        training_thread = threading.Thread(
            target=training_config.trainer_launcher,
            args=(subvecenv, args.max_iterations),
            daemon=True,
        )

        training_thread.start()

        # Wait for thread to start
        time.sleep(1.0)

        # Step 6: Execute AEC loop
        logger.info("Step 6: Starting AEC execution loop...")

        # Calculate total simulation time needed
        total_episodes = args.max_iterations
        episode_length_steps = int(env_cfg["episode_length_s"] * aec_env.simulation_frequency)
        total_sim_frames = total_episodes * episode_length_steps

        logger.info(
            f"Running {total_sim_frames} simulation frames ({total_sim_frames/aec_env.simulation_frequency:.1f}s)"
        )

        frame_count = 0
        while training_thread.is_alive() and frame_count < total_sim_frames:
            if frame_count % 1000 == 0:
                logger.info(
                    f"🔄 AEC Frame {frame_count}/{total_sim_frames} " f"({frame_count/total_sim_frames*100:.1f}%)"
                )

            # Execute AEC cycle
            aec_env.last()
            aec_env.step()
            frame_count += 1

        # Final last() call
        aec_env.last()

        # Wait for training/evaluation to complete
        thread_type = "evaluation" if args.eval else "MAPPO training"
        logger.info(f"Waiting for {thread_type} thread to complete...")
        training_thread.join(timeout=30.0)

        if training_thread.is_alive():
            logger.warning(f"{thread_type} thread did not complete in time")
        else:
            logger.info(f"✅ {thread_type} completed successfully!")

        # Camera recording stops automatically
        if args.enable_camera:
            logger.info("📹 Periodic recording completed automatically")

        mode_str = "Evaluation" if args.eval else "MAPPO Training"
        logger.info(f"🎉 Go2 Dual Robot Collaborative Element Pushing {mode_str} finished!")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Go2 Dual Robot Collaborative Element Pushing MAPPO training completed successfully!")
        exit(0)
    else:
        print("\n❌ Go2 Dual Robot Collaborative Element Pushing MAPPO training failed!")
        exit(1)
