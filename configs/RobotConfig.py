import copy
import math
import os
from abc import ABC
from dataclasses import InitVar, asdict, dataclass, field, fields
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

import gymnasium as gym
import numpy as np
import torch

from marl_logging import get_class_logger

# Removed import to avoid circular dependency
from urdf_metadata_extractor import URDFMetadataExtractor
from utils import (
    check_boolean_tensor,
    inv_quat,
    quat_to_xyz,
    transform_quat_by_quat,
    wrap_normal_function_as_closure,
    wrap_reset_function_as_closure,
    xyz_to_quat,
)

AECEnv = Any
RNAME = str  # Robot name


@dataclass
class RobotConfig:
    """Configuration for a robot in the MARL environment (V3 with functional computation)"""

    name: str
    urdf_path: str
    frequency: float  # Hz
    action_space: gym.spaces.Space = field(default=None)  # None means auto-detect
    observation_space: gym.spaces.Space = field(
        default=None
    )  # None means auto-detect through sample run of obs_function
    control_mode: str = "position"  # "position", "velocity", "force"; "force" is for both torque and force
    joint_names: List[str] = field(default_factory=list)  # Empty means auto-detect

    """
    Hook + Functions Core component
    """
    # Pre built Hook, Hook before building the scene. Can be used to modify the scene
    pre_build_hook: Optional[Callable] = None

    # Setup Function, used to setup the robot in the scene, can be deduced
    setup_function: Optional[Callable] = None

    # Post build hook, hook after building the scene. The observation shape are already deduced at this point
    post_build_hook: Optional[Callable] = None

    # Before Reset hook, hook before resetting some environment, still in the previous episode
    pre_reset_hook: Optional[Callable[[AECEnv, List[int]], None]] = None
    # Post Reset hook, hook after resetting some environment, in the new episode
    post_reset_hook: Optional[Callable[[AECEnv, List[int]], None]] = None

    # Before last()
    pre_last_hook: Optional[Callable] = None

    # Functional observation and reward computation
    info_function: Callable[[AECEnv], Dict[RNAME, Dict]] = None  # Function(mother_env) -> info_dict
    obs_function: Optional[Callable] = None  # Function(mother_env) -> obs_tensor
    reward_function: Optional[Callable] = None  # Function(mother_env) -> dict of reward_tensor
    truncation_function: Optional[Callable] = None  # Function(mother_env) -> truncation_tensor

    # By the end of last() - After autoreset
    post_last_hook: Optional[Callable] = None

    # Pre step() hook
    pre_step_hook: Optional[Callable] = None

    # Action preprocessing
    action_preprocessing_function: Optional[Callable] = None  # Function( mother_env) -> final_target_positions

    # Used for locomotion model inference
    locomotion_path: Optional[str] = field(default=None)  # locomotion path, none means action is joint values

    action_postprocessing_function: Optional[Callable] = None  # Function(action, mother_env) -> final_target_positions

    # Post step() hook
    post_step_hook: Optional[Callable] = None

    # Tuneable parameters
    initial_position: Optional[List[float]] = None  # [x, y, z]
    initial_orientation: Optional[List[float]] = None  # [x,y,z] euler or [x, y, z, w] quaternion
    initial_velocity: Optional[List[float]] = None  # [x, y, z]
    initial_angular_velocity: Optional[List[float]] = None  # [roll, pitch, yaw]
    initial_joint_pos: Optional[List[float]] = None  # Default joint positions, None means zeros
    initial_joint_vel: Optional[List[float]] = None  # Default joint velocities, None means zeros
    DP_kp: Union[float, List[float]] = 20.0  # Scalar apply to all joints, List apply to each joint
    DP_kd: Union[float, List[float]] = 0.5  # Scalar apply to all joints, List apply to each joint

    # Temporaries
    sample_obs: Optional[torch.Tensor] = None
    _initial_args: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        # Initialize independent class-specific logger
        self.logger = get_class_logger("Robot", self.name, level="INFO")

        self._initial_args = {f.name: copy.deepcopy(getattr(self, f.name)) for f in fields(self)}

        # Check if the urdf is abs path
        if not os.path.isabs(self.urdf_path):
            self.urdf_path = os.path.abspath(self.urdf_path)

        # Check if the locomotion path is abs path
        if self.locomotion_path is not None:
            if not os.path.isabs(self.locomotion_path):
                self.locomotion_path = os.path.abspath(self.locomotion_path)

        # Deduce those field that don't need AEC instance
        if self.joint_names == []:
            movable_joints = self.urdf_extractor.get_movable_joints()
            self.joint_names = [joint["name"] for joint in movable_joints]

        # Add default initial pos / orientation / velocity / angular velocity
        if self.initial_position is None:
            self.initial_position = [0.0, 0.0, 0.0]
        if self.initial_orientation is None:
            self.initial_orientation = [1.0, 0.0, 0.0]  # Euler on x direction
        if self.initial_velocity is None:
            self.initial_velocity = [0.0, 0.0, 0.0]
        if self.initial_angular_velocity is None:
            self.initial_angular_velocity = [0.0, 0.0, 0.0]

        if self.initial_joint_pos is None:
            self.initial_joint_pos = [0.0] * self.n_dofs
        if self.initial_joint_vel is None:
            self.initial_joint_vel = [0.0] * self.n_dofs

        # Convert xyz euler to xyzw quaternion for orientation
        if self.initial_orientation is not None and len(self.initial_orientation) == 3:
            self.initial_orientation = xyz_to_quat(np.array(self.initial_orientation, dtype=np.float32)).tolist()
        # No need to convert angular velocity, as it is in roll-pitch-yaw

        if self.action_space is None:
            if self.locomotion_path is None:
                joint_to_num_dofs = self.urdf_extractor.get_dof_counts_by_joint()
                total_dofs = []
                for joint_name in self.joint_names:
                    total_dofs.append(joint_to_num_dofs[joint_name])
                self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(sum(total_dofs),), dtype=np.float32)
            else:
                # Try to detect from the input of the locomotion model
                from model_inference import ModelInference

                MI = ModelInference()
                MI.load_model(self.locomotion_path)
                model_info = MI.get_model_info()
                shape = model_info["input_shape"][1:]
                self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)

        if isinstance(self.DP_kp, float):
            self.DP_kp = [self.DP_kp] * self.n_dofs
        if isinstance(self.DP_kd, float):
            self.DP_kd = [self.DP_kd] * self.n_dofs

        if self.setup_function is None:
            self.logger.debug(f"Using default setup for robot {self.name}")

            def setup_function(AEC_env):
                robot_config = AEC_env.robot_configs[self.name]

                # Convert position and orientation to tensors for interface
                pos = torch.tensor(robot_config.initial_position, device=AEC_env.device)
                quat = torch.tensor(robot_config.initial_orientation, device=AEC_env.device)

                # Use scene interface to add robot
                robot_interface = AEC_env.scene.add_robot(
                    urdf_path=robot_config.urdf_path,
                    pos=pos,
                    quat=quat,
                    fixed=False,
                )
                return robot_interface

            self.setup_function = setup_function

        # Observation
        if self.obs_function is None:
            self.logger.debug(f"Using default obs for robot {self.name}")

            def obs_function(AEC_env):
                robot = AEC_env.robots[self.name]

                # Get robot state
                pos = robot.get_pos()  # [n_envs, 3]
                quat = robot.get_quat()  # [n_envs, 4]
                vel = robot.get_vel()  # [n_envs, 3]
                ang_vel = robot.get_ang()  # [n_envs, 3]

                obs_components = [pos, quat, vel, ang_vel]

                # Add joint states if available
                if hasattr(robot, "get_dofs_position"):
                    joint_pos = robot.get_dofs_position()
                    joint_vel = robot.get_dofs_velocity()
                    obs_components.extend([joint_pos, joint_vel])

                return torch.cat(obs_components, dim=-1)

            self.obs_function = obs_function

        # Reward
        if self.reward_function is None:
            self.logger.warning(f"Using dummy reward for robot {self.name}")

            def reward_function(AEC_env):
                robot_config = AEC_env.robot_configs[self.name]

                reward_per_sec = 1
                reward = 1 / robot_config.frequency * reward_per_sec

                return {robot_config.name: torch.zeros((AEC_env.n_envs, 1), device=AEC_env.device).fill_(reward)}

            self.reward_function = reward_function

        # Truncation
        if self.truncation_function is None:
            self.logger.warning(f"Using dummy truncation for robot {self.name}")

            def truncation_function(AEC_env):
                robot_config = AEC_env.robot_configs[self.name]

                max_episode_length_s = 20  # in seconds
                truncate_if_pitch_greater_than = 10
                truncate_if_roll_greater_than = 10

                robot = AEC_env.robots[robot_config.name]
                base_init_quat = torch.tensor(robot_config.initial_orientation, device=AEC_env.device)
                base_quat = robot.get_quat()
                base_euler = quat_to_xyz(
                    transform_quat_by_quat(
                        torch.ones_like(base_quat) * inv_quat(base_init_quat),
                        base_quat,
                    ),
                    rpy=True,
                    degrees=True,
                )

                truncations = (AEC_env.episode_frame_count / AEC_env.simulation_frequency) > max_episode_length_s
                truncations |= torch.abs(base_euler[:, 1]) > truncate_if_pitch_greater_than
                truncations |= torch.abs(base_euler[:, 0]) > truncate_if_roll_greater_than
                return truncations

            self.truncation_function = truncation_function

        # Info
        if self.info_function is None:
            self.info_function = lambda AEC_env: {}

        # Action preprocessing
        if self.action_preprocessing_function is None:
            self.action_preprocessing_function = lambda AEC_env: AEC_env.action_buffers[self.name]

        # Action postprocessing
        if self.action_postprocessing_function is None:
            self.action_postprocessing_function = lambda AEC_env: AEC_env.joint_action_buffers[self.name]

        # Wrap as closure
        self.setup_function = wrap_normal_function_as_closure(self.name, self.setup_function)
        self.obs_function = wrap_normal_function_as_closure(self.name, self.obs_function)
        self.reward_function = wrap_normal_function_as_closure(self.name, self.reward_function)
        self.truncation_function = wrap_normal_function_as_closure(self.name, self.truncation_function)
        self.info_function = wrap_normal_function_as_closure(self.name, self.info_function)
        self.action_preprocessing_function = wrap_normal_function_as_closure(self.name, self.action_preprocessing_function)
        self.action_postprocessing_function = wrap_normal_function_as_closure(self.name, self.action_postprocessing_function)

        self.pre_build_hook = wrap_normal_function_as_closure(self.name, self.pre_build_hook)
        self.post_build_hook = wrap_normal_function_as_closure(self.name, self.post_build_hook)
        self.pre_reset_hook = wrap_reset_function_as_closure(self.name, self.pre_reset_hook)
        self.post_reset_hook = wrap_reset_function_as_closure(self.name, self.post_reset_hook)
        self.pre_last_hook = wrap_normal_function_as_closure(self.name, self.pre_last_hook)
        self.post_last_hook = wrap_normal_function_as_closure(self.name, self.post_last_hook)
        self.pre_step_hook = wrap_normal_function_as_closure(self.name, self.pre_step_hook)
        self.post_step_hook = wrap_normal_function_as_closure(self.name, self.post_step_hook)
        # Additional Wrappings
        # Always wrap setup function to register robot
        original_setup = self.setup_function

        # To wrap Setup Function
        def register_robot(AEC_env, robot):
            AEC_env.robots[self.name] = robot
            self.logger.debug(f"Registered robot {self.name}")

        self.setup_function = lambda AEC_env: register_robot(AEC_env, original_setup(AEC_env))

        original_truncation = self.truncation_function
        self.truncation_function = lambda AEC_env: check_boolean_tensor(original_truncation(AEC_env))

    def detect_obs_space(self, AEC_env):
        if self.observation_space is None:
            if self.sample_obs is None:
                # Remove the n_envs dimension
                self.sample_obs = self.obs_function(AEC_env)[0]
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=self.sample_obs.shape, dtype=self.sample_obs.cpu().numpy().dtype
            )

    @property
    def urdf_extractor(self):
        urdf_extractor = URDFMetadataExtractor()
        urdf_extractor.parse_urdf(self.urdf_path)
        return urdf_extractor

    @property
    def n_dofs(self):
        """Get the actual number of DOFs for joints in joint_names."""
        total_dofs = 0
        dof_counts = self.urdf_extractor.get_dof_counts_by_joint()
        for joint_name in self.joint_names:
            total_dofs += dof_counts.get(joint_name, 0)
        return total_dofs
