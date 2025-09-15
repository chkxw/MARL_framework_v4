import copy
import math
import os
from abc import ABC, abstractmethod
from dataclasses import InitVar, asdict, dataclass, field, fields
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

import genesis as gs
import gymnasium as gym
import numpy as np
import torch

from marl_logging import get_class_logger
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
from configs.RobotConfig import RobotConfig

AECEnv = Any
RNAME = str  # Robot name


@dataclass
class TrainingConfig(ABC):
    robot_cfgs: InitVar[List[RobotConfig]] = field(default=[])
    training_name: str = field(default=None)
    training_type: Literal["NORMAL", "JOINT", "SHARED"] = field(default="NORMAL")

    # Warning: Providing TrainingConfig Level hooks and functions will override original robot level hooks and functions

    # Hooks
    pre_build_hook: Callable[[AECEnv], None] = field(default=None)
    post_build_hook: Callable[[AECEnv], None] = field(default=None)
    pre_reset_hook: Callable[[AECEnv], None] = field(default=None)
    post_reset_hook: Callable[[AECEnv], None] = field(default=None)
    pre_last_hook: Callable[[AECEnv], None] = field(default=None)
    post_last_hook: Callable[[AECEnv], None] = field(default=None)
    pre_step_hook: Callable[[AECEnv], None] = field(default=None)
    post_step_hook: Callable[[AECEnv], None] = field(default=None)
    # Functions
    setup_function: Callable[[AECEnv], None] = field(default=None)
    obs_function: Callable[[AECEnv], Dict[RNAME, torch.Tensor]] = field(default=None)
    reward_function: Callable[[AECEnv], Dict[RNAME, torch.Tensor]] = field(default=None)
    truncation_function: Callable[[AECEnv], Dict[RNAME, torch.Tensor]] = field(default=None)
    info_function: Callable[[AECEnv], Dict[RNAME, Dict[str, Any]]] = field(default=None)

    # Special training level function
    shared_obs_function: Callable[[AECEnv], Dict[RNAME, torch.Tensor]] = field(default=None)

    # Robots
    robot_configs: Dict[RNAME, RobotConfig] = field(default_factory=dict, init=False)

    observation_spaces: Dict[str, gym.spaces.Space] = field(default=None)
    shared_observation_space: gym.spaces.Space = field(default=None)
    action_spaces: Dict[str, gym.spaces.Space] = field(default=None)

    # Temporaries
    sample_obs: Dict[str, torch.Tensor] = field(default=None)
    sample_shared_obs: torch.Tensor = field(default=None)
    _initial_args: Dict[str, Any] = field(default_factory=dict, init=False)

    # Task specific parameters
    task_config: Dict[str, Any] = field(default_factory=dict)

    # Tainer Specific implementation, must implement these fields
    # trainer_env_factory: Callable[[AECEnv], Any] = field(default=None)
    @property
    @abstractmethod
    def trainer_name(self):
        pass

    @property
    @abstractmethod
    def trainer_env_factory(self):
        pass

    # trainer_launcher: Callable[[AECEnv], Any] = field(default=None)
    @property
    @abstractmethod
    def trainer_launcher(self):
        pass

    # Customed  fields for trainer / reward recipe ...

    def __post_init__(self, robot_cfgs: List[RobotConfig]):
        # Initialize independent class-specific logger
        self.logger = get_class_logger("Training", self.training_name, level="INFO")

        self._initial_args = {f.name: copy.deepcopy(getattr(self, f.name)) for f in fields(self)}
        self._initial_args["robot_cfgs"] = [robot_config._initial_args for robot_config in robot_cfgs]
        """Auto complete the class initialization"""
        # Check if all robot configs have the same frequency, then set the frequency

        input_robot_configs = robot_cfgs

        # Frequency checker
        frequencies = {robot_config.frequency for robot_config in input_robot_configs}

        if len(frequencies) > 1:
            raise ValueError("All robot configs must have the same frequency to be used in a TrainingConfigBundle")

        for config in input_robot_configs:
            self.add_robot(config)

        if self.training_name is None:
            # Check if there is a name in the first robot config
            if len(self.robot_configs) > 0:
                robot_names = list(self.robot_configs.keys())
                self.training_name = "_".join(robot_names)
            else:
                self.training_name = "EmptyTrainingConfig"

        self.action_spaces = {
            robot_name: robot_config.action_space for robot_name, robot_config in self.robot_configs.items()
        }

        # Auto deduce Training Config level functions
        def build_robots_functions(function_name):
            def robots_functions(training_name, AEC_env):
                training_config = AEC_env.training_configs[training_name]
                result = {}
                all_none = True
                for robot_name, robot_config in training_config.robot_configs.items():
                    function_ = getattr(robot_config, function_name)
                    result[robot_name] = function_(AEC_env)
                    all_none &= result[robot_name] is None
                if all_none:
                    return None
                return result

            return robots_functions

        self.robots_setup_function: Callable[[AECEnv], None] = wrap_normal_function_as_closure(
            self.training_name, build_robots_functions("setup_function")
        )
        self.robots_obs_function: Callable[[AECEnv], Dict[RNAME, torch.Tensor]] = wrap_normal_function_as_closure(
            self.training_name, build_robots_functions("obs_function")
        )
        self.robots_reward_function: Callable[[AECEnv], Dict[RNAME, torch.Tensor]] = wrap_normal_function_as_closure(
            self.training_name, build_robots_functions("reward_function")
        )
        self.robots_truncation_function: Callable[[AECEnv], Dict[RNAME, torch.Tensor]] = (
            wrap_normal_function_as_closure(self.training_name, build_robots_functions("truncation_function"))
        )

        self.robots_info_function: Callable[[AECEnv], Dict[RNAME, Dict[str, Any]]] = wrap_normal_function_as_closure(
            self.training_name, build_robots_functions("info_function")
        )

        def build_robots_hooks(hook_name):
            def robots_hook(training_name, AEC_env):
                training_config = AEC_env.training_configs[training_name]
                for config in training_config.robot_configs.values():
                    hook = getattr(config, hook_name)
                    if hook is not None:
                        hook(AEC_env)

            return robots_hook

        def build_robots_reset_hooks(hook_name):
            def robots_reset_hook(training_name, AEC_env, env_indices):
                training_config = AEC_env.training_configs[training_name]
                for config in training_config.robot_configs.values():
                    hook = getattr(config, hook_name)
                    if hook is not None:
                        hook(AEC_env, env_indices)

            return robots_reset_hook

        self.robots_pre_build_hook: Callable[[AECEnv], None] = wrap_normal_function_as_closure(
            self.training_name, build_robots_hooks("pre_build_hook")
        )
        self.robots_post_build_hook: Callable[[AECEnv], None] = wrap_normal_function_as_closure(
            self.training_name, build_robots_hooks("post_build_hook")
        )
        self.robots_pre_reset_hook: Callable[[AECEnv], None] = wrap_reset_function_as_closure(
            self.training_name, build_robots_reset_hooks("pre_reset_hook")
        )
        self.robots_post_reset_hook: Callable[[AECEnv], None] = wrap_reset_function_as_closure(
            self.training_name, build_robots_reset_hooks("post_reset_hook")
        )
        self.robots_pre_last_hook: Callable[[AECEnv], None] = wrap_normal_function_as_closure(
            self.training_name, build_robots_hooks("pre_last_hook")
        )
        self.robots_post_last_hook: Callable[[AECEnv], None] = wrap_normal_function_as_closure(
            self.training_name, build_robots_hooks("post_last_hook")
        )
        self.robots_pre_step_hook: Callable[[AECEnv], None] = wrap_normal_function_as_closure(
            self.training_name, build_robots_hooks("pre_step_hook")
        )
        self.robots_post_step_hook: Callable[[AECEnv], None] = wrap_normal_function_as_closure(
            self.training_name, build_robots_hooks("post_step_hook")
        )

        # If the user don't provide training level functions, directly use per robot functions/hooks
        # If user want to tailor training level functions, but still want per robot functions, they can robot_xxx function in their implementation

        # Functions
        if self.setup_function is None:
            self.setup_function = self.robots_setup_function
        if self.obs_function is None:
            self.obs_function = self.robots_obs_function
        if self.reward_function is None:
            self.reward_function = self.robots_reward_function
        if self.truncation_function is None:
            self.truncation_function = self.robots_truncation_function
        if self.info_function is None:
            self.info_function = self.robots_info_function

        # Hooks
        if self.pre_build_hook is None:
            self.pre_build_hook = self.robots_pre_build_hook
        if self.post_build_hook is None:
            self.post_build_hook = self.robots_post_build_hook
        if self.pre_reset_hook is None:
            self.pre_reset_hook = self.robots_pre_reset_hook
        if self.post_reset_hook is None:
            self.post_reset_hook = self.robots_post_reset_hook
        if self.pre_last_hook is None:
            self.pre_last_hook = self.robots_pre_last_hook
        if self.post_last_hook is None:
            self.post_last_hook = self.robots_post_last_hook
        if self.pre_step_hook is None:
            self.pre_step_hook = self.robots_pre_step_hook
        if self.post_step_hook is None:
            self.post_step_hook = self.robots_post_step_hook

        # Wrap as closure
        original_setup_function = self.setup_function
        original_obs_function = self.obs_function
        original_reward_function = self.reward_function
        original_truncation_function = self.truncation_function
        original_info_function = self.info_function

        self.setup_function = wrap_normal_function_as_closure(self.training_name, original_setup_function)
        self.obs_function = wrap_normal_function_as_closure(self.training_name, original_obs_function)
        self.reward_function = wrap_normal_function_as_closure(self.training_name, original_reward_function)
        self.truncation_function = wrap_normal_function_as_closure(self.training_name, original_truncation_function)
        self.info_function = wrap_normal_function_as_closure(self.training_name, original_info_function)

        original_pre_build_hook = self.pre_build_hook
        original_post_build_hook = self.post_build_hook
        original_pre_reset_hook = self.pre_reset_hook
        original_post_reset_hook = self.post_reset_hook
        original_pre_last_hook = self.pre_last_hook
        original_post_last_hook = self.post_last_hook
        original_pre_step_hook = self.pre_step_hook
        original_post_step_hook = self.post_step_hook

        self.pre_build_hook = wrap_normal_function_as_closure(self.training_name, original_pre_build_hook)
        self.post_build_hook = wrap_normal_function_as_closure(self.training_name, original_post_build_hook)
        self.pre_reset_hook = wrap_reset_function_as_closure(self.training_name, original_pre_reset_hook)
        self.post_reset_hook = wrap_reset_function_as_closure(self.training_name, original_post_reset_hook)
        self.pre_last_hook = wrap_normal_function_as_closure(self.training_name, original_pre_last_hook)
        self.post_last_hook = wrap_normal_function_as_closure(self.training_name, original_post_last_hook)
        self.pre_step_hook = wrap_normal_function_as_closure(self.training_name, original_pre_step_hook)
        self.post_step_hook = wrap_normal_function_as_closure(self.training_name, original_post_step_hook)

        if self.use_shared_obs:
            original_shared_obs_function = self.shared_obs_function
            self.shared_obs_function = wrap_normal_function_as_closure(self.training_name, original_shared_obs_function)

    def add_robot(self, config: RobotConfig) -> None:
        """Add a robot to this bundle."""
        if len(self.robot_configs) != 0:
            if config.frequency != self.frequency:
                raise ValueError(
                    f"Robot {config.name} frequency {config.frequency} != bundle frequency {self.frequency}"
                )
            if config.name in self.robot_configs.keys():
                raise ValueError(f"Robot {config.name} already in bundle {self.training_name} or have a name conflict")

        self.robot_configs[config.name] = config

    def detect_obs_spaces(self, AEC_env):
        # Detect observation space for each robot
        if self.observation_spaces is None:
            self.observation_spaces = {}
            if self.sample_obs is None:
                # Get observations and remove the n_envs dimension (take first env)
                obs_dict = self.obs_function(AEC_env)
                with_n_envs_dim = True
                self.sample_obs = {}
                for robot_name, obs in obs_dict.items():
                    if obs.shape[0] != AEC_env.n_envs:
                        with_n_envs_dim = False
                for robot_name, obs in obs_dict.items():
                    if with_n_envs_dim:
                        self.sample_obs[robot_name] = obs[0]
                    else:
                        self.sample_obs[robot_name] = obs

            for robot_name, obs in self.sample_obs.items():
                self.observation_spaces[robot_name] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=obs.shape, dtype=obs.cpu().numpy().dtype
                )
                robot_config = self.robot_configs[robot_name]

                robot_config.sample_obs = obs
                robot_config.observation_space = self.observation_spaces[robot_name]

        # Detect shared observation space
        if self.shared_observation_space is None:
            if self.sample_shared_obs is None:
                if self.shared_obs_function is not None:
                    self.sample_shared_obs = self.shared_obs_function(AEC_env)
                    if self.sample_shared_obs.shape[0] == AEC_env.n_envs:
                        self.sample_shared_obs = self.sample_shared_obs[0]

                    self.shared_observation_space = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=self.sample_shared_obs.shape,
                        dtype=self.sample_shared_obs.cpu().numpy().dtype,
                    )

    @property
    def n_robots(self) -> int:
        """Get number of robots in this bundle."""
        return len(self.robot_configs)

    @property
    def robot_names(self) -> List[str]:
        """Get names of robots in this bundle."""
        return list(self.robot_configs.keys())

    @property
    def name(self) -> str:
        """Alias for training_name"""
        return self.training_name

    @property
    def frequency(self) -> float:
        """Get frequency of this bundle."""
        if len(self.robot_configs) == 0:
            return None
        return next(iter(self.robot_configs.values())).frequency

    @property
    def is_joint(self) -> bool:
        return self.training_type == "JOINT"

    @property
    def is_shared(self) -> bool:
        return self.training_type == "SHARED"

    @property
    def use_shared_obs(self) -> bool:
        return self.shared_observation_space is not None or self.shared_obs_function is not None

    @classmethod
    def single_robot_training_config(
        cls: Type["TrainingConfig"],
        robot_config: RobotConfig = None,
        **kwargs,
    ):
        if robot_config is None:
            raise ValueError("Must provide robot_config")

        return cls(robot_cfgs=[robot_config], **kwargs)

    @classmethod
    def joint_robots_training_config(
        cls: Type["TrainingConfig"],
        robot_configs: List[RobotConfig] = None,
        obs_function: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
        truncation_function: Optional[Callable] = None,
        info_function: Optional[Callable] = None,
        **kwargs,
    ):
        """Auto concat the robot's observation and actions"""
        tmp_instance = cls(robot_cfgs=robot_configs, **kwargs)
        # Just to deduce the training name
        training_name = tmp_instance.training_name

        if obs_function is None:

            def joint_obs_function(AEC_env):
                # Get individual robot observations
                training_cfg = AEC_env.training_configs[training_name]
                dict_result = training_cfg.robots_obs_function(AEC_env)

                # Try to concatenate observations on dim=1 (after n_envs)
                try:
                    obs_list = list(dict_result.values())
                    concatenated_obs = torch.cat(obs_list, dim=1)
                    return {training_cfg.robot_names[0]: concatenated_obs}
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Cannot concatenate observations for '{training_name}'. "
                        f"All robot observations must have compatible shapes for concatenation on dim=1. "
                        f"Robot obs shapes: {[obs.shape for obs in obs_list]}. Error: {e}"
                    )

            obs_function = joint_obs_function

        if reward_function is None:

            def joint_reward_function(AEC_env):
                training_cfg = AEC_env.training_configs[training_name]
                dict_result = training_cfg.robots_reward_function(AEC_env)
                # merge the nested dict
                accumulated_reward_dict = {}
                for _, reward_dict in dict_result.items():
                    for k, v in reward_dict.items():
                        if k in accumulated_reward_dict:
                            accumulated_reward_dict[k] += v
                        else:
                            accumulated_reward_dict[k] = v.clone()
                AEC_env.logger.warning(
                    f"Joint reward function for {training_name} is not implemented, using default accumulation logic, but may not be what you want"
                )
                return {training_cfg.robot_names[0]: accumulated_reward_dict}

            reward_function = joint_reward_function

        if truncation_function is None:

            def joint_truncation_function(AEC_env):
                training_cfg = AEC_env.training_configs[training_name]
                dict_result = training_cfg.robots_truncation_function(AEC_env)
                result = torch.any(torch.stack(list(dict_result.values())), dim=0)
                return {training_cfg.robot_names[0]: result}

            truncation_function = joint_truncation_function

        if info_function is None:

            def joint_info_function(AEC_env):
                training_cfg = AEC_env.training_configs[training_name]
                dict_result = training_cfg.robots_info_function(AEC_env)
                infos = {}
                for robot_name, robot_config in training_cfg.robot_configs.items():
                    for info_name, info_value in dict_result[robot_name].items():
                        if info_name not in infos:
                            infos[info_name] = info_value
                        else:
                            raise ValueError(
                                f"Info '{info_name}' is provided by multiple robots in '{training_name}'. Don't know how to handle this"
                            )
                return infos

            info_function = joint_info_function

        return cls(
            robot_cfgs=robot_configs,
            training_type="JOINT",
            obs_function=obs_function,
            reward_function=reward_function,
            truncation_function=truncation_function,
            info_function=info_function,
            **kwargs,
        )

    @classmethod
    def shared_parameter_robots_training_config(
        cls: Type["TrainingConfig"],
        base_config: RobotConfig,
        initial_positions: Optional[List[List[float]]] = None,
        initial_orientations: Optional[List[List[float]]] = None,
        initial_velocities: Optional[List[List[float]]] = None,
        initial_angular_velocities: Optional[List[List[float]]] = None,
        num_dumplicates: int = -1,
        training_name: Optional[str] = None,
        **kwargs,
    ):
        original_training_name = training_name
        if training_name is None:
            training_name = f"{base_config.name}_x{num_dumplicates}_shared"

        lengths = []
        for l in [initial_positions, initial_orientations, initial_velocities, initial_angular_velocities]:
            if l is not None:
                lengths.append(len(l))

        if num_dumplicates != -1:
            lengths.append(num_dumplicates)

        # Check if all lists have the same length
        if len(set(lengths)) > 1:
            raise ValueError("All lists' length and num_dumplicates must be the same")

        num_dumplicates = lengths[0]

        if initial_positions is None:
            # Create an array of initial positions on x y plane
            bbox = base_config.urdf_extractor.get_overall_bounding_box()
            grid_size = math.ceil(math.sqrt(num_dumplicates))
            spacing_x = (bbox['max'][0] - bbox['min'][0]) * 1.5
            spacing_y = (bbox['max'][1] - bbox['min'][1]) * 1.5
            base_pos = base_config.initial_position

            initial_positions = [
                [
                    base_pos[0] + (i % grid_size - (grid_size - 1) / 2) * spacing_x,
                    base_pos[1] + (i // grid_size - (grid_size - 1) / 2) * spacing_y,
                    base_pos[2],
                ]
                for i in range(num_dumplicates)
            ]
        if initial_orientations is None:
            initial_orientations = [base_config.initial_orientation] * num_dumplicates
        if initial_velocities is None:
            initial_velocities = [base_config.initial_velocity] * num_dumplicates
        if initial_angular_velocities is None:
            initial_angular_velocities = [base_config.initial_angular_velocity] * num_dumplicates

        # Create individual robot configs with different names and positions
        robots_configs = []
        for i, (pos, orn, vel, ang_vel) in enumerate(
            zip(initial_positions, initial_orientations, initial_velocities, initial_angular_velocities)
        ):
            robot_config_args = copy.deepcopy(base_config._initial_args)

            # Filter out fields that have init=False (like _initial_args)
            robot_config_fields = {f.name for f in fields(RobotConfig) if f.init}
            robot_config_args = {k: v for k, v in robot_config_args.items() if k in robot_config_fields}

            robot_config_args["name"] = f"{base_config.name}_{i}"
            robot_config_args["initial_position"] = pos
            robot_config_args["initial_orientation"] = orn
            robot_config_args["initial_velocity"] = vel
            robot_config_args["initial_angular_velocity"] = ang_vel

            robot_config = RobotConfig(**robot_config_args)
            robots_configs.append(robot_config)

        return cls(
            robot_cfgs=robots_configs,
            training_name=training_name,
            training_type="SHARED",
            **kwargs,
        )
