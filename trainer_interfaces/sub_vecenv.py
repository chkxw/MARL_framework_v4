#!/usr/bin/env python3
"""
SubVecEnv Classes for Genesis MARL Framework V4
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from configs.RobotConfig import RobotConfig
from configs.TrainingConfig import TrainingConfig
from marl_logging import get_class_logger
from utils import RefDict


class SubVecEnv(ABC):
    """SubVecEnv is the interface between AEC Env and training code."""

    def __init__(
        self,
        training_config: TrainingConfig,
        mother_env: Any,
    ):
        """Initialize SubVecEnv.

        Args:
            training_name: Name of the training configuration
            training_config_bundle: TrainingConfigBundle for this subVecEnv
            mother_env: Reference to the main AEC environment
        """
        # Initialize independent class-specific logger using training name
        self.logger = get_class_logger("SubVecEnv", training_config.training_name, level="INFO")

        self.training_name = training_config.training_name
        self.training_cfg = training_config
        self.mother_env = mother_env

        # Basic properties
        self.n_envs = mother_env.n_envs
        self.device = mother_env.device

        # Data buffers for receiving from AEC
        self.obs_buffer: Dict[str, torch.Tensor] = RefDict()
        self.shared_obs_buffer: torch.Tensor = None
        self.rew_buffer: Dict[str, torch.Tensor] = RefDict()
        self.term_buffer: torch.Tensor = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        self.trunc_buffer: Dict[str, torch.Tensor] = RefDict()
        self.info_buffer: Dict[str, Dict] = RefDict()
        self.episode_length_buffer: torch.Tensor = torch.zeros(
            (self.n_envs), device=self.device
        )  # This is for recieving episode lengths, not the one required by RSL_RL

        # Coordinator for synchronization with Agent
        self.coordinator = mother_env.coordinator

        self.logger.debug(f"Initializing buffers for {self.training_cfg.n_robots} robots")

        # Initialize buffers
        if self.training_cfg.use_shared_obs:
            self.shared_obs_buffer: torch.Tensor = torch.zeros(
                (self.n_envs,) + self.training_cfg.shared_observation_space.shape, device=self.device
            )
        for i, (robot_name, robot_cfg) in enumerate(self.training_cfg.robot_configs.items()):
            if self.training_cfg.is_joint and i != 0:
                # Set reference to first robot if is joint
                first_robot_name = self.training_cfg.robot_names[0]
                self.obs_buffer.add_ref(robot_name, first_robot_name)
                self.rew_buffer.add_ref(robot_name, first_robot_name)
                self.trunc_buffer.add_ref(robot_name, first_robot_name)
                self.info_buffer.add_ref(robot_name, first_robot_name)
                continue

            # Observation spaces are already detected at this point
            obs_shape = (self.n_envs,) + robot_cfg.observation_space.shape
            self.obs_buffer[robot_name] = torch.zeros(obs_shape, device=self.device)
            self.rew_buffer[robot_name] = torch.zeros((self.n_envs), device=self.device)
            self.trunc_buffer[robot_name] = torch.zeros((self.n_envs), dtype=torch.bool, device=self.device)
            self.info_buffer[robot_name] = {}

    """
    RL CORE
    """

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """Reset and get initial observations."""
        self.logger.debug(f"SubVecEnv {self.training_name} reset()")

        # Get initial obs
        self._wait_for_data()

        data = self._dict_to_tensor()

        return self._process_data(data, is_reset=True)

    def step(
        self, actions: torch.Tensor, require_data=True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        self.logger.debug(f"SubVecEnv {self.training_name} step() with actions: {actions.shape}")

        # Check action type, if is numpy, convert to tensor
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)

        if self.is_shared:
            actions = self._reshape_shared_actions(actions)
        elif self.is_joint:
            actions = self._split_joint_actions(actions)
        else:
            # Shape based check, if we only have one robot, allows [n_effective_envs, action_dim] shape, otherwise expect [n_effective_envs, n_effective_robots, action_dim]
            action_shape = actions.shape
            if len(action_shape) == 2 and action_shape[0] == self.n_effective_envs:
                actions = {self.robot_names[0]: actions}
            elif (
                len(action_shape) == 3
                and action_shape[0] == self.n_effective_envs
                and action_shape[1] == self.n_effective_robots
            ):
                actions = {self.robot_names[i]: actions[:, i] for i in range(self.n_effective_robots)}
            else:
                raise ValueError(f"Invalid actions shape: {action_shape}")

        processed_actions = self._process_actions(actions)

        self._send_actions(processed_actions)

        # Wait for new data
        if not require_data:
            return None

        self._wait_for_data()

        data = self._dict_to_tensor()

        return self._process_data(
            data,
            is_reset=False,
        )

    def _wait_for_data(self):
        """Wait for Agent to provide new data."""
        if self.coordinator is None:
            raise RuntimeError(f"SubVecEnv {self.training_name} coordinator not set")

        self.logger.debug(f"SubVecEnv {self.training_name} waiting for data")

        self.coordinator.wait_for_state(self.training_name)
        self.logger.debug(f"SubVecEnv {self.training_name} received data")

    def _send_actions(self, actions: torch.Tensor):
        """Send actions to Agent and signal completion.

        Args:
            actions: Action tensor to send
        """
        if self.coordinator is None:
            raise RuntimeError(f"SubVecEnv {self.training_name} coordinator not set")

        for robot_name, robot_action in actions.items():
            self.mother_env.action_buffers[robot_name] = robot_action  # Do we need to copy here?

        # Signal Agent that actions are ready
        self.coordinator.set_finished(self.training_name)

    def _process_actions(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Override this method to apply any customed processing to actions
        """
        return actions

    def _process_data(
        self, data, is_reset=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict, torch.Tensor]:
        """
        Override this method to apply any customed processing to data or return customed data format
        """
        obs, shared_obs, rewards, truncations, terminations, infos, episode_lengths = data

        """Process the recieved data"""
        if is_reset:
            return obs
        return obs, rewards, truncations, terminations, infos

    # Joint / Shared processing
    def _dict_to_tensor(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict, torch.Tensor]:
        data = (
            self.obs_buffer,
            self.shared_obs_buffer,
            self.rew_buffer,
            self.trunc_buffer,
            self.term_buffer,
            self.info_buffer,
            self.episode_length_buffer,
        )

        if self.is_shared:
            data = self._cat_to_n_effective_envs(data)
        elif self.is_joint:
            # Either shared or single robot, only first robot have valide data
            data = (
                self.obs_buffer[self.robot_names[0]],
                self.shared_obs_buffer,
                self.rew_buffer[self.robot_names[0]],
                self.trunc_buffer[self.robot_names[0]],
                self.term_buffer,
                self.info_buffer[self.robot_names[0]],
                self.episode_length_buffer,
            )
        else:
            obs_list = []
            rew_list = []
            trunc_list = []
            for robot_name in self.robot_names:
                obs_list.append(self.obs_buffer[robot_name])
                rew_list.append(self.rew_buffer[robot_name])
                trunc_list.append(self.trunc_buffer[robot_name])
            stacked_obs = torch.stack(obs_list)
            stacked_rew = torch.stack(rew_list)
            stacked_trunc = torch.stack(trunc_list)

            data = (
                stacked_obs,
                self.shared_obs_buffer,
                stacked_rew,
                stacked_trunc,
                self.term_buffer,
                self.info_buffer,
                self.episode_length_buffer,
            )

        return data

    def _split_joint_actions(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process actions for joint training - split actions among robots."""
        # Split actions based on each robot's action dimension
        robot_actions = {}
        start_idx = 0

        for robot_name, robot_config in self.robot_configs.items():
            action_dim = np.prod(robot_config.action_space.shape)
            end_idx = start_idx + action_dim

            robot_actions[robot_name] = actions[:, start_idx:end_idx]
            start_idx = end_idx

        return robot_actions

    def _reshape_shared_actions(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Reshape the (n_robots * n_envs, *action_shape)  to (n_envs, *action_shape) per robot
        """
        actions_reshaped = actions.view(self.n_robots, self.n_envs, -1)

        robot_actions = {}
        for i, robot_name in enumerate(self.training_cfg.robot_configs.keys()):
            robot_actions[robot_name] = actions_reshaped[i]

        return robot_actions

    def _cat_to_n_effective_envs(self, data) -> torch.Tensor:
        """
        Used for shared parameter training.
        By reshaping n_robots dim to n_envs dim, we can treat each robot as one parallel environment, the training framework will naturally use the same network to process them
        """
        obs, shared_obs, rewards, truncations, terminations, infos, episode_lengths = data

        obs = torch.cat(list(obs.values()), dim=0)
        rewards = torch.cat(list(rewards.values()), dim=0)
        truncations = torch.cat(list(truncations.values()), dim=0)
        termination_repeat_pattern = [self.n_robots] + [1] * (terminations.dim() - 1)
        terminations = terminations.repeat(*termination_repeat_pattern)

        if self.training_cfg.use_shared_obs:
            repeat_times = [self.n_robots] + [1] * shared_obs.ndim
            shared_obs = shared_obs.unsqueeze(0).repeat(*repeat_times)

        # Requirement: all info dicts have the same keys and value shapes
        keys = list(infos[self.robot_names[0]].keys())
        value_shapes = [v.shape for k, v in infos[self.robot_names[0]].items()]
        # Check if all info dicts have the same keys
        for robot_name in self.robot_names[1:]:
            if (
                keys != list(infos[robot_name].keys())
                or [v.shape for k, v in infos[robot_name].items()] != value_shapes
            ):
                raise ValueError(
                    f"When using shared parameter training, expects all info dicts to have exact same keys and value shapes, got {keys} and {list(infos[robot_name].keys())}"
                )

        concatenated_infos = {}
        for k in keys:
            tensors_to_cat = []
            for r in self.robot_names:
                tensor = infos[r][k]
                # If tensor is 0-dimensional, add a dimension
                if tensor.dim() == 0:
                    tensor = tensor.unsqueeze(0)
                tensors_to_cat.append(tensor)
            concatenated_infos[k] = torch.cat(tensors_to_cat, dim=0)

        # Special handling for episode length reshape
        reshaped_episode_lengths = (
            episode_lengths.unsqueeze(0).expand(self.n_robots, -1).reshape(-1, self.n_robots * self.n_envs).squeeze(0)
        )
        return (obs, shared_obs, rewards, truncations, terminations, concatenated_infos, reshaped_episode_lengths)

    """
    HELPERS
    """

    @property
    def n_robots(self) -> int:
        return self.training_cfg.n_robots

    @property
    def n_effective_envs(self) -> int:
        return self.n_envs if not self.is_shared else self.n_robots * self.n_envs

    @property
    def n_effective_robots(self) -> int:
        return self.n_robots if not self.is_shared else 1

    @property
    def max_episode_length(self) -> int:
        return int(self.mother_env.max_episode_length_s * self.training_cfg.frequency)

    @property
    def robot_configs(self) -> List[RobotConfig]:
        return self.training_cfg.robot_configs

    @property
    def robot_names(self) -> List[str]:
        return self.training_cfg.robot_names

    @property
    def is_joint(self) -> bool:
        return self.training_cfg.is_joint

    @property
    def is_shared(self) -> bool:
        return self.training_cfg.is_shared
    @property 
    def tensorboard_writer(self):
        """ The tensorboard writer used by specific trainer"""
        if hasattr(self, "_tensorboard_writer"):
            return self._tensorboard_writer
        else:
            raise RuntimeError(f"tensorboard writer of {self.training_cfg.training_name} not set")

    @tensorboard_writer.setter
    def tensorboard_writer(self, writer):
        self._tensorboard_writer = writer