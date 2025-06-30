#!/usr/bin/env python3
"""
SubVecEnv Classes for Genesis MARL Framework V4

This module implements the SubVecEnv classes that interface with RSL_RL training code.
SubVecEnvs handle obs/rew/termination/truncation/info format conversion between AEC
and RSL_RL, and manage data reshaping for different training paradigms.
"""

import torch
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from abc import ABC, abstractmethod

from genesis_logging import get_module_logger
from configs import TrainingConfigBundle, JointTrainingBundle
from utils import RefDict

logger = get_module_logger("sub_vecenv")


class SubVecEnv(ABC):
    """Base SubVecEnv class that interfaces with RSL_RL training code.

    The SubVecEnv class inherits from RSL_RL VecEnv interface and provides:
    - Communication with Agent through semaphores
    - Format conversion between AEC (obs,rew,termination,truncation,info) and RSL_RL (obs,rew,done,extra)
    - Data buffering and synchronization
    """

    def __init__(self, training_config_bundle: TrainingConfigBundle, mother_env: Any):  # Will be VectorizedAECEnv
        """Initialize SubVecEnv.

        Args:
            training_name: Name of the training configuration
            training_config_bundle: TrainingConfigBundle for this subVecEnv
            mother_env: Reference to the main AEC environment
        """
        self.training_name = training_config_bundle.training_name
        self.training_cfg = training_config_bundle
        self.mother_env = mother_env

        # Basic properties
        self.n_envs = mother_env.n_envs
        self.device = mother_env.device

        # Data buffers for receiving from AEC
        self.obs_buffer: Dict[str, torch.Tensor] = RefDict()
        self.rew_buffer: Dict[str, torch.Tensor] = RefDict()
        self.term_buffer: Dict[str, torch.Tensor] = RefDict()
        self.trunc_buffer: Dict[str, torch.Tensor] = RefDict()
        self.info_buffer: Dict[str, Dict] = RefDict()

        # Cached initial data for get_observations()
        self.initial_obs: Optional[torch.Tensor] = None
        self.initial_extra: Optional[Dict] = None

        # Semaphore for synchronization with Agent
        self.semaphore: Optional[threading.Semaphore] = None

        logger.info(f"Initializing SubVecEnv {self.training_name}")
        logger.info(f"  Bundle type: {type(self.training_cfg).__name__}")
        logger.info(f"  Number of robots: {self.training_cfg.n_robots}")
        logger.info(f"  Number of environments: {self.n_envs}")

        self._initialize_buffers()

        logger.debug(f"SubVecEnv {self.training_name} initialization complete")

    def _initialize_buffers(self):
        """Initialize data buffers based on bundle configuration."""
        # Observation and action dimensions will be known after obs detection
        self.obs_dim = self.training_cfg.observation_dim  # May be None initially
        self.action_dim = self.training_cfg.action_dim

        logger.debug(f"  Action dim: {self.action_dim}")
        logger.debug(f"  Obs dim: {self.obs_dim} (will be detected if None)")

        # Initialize buffers
        for i, robot_cfg in enumerate(self.training_cfg.robot_configs):
            if isinstance(self.training_cfg, JointTrainingBundle) and i != 0:
                # Set reference to first robot
                first_robot_name = self.training_cfg.robot_configs[0].name
                self.obs_buffer.add_ref(robot_cfg.name, first_robot_name)
                self.rew_buffer.add_ref(robot_cfg.name, first_robot_name)
                self.term_buffer.add_ref(robot_cfg.name, first_robot_name)
                self.trunc_buffer.add_ref(robot_cfg.name, first_robot_name)
                self.info_buffer.add_ref(robot_cfg.name, first_robot_name)
                continue
            self.obs_buffer[robot_cfg.name] = torch.zeros((self.n_envs, self.obs_dim), device=self.device)
            self.rew_buffer[robot_cfg.name] = torch.zeros((self.n_envs), device=self.device)
            self.term_buffer[robot_cfg.name] = torch.zeros((self.n_envs), device=self.device)
            self.trunc_buffer[robot_cfg.name] = torch.zeros((self.n_envs), device=self.device)
            self.info_buffer[robot_cfg.name] = {}

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """Reset and get initial observations."""
        logger.debug(f"SubVecEnv {self.training_name} reset()")

        # Wait for initial data from Agent
        self._wait_for_data()

        # Convert and cache initial data
        obs, rewards, dones, extras = self._process_pipeline(
            self.obs_buffer,
            self.rew_buffer,
            self.trunc_buffer,
            self.term_buffer,
            self.info_buffer,
        )

        self.initial_obs = obs
        self.initial_extra = extras

        logger.debug(f"SubVecEnv {self.training_name} reset complete, obs shape: {obs.shape}")
        return obs, extras

    def get_observations(self) -> tuple[torch.Tensor, Dict]:
        """Get cached initial observations."""
        if self.initial_obs is None or self.initial_extra is None:
            raise RuntimeError(f"SingleRobotSubVecEnv {self.training_name} not reset yet")
        return self.initial_obs, self.initial_extra

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        logger.debug(f"SubVecEnv {self.training_name} step() with actions: {actions.shape}")

        self._send_actions(actions)

        # Wait for new data
        self._wait_for_data()

        # Convert to RSL_RL format
        obs, rewards, dones, extras = self._process_pipeline(
            self.obs_buffer,
            self.rew_buffer,
            self.trunc_buffer,
            self.term_buffer,
            self.info_buffer,
        )

        logger.debug(f"SubVecEnv {self.training_name} step complete")
        return obs, rewards, dones, extras

    def _wait_for_data(self):
        """Wait for Agent to provide new data (wait for semaphore = 0)."""
        if self.semaphore is None:
            raise RuntimeError(f"SubVecEnv {self.training_name} semaphore not set")

        logger.debug(f"SubVecEnv {self.training_name} waiting for data (semaphore = 0)")
        self.semaphore.acquire()  # Wait for semaphore to be 0
        logger.debug(f"SubVecEnv {self.training_name} received data")

    def _send_actions(self, actions: Dict[str, torch.Tensor]):
        """Send actions to Agent and signal readiness (set semaphore = 1).

        Args:
            actions: Action tensor to send
        """
        if self.action_buffer is None:
            raise RuntimeError(f"SubVecEnv {self.training_name} action buffer not set")

        # Copy actions to the shared buffer
        for i, robot_cfg in enumerate(self.training_cfg.robot_configs):
            robot_name = robot_cfg.name
            if isinstance(self.training_cfg, JointTrainingBundle) and i != 0:
                continue
            self.mother_env.action_buffer[robot_name] = actions[robot_name]

        # Signal Agent that actions are ready (set semaphore to 1)
        logger.debug(f"SubVecEnv {self.training_name} sending actions: {actions.shape}")
        self.semaphore.release()  # Set semaphore to 1

    def _convert_to_rsl_rl_format(
        self, obs, rewards, truncations, terminations, infos
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Convert AEC format data to RSL_RL format.
        Args:
            obs: Observations tensor
            rewards: Rewards tensor
            truncations: Truncation flags tensor
            terminations: Termination flags tensor
            infos: Extra information dictionary
        Returns:
            obs: Observations tensor
            rewards: Rewards tensor
            dones: Done flags tensor (termination OR truncation)
            extras: Extra information dictionary
        """
        return obs, rewards, truncations | terminations, infos

    @abstractmethod
    def _process_pipeline(self, obs, rewards, truncations, terminations, infos) -> Tuple[torch.Tensor, Dict]:
        """Convert AEC format data to RSL_RL format."""
        pass


class SingleRobotSubVecEnv(SubVecEnv):
    """SubVecEnv for single robot training - direct pass-through implementation."""

    def __init__(self, training_config_bundle: TrainingConfigBundle, mother_env: Any):
        super().__init__(training_config_bundle, mother_env)

        if self.training_cfg.n_robots != 1:
            raise ValueError(f"SingleRobotSubVecEnv expects 1 robot, got {self.training_cfg.n_robots}")

        self.robot_name = self.training_cfg.robot_configs[0].name
        logger.info(f"SingleRobotSubVecEnv for robot: {self.robot_name}")

    def _process_pipeline(
        self, obs, rewards, truncations, terminations, infos
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        obs, rewards, dones, extras = self._convert_to_rsl_rl_format(
            obs[self.robot_name],
            rewards[self.robot_name],
            truncations[self.robot_name],
            terminations[self.robot_name],
            infos[self.robot_name],
        )

        return obs, rewards, dones, extras


class JointSubVecEnv(SubVecEnv):
    """SubVecEnv for joint training - handles concatenated observations and split actions."""

    def __init__(self, training_config_bundle: TrainingConfigBundle, mother_env: Any):
        super().__init__(training_config_bundle, mother_env)

        self.robot_names = [config.name for config in self.training_cfg.robot_configs]
        logger.info(f"JointSubVecEnv for robots: {self.robot_names}")

    def _process_pipeline(
        self, obs, rewards, truncations, terminations, infos
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        # Only the first robot is valid data, others are just references to it
        obs, rewards, dones, extras = self._convert_to_rsl_rl_format(
            obs[self.robot_names[0]],
            rewards[self.robot_names[0]],
            truncations[self.robot_names[0]],
            terminations[self.robot_names[0]],
            infos[self.robot_names[0]],
        )
        return obs, rewards, dones, extras

    def _convert_to_rsl_rl_format(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Convert joint training AEC data to RSL_RL format."""
        # Collect data from all robots
        robot_obs = []
        robot_rewards = []
        robot_terminations = []
        robot_truncations = []
        robot_infos = []

        for robot_name in self.robot_names:
            robot_obs.append(self.obs_buffer[robot_name])
            robot_rewards.append(self.rew_buffer[robot_name])
            robot_terminations.append(self.term_buffer[robot_name])
            robot_truncations.append(self.trunc_buffer[robot_name])
            if self.info_buffer:
                robot_infos.append(self.info_buffer.get(robot_name, {}))

        # Concatenate observations
        obs = self.training_cfg.concatenate_observations(robot_obs)

        # Sum rewards (or use bundle's reward function if defined)
        rewards = torch.sum(torch.stack(robot_rewards), dim=0)

        # Any robot termination/truncation = episode done
        terminations = torch.any(torch.stack(robot_terminations), dim=0)
        truncations = torch.any(torch.stack(robot_truncations), dim=0)
        dones = terminations | truncations

        # Combine infos (simplified - could be more sophisticated)
        extras = {
            'terminations': terminations,
            'truncations': truncations,
            'infos': robot_infos[0] if robot_infos else {},  # Use first robot's info
        }

        return obs, rewards, dones, extras


class SharedNetworkSubVecEnv(SubVecEnv):
    """SubVecEnv for shared network training - reshapes data to treat robots as separate environments."""

    def __init__(self, training_config_bundle: TrainingConfigBundle, mother_env: Any):
        super().__init__(training_config_bundle, mother_env)

        if self.training_cfg.n_robots < 2:
            raise ValueError(f"SharedNetworkSubVecEnv expects >= 2 robots, got {self.training_cfg.n_robots}")

        self.robot_names = [config.name for config in self.training_cfg.robot_configs]

        # For shared network, effective n_envs is multiplied by n_robots
        self.effective_n_envs = self.n_envs * self.training_cfg.n_robots

        logger.info(f"SharedNetworkSubVecEnv for robots: {self.robot_names}")
        logger.info(f"  Effective environments: {self.effective_n_envs}")

    def _process_pipeline(self, obs, rewards, truncations, terminations, infos):
        obs = torch.stack(list(obs.values()), dim=0)
        rewards = torch.stack(list(rewards.values()), dim=0)
        truncations = torch.stack(list(truncations.values()), dim=0)
        terminations = torch.stack(list(terminations.values()), dim=0)

        # Requirement: if one key appear in one info dict, it should appear in all info dicts!!!
        # TODO: Add checks here
        keys = list(infos[self.robot_names[0]].keys())
        raise NotImplementedError

        return obs, rewards, dones, extras
