#!/usr/bin/env python3
"""
HARL Training Interface for Genesis MARL Framework V4

This module provides integration between the Genesis MARL framework and HARL
(Heterogeneous-Agent Reinforcement Learning) algorithms including HAPPO, HATRPO, HAA2C.
"""

from dataclasses import asdict, dataclass, field
import os
import time
import threading
from typing import Callable, Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import gymnasium as gym

from configs.RobotConfig import RobotConfig
from configs.TrainingConfig import TrainingConfig
from trainer_interfaces.sub_vecenv import SubVecEnv
from trainer_interfaces.harl_patcher import patch_harl_envs, register_custom_env
from marl_logging import get_class_logger
from utils import RefDict

AECEnv = Any
RNAME = str  # Robot name


@dataclass
class HARLTrainConfig:
    """Training configuration"""

    n_rollout_threads: int = 1  # Keep as 1 to avoid multiprocessing
    num_env_steps: int = -1  # Need to be set manually after init
    episode_length: int = 200
    log_interval: int = 5
    eval_interval: int = 25
    use_valuenorm: bool = True
    use_linear_lr_decay: bool = False
    use_proper_time_limits: bool = True
    model_dir: Optional[str] = None


@dataclass
class HARLModelConfig:
    """Model configuration"""

    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    activation_func: str = "relu"
    use_feature_normalization: bool = True
    initialization_method: str = "orthogonal_"
    gain: float = 0.01
    use_naive_recurrent_policy: bool = False
    use_recurrent_policy: bool = False
    recurrent_n: int = 1
    data_chunk_length: int = 10
    lr: float = 0.0003
    critic_lr: float = 0.0003
    opti_eps: float = 0.00001
    weight_decay: float = 0
    std_x_coef: float = 1
    std_y_coef: float = 0.5


@dataclass
class HARLAlgoConfig:
    """Algorithm configuration"""

    ppo_epoch: int = 5
    critic_epoch: int = 5
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    actor_num_mini_batch: int = 1
    critic_num_mini_batch: int = 1
    entropy_coef: float = 0.01
    value_loss_coef: float = 1
    use_max_grad_norm: bool = True
    max_grad_norm: float = 10.0
    use_gae: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    use_huber_loss: bool = True
    use_policy_active_masks: bool = True
    huber_delta: float = 10.0
    action_aggregation: str = "prod"
    share_param: bool = False
    fixed_order: bool = False


@dataclass
class HARLEvalConfig:
    """Evaluation configuration"""
    use_eval: bool = False
    n_eval_rollout_threads: int = 1
    eval_episodes: int = 10


@dataclass
class HARLRenderConfig:
    """Render configuration"""

    use_render: bool = False
    render_episodes: int = 1


@dataclass
class HARLSeedConfig:
    """Seed configuration"""

    seed_specify: bool = True
    seed: int = 1


@dataclass
class HARLDeviceConfig:
    """Device configuration"""

    cuda: bool = True
    cuda_deterministic: bool = True
    torch_threads: int = 4


@dataclass
class HARLLoggerConfig:
    """Logger configuration"""

    log_dir: str = "./logs"


@dataclass
class HARLRunnerConfig:
    algo: str = "happo"
    env: str = "custom"  # Don't change
    exp_name: str = None  # Need to set manually


@dataclass
class HARLAlgorithmConfig:
    model: HARLModelConfig = field(default_factory=HARLModelConfig)
    algo: HARLAlgoConfig = field(default_factory=HARLAlgoConfig)
    train: HARLTrainConfig = field(default_factory=HARLTrainConfig)
    eval: HARLEvalConfig = field(default_factory=HARLEvalConfig)
    render: HARLRenderConfig = field(default_factory=HARLRenderConfig)
    logger: HARLLoggerConfig = field(default_factory=HARLLoggerConfig)
    seed: HARLSeedConfig = field(default_factory=HARLSeedConfig)
    device: HARLDeviceConfig = field(default_factory=HARLDeviceConfig)


@dataclass
class HARLEnvConfig:
    """HARL Environment Configuration"""
    custom_type: str = "genesis"  # Points to our registered environment
    state_type: str = "EP"  # Environment Provided (EP) or Feature Pruned (FP)


@dataclass
class HARLConfig:
    runner_config: HARLRunnerConfig = field(default_factory=HARLRunnerConfig)
    algorithm_config: HARLAlgorithmConfig = field(default_factory=HARLAlgorithmConfig)
    env_config: HARLEnvConfig = field(default_factory=HARLEnvConfig)


class HARLEnv(SubVecEnv):
    """Environment wrapper for HARL algorithms.

    This class adapts the Genesis AEC environment to the interface expected by HARL runners.
    It properly inherits from SubVecEnv and only overrides _process_data and _process_actions.

    HARL format requirements:
    - reset() returns (obs, share_obs, available_actions)
    - step() returns (obs, share_obs, rewards, dones, infos, available_actions)
    - obs/share_obs: (n_rollout_threads, n_agents, obs_dim) as numpy arrays
    - rewards: (n_rollout_threads, n_agents, 1) as numpy arrays
    - dones: (n_rollout_threads, n_agents) as numpy arrays
    """

    def __init__(self, training_config: Any, mother_env: Any, **kwargs):
        super().__init__(training_config=training_config, mother_env=mother_env)

        self.logger = get_class_logger("HARLEnv", self.training_name, level="INFO")

        # Validate robot configurations for HARL compatibility
        for robot_name, robot_config in self.robot_configs.items():
            if robot_config.observation_space is None:
                continue  # Not detected yet
            if len(robot_config.observation_space.shape) > 1:
                raise ValueError(
                    f"HARL expects 1D observation space, but {robot_name} got {robot_config.observation_space.shape}"
                )
            if len(robot_config.action_space.shape) > 1:
                raise ValueError(
                    f"HARL expects 1D action space, but {robot_name} got {robot_config.action_space.shape}"
                )

        # Set up properties expected by HARL runners
        self.num_agents = self.n_robots
        # Create observation/action space lists (HARL expects lists)
        self.observation_space = []
        self.action_space = []

        for robot_name in self.robot_names:
            robot_config = self.robot_configs[robot_name]
            self.observation_space.append(robot_config.observation_space)
            self.action_space.append(robot_config.action_space)

        # Shared observation space (for centralized critic)
        if self.training_cfg.shared_observation_space is not None:
            # HARL expects share_observation_space as a list with same element for all agents
            self.share_observation_space = [self.training_cfg.shared_observation_space] * self.num_agents
        else:
            # Use individual observations as shared observations
            self.share_observation_space = self.observation_space.copy()

        self.logger.info(f"HARLEnv initialized with {self.num_agents} agents")
        self.logger.info(f"Observation spaces: {[obs.shape for obs in self.observation_space]}")
        self.logger.info(f"Action spaces: {[act.shape for act in self.action_space]}")
        self.logger.info(f"Shared observation spaces: {[share.shape for share in self.share_observation_space]}")

    def _process_actions(self, actions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process actions from HARL format to SubVecEnv format.

        Args:
            actions: Dict mapping robot names to action tensors

        Returns:
            Processed actions ready for SubVecEnv
        """
        # No special processing needed - just pass through
        original_action = actions[self.robot_names[0]]
        actions = {}
        for i, robot_name in enumerate(self.robot_names):
            actions[robot_name] = original_action[:, i, :]
        return actions

    def _process_data(self, data, is_reset=False):
        """Convert SubVecEnv data format to HARL format.

        Args:
            data: (obs, shared_obs, rewards, truncations, terminations, infos, episode_lengths)
            is_reset: Whether this is a reset call

        Returns:
            For reset: (obs, share_obs, available_actions)
            For step: (obs, share_obs, rewards, dones, infos, available_actions)
        """
        obs, shared_obs, rewards, truncations, terminations, infos, episode_lengths = data

        # Convert observations to HARL format: (n_envs, n_agents, obs_dim)
        obs_list = []
        for robot_name in self.robot_names:
            obs_list.append(obs[robot_name].cpu().numpy())
        obs_np = np.stack(obs_list, axis=1)  # Stack along agent dimension

        # Convert shared observations to HARL format
        if shared_obs is not None:
            # TODO: Repeat or not based on EP/FP
            share_obs_np = shared_obs.cpu().numpy()
            # Repeat for each agent: (n_envs, n_agents, share_obs_dim)
            share_obs_np = np.expand_dims(share_obs_np, axis=1).repeat(self.num_agents, axis=1)
        else:
            # Use individual observations as shared observations
            share_obs_np = obs_np

        # Available actions (None for continuous control)
        available_actions = [None] * self.num_agents

        if is_reset:
            return obs_np, share_obs_np, available_actions

        # Convert rewards to HARL format: (n_envs, n_agents, 1)
        rewards_list = []
        for robot_name in self.robot_names:
            rewards_list.append(rewards[robot_name].cpu().numpy())
        rewards_np = np.stack(rewards_list, axis=1)
        rewards_np = np.expand_dims(rewards_np, axis=-1)  # Add reward dimension

        # Convert dones to HARL format: (n_envs, n_agents) - use truncations OR terminations
        dones_list = []
        for robot_name in self.robot_names:
            robot_truncations = truncations[robot_name].cpu().numpy()
            robot_terminations = terminations.cpu().numpy()  # Global termination
            dones_list.append(robot_truncations | robot_terminations)
        dones_np = np.stack(dones_list, axis=1)

        # Convert infos to HARL format: List of length n_envs, each element is dict with n_agents entries
        infos_harl = []
        for env_idx in range(self.n_envs):
            env_info = {}
            for agent_idx, robot_name in enumerate(self.robot_names):
                # HARL expects integer keys, not string keys
                env_info[agent_idx] = infos[robot_name][env_idx]
            infos_harl.append(env_info)
            # TODO: This logic is wrong

        return obs_np, share_obs_np, rewards_np, dones_np, infos_harl, available_actions

    def close(self):
        """Close environment."""
        self.logger.info(f"Closing HARLEnv {self.training_name}")
        # No specific cleanup needed for our wrapper
        pass

    def seed(self, seed: int):
        """Set random seed."""
        self.logger.info(f"Setting seed {seed} for HARLEnv {self.training_name}")
        # Seed is handled by parent environment
        pass
    
    @property
    def trainer_name(self):
        return "HARL"

def harl_training_thread_function(subvecenv: HARLEnv, max_iterations: int):
    """Training function that runs HARL algorithms in a separate thread."""

    training_config = subvecenv.training_cfg
    training_name = training_config.training_name

    logger = get_class_logger("HARLTrainer", training_name, level="INFO")
    logger.info(f"🚀 Starting HARL training thread for {training_name}")

    try:
        # Apply HARL patches to support custom environments
        logger.info("Applying HARL patches for custom environment support...")
        patch_harl_envs()

        # Register our HARLEnv instance as a custom environment
        register_custom_env("genesis", subvecenv)
        logger.info(f"Registered HARLEnv instance as custom environment: genesis")

        # Convert training config to HARL format
        harl_config: HARLConfig = training_config.harl_config

        runner_args = asdict(harl_config.runner_config)
        algo_args = asdict(harl_config.algorithm_config)
        env_args = asdict(harl_config.env_config)

        # Update training parameters
        algo_args["train"]["num_env_steps"] = max_iterations * algo_args["train"]["episode_length"] * subvecenv.n_envs
        algo_args["train"]["n_rollout_threads"] = subvecenv.n_envs

        # Import and create HARL runner
        from harl.runners import RUNNER_REGISTRY

        # Create runner - patched HARL will automatically use our registered environment
        runner = RUNNER_REGISTRY[runner_args["algo"]](runner_args, algo_args, env_args)

        logger.info(f"🎯 Starting HARL training for {training_name}...")

        # Run training
        runner.run()

        logger.info(f"✅ HARL training completed for {training_name}")

    except Exception as e:
        logger.error(f"❌ HARL training failed for {training_name}: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


@dataclass
class HARLTrainingConfig(TrainingConfig):
    """Configuration for HARL training."""

    harl_config: HARLConfig = field(default_factory=HARLConfig)

    # Fields required by TrainingConfig interface
    trainer_name: str = "HARL"
    trainer_env_factory: Callable[[AECEnv], HARLEnv] = None
    trainer_launcher: Callable[[HARLEnv, int], None] = harl_training_thread_function

    def __post_init__(self, robot_cfgs):
        super().__post_init__(robot_cfgs)
        if self.harl_config.runner_config.exp_name is None:
            self.harl_config.runner_config.exp_name = self.training_name
        if self.trainer_env_factory is None:
            training_config_ref = self  # Capture self in closure

            def trainer_env_factory(AEC_env):
                return HARLEnv(training_config=training_config_ref, mother_env=AEC_env)

            self.trainer_env_factory = trainer_env_factory
