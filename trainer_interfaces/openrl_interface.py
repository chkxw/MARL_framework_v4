#!/usr/bin/env python3
"""
OpenRL Training Interface for Genesis MARL Framework V4

This module provides integration between the Genesis MARL framework and OpenRL
for multi-agent reinforcement learning with MAPPO and other algorithms.
"""

from dataclasses import asdict, dataclass, field
import os
import shutil
from types import SimpleNamespace
from typing import Callable, Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym

from configs.RobotConfig import RobotConfig
from configs.TrainingConfig import TrainingConfig
from trainer_interfaces.sub_vecenv import SubVecEnv
from marl_logging import get_class_logger
from utils import RefDict

AECEnv = Any
RNAME = str  # Robot name


@dataclass
class OpenRLAlgorithmConfig:
    """PPO algorithm configuration - matches OpenRL's actual parameters"""

    # Core PPO parameters
    lr: float = 7e-4
    critic_lr: float = 7e-4
    opti_eps: float = 1e-5
    weight_decay: float = 0.0

    # PPO-specific
    clip_param: float = 0.2
    ppo_epoch: int = 5
    num_mini_batch: int = 4

    # Value function
    use_clipped_value_loss: bool = True
    value_loss_coef: float = 0.5
    use_huber_loss: bool = True
    huber_delta: float = 10.0

    # GAE parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    use_gae: bool = True

    # Regularization
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    # Training options
    use_linear_lr_decay: bool = False


@dataclass
class OpenRLPolicyConfig:
    """Network architecture configuration - matches OpenRL's actual parameters"""

    # Network architecture
    hidden_size: int = 64
    layer_N: int = 2  # OpenRL uses layer_N instead of hidden_layer
    activation_id: int = 3  # 0: tanh, 1: relu, 2: leaky_relu, 3: selu
    use_orthogonal: bool = True
    gain: float = 0.01

    # Recurrent network options
    use_recurrent_policy: bool = False
    use_naive_recurrent_policy: bool = False
    recurrent_N: int = 1
    data_chunk_length: int = 10

    # Multi-agent specific
    use_centralized_V: bool = True
    use_share_model: bool = True

    # Normalization options
    use_feature_normalization: bool = True
    use_popart: bool = True
    use_valuenorm: bool = False  # popart and valuenorm can not be set True simultaneously, usually popart is better
    use_adv_normalize: bool = True


@dataclass
class OpenRLRunnerConfig:
    """Training runner configuration - matches OpenRL's actual parameters"""

    # Core training setup
    algorithm_name: str = "mappo"
    env_name: str = "genesis"
    experiment_name: str = ""
    scenario_name: str = "genesis_marl"
    seed: int = 1

    # Training parameters
    num_env_steps: int = 10000000
    episode_length: int = 200
    n_rollout_threads: int = 32  # 32 core cpu
    n_eval_rollout_threads: int = 1

    # Logging and evaluation
    disable_wandb: bool = True  # This means enable tensorboard
    use_render: bool = False

    # Multi-agent parameters
    num_agents: int = 1  # Set by environment


@dataclass
class OpenRLTrainConfig:
    """Training loop configuration - simplified"""

    save_interval: int = 1000
    log_interval: int = 1
    use_eval: bool = False
    model_dir: Optional[str] = None


@dataclass
class OpenRLEnvConfig:
    """Environment configuration - simplified"""

    seed_specify: bool = True


@dataclass
class OpenRLDeviceConfig:
    """Device and computation configuration - simplified"""

    cuda: bool = True
    cuda_deterministic: bool = True
    torch_threads: int = 1


@dataclass
class OpenRLConfig(
    OpenRLAlgorithmConfig,
    OpenRLPolicyConfig,
    OpenRLRunnerConfig,
    OpenRLTrainConfig,
    OpenRLEnvConfig,
    OpenRLDeviceConfig,
):
    """Complete OpenRL configuration"""

    # Additional fields
    run_dir: str = "./logs"
    log_dir: str = "./logs"

    def to_openrl_args(self):
        """Convert this config to OpenRL args using OpenRL's own config parser.

        This method uses OpenRL's create_config_parser() and feeds it our dataclass
        as a dictionary, which should be the cleanest approach.
        """
        from openrl.configs.config import create_config_parser

        # Convert our dataclass to dictionary
        config_dict = asdict(self)

        # Create OpenRL's config parser
        parser = create_config_parser()

        # Parse using OpenRL's parser to get properly formatted args
        args = vars(parser.parse_args({}))

        args.update(config_dict)

        return SimpleNamespace(**args)


class OpenRLEnv(SubVecEnv):
    """Environment wrapper for OpenRL algorithms.

    This class adapts the Genesis AEC environment to the interface expected by OpenRL.
    It properly inherits from SubVecEnv and only overrides _process_data and _process_actions.

    OpenRL automatically handles multi-agent scenarios by detecting the environment type,
    so we need to provide the appropriate interface.
    """

    def __init__(self, training_config: Any, mother_env: Any):
        super().__init__(training_config=training_config, mother_env=mother_env)

        self.logger = get_class_logger("OpenRLEnv", self.training_name, level="INFO")

        self.agent = None  # The OpenRL agent class, should be set manually before training start

        # Create unified observation and action spaces
        # For multi-agent, OpenRL expects these to be the space for a single agent
        if self.is_joint:
            # Joint training: concatenate all observation/action spaces
            obs_dim = sum(np.prod(cfg.observation_space.shape) for cfg in self.robot_configs.values())
            act_dim = sum(np.prod(cfg.action_space.shape) for cfg in self.robot_configs.values())

            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        else:
            # Use first robot's spaces as reference (all should be same for shared param)
            first_robot_config = self.robot_configs[self.robot_names[0]]
            self.observation_space = first_robot_config.observation_space
            self.action_space = first_robot_config.action_space

        # Handle shared observations for centralized critic
        if self.training_cfg.use_shared_obs and self.training_cfg.shared_observation_space is not None:
            self.share_observation_space = self.training_cfg.shared_observation_space
            # For multi-agent, create a list of shared obs spaces
            self.share_observation_space_list = [self.share_observation_space] * self.agent_num
        else:
            self.share_observation_space = None
            self.share_observation_space_list = None

        # Additional properties for OpenRL multi-agent compatibility
        self.possible_agents = self.agents.copy()
        self.agent_name_mapping = {name: i for i, name in enumerate(self.agents)}

        # Properties expected by some OpenRL components
        self.world = None  # Some OpenRL envs have this
        self.scenario_name = training_config.training_name

        self.logger.info(f"OpenRLEnv initialized with {self.agent_num} agents")
        self.logger.info(f"Agents list: {self.agents}")
        self.logger.info(f"Observation space: {self.observation_space}")
        self.logger.info(f"Action space: {self.action_space}")
        if self.share_observation_space:
            self.logger.info(f"Shared observation space: {self.share_observation_space}")

    @property
    def trainer_name(self):
        return "OpenRL"

    @property
    def env_name(self):
        return self.training_name

    @property
    def agent_num(self):
        """Number of agents - critical for OpenRL's multi-agent detection."""
        return self.n_effective_robots

    @property
    def agents(self):
        """List of agent IDs - critical for OpenRL's multi-agent detection."""
        return self.robot_names

    @property
    def parallel_env_num(self):
        return self.n_effective_envs

    @property
    def is_vector_env(self):
        return True

    @property
    def use_monitor(self):
        return False

    def batch_rewards(self, buffer):
        return {}

    def close(self, **kwargs):
        pass

    def reset(self, seed=None):
        """Just for compability, as OpenRL will try to pass seed as an argument"""
        # Seed is ignored since our environment handles seeding differently
        return super().reset()

    def _process_actions(self, actions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process actions from OpenRL format to SubVecEnv format.

        OpenRL provides actions as tensors, we need to ensure they're properly
        distributed to each robot.

        Args:
            actions: Dict mapping robot names to action tensors

        Returns:
            Processed actions ready for SubVecEnv
        """
        # For shared parameter training, actions are already split by SubVecEnv
        # For joint training, actions are already split by SubVecEnv
        # Just pass through
        return actions

    def _process_data(self, data, is_reset=False):
        """Convert SubVecEnv data format to OpenRL format.

        OpenRL expects:
        - For reset: observations (and info dict)
        - For step: (obs, rewards, terminated, truncated, info)

        Args:
            data: (obs, shared_obs, rewards, truncations, terminations, infos, episode_lengths)
            is_reset: Whether this is a reset call

        Returns:
            Data in OpenRL expected format
        """
        obs, shared_obs, rewards, truncations, terminations, infos, episode_lengths = data  # episode_lengths unused
        # Default stacking is [n_robots, n_envs], but OpenRL expects [n_envs, n_robots]

        if self.is_shared:
            # Fill the missing n_robots dim at position 0
            obs = obs.unsqueeze(0)  # [n_envs, obs_dim] -> [1, n_envs, obs_dim]
            rewards = rewards.unsqueeze(0)  # [n_envs] -> [1, n_envs]
            truncations = truncations.unsqueeze(0)  # [n_envs] -> [1, n_envs]
            original_info = {**infos}
            del infos
            infos = {self.robot_names[0]: original_info}  # Wrap with a robot name

            if shared_obs is not None:
                shared_obs = shared_obs.unsqueeze(0)

        # Transpose to OpenRL format: [n_robots, n_envs, ...] -> [n_envs, n_robots, ...]
        obs = obs.transpose(0, 1)
        rewards = rewards.transpose(0, 1)

        # Handle terminations (always [n_envs] originally)
        terminations = terminations.unsqueeze(0).expand(self.n_effective_robots, -1)  # [n_envs] -> [n_robots, n_envs]

        # Combine terminations and truncations (both now [n_robots, n_envs])
        dones = terminations | truncations
        dones = dones.transpose(0, 1)  # [n_robots, n_envs] -> [n_envs, n_robots]

        # Add reward dimension for OpenRL (expects [n_envs, n_robots, 1])
        if rewards.dim() == 2:  # [n_envs, n_robots]
            rewards = rewards.unsqueeze(-1)  # -> [n_envs, n_robots, 1]

        obs_np = obs.cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        dones_np = dones.cpu().numpy()

        # OpenRL expect list of dictional for info
        # openrl_info is List[Dict[field_name,Dict[robot_name,field_value]]]
        openrl_info = [{}] * self.n_effective_envs
        for robot_name, key_infos in infos.items():
            robot_idx = self.robot_names.index(robot_name)
            for info_name, info_tensor in key_infos.items():
                if not isinstance(info_tensor, torch.Tensor):
                    raise ValueError(f"Info '{info_name}' of robot '{robot_name}' is not torch.Tensor")
                if info_name == "termination_obs":
                    info_name = "final_observation"  # Use the OpenRL expected name for the last obs in episode
                if info_name == "episode_reward" and self.tensorboard_writer is not None:
                    none_nan_episode_reward = info_tensor[~torch.isnan(info_tensor)]
                    self.tensorboard_writer.add_scalar(
                        f"{robot_name}/episode_reward_mean",
                        none_nan_episode_reward.mean(),
                        self.mother_env.global_frame_count,
                    )
                    if len(none_nan_episode_reward) > 1:
                        self.tensorboard_writer.add_scalar(
                            f"{robot_name}/episode_reward_std",
                            none_nan_episode_reward.std(),
                            self.mother_env.global_frame_count,
                        )
                    continue
                if info_name.startswith("tensorboard/") and self.tensorboard_writer is not None:
                    real_info_name = robot_name + info_name[len("tensorboard") :]
                    if (
                        self.is_shared
                        and hasattr(info_tensor, 'ndim')
                        and info_tensor.ndim == 1
                        and info_tensor.shape[0] == self.n_robots
                    ):
                        info_tensor = info_tensor.mean()
                    self.tensorboard_writer.add_scalar(real_info_name, info_tensor, self.mother_env.global_frame_count)
                    continue

                # Check if info_tensor is a scalar (0-dimensional tensor) or python scalar
                if (
                    isinstance(info_tensor, torch.Tensor) and len(info_tensor.shape) == 0
                ):  # 0-dimensional tensor (scalar)
                    # Handle scalar tensor - broadcast to all envs
                    scalar_value = info_tensor
                    if isinstance(info_tensor, torch.Tensor):
                        scalar_value = info_tensor.item()  # Convert to python scalar
                    for env_idx in range(self.n_effective_envs):
                        if info_name in openrl_info[env_idx]:
                            openrl_info[env_idx][info_name][robot_idx] = scalar_value
                        else:
                            openrl_info[env_idx][info_name] = {robot_idx: scalar_value}
                elif info_tensor.ndim == 1 and info_tensor.shape[0] == self.n_effective_envs:
                    for env_idx in range(self.n_effective_envs):
                        value = info_tensor[env_idx]
                        # Check if it is all nan, if so, skip, it is a padding
                        if torch.isnan(value).all():
                            continue
                        if info_name in openrl_info[env_idx]:
                            openrl_info[env_idx][info_name][robot_idx] = value.cpu().numpy()
                        else:
                            openrl_info[env_idx][info_name] = {robot_idx: value.cpu().numpy()}
                elif (
                    info_tensor.ndim == 1 and self.is_shared and info_tensor.shape[0] == self.n_robots
                ):  # Only case: is shared, each robot provide one scalar
                    value = info_tensor
                    # Check if it is all nan, if so, skip, it is a padding
                    if torch.isnan(value).all():
                        continue
                    # Broadcast to all envs
                    for env_idx in range(self.n_effective_envs):
                        if info_name in openrl_info[env_idx]:
                            openrl_info[env_idx][info_name][robot_idx] = value.cpu().numpy()
                        else:
                            openrl_info[env_idx][info_name] = {robot_idx: value.cpu().numpy()}
                else:
                    raise ValueError(f"Unexpected info shape: {info_tensor.shape}")

        if is_reset:
            return obs_np, openrl_info
        return obs_np, rewards_np, dones_np, openrl_info

    @property
    def unwrapped(self):
        """Return self as the unwrapped environment for OpenRL."""
        return self

    def render(self):
        """Render method for compatibility."""
        # Rendering is handled by the mother environment if needed
        pass

    def close(self):
        """Close environment."""
        self.logger.info(f"Closing OpenRLEnv {self.training_name}")
        # No specific cleanup needed for our wrapper
        pass

    def seed(self, seed: int):
        """Set random seed."""
        self.logger.info(f"Setting seed {seed} for OpenRLEnv {self.training_name}")
        # Seed is handled by parent environment
        pass

    @property
    def tensorboard_writer(self):
        """The tensorboard writer used by specific trainer"""
        if self.agent and hasattr(self.agent, 'logger'):
            return self.agent.logger.writter
        else:
            return None


def openrl_training_thread_function(subvecenv: OpenRLEnv, max_iterations: int):
    """Training function that runs OpenRL's PPO in a separate thread."""

    training_config = subvecenv.training_cfg
    training_name = training_config.training_name
    openrl_config: OpenRLConfig = training_config.openrl_config

    logger = get_class_logger("OpenRLTrainer", training_name, level="INFO")
    logger.info(f"🚀 Starting OpenRL training thread for {training_name}")

    try:
        # Import OpenRL components
        from openrl.modules.common import PPONet as Net
        from openrl.runners.common import PPOAgent as Agent
        from openrl.utils.callbacks.checkpoint_callback import CheckpointCallback

        # Update config with environment-specific values
        openrl_config.episode_length = subvecenv.max_episode_length
        openrl_config.num_env_steps = max_iterations * openrl_config.episode_length * subvecenv.n_envs
        openrl_config.num_agents = subvecenv.agent_num

        # Create log directory
        log_dir = os.path.join(openrl_config.log_dir, training_name)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        # Add some additional configs that OpenRL expects
        openrl_config.cuda = "cuda" in str(subvecenv.device)

        logger.info(f"Initializing PPONet for {training_name}...")
        logger.info(f"Environment has {subvecenv.agent_num} agents: {subvecenv.agents}")

        # Convert our config to OpenRL's expected format using their parser
        args = openrl_config.to_openrl_args()
        logger.info("✅ Successfully converted config using OpenRL's parser")

        # Create network
        # OpenRL will automatically detect multi-agent scenario from the environment
        device = "cuda" if openrl_config.cuda else "cpu"
        net = Net(subvecenv, cfg=args, device=device)

        logger.info(f"Initializing PPOAgent for {training_name}...")

        # Create agent
        agent = Agent(net, use_wandb=not openrl_config.disable_wandb, use_tensorboard=openrl_config.disable_wandb)

        # Create checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=openrl_config.save_interval,  # Save model every 10000 steps
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix="openrl_model",
            verbose=2,  # Print message when saving
        )

        # Calculate total timesteps
        total_timesteps = openrl_config.num_env_steps

        logger.info(f"🎯 Starting OpenRL training for {training_name} with {total_timesteps} timesteps...")
        logger.info(
            f"📁 Checkpoints will be saved every {openrl_config.save_interval} steps to {os.path.join(log_dir, 'checkpoints')}"
        )

        # Register the agent class
        subvecenv.agent = agent

        # Start training with checkpoint callback
        agent.train(total_time_steps=total_timesteps, callback=checkpoint_callback)

        # Save final model
        final_model_path = os.path.join(log_dir, "final_model")
        agent.save(final_model_path)
        logger.info(f"Model saved to {final_model_path}")

        logger.info(f"✅ OpenRL training completed for {training_name}")

    except Exception as e:
        logger.error(f"❌ OpenRL training failed for {training_name}: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


@dataclass
class OpenRLTrainingConfig(TrainingConfig):
    """Configuration for OpenRL training."""

    openrl_config: OpenRLConfig = field(default_factory=OpenRLConfig)

    # Fields required by TrainingConfig interface
    trainer_name: str = "OpenRL"
    trainer_env_factory: Callable[[AECEnv], OpenRLEnv] = None
    trainer_launcher: Callable[[OpenRLEnv, int], None] = openrl_training_thread_function

    def __post_init__(self, robot_cfgs):
        super().__post_init__(robot_cfgs)

        # Update OpenRL config with actual environment parameters
        self.openrl_config.experiment_name = f"{self.training_name}_{self.trainer_name}"
        self.openrl_config.scenario_name = self.training_name

        if self.trainer_env_factory is None:
            training_config_ref = self  # Capture self in closure

            def trainer_env_factory(AEC_env):
                return OpenRLEnv(training_config=training_config_ref, mother_env=AEC_env)

            self.trainer_env_factory = trainer_env_factory
