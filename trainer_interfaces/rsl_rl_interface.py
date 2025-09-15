#!/usr/bin/env python3
from dataclasses import asdict, dataclass, field
import os
import cloudpickle as pickle
import shutil
from typing import Callable, Dict, Any, List, Optional, Tuple

import numpy as np
from configs.RobotConfig import RobotConfig
from configs.TrainingConfig import TrainingConfig
from trainer_interfaces.sub_vecenv import SubVecEnv
from marl_logging import get_class_logger
from abc import abstractmethod
import torch
from utils import RefDict

AECEnv = Any
RNAME = str  # Robot name


@dataclass
class AlgorithmConfig:
    class_name: str = "PPO"
    clip_param: float = 0.2
    desired_kl: float = 0.01
    entropy_coef: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 0.001
    max_grad_norm: float = 1.0
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    schedule: str = "adaptive"
    use_clipped_value_loss: bool = True
    value_loss_coef: float = 1.0


@dataclass
class PolicyConfig:
    activation: str = "elu"
    actor_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    init_noise_std: float = 1.0
    class_name: str = "ActorCritic"


@dataclass
class RunnerConfig:
    checkpoint: int = -1
    experiment_name: str = ""
    load_run: int = -1
    log_interval: int = 1
    max_iterations: int = 0
    record_interval: int = -1
    resume: bool = False
    resume_path: Optional[str] = None
    run_name: str = ""


@dataclass
class RSL_RLConfig:
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    init_member_classes: Dict[str, Any] = field(default_factory=dict)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    runner_class_name: str = "OnPolicyRunner"
    num_steps_per_env: int = 24
    save_interval: int = 100
    empirical_normalization: Optional[Any] = None
    seed: int = 1


class RSL_RLEnv(SubVecEnv):
    def __init__(self, training_config: Any, mother_env: Any, **kwargs):
        super().__init__(training_config=training_config, mother_env=mother_env)

        # RSL_RL expect 1 dim obs, check the obshape
        for robot_name, robot_config in self.robot_configs.items():
            if robot_config.observation_space is None:
                continue  # Not detected yet or not used
            if len(robot_config.observation_space.shape) > 1:
                raise ValueError(
                    f"RSL_RL expects 1D observation space, but {robot_name} got {robot_config.observation_space.shape}"
                )
            if len(robot_config.action_space.shape) > 1:
                raise ValueError(
                    f"RSL_RL expects 1D action space, but {robot_name} got {robot_config.action_space.shape}"
                )
        if training_config.shared_observation_space is not None:
            if len(training_config.shared_observation_space.shape) > 1:
                raise ValueError(
                    f"shared observation space are used as privileged observations for RSL_RL, and RSL_RL expect 1D privileged space, but got {training_config.shared_observation_space.shape}"
                )

        # Properties expected by RSL_RL
        self.episode_length_buf: torch.Tensor = torch.zeros((self.n_envs), device=self.device)

        # No matter it is joint/shared/single robot, the number of observations and actions are always represented by the first robot
        primary_robot_config = self.robot_configs[self.robot_names[0]]
        self.num_obs = np.prod(primary_robot_config.observation_space.shape)
        self.num_actions = np.prod(primary_robot_config.action_space.shape)
        if "num_privileged_obs" in kwargs:
            self.num_privileged_obs = kwargs["num_privileged_obs"]
        else:
            self.num_privileged_obs = None

        # Cached initial return of reset() and use for get_observations() - Due to strange function call order in rsl_rl.runner.onPolicyRunner
        # The onPolicyRunner.learn() doens't call reset(), instead it did two calls of get_observations() and use the second one as initial observation
        # Warning: We need to ensure the env.reset() is manually called before calling onPolicyRunner.learn()
        self.initial_obs: Optional[torch.Tensor] = None
        self.initial_extra: Optional[Dict] = None

    def get_observations(self) -> tuple[torch.Tensor, Dict]:
        """Get cached initial observations."""
        if self.initial_obs is None or self.initial_extra is None:
            raise RuntimeError(f"SingleRobotSubVecEnv {self.training_name} not reset yet")
        return self.initial_obs, self.initial_extra

    def _process_data(self, data, is_reset=False):
        obs, shared_obs, rewards, truncations, terminations, infos, episode_lengths = data
        observations, rewards, dones, extras = self._convert_to_rsl_rl_format(
            obs, shared_obs, rewards, truncations, terminations, infos
        )
        # RSL_RL doesn't support multi-agent training, check if the first dim of observations, rewards, truncations
        # If is all 1, if yes squeeze them
        if observations.shape[0] == 1 and rewards.shape[0] == 1 and dones.shape[0] == 1:
            observations = observations.squeeze(0)
            rewards = rewards.squeeze(0)
            dones = dones.squeeze(0)

        if (
            observations.shape[0] == self.n_effective_envs
            and rewards.shape[0] == self.n_effective_envs
            and dones.shape[0] == self.n_effective_envs
        ):
            pass
        else:
            raise ValueError(f"RSL_RL doesn't support multi-agent training")

        self.episode_length_buf = episode_lengths
        if is_reset:
            self.initial_obs = observations
            self.initial_extra = extras
            return observations, extras
        return observations, rewards, dones, extras

    def _convert_to_rsl_rl_format(
        self, obs, shared_obs, rew, trunc, term, info
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
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
        extra = info
        if "observations" not in extra:
            extra["observations"] = {}
        return obs, rew, term | trunc, extra

    @property
    def num_envs(self):
        """Alias expected by rsl_rl"""
        return self.n_effective_envs

    @property
    def trainer_name(self):
        return "RSL_RL"


def training_thread_function(subvecenv: SubVecEnv, max_iterations: int):
    """Training function that runs in a separate thread using OnPolicyRunner."""
    from rsl_rl.runners import OnPolicyRunner
    import genesis as gs

    training_config = subvecenv.training_cfg
    training_name = training_config.training_name
    exp_name = training_name

    # logger.info(f"🚀 Starting training thread for {training_name}")

    try:
        # Reset subVecEnv to get initial observations
        # logger.info(f"Resetting {training_name}...")
        obs, extras = subvecenv.reset()
        # logger.info(f"Initial reset complete for {training_name}, obs shape: {obs.shape}")

        # Create training configuration
        # Create log directory
        log_dir = f"logs/{exp_name}"
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        pickle.dump(training_config._initial_args, open(f"{log_dir}/cfgs.pkl", "wb"))

        # Create OnPolicyRunner
        # logger.info(f"Creating OnPolicyRunner for {training_name}...")
        # Convert TrainConfig dataclass to dictionary for RSL_RL compatibility

        runner = OnPolicyRunner(subvecenv, asdict(training_config.rsl_rl_config), log_dir, device=gs.device)

        # Start training
        # logger.info(f"🎯 Starting training for {exp_name} with {max_iterations} iterations...")
        if runner.cfg.get("logger", "tensorboard")== "tensorboard":
            subvecenv.tensorboard_writer = runner.writer

        runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=False)

        subvecenv.logger.info(f"✅ Training completed for {training_name}")

    except Exception as e:
        # logger.error(f"❌ Training failed for {training_name}: {e}")
        # logger.error(f"Traceback: {traceback.format_exc()}")
        raise


@dataclass
class RSL_RLTrainingConfig(TrainingConfig):
    """Configuration for RSL_RL training"""

    rsl_rl_config: RSL_RLConfig = field(default_factory=RSL_RLConfig)

    # Fields required by TrainingConfig interface
    trainer_name: str = "RSL_RL"
    trainer_env_factory: Callable[[AECEnv], RSL_RLEnv] = None
    trainer_launcher: Callable[[RSL_RLEnv, "RSL_RLTrainingConfig", int], None] = training_thread_function

    def __post_init__(self, robot_cfgs):
        super().__post_init__(robot_cfgs)
        if self.trainer_env_factory is None:
            training_config_ref = self  # Capture self in closure

            def trainer_env_factory(AEC_env):
                return RSL_RLEnv(training_config=training_config_ref, mother_env=AEC_env)

            self.trainer_env_factory = trainer_env_factory
