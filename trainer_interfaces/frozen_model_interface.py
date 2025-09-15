#!/usr/bin/env python3
"""
Frozen Model Interface for Genesis MARL Framework V4

This module provides a special interface for running pre-trained models
in inference-only mode. It acts like a training interface but only performs
model inference without any actual training.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple
import time

import torch
import numpy as np

from configs.TrainingConfig import TrainingConfig
from trainer_interfaces.sub_vecenv import SubVecEnv
from marl_logging import get_class_logger
from model_inference import ModelInference

AECEnv = Any


class FrozenModelSubVecEnv(SubVecEnv):
    """SubVecEnv for frozen model inference.
    
    This class follows the standard SubVecEnv pattern but doesn't need
    to override step/reset. It just provides empty _process_actions.
    """
    
    def __init__(self, training_config: 'FrozenModelConfig', mother_env: Any):
        """Initialize frozen model inference environment."""
        super().__init__(training_config=training_config, mother_env=mother_env)
        
        self.logger = get_class_logger("FrozenModelSubVecEnv", self.training_name, level="INFO")
        
        # The model will be loaded in the launcher, not here
        self.model_inference = None
        
    def _process_actions(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Pass through actions without processing (they come from inference)."""
        return actions
    
    def _process_data(self, data, is_reset=False):
        """Standard data processing."""
        obs, shared_obs, rewards, truncations, terminations, infos, episode_lengths = data
        
        if is_reset:
            return obs
        return obs, rewards, truncations, terminations, infos


@dataclass
class FrozenModelConfig(TrainingConfig):
    """Configuration for frozen model inference."""
    
    model_path: str = field(default=None)
    trainer_name: str = field(default="frozen_model", init=False)
    
    def __post_init__(self, robot_cfgs):
        """Validate frozen model configuration."""
        super().__post_init__(robot_cfgs)
        
        if self.model_path is None:
            raise ValueError("model_path must be provided for FrozenModelConfig")
        
        import os
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if not (self.model_path.endswith('.onnx') or 
                self.model_path.endswith('.pt') or 
                self.model_path.endswith('.pth')):
            raise ValueError(f"Model file must be .onnx, .pt, or .pth, got: {self.model_path}")
    
    @property
    def trainer_env_factory(self):
        """Factory for creating FrozenModelSubVecEnv."""
        config_ref = self
        
        def factory(AEC_env):
            return FrozenModelSubVecEnv(training_config=config_ref, mother_env=AEC_env)
        
        return factory
    
    @property
    def trainer_launcher(self):
        """Launcher for frozen model inference loop."""
        return frozen_model_launcher


def frozen_model_launcher(subvecenv: FrozenModelSubVecEnv, max_iterations: int):
    """Launcher function for frozen model inference.
    
    This function runs the inference loop, generating actions from the model
    based on observations, following the training_type pattern.
    """
    logger = get_class_logger("FrozenModelLauncher", subvecenv.training_name, level="INFO")
    logger.info(f"🚀 Starting frozen model inference for {subvecenv.training_name}")
    
    # Load the model
    model_inference = ModelInference(force_cpu=False)
    logger.info(f"Loading frozen model from: {subvecenv.training_cfg.model_path}")
    model_inference.load_model(subvecenv.training_cfg.model_path)
    
    model_info = model_inference.get_model_info()
    logger.info(f"Model loaded: {model_info}")
    
    try:
        # Initial reset to get observations
        logger.info("Performing initial reset...")
        obs = subvecenv.reset()
        logger.info(f"Initial observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        
        # Run inference loop
        episode_count = 0
        step_count = 0
        episode_rewards = []
        current_episode_reward = 0.0
        
        logger.info("Starting inference loop...")
        logger.info(f"Training type: {subvecenv.training_cfg.training_type}")
        logger.info(f"Number of robots: {subvecenv.n_robots}")
        logger.info(f"Number of environments: {subvecenv.n_envs}")
        
        while True:
            # Prepare observations for inference based on training_type
            if subvecenv.training_cfg.training_type == "NORMAL":
                # For NORMAL (like MAPPO), each robot has independent policy
                if subvecenv.n_robots == 1:
                    # Single robot: obs is [n_envs, obs_dim]
                    obs_for_inference = obs
                else:
                    # Multi-robot: obs is [n_robots, n_envs, obs_dim]
                    # Flatten to [n_robots * n_envs, obs_dim] for batch inference
                    obs_for_inference = obs.reshape(-1, obs.shape[-1])
                    
            elif subvecenv.training_cfg.training_type == "JOINT":
                # JOINT: single policy controls all robots
                # obs is already concatenated [n_envs, total_obs_dim]
                obs_for_inference = obs
                
            elif subvecenv.training_cfg.training_type == "SHARED":
                # SHARED: obs is already [n_robots * n_envs, obs_dim]
                obs_for_inference = obs
            else:
                raise ValueError(f"Unknown training type: {subvecenv.training_cfg.training_type}")
            
            # Run inference
            with torch.no_grad():
                actions = model_inference.predict(obs_for_inference)
            
            # Reshape actions based on training_type
            if subvecenv.training_cfg.training_type == "NORMAL" and subvecenv.n_robots > 1:
                # Reshape from [n_robots * n_envs, action_dim] to [n_envs, n_robots, action_dim]
                n_robots = subvecenv.n_robots
                n_envs = subvecenv.n_envs
                action_dim = actions.shape[-1]
                actions = actions.reshape(n_robots, n_envs, action_dim).transpose(0, 1)
                
            # Step the environment
            result = subvecenv.step(actions)
            
            if result is None:
                continue
                
            obs, rewards, truncations, terminations, infos = result
            
            # Track rewards
            if isinstance(rewards, torch.Tensor):
                current_episode_reward += rewards.mean().item()
            
            step_count += 1
            
            # Check for episode end
            episode_ended = False
            if isinstance(terminations, torch.Tensor):
                episode_ended = terminations.any().item()
            
            if episode_ended:
                episode_count += 1
                episode_rewards.append(current_episode_reward)
                
                if episode_count % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                    logger.info(f"Episode {episode_count}: Steps={step_count}, "
                              f"Avg Reward (last 10): {avg_reward:.2f}")
                
                current_episode_reward = 0.0
            
            # Check if we should stop
            if max_iterations > 0 and episode_count >= max_iterations:
                logger.info(f"Reached max iterations ({max_iterations} episodes)")
                break
    
    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
    except Exception as e:
        logger.error(f"Error during frozen model inference: {e}")
        raise
    finally:
        logger.info(f"Frozen model inference completed. Total episodes: {episode_count}, Total steps: {step_count}")
        if episode_rewards:
            logger.info(f"Average episode reward: {np.mean(episode_rewards):.2f}")