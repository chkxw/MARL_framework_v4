#!/usr/bin/env python3
"""
Joint Training SubVecEnv for Genesis MARL Framework V3

This module implements a specialized SubVecEnv wrapper for joint training scenarios
where multiple robots are controlled by a single policy with concatenated observations/actions.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

from .sub_vecenv_wrapper import SubVecEnvWrapper
from .robot_config import JointTrainingBundle
from .genesis_logging import get_module_logger


logger = get_module_logger("joint_training_subvecenv")


class JointTrainingSubVecEnv(SubVecEnvWrapper):
    """SubVecEnv wrapper for joint training scenarios (V3).
    
    In joint training:
    - Multiple robots share a single policy
    - Observations from all robots are concatenated per environment  
    - Actions are split and applied to individual robots
    - Single reward combines contributions from all robots
    - Environment count stays the same (n_envs)
    """
    
    def __init__(
        self,
        bundle: JointTrainingBundle,
        mother_env,
        n_envs: int,
        device: str = "cuda",
        timeout: float = 10.0,
    ):
        """Initialize joint training SubVecEnv wrapper.
        
        Args:
            bundle: JointTrainingBundle containing robot configurations
            mother_env: Direct reference to VectorizedAECEnv
            n_envs: Number of parallel environments
            device: PyTorch device
            timeout: Timeout for coordination
        """
        # Calculate total observation and action dimensions
        total_obs_dim = bundle.get_observation_dim()
        total_action_dim = bundle.get_action_dim()
        
        # Initialize base wrapper with concatenated dimensions
        super().__init__(
            agent_name=bundle.bundle_name,
            mother_env=mother_env,
            n_envs=n_envs,
            obs_dim=total_obs_dim,
            action_dim=total_action_dim,
            device=device,
            timeout=timeout
        )
        
        self.bundle = bundle
        self.n_robots = len(bundle.robot_configs)
        
        logger.info(f"Initialized JointTrainingSubVecEnv for '{bundle.bundle_name}'")
        logger.info(f"  {self.n_robots} robots, total obs_dim: {total_obs_dim}, total action_dim: {total_action_dim}")
    
    def _update_from_mother_env(self):
        """Update local buffers from mother environment data (joint training version)."""
        # Get the agent (bundle) from mother environment
        agent_name = self.agent_name
        
        if agent_name not in self.mother_env.possible_agents:
            logger.error(f"Agent {agent_name} not found in mother environment")
            return
        
        # Get concatenated observation directly from mother env observe method
        # The mother env should handle concatenation through the Agent class
        obs = self.mother_env.observe(agent_name)
        
        # Get other data directly (these are already combined for joint training)
        reward = self.mother_env._cumulative_rewards[agent_name]
        terminated = self.mother_env.terminations[agent_name]
        truncated = self.mother_env.truncations[agent_name]
        
        # Update local buffers
        self.obs_buf = obs.clone()
        self.rew_buf = reward.clone()
        
        # Joint training death detection:
        # If ANY robot in the joint group dies, mark as dead
        # This is handled by the mother environment's Agent class
        old_dead_mask = self.agent_dead_mask.clone()
        
        # Step 1: Mark as dead if terminated or truncated
        self.agent_dead_mask |= (terminated | truncated)
        
        # Step 2: Remove from dead mask if observation is not NaN (autoreset)
        nan_mask = torch.isnan(obs).any(dim=1)
        self.agent_dead_mask &= nan_mask
        
        # Log transitions
        newly_dead = self.agent_dead_mask & ~old_dead_mask
        newly_alive = ~self.agent_dead_mask & old_dead_mask
        
        if newly_dead.any():
            logger.info(f"Joint agent {self.agent_name} died in {newly_dead.sum().item()} environments")
        if newly_alive.any():
            logger.info(f"Joint agent {self.agent_name} revived in {newly_alive.sum().item()} environments")
        
        # For dead agents, use zero observations and rewards
        self.obs_buf[self.agent_dead_mask] = 0.0
        self.rew_buf[self.agent_dead_mask] = 0.0
        
        # Update reset buffer
        self.reset_buf = (terminated | truncated) & ~self.agent_dead_mask
        
        # Update extras for rsl_rl compatibility
        self.extras = {"observations": {"critic": self.obs_buf}}
        if truncated.any():
            self.extras["time_outs"] = truncated.float()
        
        logger.debug(f"Updated joint training buffers for {self.agent_name}: "
                    f"{self.agent_dead_mask.sum().item()}/{self.num_envs} dead agents")
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the environment with joint actions.
        
        Args:
            actions: Joint action tensor [num_envs, total_action_dim]
        
        Returns:
            observations: Concatenated observations [num_envs, total_obs_dim]
            rewards: Combined rewards [num_envs]
            dones: Done flags [num_envs]
            extras: Additional info
        """
        logger.debug(f"Joint training step for {self.agent_name}, action shape: {actions.shape}")
        
        # Validate joint action dimensions
        expected_shape = (self.num_envs, self.num_actions)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected joint action shape {expected_shape}, got {actions.shape}")
        
        # The mother environment's Agent class will handle action splitting
        # We just pass the concatenated actions as-is
        return super().step(actions)
    
    def get_individual_robot_info(self) -> Dict[str, Any]:
        """Get information about individual robots in the joint training setup.
        
        Returns:
            Dictionary with information about each robot in the bundle
        """
        robot_info = {
            "bundle_type": "joint_training",
            "n_robots": self.n_robots,
            "total_action_dim": self.num_actions,
            "total_obs_dim": self.num_obs,
            "robots": []
        }
        
        for i, config in enumerate(self.bundle.robot_configs):
            robot_info["robots"].append({
                "name": config.name,
                "action_dim": config.action_dim,
                "obs_dim": config.observation_dim,
                "frequency": config.frequency,
                "control_mode": config.control_mode
            })
        
        return robot_info
    
    def split_joint_action_for_analysis(self, joint_action: torch.Tensor) -> List[torch.Tensor]:
        """Split joint action tensor for analysis purposes.
        
        Args:
            joint_action: Joint action tensor [num_envs, total_action_dim]
            
        Returns:
            List of individual robot actions
        """
        return self.bundle.split_action(joint_action)