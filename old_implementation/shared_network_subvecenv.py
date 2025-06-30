#!/usr/bin/env python3
"""
Shared Network SubVecEnv for Genesis MARL Framework V3

This module implements a specialized SubVecEnv wrapper for shared network training scenarios
where multiple robots of the same type use the same policy network.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

from .sub_vecenv_wrapper import SubVecEnvWrapper
from .robot_config import SharedNetworkBundle
from .genesis_logging import get_module_logger


logger = get_module_logger("shared_network_subvecenv")


class SharedNetworkSubVecEnv(SubVecEnvWrapper):
    """SubVecEnv wrapper for shared network training scenarios (V3).
    
    In shared network training:
    - Multiple robots of the same type use the same policy network
    - Each robot appears as a separate environment to the policy
    - n_envs gets multiplied by number of robots (m robots x n envs = mxn total envs)
    - obs/action dimensions stay the same as individual robot
    """
    
    def __init__(
        self,
        bundle: SharedNetworkBundle,
        mother_env,
        n_envs: int,
        device: str = "cuda",
        timeout: float = 10.0,
    ):
        """Initialize shared network SubVecEnv wrapper.
        
        Args:
            bundle: SharedNetworkBundle containing robot configurations
            mother_env: Direct reference to VectorizedAECEnv
            n_envs: Number of parallel environments per robot
            device: PyTorch device
            timeout: Timeout for coordination
        """
        # Get individual robot dimensions (same for all robots in shared network)
        individual_obs_dim = bundle.get_observation_dim()
        individual_action_dim = bundle.get_action_dim()
        
        # Calculate total environments (n_robots x n_envs)
        self.n_robots = len(bundle.robot_configs)
        total_envs = n_envs * self.n_robots
        
        # Initialize base wrapper with expanded environment count
        super().__init__(
            agent_name=bundle.bundle_name,
            mother_env=mother_env,
            n_envs=total_envs,  # Expanded environment count
            obs_dim=individual_obs_dim,
            action_dim=individual_action_dim,
            device=device,
            timeout=timeout
        )
        
        self.bundle = bundle
        self.base_n_envs = n_envs  # Original number of environments per robot
        
        logger.info(f"Initialized SharedNetworkSubVecEnv for '{bundle.bundle_name}'")
        logger.info(f"  {self.n_robots} robots x {n_envs} envs = {total_envs} total envs")
        logger.info(f"  Individual robot obs_dim: {individual_obs_dim}, action_dim: {individual_action_dim}")
    
    def _update_from_mother_env(self):
        """Update local buffers from mother environment data (shared network version)."""
        # Get the agent (bundle) from mother environment
        agent_name = self.agent_name
        
        if agent_name not in self.mother_env.possible_agents:
            logger.error(f"Agent {agent_name} not found in mother environment")
            return
        
        # Get reshaped observation from mother env
        # The mother env's Agent class should handle reshaping to shared format
        obs = self.mother_env.observe(agent_name)  # Shape: [n_robots * n_envs, obs_dim]
        
        # Get other data (also reshaped)
        reward = self.mother_env._cumulative_rewards[agent_name]  # Shape: [n_robots * n_envs]
        terminated = self.mother_env.terminations[agent_name]     # Shape: [n_robots * n_envs]
        truncated = self.mother_env.truncations[agent_name]       # Shape: [n_robots * n_envs]
        
        # Update local buffers (already in correct shared network format)
        self.obs_buf = obs.clone()
        self.rew_buf = reward.clone()
        
        # Shared network death detection:
        # Each robot instance is tracked independently
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
            logger.info(f"Shared network agent {self.agent_name} died in {newly_dead.sum().item()} robot-env instances")
        if newly_alive.any():
            logger.info(f"Shared network agent {self.agent_name} revived in {newly_alive.sum().item()} robot-env instances")
        
        # For dead agents, use zero observations and rewards
        self.obs_buf[self.agent_dead_mask] = 0.0
        self.rew_buf[self.agent_dead_mask] = 0.0
        
        # Update reset buffer
        self.reset_buf = (terminated | truncated) & ~self.agent_dead_mask
        
        # Update extras for rsl_rl compatibility
        self.extras = {"observations": {"critic": self.obs_buf}}
        if truncated.any():
            self.extras["time_outs"] = truncated.float()
        
        logger.debug(f"Updated shared network buffers for {self.agent_name}: "
                    f"{self.agent_dead_mask.sum().item()}/{self.num_envs} dead robot-env instances")
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the environment with shared network actions.
        
        Args:
            actions: Action tensor [n_robots * n_envs, action_dim]
        
        Returns:
            observations: Observations [n_robots * n_envs, obs_dim]
            rewards: Rewards [n_robots * n_envs]
            dones: Done flags [n_robots * n_envs]
            extras: Additional info
        """
        logger.debug(f"Shared network step for {self.agent_name}, action shape: {actions.shape}")
        
        # Validate shared network action dimensions
        expected_shape = (self.num_envs, self.num_actions)  # num_envs = n_robots * base_n_envs
        if actions.shape != expected_shape:
            raise ValueError(f"Expected shared network action shape {expected_shape}, got {actions.shape}")
        
        # The mother environment's Agent class will handle action reshaping
        # We pass the shared network format actions as-is
        return super().step(actions)
    
    def get_robot_environment_mapping(self) -> Dict[str, Any]:
        """Get mapping information between robots and environment indices.
        
        Returns:
            Dictionary with robot-environment mapping information
        """
        mapping_info = {
            "bundle_type": "shared_network",
            "n_robots": self.n_robots,
            "base_n_envs": self.base_n_envs,
            "total_envs": self.num_envs,
            "individual_action_dim": self.num_actions,
            "individual_obs_dim": self.num_obs,
            "robot_env_ranges": []
        }
        
        # Each robot occupies a range of environment indices
        for i, config in enumerate(self.bundle.robot_configs):
            start_env = i * self.base_n_envs
            end_env = (i + 1) * self.base_n_envs
            mapping_info["robot_env_ranges"].append({
                "robot_name": config.name,
                "robot_index": i,
                "env_start": start_env,
                "env_end": end_env,
                "env_indices": list(range(start_env, end_env))
            })
        
        return mapping_info
    
    def get_robot_observations(self, robot_index: int) -> torch.Tensor:
        """Get observations for a specific robot.
        
        Args:
            robot_index: Index of the robot (0 to n_robots-1)
            
        Returns:
            Observations for the specified robot [base_n_envs, obs_dim]
        """
        if robot_index >= self.n_robots:
            raise ValueError(f"Robot index {robot_index} >= n_robots {self.n_robots}")
        
        start_env = robot_index * self.base_n_envs
        end_env = (robot_index + 1) * self.base_n_envs
        
        return self.obs_buf[start_env:end_env]
    
    def get_robot_rewards(self, robot_index: int) -> torch.Tensor:
        """Get rewards for a specific robot.
        
        Args:
            robot_index: Index of the robot (0 to n_robots-1)
            
        Returns:
            Rewards for the specified robot [base_n_envs]
        """
        if robot_index >= self.n_robots:
            raise ValueError(f"Robot index {robot_index} >= n_robots {self.n_robots}")
        
        start_env = robot_index * self.base_n_envs
        end_env = (robot_index + 1) * self.base_n_envs
        
        return self.rew_buf[start_env:end_env]
    
    def reshape_actions_to_individual_robots(self, shared_actions: torch.Tensor) -> List[torch.Tensor]:
        """Reshape shared network actions back to individual robot format.
        
        Args:
            shared_actions: Actions in shared format [n_robots * n_envs, action_dim]
            
        Returns:
            List of action tensors for each robot [base_n_envs, action_dim]
        """
        return self.bundle.reshape_from_shared_format(shared_actions, self.base_n_envs)