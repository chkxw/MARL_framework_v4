"""Direct-access Sub-VecEnv wrapper for individual agents in MARL framework V3.

This wrapper implements the rsl_rl VecEnv interface for each agent,
communicating with the mother AEC environment through direct object references
and condition variables for coordination.
Handles dead agents through NaN observation detection.
No tensor serialization needed - direct tensor sharing.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import threading
import time
from rsl_rl.env.vec_env import VecEnv

from .genesis_logging import get_module_logger


logger = get_module_logger("sub_vecenv_wrapper")


class SubVecEnvWrapper:
    """VecEnv interface wrapper for a single agent in the MARL system (V3).
    
    This wrapper:
    - Implements the rsl_rl VecEnv interface
    - Uses direct object reference to mother AEC environment (no queues)
    - Uses condition variables for coordination
    - Detects dead agents through NaN observations
    - Continues operating even when agent is dead
    - Shares tensors directly without serialization
    """
    
    def __init__(
        self,
        agent_name: str,
        mother_env,              # Direct reference to VectorizedAECEnv
        n_envs: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cuda",
        timeout: float = 10.0,   # Timeout for waiting for observations (seconds)
    ):
        """Initialize the sub-VecEnv wrapper with direct mother env access (V3).
        
        Args:
            agent_name: Name of the agent this wrapper represents
            mother_env: Direct reference to VectorizedAECEnv
            n_envs: Number of parallel environments
            obs_dim: Observation dimension
            action_dim: Action dimension
            device: PyTorch device
            timeout: Timeout for waiting for observations from mother environment (seconds)
        """
        self.agent_name = agent_name
        self.mother_env = mother_env
        self.device = device
        self.timeout = timeout
        
        logger.info(f"Initializing SubVecEnvWrapper for agent {agent_name} (V3 direct access)")
        logger.info(f"  n_envs: {n_envs}, obs_dim: {obs_dim}, action_dim: {action_dim}")
        
        # Get Agent bundle and coordination references for V3 flag coordination
        self.agent_bundle = None
        for agent in mother_env.agents:
            if agent.agent_name == agent_name:
                self.agent_bundle = agent
                break
        
        if self.agent_bundle is None:
            raise ValueError(f"Agent bundle {agent_name} not found in mother environment")
        
        # Use Agent's coordination references
        self.condition = self.agent_bundle.coordination_flag
        if self.condition is None:
            raise ValueError(f"No coordination flag found for agent {agent_name}")
        
        # Environment parameters (for rsl_rl compatibility)
        self.num_envs = n_envs
        self.num_obs = obs_dim
        self.num_actions = action_dim
        self.max_episode_length = mother_env.max_episode_length
        self.episode_length_buf = torch.zeros(n_envs, device=device, dtype=torch.int32)
        
        # Initialize buffers (these will be updated from mother env data)
        self.obs_buf = torch.zeros((n_envs, obs_dim), device=device, dtype=torch.float32)
        self.rew_buf = torch.zeros((n_envs,), device=device, dtype=torch.float32)
        self.reset_buf = torch.zeros((n_envs,), device=device, dtype=torch.bool)
        
        # Track dead agent states
        self.agent_dead_mask = torch.zeros((n_envs,), device=device, dtype=torch.bool)
        
        # For rsl_rl compatibility
        self.extras = {}
        
        # State tracking
        self.is_initialized = False
        self.step_count = 0
        
        # Cache for initial observation (OnPolicyRunner workaround)
        self.cached_initial_obs = None
        self.cached_initial_extras = None
        
        logger.info(f"SubVecEnvWrapper {agent_name} initialized with V3 flag coordination")
    
    def _wait_for_observation_update(self, timeout: Optional[float] = None) -> bool:
        """Wait for mother environment to signal observations are ready (V3).
        
        According to DESIGN.md, SubVecEnv should wait for flag to be "SUBVECENV_SIDE"
        which indicates AEC has computed observations and they're ready to consume.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if observations were updated, False if timeout
        """
        if timeout is None:
            timeout = self.timeout
        
        # BUG FIX: Wait for proper flag coordination, not agent selection
        return self.agent_bundle.wait_for_flag_change(expected_state="SUBVECENV_SIDE", timeout=timeout)
    
    def _update_from_mother_env(self):
        """Update local buffers from mother environment data (V3)."""
        # Direct access to mother environment tensors - no copying!
        agent_idx = self.mother_env.agent_name_mapping[self.agent_name]
        
        # Get observation directly from mother env
        obs = self.mother_env.observe(self.agent_name)  # This handles NaN for dead agents
        
        # Get other data directly
        reward = self.mother_env._cumulative_rewards[self.agent_name]
        terminated = self.mother_env.terminations[self.agent_name]
        truncated = self.mother_env.truncations[self.agent_name]
        info_list = self.mother_env.infos[self.agent_name]
        
        # Update local buffers (these are just references/copies for rsl_rl compatibility)
        self.obs_buf = obs.clone()  # Clone to avoid accidental modification
        self.rew_buf = reward.clone()
        
        # Death detection logic (same as V2)
        old_dead_mask = self.agent_dead_mask.clone()
        
        # Step 1: Mark agents as dead if they terminated or truncated
        self.agent_dead_mask |= (terminated | truncated)
        
        # Step 2: Remove from dead mask if observation is not NaN (autoreset happened)
        nan_mask = torch.isnan(obs).any(dim=1)
        self.agent_dead_mask &= nan_mask  # Only keep dead if obs is NaN
        
        # Log transitions
        newly_dead = self.agent_dead_mask & ~old_dead_mask
        newly_alive = ~self.agent_dead_mask & old_dead_mask
        
        if newly_dead.any():
            logger.info(f"Agent {self.agent_name} died in {newly_dead.sum().item()} environments")
        if newly_alive.any():
            logger.info(f"Agent {self.agent_name} revived (autoreset) in {newly_alive.sum().item()} environments")
        
        # For dead agents, use zero observations for RL algorithm
        self.obs_buf[self.agent_dead_mask] = 0.0
        self.rew_buf[self.agent_dead_mask] = 0.0
        
        # Update reset buffer
        self.reset_buf = (terminated | truncated) & ~self.agent_dead_mask
        
        # Update extras for rsl_rl compatibility
        self.extras = {"observations": {"critic": self.obs_buf}}
        if truncated.any():
            self.extras["time_outs"] = truncated.float()
        
        logger.debug(f"Updated buffers for {self.agent_name}: "
                    f"{self.agent_dead_mask.sum().item()}/{self.num_envs} dead agents")
    
    def reset(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the environment (rsl_rl VecEnv interface - V3).
        
        Returns:
            observations: Tensor of shape [num_envs, num_obs]
            extras: Dictionary with additional info
        """
        logger.debug(f"🔄 {self.agent_name}: Reset called - waiting for initial observations...")
        
        # V3: Direct access to mother environment data
        # Wait for mother environment to have data ready for this agent
        if not self._wait_for_observation_update(timeout=self.timeout):
            logger.error(f"⏰ {self.agent_name}: TIMEOUT waiting for reset observations")
            return self.obs_buf, self.extras
        
        logger.debug(f"✅ {self.agent_name}: Received reset observations, updating buffers...")
        
        # Update buffers from mother environment
        self._update_from_mother_env()
        
        # Clear tracking states
        self.agent_dead_mask.fill_(False)
        self.episode_length_buf.zero_()
        self.step_count = 0
        self.is_initialized = True
        
        # Cache initial observation for get_observations() calls
        self.cached_initial_obs = self.obs_buf.clone()
        self.cached_initial_extras = self.extras.copy()
        
        logger.info(f"✅ {self.agent_name}: Reset complete - obs_shape={self.obs_buf.shape}")
        logger.debug(f"💾 {self.agent_name}: Cached initial observation for get_observations() calls")
        
        return self.obs_buf, self.extras
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the environment with given actions (rsl_rl VecEnv interface - V3).
        
        Args:
            actions: Tensor of shape [num_envs, num_actions]
        
        Returns:
            observations: Tensor of shape [num_envs, num_obs]
            rewards: Tensor of shape [num_envs]
            dones: Tensor of shape [num_envs]
            extras: Dictionary with additional info
        """
        self.step_count += 1
        
        if not self.is_initialized:
            logger.warning("Step called before reset, initializing...")
            self.reset()
        
        logger.debug(f"Step {self.step_count} for agent {self.agent_name}")
        
        # Validate actions
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"Expected action shape {(self.num_envs, self.num_actions)}, "
                           f"got {actions.shape}")
        
        # V3 Flag Coordination: Submit action to AEC cache and set flag to AEC side
        logger.debug(f"🚀 {self.agent_name}: Caching action (shape: {actions.shape}) and setting flag to AEC side")
        
        # Write action to cache
        self.agent_bundle.action_cache_ref[self.agent_name] = actions
        logger.debug(f"💾 {self.agent_name}: Action cached successfully")
        
        # Set flag to AEC side and notify
        logger.debug(f"🔄 {self.agent_name}: Setting flag SUBVECENV_SIDE -> AEC_SIDE and notifying")
        self.agent_bundle.set_flag_state("AEC_SIDE")
        self.agent_bundle.notify_coordination_flag()
        
        # Wait for flag to flip back to SubVecEnv side (observations ready)
        logger.debug(f"⏳ {self.agent_name}: Waiting for flag flip to SUBVECENV_SIDE (observations ready)...")
        
        success = self.agent_bundle.wait_for_flag_change(expected_state="SUBVECENV_SIDE", timeout=self.timeout)
        if not success:
            logger.error(f"⏰ {self.agent_name}: TIMEOUT waiting for observations")
            return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
        
        # Update buffers from mother environment
        logger.debug(f"📊 {self.agent_name}: Updating buffers from mother environment...")
        self._update_from_mother_env()
        
        # Track episode length
        self.episode_length_buf += 1
        self.episode_length_buf[self.reset_buf] = 0
        
        logger.debug(f"✅ {self.agent_name}: Step completed - obs_shape={self.obs_buf.shape}, "
                    f"rewards_sum={self.rew_buf.sum().item():.3f}, "
                    f"resets={self.reset_buf.sum().item()}/{self.num_envs}")
        
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
        
    
    def get_observations(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Special function for OnPolicyRunner compatibility (V3).
        
        In the whole training process, this function only gets called for two purposes:
        1. before very first reset to determine the shape of obs
        2. after reset as the very first observation 
        
        However, there's a mismatch with standard API:
        1. standard API expect the very first obs get returned by reset()
        2. the obs is synced with mother AEC env, we cannot get the initial obs several times
        
        V3 Solution:
        1. in reset(), we cache the obs and extras
        2. in get_observations(), if we don't find cached obs and extras, we get immediate 
           obs directly from mother env for dimension discovery; after reset, it returns 
           the cached initial obs
        """
        if self.cached_initial_obs is not None and self.cached_initial_extras is not None:
            # Consume the cached initial obs
            cached_initial_obs = self.cached_initial_obs
            cached_initial_extras = self.cached_initial_extras
            self.cached_initial_obs = None
            self.cached_initial_extras = None
            return cached_initial_obs, cached_initial_extras
        else:
            # V3: Get immediate observation directly from mother env (for dimension discovery)
            try:
                obs = self.mother_env.observe(self.agent_name)
                if obs is None:
                    raise ValueError("Failed to get immediate observation")
                
                # Create minimal extras for compatibility
                extras = {"observations": {"critic": obs}}
                
                logger.debug(f"Provided immediate observation for {self.agent_name} (dimension discovery)")
                return obs, extras
                
            except Exception as e:
                logger.error(f"Failed to get immediate observation for {self.agent_name}: {e}")
                # Fallback to dummy observation
                dummy_obs = torch.zeros((self.num_envs, self.num_obs), device=self.device)
                dummy_extras = {"observations": {"critic": dummy_obs}}
                return dummy_obs, dummy_extras
            
    def get_privileged_observations(self) -> Optional[torch.Tensor]:
        """Get privileged observations (not used)."""
        return None
    
    def get_dead_environments(self) -> List[int]:
        """Get list of environment indices where this agent is dead.
        
        Returns:
            List of environment indices where agent is currently dead
        """
        if hasattr(self, 'agent_dead_mask'):
            dead_envs = self.agent_dead_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
            # Handle single element case
            if isinstance(dead_envs, int):
                dead_envs = [dead_envs]
            return dead_envs
        return []
    
    def close(self):
        """Close the environment (V3)."""
        logger.info(f"Closing SubVecEnvWrapper for agent {self.agent_name}")
        # V3: No command needed, just clean up local references
        self.mother_env = None
        self.condition = None


def create_subvecenv_v3(
    agent_name: str,
    mother_env,
    config: Dict[str, Any]
) -> SubVecEnvWrapper:
    """Create a SubVecEnvWrapper with direct access to mother environment (V3).
    
    Args:
        agent_name: Name of the agent
        mother_env: Direct reference to VectorizedAECEnv
        config: Configuration dictionary
        
    Returns:
        SubVecEnvWrapper instance
    """
    logger.info(f"Creating V3 SubVecEnv for {agent_name}")
    
    return SubVecEnvWrapper(
        agent_name=agent_name,
        mother_env=mother_env,
        n_envs=config["n_envs"],
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"],
        device=config.get("device", "cuda"),
        timeout=config.get("timeout", 10.0)
    )


# Legacy functions for V2 compatibility (deprecated in V3)
def run_sub_vecenv_thread(
    agent_name: str,
    config: Dict[str, Any],
    command_queue=None,  # Deprecated
    response_queue=None,  # Deprecated 
    ready_event=None,    # Deprecated
    stop_event=None,     # Deprecated
):
    """Deprecated: V3 uses direct object references instead of threads.
    
    This function is kept for backward compatibility but should not be used in V3.
    Use create_subvecenv_v3() instead.
    """
    logger.warning("run_sub_vecenv_thread is deprecated in V3. Use create_subvecenv_v3() instead.")
    raise NotImplementedError("V3 uses direct object references. Use create_subvecenv_v3().")


def run_sub_vecenv_process(*args, **kwargs):
    """Deprecated: V3 uses direct object references instead of processes."""
    logger.warning("run_sub_vecenv_process is deprecated in V3. Use create_subvecenv_v3() instead.")
    raise NotImplementedError("V3 uses direct object references. Use create_subvecenv_v3().")