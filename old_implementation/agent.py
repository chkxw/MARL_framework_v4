#!/usr/bin/env python3
"""
Agent class for Genesis MARL Framework V3

This module defines the Agent class that manages robot bundles for joint/shared training.
Each agent represents a bundle of robots that are trained together.
"""

import torch
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import genesis as gs

from .robot_config import RobotConfig, RobotConfigBundle, JointTrainingBundle, SharedNetworkBundle
from .joint_detector import get_joint_names_from_config
from .genesis_logging import get_module_logger


logger = get_module_logger("agent")


class Agent:
    """Agent class that manages multiple user bundles for coordinated training (V3).
    
    The Agent class:
    - Manages multiple original user bundles (preserves user intention)
    - Creates flattened robot configs for easier mother env processing
    - Knows which SubVecEnvs it corresponds to
    - Coordinates robot setup and action application for all its bundles
    """
    
    def __init__(self, original_bundles: List[Union[RobotConfigBundle, RobotConfig]], 
                 frequency: float, agent_name: str, mother_env):
        """Initialize agent with original user bundles.
        
        Args:
            original_bundles: List of original user bundles/configs with same frequency
            frequency: The frequency for this agent (all bundles must have same frequency)
            agent_name: Name for this agent (e.g., "agent_50Hz")
            mother_env: Reference to VectorizedAECEnv
        """
        self.original_bundles = original_bundles
        self.frequency = frequency
        self.agent_name = agent_name
        self.mother_env = mother_env
        
        # Create flattened robot configs for easier processing by mother env
        self.flattened_robot_configs = []
        self.bundle_to_robot_mapping = {}  # {bundle_index: [robot_indices_in_flattened]}
        
        for bundle_idx, item in enumerate(original_bundles):
            start_idx = len(self.flattened_robot_configs)
            
            if isinstance(item, RobotConfigBundle):
                # Extract configs from bundle
                robot_configs = item.get_robot_configs()
                self.flattened_robot_configs.extend(robot_configs)
            elif isinstance(item, RobotConfig):
                # Individual robot config
                self.flattened_robot_configs.append(item)
                robot_configs = [item]
            else:
                raise ValueError(f"Invalid robot config type: {type(item)}")
            
            end_idx = len(self.flattened_robot_configs)
            self.bundle_to_robot_mapping[bundle_idx] = list(range(start_idx, end_idx))
            logger.debug(f"Bundle {bundle_idx} maps to robot indices {start_idx}-{end_idx-1}")
        
        # Robot references (will be populated by setup_robots)
        self.robots = []
        self.robot_indices = []  # Indices in mother_env.robots list
        
        # Check bundle types for compatibility (all must have same frequency)
        for item in original_bundles:
            if isinstance(item, RobotConfigBundle):
                if item.frequency != frequency:
                    raise ValueError(f"Bundle {item.bundle_name} frequency {item.frequency} != agent frequency {frequency}")
            elif isinstance(item, RobotConfig):
                if item.frequency != frequency:
                    raise ValueError(f"Robot {item.name} frequency {item.frequency} != agent frequency {frequency}")
        
        # Determine if this agent has joint training or shared network bundles
        self.has_joint_training = any(isinstance(item, JointTrainingBundle) for item in original_bundles)
        self.has_shared_network = any(isinstance(item, SharedNetworkBundle) for item in original_bundles)
        
        # For simplified bundle interface, create a virtual bundle that represents the whole agent
        class VirtualAgentBundle:
            def __init__(self, agent):
                self.bundle_name = agent.agent_name
                self.frequency = agent.frequency
                self.robot_configs = agent.flattened_robot_configs
            
            def get_action_dim(self):
                return sum(c.action_dim for c in self.robot_configs)
            
            def get_observation_dim(self):
                return sum(c.observation_dim for c in self.robot_configs)
        
        self.bundle = VirtualAgentBundle(self)  # For backward compatibility
        
        # IMPORTANT CHANGE 1: References to coordination flags (set by mother_env)
        self.coordination_flag = None  # threading.Condition, set by mother_env
        self.flag_state_ref = None  # Reference to mother_env.flag_states[agent_name]
        self.action_cache_ref = None  # Reference to mother_env.action_cache[agent_name]
        
        logger.info(f"Created Agent '{self.agent_name}'")
        logger.info(f"  Original bundles: {len(self.original_bundles)}")
        logger.info(f"  Flattened robots: {len(self.flattened_robot_configs)}")
        logger.info(f"  Frequency: {self.frequency}Hz")
        logger.info(f"  Total action dim: {self.bundle.get_action_dim()}")
        logger.info(f"  Total observation dim: {self.bundle.get_observation_dim()}")
        
        # Log details about original bundles
        for i, bundle_item in enumerate(self.original_bundles):
            if isinstance(bundle_item, RobotConfigBundle):
                logger.info(f"    Bundle {i}: {bundle_item.bundle_name} ({type(bundle_item).__name__}) - {len(bundle_item.get_robot_configs())} robots")
            else:
                logger.info(f"    Robot {i}: {bundle_item.name} (individual config)")
    
    def get_subvecenv_specs(self) -> List[Dict[str, Any]]:
        """Get specifications for SubVecEnvs that this Agent should manage.
        
        Returns:
            List of SubVecEnv specifications, one for each original bundle
        """
        subvecenv_specs = []
        
        # OPT: Use class mapping
        for i, bundle_item in enumerate(self.original_bundles): 
            if isinstance(bundle_item, JointTrainingBundle):
                # Joint training bundle needs a JointTrainingSubVecEnv
                spec = {
                    "type": "joint_training",
                    "bundle": bundle_item,
                    "agent_name": f"{self.agent_name}_joint_{i}",
                    "obs_dim": bundle_item.get_observation_dim(),
                    "action_dim": bundle_item.get_action_dim(),
                    "n_envs_multiplier": bundle_item.get_n_envs_multiplier()
                }
            elif isinstance(bundle_item, SharedNetworkBundle):
                # Shared network bundle needs a SharedNetworkSubVecEnv
                spec = {
                    "type": "shared_network", 
                    "bundle": bundle_item,
                    "agent_name": f"{self.agent_name}_shared_{i}",
                    "obs_dim": bundle_item.get_observation_dim(),
                    "action_dim": bundle_item.get_action_dim(),
                    "n_envs_multiplier": bundle_item.get_n_envs_multiplier()
                }
            elif isinstance(bundle_item, RobotConfigBundle):
                # Basic bundle needs a basic SubVecEnvWrapper
                spec = {
                    "type": "basic",
                    "bundle": bundle_item,
                    "agent_name": f"{self.agent_name}_basic_{i}",
                    "obs_dim": bundle_item.get_observation_dim(),
                    "action_dim": bundle_item.get_action_dim(),
                    "n_envs_multiplier": bundle_item.get_n_envs_multiplier()
                }
            else:
                # Individual robot config needs a basic SubVecEnvWrapper
                spec = {
                    "type": "single_robot",
                    "robot_config": bundle_item,
                    "agent_name": f"{self.agent_name}_robot_{i}",
                    "obs_dim": bundle_item.observation_dim,
                    "action_dim": bundle_item.action_dim,
                    "n_envs_multiplier": 1
                }
            
            subvecenv_specs.append(spec)
            logger.debug(f"SubVecEnv spec {i}: {spec['type']} - {spec['agent_name']}")
        
        return subvecenv_specs
    
    def set_coordination_references(self, flag: 'threading.Condition', 
                                   flag_state_dict: Dict[str, str], 
                                   action_cache_dict: Dict[str, Any]):
        """Set references to coordination flags (called by mother_env).
        
        Args:
            flag: threading.Condition for this agent
            flag_state_dict: Reference to mother_env.flag_states
            action_cache_dict: Reference to mother_env.action_cache
        """
        self.coordination_flag = flag
        self.flag_state_ref = flag_state_dict
        self.action_cache_ref = action_cache_dict
        current_state = flag_state_dict.get(self.agent_name, "UNKNOWN")
        logger.debug(f"🔗 {self.agent_name}: Coordination references set - current_flag='{current_state}'")
    
    def get_flag_state(self) -> str:
        """Get current flag state for this agent."""
        return self.flag_state_ref.get(self.agent_name, "AEC_SIDE")
    
    def set_flag_state(self, state: str):
        """Set flag state for this agent."""
        old_state = self.flag_state_ref.get(self.agent_name, "UNKNOWN")
        self.flag_state_ref[self.agent_name] = state
        logger.debug(f"🔄 {self.agent_name}: Flag state changed: '{old_state}' -> '{state}'")
    
    def notify_coordination_flag(self):
        """Notify coordination flag (wake up waiting threads)."""
        if self.coordination_flag:
            with self.coordination_flag:
                logger.debug(f"📢 {self.agent_name}: Notifying all waiting threads")
                self.coordination_flag.notify_all()
        else:
            logger.warning(f"🚨 {self.agent_name}: Cannot notify - no coordination flag")
    
    def wait_for_flag_change(self, expected_state: str, timeout: float = 30.0) -> bool:
        """Wait for flag state to change to expected state.
        
        Args:
            expected_state: The state to wait for ("AEC_SIDE" or "SUBVECENV_SIDE")
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if flag reached expected state, False if timeout
        """
        if not self.coordination_flag:
            logger.warning(f"🚨 {self.agent_name}: No coordination flag - cannot wait")
            return False
        
        logger.debug(f"🔄 {self.agent_name}: Starting wait for flag '{expected_state}' (timeout={timeout}s)")
        
        with self.coordination_flag:
            start_time = time.time()
            current_state = self.get_flag_state()
            
            logger.debug(f"🔍 {self.agent_name}: Current flag state '{current_state}', waiting for '{expected_state}'")
            
            # Proper condition variable usage: wait while predicate is false
            wait_iteration = 0
            while self.get_flag_state() != expected_state:
                wait_iteration += 1
                elapsed = time.time() - start_time
                remaining_timeout = timeout - elapsed
                
                if remaining_timeout <= 0:
                    logger.warning(f"⏰ {self.agent_name}: TIMEOUT waiting for flag '{expected_state}' after {elapsed:.2f}s (iterations: {wait_iteration})")
                    return False
                
                logger.debug(f"⏳ {self.agent_name}: Wait iteration {wait_iteration}, remaining timeout: {remaining_timeout:.2f}s")
                
                # Wait for notification with remaining timeout
                notified = self.coordination_flag.wait(timeout=remaining_timeout)
                elapsed_after_wait = time.time() - start_time
                new_state = self.get_flag_state()
                
                if notified:
                    logger.debug(f"🔔 {self.agent_name}: Notified after {elapsed_after_wait:.2f}s, new state: '{new_state}'")
                else:
                    logger.debug(f"⏸️ {self.agent_name}: Wait timeout (partial) after {elapsed_after_wait:.2f}s, state: '{new_state}'")
            
            final_elapsed = time.time() - start_time
            logger.debug(f"✅ {self.agent_name}: Flag reached '{expected_state}' after {final_elapsed:.2f}s (iterations: {wait_iteration})")
            return True
    
    def setup_robots(self, scene: gs.Scene) -> List:
        """Setup robots in the Genesis scene using flattened robot configs.
        
        Args:
            scene: Genesis scene to add robots to
            
        Returns:
            List of robot entities created
        """
        logger.debug(f"Setting up robots for agent {self.agent_name}")
        
        self.robots = []
        for i, config in enumerate(self.flattened_robot_configs):
            logger.debug(f"  Adding robot {config.name}")
            
            robot = scene.add_entity(
                gs.morphs.URDF(
                    file=config.urdf_path,
                    pos=config.initial_position,
                    quat=config.initial_orientation,
                )
            )
            self.robots.append(robot)
        
        logger.info(f"Setup {len(self.robots)} robots for agent {self.agent_name}")
        logger.debug(f"  From {len(self.original_bundles)} original bundles -> {len(self.flattened_robot_configs)} robots")
        return self.robots
    
    def setup_control_parameters(self):
        """Setup control parameters for all robots after scene.build()."""
        logger.debug(f"Setting up control parameters for agent {self.agent_name}")
        
        for i, (robot, config) in enumerate(zip(self.robots, self.flattened_robot_configs)):
            if config.control_mode == "position":
                # Auto-detect controllable joints
                joint_names, motor_dofs = get_joint_names_from_config(robot, config)
                
                if len(motor_dofs) > 0:
                    logger.debug(f"Setting PD control for {config.name}: {len(motor_dofs)} DOFs")
                    
                    try:
                        robot_dof_pos = robot.get_dofs_position()
                        robot_dof_count = robot_dof_pos.shape[-1] if robot_dof_pos.ndim > 1 else len(robot_dof_pos)
                        
                        # Convert scene-wide indices to robot-relative indices
                        # BUG FIX: Calculate proper DOF offset for heterogeneous robots
                        # Get cumulative DOF offset by summing DOFs of all previous robots in the scene
                        robot_dof_offset = 0
                        robot_idx_in_scene = self.robot_indices[i]
                        
                        # Sum DOFs of all robots before this one in the scene
                        for prev_robot_idx in range(robot_idx_in_scene):
                            if prev_robot_idx < len(self.mother_env.robots):
                                prev_robot = self.mother_env.robots[prev_robot_idx]
                                prev_dof_pos = prev_robot.get_dofs_position()
                                prev_dof_count = prev_dof_pos.shape[-1] if prev_dof_pos.ndim > 1 else len(prev_dof_pos)
                                robot_dof_offset += prev_dof_count
                        
                        robot_relative_dofs = [dof - robot_dof_offset for dof in motor_dofs 
                                              if dof >= robot_dof_offset and dof < robot_dof_offset + robot_dof_count]
                        
                        if len(robot_relative_dofs) > 0:
                            robot.set_dofs_kp([150.0] * len(robot_relative_dofs), robot_relative_dofs)
                            robot.set_dofs_kv([10.0] * len(robot_relative_dofs), robot_relative_dofs)
                            logger.debug(f"  Applied PD control to {len(robot_relative_dofs)} DOFs")
                        
                    except Exception as e:
                        logger.warning(f"Could not set PD control for {config.name}: {e}")
        
        logger.debug(f"Control parameters setup complete for agent {self.agent_name}")
    
    def set_robot_indices(self, indices: List[int]):
        """Set the indices of robots in the mother environment's robot list.
        
        Args:
            indices: List of indices corresponding to self.robots in mother_env.robots
        """
        self.robot_indices = indices
        logger.debug(f"Set robot indices for {self.agent_name}: {indices}")
    
    def calculate_observations(self, env_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate observations for this agent's robots.
        
        Args:
            env_indices: Environment indices to compute for (None = all envs)
            
        Returns:
            Observation tensor with shape depending on bundle type:
            - JointTraining: [n_envs, total_obs_dim] (concatenated)
            - SharedNetwork: [n_robots * n_envs, obs_dim] (reshaped)
        """
        if env_indices is None:
            env_indices = torch.arange(self.mother_env.n_envs, device=self.mother_env.device)
        
        # Compute observations for each robot
        robot_observations = []
        for i, (robot, config) in enumerate(zip(self.robots, self.flattened_robot_configs)):
            if config.obs_function is not None:
                # Use custom observation function
                obs = config.obs_function(self.mother_env, self.robot_indices[i], env_indices)
            else:
                # Use default observation computation
                obs = self._compute_default_observation(robot, config, env_indices)
            
            robot_observations.append(obs)
        
        # For Agent with multiple bundles, we concatenate all observations (simplified approach)
        # TODO: More sophisticated processing based on original bundle types
        if robot_observations:
            return torch.cat(robot_observations, dim=-1)
        else:
            return torch.zeros((len(env_indices), 0), device=self.mother_env.device)
    
    def calculate_rewards(self, env_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate rewards for this agent's robots.
        
        Args:
            env_indices: Environment indices to compute for (None = all envs)
            
        Returns:
            Reward tensor with shape depending on bundle type:
            - JointTraining: [n_envs] (combined reward)
            - SharedNetwork: [n_robots * n_envs] (individual rewards)
        """
        if env_indices is None:
            env_indices = torch.arange(self.mother_env.n_envs, device=self.mother_env.device)
        
        # Compute rewards for each robot
        robot_rewards = []
        for i, (robot, config) in enumerate(zip(self.robots, self.flattened_robot_configs)):
            if config.reward_function is not None:
                # Use custom reward function
                reward = config.reward_function(self.mother_env, self.robot_indices[i], env_indices)
            else:
                # Use default reward computation
                reward = self._compute_default_reward(robot, config, env_indices)
            
            robot_rewards.append(reward)
        
        # For Agent with multiple bundles, we sum all rewards (simplified approach)
        # TODO: More sophisticated processing based on original bundle types
        if robot_rewards:
            return sum(robot_rewards)
        else:
            return torch.zeros(len(env_indices), device=self.mother_env.device)
    
    def apply_actions(self, actions: torch.Tensor, env_indices: Optional[torch.Tensor] = None):
        """Apply actions to this agent's robots.
        
        Args:
            actions: Action tensor with shape depending on bundle type
            env_indices: Environment indices to apply to (None = all envs)
        """
        if env_indices is None:
            env_indices = torch.arange(self.mother_env.n_envs, device=self.mother_env.device)
        
        # For Agent with multiple bundles, split actions based on robot action dimensions
        # TODO: More sophisticated action splitting based on original bundle types
        
        # Split actions for each robot based on action dimensions
        robot_actions = []
        start_idx = 0
        for config in self.flattened_robot_configs:
            end_idx = start_idx + config.action_dim
            robot_action = actions[:, start_idx:end_idx]
            robot_actions.append(robot_action)
            start_idx = end_idx
        
        # Apply actions to each robot
        for i, (robot, config, robot_action) in enumerate(zip(self.robots, self.flattened_robot_configs, robot_actions)):
            self._apply_robot_action(robot, config, robot_action, env_indices, i)
    
    def reset_robots(self, env_indices: torch.Tensor):
        """Reset robots to initial positions for specified environments.
        
        Args:
            env_indices: Environment indices to reset
        """
        for i, (robot, config) in enumerate(zip(self.robots, self.flattened_robot_configs)):
            batch_size = len(env_indices)
            init_pos = torch.tensor(config.initial_position, device=self.mother_env.device).unsqueeze(0).expand(batch_size, -1)
            init_quat = torch.tensor(config.initial_orientation, device=self.mother_env.device).unsqueeze(0).expand(batch_size, -1)
            
            robot.set_pos(init_pos, envs_idx=env_indices)
            robot.set_quat(init_quat, envs_idx=env_indices)
            robot.zero_all_dofs_velocity(envs_idx=env_indices)
            
            # Reset joint DOF positions
            try:
                current_dof_pos = robot.get_dofs_position()
                if current_dof_pos.ndim > 1:
                    actual_dof_count = current_dof_pos.shape[-1]
                else:
                    actual_dof_count = len(current_dof_pos)
                
                base_dof_count = 6  # Standard base DOFs
                if actual_dof_count > base_dof_count:
                    current_pos = robot.get_dofs_position()[env_indices]
                    reset_pos = current_pos.clone()
                    reset_pos[:, base_dof_count:] = 0.0
                    robot.set_dofs_position(reset_pos, zero_velocity=True, envs_idx=env_indices)
                else:
                    default_pos = torch.zeros((batch_size, actual_dof_count), device=self.mother_env.device)
                    robot.set_dofs_position(default_pos, zero_velocity=True, envs_idx=env_indices)
                    
            except Exception as e:
                logger.warning(f"Could not reset DOF positions for {config.name}: {e}")
    
    def _compute_default_observation(self, robot, config: RobotConfig, env_indices: torch.Tensor) -> torch.Tensor:
        """Compute default observation for a robot."""
        batch_size = len(env_indices)
        
        # Get robot state
        pos = robot.get_pos()[env_indices]       # [batch, 3]
        quat = robot.get_quat()[env_indices]     # [batch, 4]
        vel = robot.get_vel()[env_indices]       # [batch, 3]
        ang_vel = robot.get_ang()[env_indices]   # [batch, 3]
        
        obs_components = [pos, quat, vel, ang_vel]
        
        # Add joint states if available
        if hasattr(robot, 'get_dofs_position'):
            joint_pos = robot.get_dofs_position()[env_indices]
            joint_vel = robot.get_dofs_velocity()[env_indices]
            obs_components.extend([joint_pos, joint_vel])
        
        # Concatenate all components
        obs = torch.cat(obs_components, dim=-1)
        
        # Pad or trim to expected dimension
        current_dim = obs.shape[-1]
        expected_dim = config.observation_dim
        
        if current_dim < expected_dim:
            padding = torch.zeros((batch_size, expected_dim - current_dim), device=self.mother_env.device)
            obs = torch.cat([obs, padding], dim=-1)
        elif current_dim > expected_dim:
            obs = obs[:, :expected_dim]
        
        return obs
    
    def _compute_default_reward(self, robot, config: RobotConfig, env_indices: torch.Tensor) -> torch.Tensor:
        """Compute default reward for a robot."""
        # Get robot state
        pos = robot.get_pos()[env_indices]
        quat = robot.get_quat()[env_indices]
        vel = robot.get_vel()[env_indices]
        ang_vel = robot.get_ang()[env_indices]
        
        # Simple reward: height + stability - velocity penalties
        height_reward = torch.where(pos[:, 2] > 0.3, 1.0, -1.0)
        stability_reward = quat[:, 3] * 2.0 - 1.0  # w component of quaternion
        vel_penalty = -0.1 * torch.norm(vel, dim=-1)
        ang_vel_penalty = -0.1 * torch.norm(ang_vel, dim=-1)
        
        reward = height_reward + 0.5 * stability_reward + vel_penalty + ang_vel_penalty
        
        return reward
    
    def _apply_robot_action(self, robot, config: RobotConfig, actions: torch.Tensor, 
                           env_indices: torch.Tensor, robot_idx: int):
        """Apply actions to a specific robot."""
        # Get alive mask for this robot from mother environment
        if hasattr(self.mother_env, 'agent_alive_mask') and robot_idx < self.mother_env.agent_alive_mask.shape[1]:
            alive_mask = self.mother_env.agent_alive_mask[env_indices, robot_idx]
            alive_envs = env_indices[alive_mask]
        else:
            alive_envs = env_indices
        
        if len(alive_envs) == 0:
            return
        
        # Apply actions only to alive agents
        alive_actions = actions[alive_mask] if hasattr(self.mother_env, 'agent_alive_mask') else actions
        
        # Handle action dimension mismatch
        current_pos = robot.get_dofs_position()[alive_envs]
        current_pos = current_pos.to(alive_actions.device)
        actual_dof_count = current_pos.shape[-1]
        action_dim = alive_actions.shape[-1]
        
        if action_dim != actual_dof_count:
            if action_dim < actual_dof_count:
                # Pad actions with zeros
                padding = torch.zeros((alive_actions.shape[0], actual_dof_count - action_dim), 
                                    device=alive_actions.device)
                alive_actions = torch.cat([alive_actions, padding], dim=-1)
            else:
                # Truncate actions
                alive_actions = alive_actions[:, :actual_dof_count]
        
        # Scale actions
        scaled_actions = alive_actions * config.action_scale
        
        # Apply control based on mode
        if config.control_mode == "position" and hasattr(robot, 'control_dofs_position'):
            target_pos = current_pos + scaled_actions
            robot.control_dofs_position(target_pos, envs_idx=alive_envs)
        elif config.control_mode == "velocity" and hasattr(robot, 'control_dofs_velocity'):
            robot.control_dofs_velocity(scaled_actions, envs_idx=alive_envs)
        elif config.control_mode == "torque" and hasattr(robot, 'control_dofs_force'):
            robot.control_dofs_force(scaled_actions, envs_idx=alive_envs)