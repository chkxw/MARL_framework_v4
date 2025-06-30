#!/usr/bin/env python3
"""
Agent Class for Genesis MARL Framework V4

This module implements the Agent class which is the basic unit of interaction
with the AEC environment. Agents handle communication with SubVecEnv instances
and manage robot actions in the Genesis simulator.
"""

import torch
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from genesis_logging import get_module_logger
from configs import TrainingConfigBundle, JointTrainingBundle

from genesis.engine.entities import RigidEntity

logger = get_module_logger("agent")


class Agent:
    """Agent class for MARL environment interaction.

    The Agent is the basic unit of interaction with AEC environment.
    Based on a list of TrainingConfigBundle, each corresponding to a subVecEnv.

    Key responsibilities:
    - Communicate with subVecEnv instances through semaphores
    - Get actions from subVecEnv and apply them to Genesis simulator
    - Handle two-layer control with localmotion models
    - Batch localmotion inference for optimization
    """

    def __init__(
        self, agent_name: str, frequency: float, training_config_bundles: List[TrainingConfigBundle], mother_env: Any
    ):  # Will be VectorizedAECEnv
        """Initialize Agent.

        Args:
            agent_name: Name of the agent (contains frequency info)
            frequency: Control frequency for this agent (Hz)
            training_config_bundles: List of training bundles for this agent
            mother_env: Reference to the main AEC environment
        """
        self.agent_name = agent_name
        self.frequency = frequency
        self.training_config_bundles = training_config_bundles
        self.mother_env = mother_env

        # References to subVecEnvs and semaphores
        self.subvecenvs: Dict[str, Any] = {}  # training_name -> subVecEnv
        self.semaphores: Dict[str, threading.Semaphore] = {}  # training_name -> semaphore

        self.motor_dofs_idx_locals: Dict[str, List[int]] = {}  # robot_name -> list of motor dof indexes

        # Robot mappings for this agent
        self.robots: Dict[str, RigidEntity] = {}  # robot_name -> Robot Entity
        self.robot_names: List[str] = []  # List of robot names
        self.robot_configs: Dict[str, Any] = {}  # robot_name -> RobotConfig

        # Localmotion model optimization
        self.localmotion_models: Dict[str, Any] = {}  # path -> model instance
        self.localmotion_groups: Dict[str, List[str]] = {}  # path -> list of robot names

        logger.info(f"Initializing Agent {agent_name} at {frequency}Hz")
        logger.info(f"  Training bundles: {[bundle.training_name for bundle in training_config_bundles]}")

        for bundle in self.training_config_bundles:
            for robot_config in bundle.robot_configs:
                robot_name = robot_config.name
                self.robot_names.append(robot_name)
                self.robot_configs[robot_name] = robot_config

                logger.debug(f"  Added robot {robot_name} to agent {self.agent_name}")

        self._setup_localmotion_optimization()

        logger.debug(f"Agent {agent_name} initialization complete")
        logger.debug(f"  Total robots: {len(self.robot_names)}")
        logger.debug(f"  Robot names: {self.robot_names}")

    def _setup_localmotion_optimization(self):
        """
        Setup localmotion model grouping for batch optimization.
        Work on robot level
        """
        # Group robots by localmotion model path for batch inference
        localmotion_groups = {}

        for robot_name, robot_config in self.robot_configs.items():
            if robot_config.localmotion_path is not None:
                path = robot_config.localmotion_path
                if path not in localmotion_groups:
                    localmotion_groups[path] = []
                localmotion_groups[path].append(robot_name)

        self.localmotion_groups = localmotion_groups

        if localmotion_groups:
            logger.info(f"Agent {self.agent_name} localmotion optimization groups:")
            for path, robots in localmotion_groups.items():
                logger.info(f"  {path}: {robots}")

        # TODO: Load localmotion models - either .onnx or .pt

    def setup_robots(self):
        """Setup robots in the Genesis scene.

        This method is called by the AEC environment during initialization
        to add robots to the Genesis simulator.
        """
        logger.info(f"Setting up robots for agent {self.agent_name}")

        for training_bundle in self.training_config_bundles:
            training_bundle.setup_function(self.mother_env)

            for robot_cfg in training_bundle.robot_configs:
                robot_name = robot_cfg.name
                self.robots[robot_name] = self.mother_env.robots[robot_name]
                dofs_idx_local = []
                for joint_name in robot_cfg.joint_names:
                    dofs_idx_local.extend(self.robots[robot_name].get_joint(joint_name).dofs_idx_local)

                self.motor_dofs_idx_locals[robot_name] = torch.tensor(dofs_idx_local, device=self.mother_env.device)

    def _setup_robot_dp_parameters(self, robot_name):
        """Setup DP (Differential Position) parameters for robot control.

        Should be called after scene is built

        Args:
            robot: Genesis robot entity
            robot_config: RobotConfig with DP parameters
        """

        robot, robot_config = self.mother_env.robots[robot_name], self.robot_configs[robot_name]

        # Set PD control gains for all DOFs
        if hasattr(robot, 'set_dofs_kp') and hasattr(robot, 'set_dofs_kv'):
            robot.set_dofs_kp(robot_config.DP_kp, dofs_idx_local=self.motor_dofs_idx_locals[robot_name])
            robot.set_dofs_kv(robot_config.DP_kd, dofs_idx_local=self.motor_dofs_idx_locals[robot_name])
        else:
            logger.warning(f"Robot {robot_config.name} does not support PD control parameter setting")

    def _reset_robots(self, env_indices: torch.Tensor):
        """Reset robots in specified environments.

        Args:
            env_indices: Tensor of environment indices to reset
        """
        logger.debug(f"Agent {self.agent_name} resetting envs: {env_indices.tolist()}")

        for robot_name in self.robot_names:
            robot = self.mother_env.robots[robot_name]
            robot_config = self.robot_configs[robot_name]

            # TODO allow more options for initial pos / velocity / joint values
            n_dofs = robot.get_dofs_position().shape[1]
            initial_joint_pos = torch.zeros(n_dofs, device=env_indices.device, dtype=torch.float32)
            initial_joint_pos_expanded = initial_joint_pos.unsqueeze(0).expand(len(env_indices), -1)

            robot.set_dofs_position(initial_joint_pos_expanded, envs_idx=env_indices, zero_velocity=True)

            initial_pos = torch.tensor(robot_config.initial_position, device=env_indices.device, dtype=torch.float32)
            initial_quat = torch.tensor(
                robot_config.initial_orientation, device=env_indices.device, dtype=torch.float32
            )

            # Expand to match environment indices
            initial_pos_expanded = initial_pos.unsqueeze(0).expand(len(env_indices), -1)
            initial_quat_expanded = initial_quat.unsqueeze(0).expand(len(env_indices), -1)

            # Reset robot to initial position/orientation
            robot.set_pos(initial_pos_expanded, envs_idx=env_indices)
            robot.set_quat(initial_quat_expanded, envs_idx=env_indices)

            logger.debug(f"  Reset robot {robot_name} in envs {env_indices.tolist()}")

    def step(self):
        """Execute one step of the agent.

        1. Get actions from all subVecEnvs (wait if not available)
        2. Split actions per robot if needed
        3. Apply localmotion if configured
        4. Apply actions to Genesis simulator
        """
        logger.debug(f"Agent {self.agent_name} stepping")

        # 1. Wait for actions from all subVecEnvs
        self.wait_subvecenvs()

        robot_actions = {self.mother_env.action_buffers[robot_name] for robot_name in self.robot_names}

        # 3. Apply localmotion models if configured
        joint_actions = self._apply_localmotion_models(robot_actions)

        # 4. Apply joint actions to Genesis simulator
        for robot_name, joint_action in joint_actions.items():
            robot = self.robots[robot_name]
            robot_config = self.robot_configs[robot_name]

            # Scale actions
            scaled_action = joint_action * robot_config.action_scale

            # Apply to simulator based on control mode
            self._apply_robot_action(robot, robot_config, scaled_action)

            logger.debug(f"Applied action to robot {robot_name}: shape {scaled_action.shape}")

        logger.debug(f"Agent {self.agent_name} step complete")

    def _apply_localmotion_models(self, robot_actions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply localmotion models to convert high-level actions to joint values.

        Args:
            robot_actions: Dictionary mapping robot_name -> action tensor

        Returns:
            Dictionary mapping robot_name -> joint_action tensor
        """

        # TODO: haven't check yet
        joint_actions = {}

        # Process robots with localmotion models in batches
        for localmotion_path, robot_names in self.localmotion_groups.items():
            if localmotion_path in self.localmotion_models:
                # Batch inference for efficiency
                model = self.localmotion_models[localmotion_path]

                # Collect actions for this group
                group_actions = []
                for robot_name in robot_names:
                    group_actions.append(robot_actions[robot_name])

                # Batch inference
                batched_actions = torch.cat(group_actions, dim=0)
                batched_joint_actions = model.inference(batched_actions)

                # Split results back to individual robots
                start_idx = 0
                for robot_name in robot_names:
                    n_envs = robot_actions[robot_name].shape[0]
                    end_idx = start_idx + n_envs
                    joint_actions[robot_name] = batched_joint_actions[start_idx:end_idx]
                    start_idx = end_idx

                logger.debug(f"Applied localmotion model {localmotion_path} to robots {robot_names}")

        # For robots without localmotion, actions are already joint values
        for robot_name in self.robot_names:
            if robot_name not in joint_actions:
                joint_actions[robot_name] = robot_actions[robot_name]

        return joint_actions

    def _apply_robot_action(self, robot_name, action: torch.Tensor):
        """Apply action to a single robot in the simulator.

        Args:
            robot: Genesis robot entity
            robot_config: RobotConfig
            action: Joint action tensor [n_envs, n_joints]
        """
        robot_config = self.robot_configs[robot_name]
        robot = self.robots[robot_name]

        control_mode = robot_config.control_mode
        logger.debug(f"Applying {control_mode} action to robot {robot_config.name}, shape: {action.shape}")

        dofs_idx_local = self.motor_dofs_idx_locals[robot_name]

        if len(dofs_idx_local) != action.shape[1]:
            logger.warning(
                f"Robot {robot_config.name} has {len(dofs_idx_local)} DOFs but action has {action.shape[1]} DOFs"
            )

        if control_mode == "position":
            robot.control_dofs_position(position=action, dofs_idx_local=dofs_idx_local)
        elif control_mode == "velocity":
            robot.control_dofs_velocity(velocity=action, dofs_idx_local=dofs_idx_local)
        elif control_mode == "force":
            robot.control_dofs_force(force=action, dofs_idx_local=dofs_idx_local)
        else:
            raise ValueError(f"Unknown control mode: {control_mode}")

    def last(self):
        """Send observations/rewards/terminations/truncations/info to subVecEnvs.

        Args:
            dont_send: If True, don't set semaphores (used for initial setup)
        """
        logger.debug(f"Agent {self.agent_name} last() called")

        # Move data from AEC's buffers to subVecEnv buffers
        for training_cfg in self.training_config_bundles:
            training_name = training_cfg.training_name
            subvecenv = self.subvecenvs[training_name]

            for i, robot_cfg in enumerate(training_cfg.robot_configs):
                subvecenv.obs_buffer[robot_cfg.name] = self.mother_env.observations[robot_cfg.name]
                subvecenv.rew_buffer[robot_cfg.name] = self.mother_env._cumulative_rewards[robot_cfg.name]
                subvecenv.term_buffer[robot_cfg.name] = self.mother_env.terminations[robot_cfg.name]
                subvecenv.trunc_buffer[robot_cfg.name] = self.mother_env.truncations[robot_cfg.name]
                subvecenv.info_buffer[robot_cfg.name] = self.mother_env.infos[robot_cfg.name]

                if i == 0 and isinstance(training_cfg, JointTrainingBundle):
                    break  # Only first robot in bundle has valid obs/rew/term/trunc/info

        # Notify subVecEnvs that data is ready
        self.notify_subvecenvs()

        logger.debug(f"Agent {self.agent_name} last() complete")

    def wait_subvecenvs(self):
        """Wait for all subVecEnv semaphores to be 1 (actions ready)."""
        logger.debug(f"Agent {self.agent_name} waiting for subVecEnvs")

        for training_name in [bundle.training_name for bundle in self.training_config_bundles]:
            semaphore = self.semaphores[training_name]
            semaphore.acquire()  # Wait for semaphore to be 1
            logger.debug(f"  Received signal from {training_name}")

    def notify_subvecenvs(self):
        """Set all subVecEnv semaphores to 0 (data ready)."""
        logger.debug(f"Agent {self.agent_name} notifying subVecEnvs")

        for training_name in [bundle.training_name for bundle in self.training_config_bundles]:
            semaphore = self.semaphores[training_name]
            semaphore.release()  # Set semaphore to 0
            logger.debug(f"  Notified {training_name}")


if __name__ == "__main__":
    print("Agent class implementation complete")
    print("This module should be imported and used within the MARL framework")
