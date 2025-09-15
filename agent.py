#!/usr/bin/env python3
"""
Agent Class for Genesis MARL Framework V4

This module implements the Agent class which is the basic unit of interaction
with the AEC environment. Agents handle communication with SubVecEnv instances
and manage robot actions through the simulator interface.
"""

from typing import Any, Dict, List

import torch

from configs.TrainingConfig import TrainingConfig
from marl_logging import get_class_logger
from model_inference import ModelInference
from simulation_interface import RobotInterface


class Agent:
    """Agent class for MARL environment interaction.

    The Agent is the basic unit of interaction with AEC environment.
    Based on a list of TrainingConfigBundle, each corresponding to a subVecEnv.

    Key responsibilities:
    - Communicate with subVecEnv instances through semaphores
    - Get actions from subVecEnv and apply them through simulator interface
    - Handle two-layer control with locomotion models
    - Batch locomotion inference for optimization
    """

    def __init__(
        self, agent_name: str, frequency: float, training_configs: List[TrainingConfig], mother_env: Any
    ):  # Will be VectorizedAECEnv
        """Initialize Agent.

        Args:
            agent_name: Name of the agent (contains frequency info)
            frequency: Control frequency for this agent (Hz)
            training_configs: List of training bundles for this agent
            mother_env: Reference to the main AEC environment
        """
        # Initialize independent class-specific logger
        self.logger = get_class_logger("Agent", agent_name, level="INFO")

        self.agent_name = agent_name
        self.frequency = frequency
        self.training_configs = training_configs
        self.mother_env = mother_env
        self.coordinator = None

        # References to subVecEnvs and coordinator
        self.subvecenvs: Dict[str, Any] = {}  # training_name -> subVecEnv

        # Robot mappings for this agent
        self.robots: Dict[str, RobotInterface] = {}  # robot_name -> RobotInterface
        self.robot_names: List[str] = []  # List of robot names
        self.robot_configs: Dict[str, Any] = {}  # robot_name -> RobotConfig

        # locomotion model optimization
        self.locomotion_models: Dict[str, ModelInference] = {}  # path -> ModelInference instance
        self.locomotion_groups: Dict[str, List[str]] = {}  # path -> list of robot names

        self.logger.info(f"Initializing Agent {agent_name} at {frequency}Hz")
        self.logger.info(f"  Training bundles: {[bundle.training_name for bundle in training_configs]}")

        for bundle in self.training_configs:
            for robot_name, robot_config in bundle.robot_configs.items():
                self.robot_names.append(robot_name)
                self.robot_configs[robot_name] = robot_config

        self._setup_locomotion_optimization()

        self.logger.debug(f"Agent {agent_name} initialization complete")
        self.logger.debug(f"  Total robots: {len(self.robot_names)}")
        self.logger.debug(f"  Robot names: {self.robot_names}")

    def set_coordinator(self, coordinator):
        """Set the coordinator reference after initialization."""
        self.coordinator = coordinator
        self.logger.debug(f"Agent {self.agent_name} coordinator set")

    def _setup_locomotion_optimization(self):
        """
        Setup locomotion model grouping for batch optimization.
        Work on robot level
        """
        # Group robots by locomotion model path for batch inference
        locomotion_groups = {}

        for robot_name, robot_config in self.robot_configs.items():
            if robot_config.locomotion_path is not None:
                path = robot_config.locomotion_path
                if path not in locomotion_groups:
                    locomotion_groups[path] = []
                locomotion_groups[path].append(robot_name)

        self.locomotion_groups = locomotion_groups

        if locomotion_groups:
            self.logger.info(f"Agent {self.agent_name} locomotion optimization groups:")
            for path, robots in locomotion_groups.items():
                self.logger.info(f"  {path}: {robots}")

        # Load locomotion models
        for locomotion_path in self.locomotion_groups.keys():

            self.logger.info(f"Loading locomotion model: {locomotion_path}")

            # Simple enhanced initialization
            inference = ModelInference(force_cpu=False)
            inference.load_model(locomotion_path)  # Auto-detects ONNX/PyTorch
            self.logger.info(f"Loaded locomotion model: {locomotion_path}")

            # Store the enhanced inference instance
            self.locomotion_models[locomotion_path] = inference

            robot_names = self.locomotion_groups[locomotion_path]
            action_dim = self.robot_configs[robot_names[0]].action_space.shape[0]

            sample_input = torch.randn(action_dim, dtype=torch.float32)
            inference.record_performance_analysis(sample_input, max_batch_size=len(robot_names), verbose=False)

            # Log optimization status
            status = inference.get_performance_status()
            self.logger.info(f"  Ultra-efficient optimization: {status['is_analyzed']}")
            self.logger.info(f"  GPU beneficial types: {status['gpu_beneficial_types']}")
            self.logger.info(f"  CPU-only types: {status['cpu_only_types']}")

            for input_type, threshold in status['crossover_thresholds'].items():
                if threshold == "never":
                    self.logger.info(f"    {input_type}: CPU always optimal")
                else:
                    self.logger.info(f"    {input_type}: GPU optimal at batch >= {threshold}")

    def setup_robots(self):
        """Setup robots in the Genesis scene.

        This method is called by the AEC environment during initialization
        to add robots to the Genesis simulator.
        """
        self.logger.info(f"Setting up robots for agent {self.agent_name}")

        for training_bundle in self.training_configs:
            training_bundle.setup_function(self.mother_env)

            for robot_name, robot_cfg in training_bundle.robot_configs.items():
                self.robots[robot_name] = self.mother_env.robots[robot_name]
                robot_interface = self.robots[robot_name]

                dofs_idx_local = []
                for joint_name in robot_cfg.joint_names:
                    joint_new = robot_interface.get_joint(joint_name)
                    dofs_idx_local_new = joint_new.dofs_idx_local

                    dofs_idx_local.extend(dofs_idx_local_new)

                self.mother_env.joint_dofs_idx_locals[robot_name] = torch.tensor(
                    dofs_idx_local, device=self.mother_env.device
                )

    def _setup_robot_dp_parameters(self, robot_name):
        """Setup DP (Differential Position) parameters for robot control.

        Should be called after scene is built

        Args:
            robot: Genesis robot entity
            robot_config: RobotConfig with DP parameters
        """

        robot_interface, robot_config = self.mother_env.robots[robot_name], self.robot_configs[robot_name]

        # Set PD control gains for all DOFs using both old and new interfaces for validation
        joint_indices = self.mother_env.joint_dofs_idx_locals[robot_name]

        robot_interface.set_kp_gains(robot_config.DP_kp, joint_indices=joint_indices)
        robot_interface.set_kd_gains(robot_config.DP_kd, joint_indices=joint_indices)

    def _reset_robots(self, env_indices: torch.Tensor):
        """Reset robots in specified environments.

        Args:
            env_indices: Tensor of environment indices to reset
        """
        self.logger.debug(f"Agent {self.agent_name} resetting envs: {env_indices.tolist()}")

        for robot_name in self.robot_names:
            robot_interface = self.mother_env.robots[robot_name]
            robot_config = self.robot_configs[robot_name]

            # Use the configured default joint positions for controllable joints only
            initial_joint_pos = torch.tensor(
                robot_config.initial_joint_pos, device=env_indices.device, dtype=torch.float32
            )
            initial_joint_vel = torch.tensor(
                robot_config.initial_joint_vel, device=env_indices.device, dtype=torch.float32
            )
            initial_joint_pos_expanded = initial_joint_pos.unsqueeze(0).expand(len(env_indices), -1)
            initial_joint_vel_expanded = initial_joint_vel.unsqueeze(0).expand(len(env_indices), -1)

            joint_indices = self.mother_env.joint_dofs_idx_locals[robot_name]

            robot_interface.set_joint_pos(
                initial_joint_pos_expanded,
                joint_indices=joint_indices,
                env_indices=env_indices,
                zero_velocity=True,
            )
            robot_interface.set_joint_vel(
                initial_joint_vel_expanded,
                joint_indices=joint_indices,
                env_indices=env_indices,
            )

            initial_pos = torch.tensor(robot_config.initial_position, device=env_indices.device, dtype=torch.float32)
            initial_quat = torch.tensor(
                robot_config.initial_orientation, device=env_indices.device, dtype=torch.float32
            )
            initial_vel = torch.tensor(robot_config.initial_velocity, device=env_indices.device, dtype=torch.float32)
            initial_ang_vel = torch.tensor(
                robot_config.initial_angular_velocity, device=env_indices.device, dtype=torch.float32
            )

            # Expand to match environment indices
            initial_pos_expanded = initial_pos.unsqueeze(0).expand(len(env_indices), -1)
            initial_quat_expanded = initial_quat.unsqueeze(0).expand(len(env_indices), -1)
            initial_vel_expanded = initial_vel.unsqueeze(0).expand(len(env_indices), -1)
            initial_ang_vel_expanded = initial_ang_vel.unsqueeze(0).expand(len(env_indices), -1)

            # Reset robot to initial position/orientation using both old and new interfaces
            robot_interface.set_pos(initial_pos_expanded, env_indices=env_indices)
            robot_interface.set_orientation(initial_quat_expanded, env_indices=env_indices)

            # Check if vel and angular vel should be set (All 0 means no need to be set)
            no_need_to_set_vel = torch.allclose(initial_vel, torch.zeros_like(initial_vel))
            no_need_to_set_ang_vel = torch.allclose(initial_ang_vel, torch.zeros_like(initial_ang_vel))
            if not (no_need_to_set_vel and no_need_to_set_ang_vel):
                robot_interface.set_lin_vel(initial_vel_expanded, env_indices=env_indices)
                robot_interface.set_ang_vel(initial_ang_vel_expanded, env_indices=env_indices)

            self.logger.debug(f"  Reset robot {robot_name} in envs {env_indices.tolist()}")

    def _apply_locomotion_models(self, robot_actions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply locomotion models to convert high-level actions to joint values.

        Args:
            robot_actions: Dictionary mapping robot_name -> action tensor

        Returns:
            Dictionary mapping robot_name -> joint_action tensor
        """
        joint_actions = {}
        n_envs = self.mother_env.n_envs

        # Process robots with locomotion models in batches
        for locomotion_path, robot_names in self.locomotion_groups.items():
            # Batch inference for efficiency
            inference = self.locomotion_models[locomotion_path]

            # Collect actions for this group
            group_actions = []
            for robot_name in robot_names:
                n_envs_action = robot_actions[robot_name]
                # Reshape to (n_envs, actions...)
                group_actions.append(n_envs_action.view(n_envs_action.shape[0], -1))

            # Use n_envs dim as batch dim, stack actions from all robots
            batched_actions = torch.cat(group_actions, dim=0)

            # Universal inference with automatic GPU/CPU optimization
            batched_joint_actions = inference.predict(batched_actions)

            # Dispatch inferred actions to individual robots
            split_joint_actions = torch.split(batched_joint_actions, n_envs, dim=0)
            for i, robot_name in enumerate(robot_names):
                self.mother_env.joint_action_buffers[robot_name] = split_joint_actions[i]
                joint_actions[robot_name] = split_joint_actions[i]

        # For robots without locomotion, actions are already joint values
        for robot_name in self.robot_names:
            if robot_name not in joint_actions:
                self.mother_env.joint_action_buffers[robot_name] = robot_actions[robot_name]
                joint_actions[robot_name] = robot_actions[robot_name]

        # Check if there's nan value in joint actions
        for robot_name, joint_action in joint_actions.items():
            if torch.isnan(joint_action).any():
                self.logger.warning(f"NaN value detected in joint actions for robot {robot_name}")
        return joint_actions

    def _apply_robot_action(self, robot_name, action: torch.Tensor):
        """Apply action to a single robot in the simulator.

        Args:
            robot: Genesis robot entity
            robot_config: RobotConfig
            action: Joint action tensor [n_envs, n_joints]
        """
        robot_config = self.robot_configs[robot_name]
        robot_interface = self.robots[robot_name]

        control_mode = robot_config.control_mode
        self.logger.debug(f"Applying {control_mode} action to robot {robot_config.name}, shape: {action.shape}")

        dofs_idx_local = self.mother_env.joint_dofs_idx_locals[robot_name]

        if len(dofs_idx_local) != action.shape[1]:
            self.logger.warning(
                f"Robot {robot_config.name} has {len(dofs_idx_local)} DOFs but action has {action.shape[1]} DOFs"
            )

        # Apply control using both old and new interfaces for validation
        if control_mode == "position":
            robot_interface.control_joint_pos(target_pos=action, joint_indices=dofs_idx_local)
        elif control_mode == "velocity":
            robot_interface.control_joint_vel(target_vel=action, joint_indices=dofs_idx_local)
        elif control_mode == "force":
            robot_interface.control_joint_force(forces=action, joint_indices=dofs_idx_local)
        else:
            raise ValueError(f"Unknown control mode: {control_mode}")

    def _wait_for_actions(self):
        """Wait for all subVecEnvs to provide actions."""
        if self.coordinator is None:
            raise RuntimeError(f"Agent {self.agent_name} coordinator not set")

        # Get list of training states we need to wait for
        training_states = [bundle.training_name for bundle in self.training_configs]

        self.logger.debug(f"Agent {self.agent_name} waiting for actions from: {training_states}")

        # Wait for all training states to finish providing actions
        self.coordinator.wait(training_states)

        self.logger.debug(f"Agent {self.agent_name} received all actions")

    def _notify_data_ready(self):
        """Notify all subVecEnvs that observation/reward data is ready."""
        if self.coordinator is None:
            raise RuntimeError(f"Agent {self.agent_name} coordinator not set")

        # Get list of training states to wake up
        training_states = [training_cfg.training_name for training_cfg in self.training_configs]

        self.logger.debug(f"Agent {self.agent_name} notifying data ready to: {training_states} subVecEnvs")

        # Reset finished flags and wake up training threads
        self.coordinator.set_unfinished(training_states)
        self.coordinator.wake(training_states)

        self.logger.debug(f"Agent {self.agent_name} notification complete")


if __name__ == "__main__":
    print("Agent class implementation complete")
    print("This module should be imported and used within the MARL framework")
