#!/usr/bin/env python3
"""
Unified Go1/Go2 MARL Training Script using Genesis MARL Framework V4

This script replicates the original go1/go_rsl_rl_training functionality using
the new Genesis MARL framework architecture. It demonstrates how to create
equivalent Go1 or Go2 locomotion training using the VectorizedAECEnv approach.
"""

import argparse
import os
import pickle
import shutil
import threading
import time
from dataclasses import asdict
from typing import Any, Dict

import genesis as gs
import torch
from utils import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat

from configs.SimulatorConfig import SimulatorConfig
from configs.RobotConfig import RobotConfig
from marl_logging import get_class_logger
from rsl_rl.runners import OnPolicyRunner
from utils import gs_rand_float
from trainer_interfaces.rsl_rl_interface import (
    RSL_RLTrainingConfig,
    AlgorithmConfig,
    RunnerConfig,
    RSL_RLConfig,
    PolicyConfig,
)

# Import MARL framework components
from vectorized_aec_env import VectorizedAECEnv
from configs.CameraConfig import CameraConfig


# Go1/Go2 configuration parameters (from original training scripts)
env_cfg = {
    "num_actions": 12,
    "default_joint_angles": {  # [rad]
        "FL_hip_joint": 0.0,
        "FR_hip_joint": 0.0,
        "RL_hip_joint": 0.0,
        "RR_hip_joint": 0.0,
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
    },
    "joint_names": [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ],
    "kp": 20.0,
    "kd": 0.5,
    "termination_if_roll_greater_than": 10,  # degree
    "termination_if_pitch_greater_than": 10,
    "base_init_pos": [0.0, 0.0, 0.42],
    "base_init_quat": [1.0, 0.0, 0.0, 0.0],
    "episode_length_s": 20.0,
    "resampling_time_s": 4.0,
    "action_scale": 0.25,
    "simulate_action_latency": False,  # Enable action latency simulation like original
    "clip_actions": 100.0,
}

obs_cfg = {
    "num_obs": 45,
    "obs_scales": {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
    },
}

reward_cfg = {
    "tracking_sigma": 0.25,
    "base_height_target": 0.3,
    "feet_height_target": 0.075,
    "reward_scales": {
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.2,
        "lin_vel_z": -1.0,
        "base_height": -50.0,
        "action_rate": -0.005,
        "similar_to_default": -0.1,
    },
}

command_cfg = {
    "num_commands": 3,
    "lin_vel_x_range": [-1.0, 2.5],
    "lin_vel_y_range": [-0.8, 0.8],
    "ang_vel_range": [-0.9, 0.9],
}


def create_go_robot_config(name: str, robot_type: str = "go2", frequency: int = 50) -> RobotConfig:
    """Create a Go1/Go2 robot configuration with proper obs/reward/termination functions.

    This replicates the functionality from the original go1/go_env.py but in a functional
    approach compatible with the Genesis MARL framework.

    Args:
        name: Name of the robot
        robot_type: Either "go1" or "go2"
    """

    # Set URDF path based on robot type
    if robot_type == "go1":
        urdf_path = "go1_standalone/urdf/go1.urdf"
    elif robot_type == "go2":
        urdf_path = "go2_standalone/urdf/go2.urdf"
    else:
        raise ValueError(f"Unknown robot type: {robot_type}. Must be 'go1' or 'go2'")

    def go_setup_function(robot_name, env):
        """Setup function for Go robot."""
        robot_cfg = env.robot_configs[robot_name]
        # Use the robot's URDF path
        robot = env.scene.add_robot(
            name=robot_name,
            urdf_path=robot_cfg.urdf_path,
            pos=robot_cfg.initial_position,
            quat=robot_cfg.initial_orientation,
            fixed=False,
        )

        # Register _last_actions field for action latency simulation (only once for all robots)
        if not hasattr(env, '_last_actions'):
            sample_action = torch.zeros(robot_cfg.action_space.shape, device=env.device, dtype=torch.float32)
            env.register_tensor_field(
                "_last_actions",
                sample_action,
                per_robot=True,  # Creates per-robot RefDict automatically
                per_env=True,  # One tensor per environment
                clear_in_reset=True,  # Clear when environments reset
                clear_after_use=False,  # Don't clear after each step
            )

        # Register reward fields for debugging
        env.reward_components = [
            "rew_tracking_lin_vel",
            "rew_tracking_ang_vel",
            "rew_lin_vel_z",
            "rew_base_height",
            "rew_action_rate",
            "rew_similar_to_default",
            "rew_total",
        ]
        for key in env.reward_components:
            env.register_tensor_field(
                key,
                torch.tensor(0.0, device=env.device, dtype=torch.float32),
                per_robot=True,
                per_env=True,
                clear_in_reset=True,
                clear_after_use=False,
            )

        # Register command field
        env._commands = {}
        env._commands[name] = torch.zeros(
            (env.n_envs, command_cfg["num_commands"]), device=env.device, dtype=torch.float32
        )
        # Initial command sampling
        env._commands[name][:, 0] = gs_rand_float(*command_cfg["lin_vel_x_range"], (env.n_envs,), env.device)
        env._commands[name][:, 1] = gs_rand_float(*command_cfg["lin_vel_y_range"], (env.n_envs,), env.device)
        env._commands[name][:, 2] = gs_rand_float(*command_cfg["ang_vel_range"], (env.n_envs,), env.device)

        return robot

    def go_pre_reset_hook(env, env_indices):
        # We cannot write to infos directly, since info will be clear during reset, we need to do it after reset

        if "episode" not in env.infos:
            env.infos[name]["episode"] = {}
        for key in env.reward_components:
            tensor = env.__dict__[key][name][env_indices]
            mean = torch.mean(tensor).item()
            env.infos[name]["episode"][key] = mean
            tensor.zero_()

    def go_post_reset_hook(env, env_indices):
        if hasattr(env, '_commands') and name in env._commands:
            env._commands[name][env_indices, 0] = gs_rand_float(
                *command_cfg["lin_vel_x_range"], (len(env_indices),), env.device
            )
            env._commands[name][env_indices, 1] = gs_rand_float(
                *command_cfg["lin_vel_y_range"], (len(env_indices),), env.device
            )
            env._commands[name][env_indices, 2] = gs_rand_float(
                *command_cfg["ang_vel_range"], (len(env_indices),), env.device
            )

    def go_post_step_hook(env):
        """Calculate all intermediates"""
        # Resample commands every resampling_time_s using existing episode_frame_count
        # Convert episode frames to robot control steps (episode_frame_count is at simulation frequency)
        robot_config = env.robot_configs[name]

        robot_steps = ((env.episode_frame_count.float() / env.simulation_frequency) * robot_config.frequency).int()
        resampling_steps = int(env_cfg["resampling_time_s"] * robot_config.frequency)
        envs_to_resample = (robot_steps % resampling_steps == 0).nonzero(as_tuple=False).flatten()

        if len(envs_to_resample) > 0:
            env._commands[name][envs_to_resample, 0] = gs_rand_float(
                *command_cfg["lin_vel_x_range"], (len(envs_to_resample),), env.device
            )
            env._commands[name][envs_to_resample, 1] = gs_rand_float(
                *command_cfg["lin_vel_y_range"], (len(envs_to_resample),), env.device
            )
            env._commands[name][envs_to_resample, 2] = gs_rand_float(
                *command_cfg["ang_vel_range"], (len(envs_to_resample),), env.device
            )

    def go_obs_function(env) -> torch.Tensor:
        """Go observation function - 45 dimensional observation."""
        robot_interface = env.robots[name]
        robot_config = env.robot_configs[name]
        n_envs = env.n_envs
        device = env.device

        base_pos = robot_interface.get_pos()  # [n_envs, 3]
        base_quat = robot_interface.get_orientation(format="quat")  # [n_envs, 4]
        base_lin_vel = robot_interface.get_lin_vel()  # [n_envs, 3]
        base_ang_vel = robot_interface.get_ang_vel()  # [n_envs, 3]

        # Get joint indices for the controllable joints
        dofs_idx_local = env.joint_dofs_idx_locals[name]

        dof_pos = robot_interface.get_joint_pos(joint_indices=dofs_idx_local)  # [n_envs, total_dofs]
        dof_vel = robot_interface.get_joint_vel(joint_indices=dofs_idx_local)  # [n_envs, total_dofs]

        # Transform velocities to robot frame
        inv_base_quat = inv_quat(base_quat)
        base_lin_vel = transform_by_quat(base_lin_vel, inv_base_quat)
        base_ang_vel = transform_by_quat(base_ang_vel, inv_base_quat)

        # Calculate projected gravity
        global_gravity = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32).repeat(n_envs, 1)
        projected_gravity = transform_by_quat(global_gravity, inv_base_quat)

        # Use dynamic commands
        commands = env._commands[name]

        # Default joint positions
        default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][joint] for joint in env_cfg["joint_names"]],
            device=device,
            dtype=torch.float32,
        )

        # Now the previous action should in the action buffer
        if name not in env.action_buffers:
            prev_actions = torch.zeros((n_envs, *robot_config.action_space.shape), device=device, dtype=torch.float32)
        else:
            prev_actions = env.action_buffers[name]

        # Scale observations
        commands_scale = torch.tensor(
            [obs_cfg["obs_scales"]["lin_vel"], obs_cfg["obs_scales"]["lin_vel"], obs_cfg["obs_scales"]["ang_vel"]],
            device=device,
            dtype=torch.float32,
        )

        # Concatenate observations [45 dims total]
        obs = torch.cat(
            [
                base_ang_vel * obs_cfg["obs_scales"]["ang_vel"],  # 3
                projected_gravity,  # 3
                commands * commands_scale,  # 3
                (dof_pos - default_dof_pos) * obs_cfg["obs_scales"]["dof_pos"],  # 12
                dof_vel * obs_cfg["obs_scales"]["dof_vel"],  # 12
                prev_actions,  # 12
            ],
            dim=-1,
        )

        # Also manually set observation critic
        if name not in env.infos:
            env.infos[name] = {}
        if "observations" not in env.infos[name]:
            env.infos[name]["observations"] = {}
        env.infos[name]["observations"]["critic"] = obs

        return obs

    def go_reward_function(env) -> Dict[str, torch.Tensor]:
        """Go reward function - implements all reward components from original."""
        robot_interface = env.robots[name]
        robot_cfg = env.robot_configs[name]
        n_envs = env.n_envs
        device = env.device

        base_pos = robot_interface.get_pos()
        base_quat = robot_interface.get_orientation(format="quat")
        base_lin_vel = robot_interface.get_lin_vel()
        base_ang_vel = robot_interface.get_ang_vel()

        # Get joint indices for the controllable joints (same as in obs function)
        dofs_idx_local = env.joint_dofs_idx_locals[name]

        # Get only controllable joint positions using both APIs
        dof_pos = robot_interface.get_joint_pos(joint_indices=dofs_idx_local)

        # Transform to robot frame
        inv_base_quat = inv_quat(base_quat)
        base_lin_vel = transform_by_quat(base_lin_vel, inv_base_quat)
        base_ang_vel = transform_by_quat(base_ang_vel, inv_base_quat)

        # Get commands (use the same dynamic commands from obs function)
        if hasattr(env, '_commands') and name in env._commands:
            commands = env._commands[name]  # Use raw commands for reward (like original)
        else:
            # Fallback to static commands if not initialized
            commands = torch.zeros((n_envs, 3), device=device, dtype=torch.float32)
            commands[:, 0] = command_cfg["lin_vel_x_range"][0]

        if hasattr(env, 'action_buffers') and name in env.action_buffers:
            actions = env.action_buffers[name]
        else:
            actions = torch.zeros((n_envs, env_cfg["num_actions"]), device=device, dtype=torch.float32)

        # Get last actions (for action rate penalty) - use framework's _last_actions
        if hasattr(env, '_last_actions') and name in env._last_actions:
            last_actions = env._last_actions[name].clone()
        else:
            last_actions = torch.zeros_like(actions)

        # Update last actions for next step
        env._last_actions[name].copy_(actions)
        # Default joint positions
        default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][joint] for joint in env_cfg["joint_names"]],
            device=device,
            dtype=torch.float32,
        )

        # Reward components
        total_reward = torch.zeros((n_envs, 1), device=device, dtype=torch.float32)

        # Very very weird thing: We need to multiply dt into reward_cfg["reward_scales"] first. If we multiply dt to scaled reward, it will crush training???????????????????????

        dt = 1.0 / robot_cfg.frequency
        real_scale_factor = {key: value * dt for key, value in reward_cfg["reward_scales"].items()}
        # 1. Tracking linear velocity (xy)
        lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        tracking_lin_vel = torch.exp(-lin_vel_error / reward_cfg["tracking_sigma"])
        rew_tracking_lin_vel = tracking_lin_vel * real_scale_factor["tracking_lin_vel"]

        # 2. Tracking angular velocity (yaw)
        ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
        tracking_ang_vel = torch.exp(-ang_vel_error / reward_cfg["tracking_sigma"])
        rew_tracking_ang_vel = tracking_ang_vel * real_scale_factor["tracking_ang_vel"]

        # 3. Penalize z-axis linear velocity
        lin_vel_z_penalty = torch.square(base_lin_vel[:, 2])
        rew_lin_vel_z = lin_vel_z_penalty * real_scale_factor["lin_vel_z"]

        # 4. Base height penalty
        base_height_error = torch.square(base_pos[:, 2] - reward_cfg["base_height_target"])
        rew_base_height = base_height_error * real_scale_factor["base_height"]

        # 5. Action rate penalty (change in actions)
        action_rate = torch.sum(torch.square(last_actions - actions), dim=1)
        rew_action_rate = action_rate * real_scale_factor["action_rate"]

        # 6. Joint position similarity to default
        joint_deviation = torch.sum(torch.abs(dof_pos - default_dof_pos), dim=1)
        rew_similar_to_default = joint_deviation * real_scale_factor["similar_to_default"]

        # Total reward
        total_reward = (
            rew_tracking_lin_vel
            + rew_tracking_ang_vel
            + rew_lin_vel_z
            + rew_base_height
            + rew_action_rate
            + rew_similar_to_default
        )

        # Store individual reward components in env.infos for TensorBoard logging
        if name not in env.infos:
            env.infos[name] = {}

        env.infos[name]["reward_components"] = {}

        env.infos[name]["reward_components"]["tracking_lin_vel"] = rew_tracking_lin_vel
        env.infos[name]["reward_components"]["tracking_ang_vel"] = rew_tracking_ang_vel
        env.infos[name]["reward_components"]["lin_vel_z"] = rew_lin_vel_z
        env.infos[name]["reward_components"]["base_height"] = rew_base_height
        env.infos[name]["reward_components"]["action_rate"] = rew_action_rate
        env.infos[name]["reward_components"]["similar_to_default"] = rew_similar_to_default

        env.rew_tracking_lin_vel[name] += rew_tracking_lin_vel
        env.rew_tracking_ang_vel[name] += rew_tracking_ang_vel
        env.rew_lin_vel_z[name] += rew_lin_vel_z
        env.rew_base_height[name] += rew_base_height
        env.rew_action_rate[name] += rew_action_rate
        env.rew_similar_to_default[name] += rew_similar_to_default
        env.rew_total[name] += total_reward

        return {name: total_reward}

    def go_truncation_function(env) -> torch.Tensor:
        """Go truncation function - robot falls or episode time exceeded."""
        robot_interface = env.robots[name]

        base_quat = robot_interface.get_orientation(format="quat")

        # Calculate base orientation relative to initial orientation
        base_init_quat = torch.tensor(env_cfg["base_init_quat"], device=env.device, dtype=torch.float32)
        inv_base_init_quat = inv_quat(base_init_quat)
        base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(base_quat) * inv_base_init_quat, base_quat),
            rpy=True,
            degrees=True,
        )

        # Episode time check
        episode_time = env.episode_frame_count.float() / env.simulation_frequency
        time_exceeded = episode_time >= env_cfg["episode_length_s"]

        # Orientation checks
        roll_exceeded = torch.abs(base_euler[:, 0]) > env_cfg["termination_if_roll_greater_than"]
        pitch_exceeded = torch.abs(base_euler[:, 1]) > env_cfg["termination_if_pitch_greater_than"]

        # Set timeout info
        time_out_idx = (episode_time >= env_cfg["episode_length_s"]).nonzero(as_tuple=False).flatten()
        env.infos[name]["time_outs"] = torch.zeros_like(episode_time, device=gs.device, dtype=gs.tc_float)
        env.infos[name]["time_outs"][time_out_idx] = 1.0

        # Combine truncation conditions
        truncated = time_exceeded | roll_exceeded | pitch_exceeded

        return truncated

    def go_info_function(env) -> Dict[str, Any]:
        """Go info function - provides episode statistics and critic observations."""

        # No specific info for Go2 training
        return {}

    def go_action_preprocessing_function(env):
        """Go action preprocessing function with proper action latency simulation."""
        # Get current action from action buffer
        action = env.action_buffers[name].clone()

        # Apply action clipping
        action = torch.clip(action, -env_cfg["clip_actions"], env_cfg["clip_actions"])

        # Apply action latency simulation using registered field
        if hasattr(env, '_last_actions') and name in env._last_actions:
            last_actions_tensor = env._last_actions[name]
            if env_cfg["simulate_action_latency"]:
                # Use last actions for execution (1-step latency)
                exec_action = last_actions_tensor.clone()
            else:
                # Use current actions
                exec_action = action.clone()
        else:
            # Fallback if field not registered
            exec_action = action.clone()
            # logger.warning(f"_last_actions field not registered for {name}, skipping action latency simulation")

        # Apply action scaling and add default joint positions
        scaled_action = exec_action * env_cfg["action_scale"]
        if env_cfg.get("default_joint_angles"):
            default_dof_pos = torch.tensor(
                [env_cfg["default_joint_angles"][joint] for joint in env_cfg["joint_names"]],
                device=env.device,
                dtype=torch.float32,
            )
            target_dof_pos = scaled_action + default_dof_pos
        else:
            target_dof_pos = scaled_action

        return target_dof_pos

    # Extract default joint positions in the correct order for the joint_names
    initial_joint_pos = [env_cfg["default_joint_angles"][joint] for joint in env_cfg["joint_names"]]

    # Create robot config
    config = RobotConfig(
        name=name,
        urdf_path=urdf_path,
        frequency=frequency,  # Configurable control frequency
        initial_position=env_cfg["base_init_pos"],
        initial_orientation=env_cfg["base_init_quat"],
        action_space=None,  # Auto-detect from robot joints (12 actions)
        observation_space=None,  # Will be 45D as calculated
        control_mode="position",
        joint_names=env_cfg["joint_names"],
        pre_reset_hook=go_pre_reset_hook,
        post_reset_hook=go_post_reset_hook,
        post_step_hook=go_post_step_hook,
        obs_function=go_obs_function,
        reward_function=go_reward_function,
        truncation_function=go_truncation_function,
        info_function=go_info_function,
        setup_function=go_setup_function,
        action_preprocessing_function=go_action_preprocessing_function,  # Add action preprocessing
        initial_joint_pos=initial_joint_pos,  # Add default joint positions
        DP_kp=env_cfg["kp"],
        DP_kd=env_cfg["kd"],
    )

    return config


def get_go_train_cfg(exp_name: str, max_iterations: int) -> RSL_RLConfig:
    """Get RSL_RL training configuration matching original go1/go2_train.py."""
    alg_config = AlgorithmConfig(
        class_name="PPO",
        clip_param=0.2,
        desired_kl=0.01,
        entropy_coef=0.01,
        gamma=0.99,
        lam=0.95,
        learning_rate=0.001,
        max_grad_norm=1.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        schedule="adaptive",
        use_clipped_value_loss=True,
        value_loss_coef=1.0,
    )
    policy_config = PolicyConfig(
        activation="elu",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        init_noise_std=1.0,
        class_name="ActorCritic",
    )
    runner_config = RunnerConfig(
        checkpoint=-1,
        experiment_name=exp_name,
        load_run=-1,
        log_interval=1,
        max_iterations=max_iterations,
        record_interval=-1,
        resume=False,
        resume_path=None,
        run_name="",
    )
    rsl_rl_config = RSL_RLConfig(
        algorithm=alg_config,
        init_member_classes={},
        policy=policy_config,
        runner=runner_config,
        runner_class_name="OnPolicyRunner",
        num_steps_per_env=24,
        save_interval=100,
        empirical_normalization=None,
        seed=1,
    )
    return rsl_rl_config


def main():
    """Main training function."""
    # Setup main script logger
    logger = get_class_logger("TrainingScript", "go_locomotion", level="INFO")

    parser = argparse.ArgumentParser(description="Go1/Go2 MARL Training with Genesis Framework")
    parser.add_argument("-r", "--robot", type=str, default="go1", choices=["go1", "go2"], help="Robot type to train")
    parser.add_argument("-e", "--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("-B", "--num_envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=101, help="Maximum training iterations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--frequency", type=int, default=50, help="Robot control frequency in Hz (default: 50)")
    parser.add_argument("--render", action="store_true",
                        help="Enable real-time rendering. This will slow down training but allows visual inspection")
    parser.add_argument("--enable_camera", action="store_true", help="Enable periodic video recording")
    parser.add_argument("--camera_interval", type=int, default=150, help="Recording interval in seconds")
    parser.add_argument("--camera_duration", type=int, default=30, help="Recording duration in seconds")
    parser.add_argument("--camera_res", type=str, default="1280x720", help="Camera resolution (WxH)")
    args = parser.parse_args()

    # Set default experiment name based on robot type if not provided
    if args.exp_name is None:
        args.exp_name = f"{args.robot}-locomotion-train"

    logger.info(f"🤖 Starting {args.robot.upper()} MARL Training")
    logger.info(f"  Robot: {args.robot}")
    logger.info(f"  Experiment: {args.exp_name}")
    logger.info(f"  Environments: {args.num_envs}")
    logger.info(f"  Iterations: {args.max_iterations}")
    logger.info(f"  Control frequency: {args.frequency} Hz")
    logger.info(f"  Seed: {args.seed if args.seed is not None else 'None (random)'}")
    logger.info(f"  Real-time rendering: {args.render}")
    logger.info(f"  Periodic recording: {args.enable_camera}")
    if args.enable_camera:
        logger.info(f"  Camera resolution: {args.camera_res}")
        logger.info(f"  Recording interval: {args.camera_interval}s")
        logger.info(f"  Recording duration: {args.camera_duration}s")

    # Step 1: Create robot configuration based on robot type
    logger.info(f"Step 1: Creating {args.robot.upper()} robot configuration...")
    robot_config = create_go_robot_config(f"{args.robot}_robot", robot_type=args.robot, frequency=args.frequency)

    # Step 1.5: Create camera configuration
    camera_config = None
    if args.enable_camera:
        width, height = map(int, args.camera_res.split('x'))
        camera_config = CameraConfig(
            res=(width, height),
            fov=45.0,
            offset=(3.0, 3.0, 2.0),  # Camera positioned behind and above robot
            enable_recording=True,
            record_interval_s=args.camera_interval,
            record_duration_s=args.camera_duration,
            video_prefix=f'{args.exp_name}_progress',
            track_targets=[robot_config.name],
        )

    # Step 2: Create training bundle
    logger.info("Step 2: Creating single robot training bundle...")
    rsl_rl_cfg = get_go_train_cfg(args.exp_name, args.max_iterations)
    training_config = RSL_RLTrainingConfig(
        training_name=args.exp_name,
        robot_cfgs=[robot_config],
        rsl_rl_config=rsl_rl_cfg,
        task_config={"env_cfg": env_cfg, "obs_cfg": obs_cfg, "reward_cfg": reward_cfg, "command_cfg": command_cfg},
    )

    # Step 3: Initialize AEC Environment
    logger.info("Step 3: Initializing VectorizedAECEnv...")
    try:
        # Prepare camera configs if enabled
        camera_configs = {"tracking_camera": camera_config} if args.enable_camera else None

        aec_env = VectorizedAECEnv(
            training_configs=[training_config],
            n_envs=args.num_envs,
            max_episode_length_s=env_cfg["episode_length_s"],
            render=args.render,  # Use command-line option
            seed=args.seed,
            camera_configs=camera_configs,
            simulator_config=SimulatorConfig(
                show_viewer=args.render,
                dt=1.0 / args.frequency,
                n_envs=args.num_envs,
                device="cuda" if torch.cuda.is_available() else "cpu"
            ),
        )
        logger.info("✅ VectorizedAECEnv created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create VectorizedAECEnv: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

    # Step 4: Reset AEC Environment
    logger.info("Step 4: Resetting AEC Environment...")
    aec_env.reset()
    logger.info("✅ AEC Environment initialized and reset")

    # Step 5: Start training thread
    logger.info("Step 5: Starting training thread...")
    subvecenv = aec_env.subvecenvs[args.exp_name]

    training_thread = threading.Thread(
        target=training_config.trainer_launcher,
        args=(subvecenv, args.max_iterations),
        daemon=True,
    )

    training_thread.start()

    # Wait for thread to start
    time.sleep(1.0)

    # Step 6: Execute AEC loop
    logger.info("Step 6: Starting AEC execution loop...")

    # Calculate total simulation time needed
    # RSL_RL will run max_iterations * num_steps_per_env steps
    # With configurable robot frequency, we need enough simulation time
    rsl_rl_cfg = training_config.rsl_rl_config
    total_steps_needed = args.max_iterations * rsl_rl_cfg.num_steps_per_env
    total_sim_time = total_steps_needed / args.frequency  # Use configurable robot frequency
    total_sim_frames = int(total_sim_time * aec_env.simulation_frequency)

    logger.info(f"Running {total_sim_frames} simulation frames ({total_sim_time:.1f}s)")

    frame_count = 0
    while training_thread.is_alive() and frame_count < total_sim_frames:
        if frame_count % 1000 == 0:
            logger.info(f"🔄 AEC Frame {frame_count}/{total_sim_frames} " f"({frame_count/total_sim_frames*100:.1f}%)")

        # Execute AEC cycle
        aec_env.last()
        aec_env.step()
        frame_count += 1

    # Final last() call
    aec_env.last()

    # Wait for training to complete
    logger.info("Waiting for training thread to complete...")
    training_thread.join(timeout=30.0)

    if training_thread.is_alive():
        logger.warning("Training thread did not complete in time")
    else:
        logger.info("✅ Training completed successfully!")

    # Camera recording stops automatically - no manual intervention needed
    if args.enable_camera:
        logger.info("📹 Periodic recording completed automatically")

    logger.info(f"🎉 {args.robot.upper()} MARL Training finished!")

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 MARL training completed successfully!")
        exit(0)
    else:
        print("\n❌ MARL training failed!")
        exit(1)
