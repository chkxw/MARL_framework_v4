#!/usr/bin/env python3
"""
Vectorized AEC Environment for Genesis MARL Framework V4

This is the core orchestrator that manages all components:
- Genesis simulator scene
- Agents (grouped by frequency)
- SubVecEnvs (for RSL_RL interface)
- Centralized data storage with RefDict
- Scheduler for multi-frequency control
- Soft robot death and auto-reset
"""

import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from agent import Agent
from centralized_scheduler import CentralizedFrequencyScheduler, FrequencyOptimizer
from configs.RobotConfig import RobotConfig
from configs.TrainingConfig import TrainingConfig
from coordinator import CoordinatorBase, ThreadingCoordinator
from marl_logging import get_class_logger
from simulation_interface import SceneInterface
from configs.SimulatorConfig import SimulatorConfig, create_default_config
from utils import RefDict
from simulation_interface import RobotInterface, SceneInterface


class VectorizedAECEnv:
    """Vectorized AEC Environment - the core orchestrator.

    This class is the main entry point for the Genesis MARL framework.
    It manages all components and follows a modified AEC pattern:
    - reset() -> last() -> step() -> last() -> step() ...

    Key modifications from standard AEC:
    - Agents never die (only robots can have soft death)
    - Robot level soft death with auto-reset
    - Automatic agent selection based on scheduler
    - Auto-reset for terminated environments
    - No action argument in step() (actions come via semaphores)
    """

    def __init__(
        self,
        training_configs: List[TrainingConfig],
        n_envs: int = 64,
        device: str = "cuda",
        max_episode_length_s: float = 20.0,
        simulation_frequency: Optional[int] = None,
        render: bool = False,
        seed: Optional[int] = None,
        record_termination_obs: bool = False,
        simulator_config: Optional[SimulatorConfig] = None,
        camera_configs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Vectorized AEC Environment.

        Args:
            training_configs: List of training configurations
            n_envs: Number of parallel environments
            device: Device for computations
            max_episode_length_s: Maximum episode length in seconds
            simulation_frequency: Simulation frequency, leave None for automatic optimization
            render: Whether to enable rendering
            seed: Random seed for reproducibility (sets numpy, torch, and simulator seeds)
            simulator_config: Optional simulator configuration. If None, defaults will be used.
        """

        """Core properties"""
        # Initialize independent main logger
        self.logger = get_class_logger("VectorizedAECEnv", "main", level="INFO")

        self.training_configs = {cfg.training_name: cfg for cfg in training_configs}
        self.robot_2_training_name = {
            robot_name: trianing_name
            for trianing_name, training_cfg in self.training_configs.items()
            for robot_name in training_cfg.robot_configs.keys()
        }
        self.n_envs = n_envs
        self.device = device
        self.max_episode_length_s = max_episode_length_s
        self.simulation_frequency = self._set_frequency(simulation_frequency)
        self.render = render
        self.seed = self._set_seed(seed)
        self.record_termination_obs = record_termination_obs  # If enabled, record termination observation in info dict

        # Setup simulator configuration
        default_config = SimulatorConfig(
            dt=1.0 / self.simulation_frequency, show_viewer=self.render, n_envs=self.n_envs, device=self.device
        )
        if simulator_config is None:
            # Create default configuration
            self.simulator_config = default_config
        else:
            # Use provided config, merged with defaults
            self.simulator_config = simulator_config.merge_with_defaults(default_config)

        """Book keeping"""
        # Reference fields (centralized data storage)
        self.robot_configs: Dict[str, RobotConfig] = {}  # robot_name -> robot config
        self.joint_dofs_idx_locals: Dict[str, List[int]] = {}  # robot_name -> list of motor dof indexes
        self.agents: Dict[str, Agent] = {}  # agent_name -> agent instance
        self.subvecenvs: Dict[str, Any] = {}  # training_name -> subVecEnv instance

        # AEC related fields (using RefDict for shared data)
        self.terminations: torch.Tensor = torch.zeros(n_envs, dtype=torch.bool, device=device)  # tensor[n_envs, bool]
        self.truncations: Dict[str, torch.Tensor] = RefDict()  # robot_name -> tensor[n_envs, bool]
        self.rewards: Dict[str, Dict[str, torch.Tensor]] = RefDict()  # robot_name -> Dict[str, tensor[n_envs, float]]
        self._cumulative_rewards: Dict[str, torch.Tensor] = RefDict()  # robot_name -> tensor[n_envs, float]
        self.episode_rewards: Dict[str, torch.Tensor] = RefDict()  # robot_name -> tensor[n_envs, float]
        # Difference between rewards, _cumulative_rewards and episode_rewards
        # rewards: a temporary varaible used in last() to store reward generated by a specific robot, it's a nested dict because robot can generate reward for other robot
        # _cumulative_rewards: the cumulative reward for each robot since its last action
        # episode_rewards: the cumulative reward for each robot since the start of the episode
        self.infos: Dict[str, Dict[str, Any]] = RefDict()  # robot_name -> dict
        self.observations: Dict[str, torch.Tensor] = RefDict()  # robot_name -> tensor[n_envs, obs_dim]

        # Special field
        self.shared_observations: Dict[str, torch.Tensor] = RefDict()  # training_name -> tensor[n_envs, obs_dim]

        # Action recievers (Not ref dict, all robots have independent actions)
        self.action_buffers: Dict[str, List[torch.Tensor]] = dict()  # robot_name -> list[torch.Tensor]
        self.joint_action_buffers: Dict[str, torch.Tensor] = dict()  # Temporary storage for joint actions

        # Soft death related fields
        self.dead_robots: Dict[str, torch.Tensor] = RefDict()  # robot_name -> tensor[n_envs, bool]

        # AEC state
        self.agent_selection: str = None  # Current selected agent
        self.scheduler: CentralizedFrequencyScheduler
        self.coordinator: CoordinatorBase

        # Episode tracking
        self.episode_frame_count: torch.Tensor = torch.zeros(n_envs, device=device)  # Time in seconds
        self.episode_count: torch.Tensor = torch.zeros(n_envs, device=device)  # Number of episodes in each env

        self.global_frame_count: int = 0

        # Simulation scene interface (the only place simulation-specific code should exist)
        self.scene: Optional[SceneInterface] = None

        # Camera system
        self.camera_configs: Optional[Dict[str, Any]] = camera_configs

        """Customed Runtime fields"""
        self._registered_fields = {}

        """Initialization Steps"""
        # Step 1: Initialize coordinator
        self.logger.debug("Step 1: Initializing coordinator...")
        self._initialize_coordinator()

        # Step 2: Group TrainingConfigBundles by frequency and initialize agents
        self.logger.debug("Step 3: Grouping bundles by frequency and creating agents...")
        self._create_agents()

        # Step 4: Setup scheduler
        self.logger.debug("Step 4: Setting up scheduler...")
        self._setup_scheduler()

        # Step 5: Setup Simulation scene
        self.logger.debug("Step 5: Setting up Simulation scene...")
        self._setup_simulation_scene()

        # Step 6: Add robots by calling agent's setup_robots method
        self.logger.debug("Step 6: Adding robots to scene...")
        self._setup_robots()

        # Step 6.5: Setup cameras if provided
        if self.camera_configs:
            self.logger.debug("Step 6.5: Setting up cameras...")
            self._setup_cameras()

        # Step 7: Build the scene (after all entities are added)
        self.logger.debug("Step 7: Building Genesis scene...")

        # Step 8: Initialize robots after scene build
        self.logger.debug("Step 8: Initializing robots after scene build...")
        self._build_scene()

        # Step 9: Setup DP parameters according to each RobotConfig
        self.logger.debug("Step 9: Setting up DP parameters...")
        self._setup_dp_parameters()

        # Step 10: Detect observation spaces
        self.logger.debug("Step 10: Detecting observation spaces...")
        self._detect_obs_spaces()

        # Step 11: Initialize all buffers (now we know obs dims)
        self.logger.debug("Step 11: Initializing data buffers...")
        self._initialize_buffers()

        # Step 12: Create subVecEnv entities based on TrainingConfigBundles
        self.logger.debug("Step 12: Creating subVecEnv entities...")
        self._create_subvecenvs()

        # TODO: print summary

    def _set_seed(self, seed: Optional[int] = None):
        """Set random seeds for numpy, torch and python."""

        if seed is None:
            seed = int(time.time())
            self.logger.debug(f"No seed provided, using time based seed: {seed}")

        # Set numpy seed
        np.random.seed(seed)

        # Set torch seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Set python random seed
        random.seed(seed)

        return seed

    def _set_frequency(self, simulation_frequency: Optional[int] = None, max_frequency_allowed: int = 150):
        if simulation_frequency is None:
            training_frequencies = {
                training_name: training_cfg.frequency for training_name, training_cfg in self.training_configs.items()
            }
            simulation_frequency, frequency_info = FrequencyOptimizer.find_optimal_genesis_frequency(
                robot_frequencies=list(training_frequencies.values()),
                max_genesis_freq=max_frequency_allowed,
                tolerance=0.05,
            )
            # Print frequency info
            self.logger.debug("Automatic optimized frequency:")
            self.logger.debug(f"{'Agent':<10}{'Desired':<10}{'Actual':<10}{'Period':<10}{'Error':<10}")
            for name, info in frequency_info.items():
                self.logger.debug(
                    f"{name:<10}{info['desired_frequency']:<10.1f}Hz"
                    f"{info['actual_frequency']:<10.1f}Hz"
                    f"{info['period_frames']:<10}"
                    f"{info['frequency_error']*100:<10.1f}%"
                )
            self.logger.debug(
                f"Genesis frequency: {simulation_frequency:.1f} Hz (dt = {1.0 / simulation_frequency:.4f}s)"
            )

        return simulation_frequency

    def _initialize_coordinator(self):
        """Initialize coordinator and other resources."""
        # Create list of thread/state names for coordinator
        thread_names = list(self.training_configs.keys())
        thread_names.append("main")

        # Initialize coordinator with all training threads + main thread
        self.coordinator = ThreadingCoordinator(thread_names)
        self.coordinator.wake("main")  # Main thread starts active

    def _create_agents(self):
        # Group bundles by frequency
        frequency_groups = defaultdict(list)
        for training_cfg in self.training_configs.values():
            frequency = training_cfg.frequency
            frequency_groups[frequency].append(training_cfg)

        # Create agents for each frequency group
        for frequency, bundles in frequency_groups.items():
            agent_name = f"agent_{frequency}hz"

            # Create agent with all bundles of this frequency
            agent = Agent(agent_name=agent_name, frequency=frequency, training_configs=bundles, mother_env=self)

            self.agents[agent_name] = agent

            self.logger.debug(f"Created agent: {agent_name} with {bundles}")

        # Set coordinator reference for all agents
        for agent in self.agents.values():
            agent.set_coordinator(self.coordinator)

    def _setup_scheduler(self):
        agent_frequencies = {}
        for agent_name, agent in self.agents.items():
            agent_frequencies[agent_name] = agent.frequency

        self.scheduler = CentralizedFrequencyScheduler(agent_frequencies=agent_frequencies)

        self.logger.info(f"Scheduler setup with {len(agent_frequencies)} agents")

    def _setup_simulation_scene(self):
        """Create simulation scene using the appropriate interface."""

        if self.simulator_type == "genesis":
            from simulation_interface.genesis_interface import GenesisSceneInterface

            self.scene = GenesisSceneInterface(
                config=self.simulator_config,
            )

        elif self.simulator_type == "mujoco":
            from simulation_interface.mujoco_interface import MuJoCoSceneInterface

            self.scene = MuJoCoSceneInterface(config=self.simulator_config)

        elif self.simulator_type == "isaac_gym":
            from simulation_interface.isaac_interface import IsaacGymSceneInterface

            self.scene = IsaacGymSceneInterface(config=self.simulator_config)

        elif self.simulator_type == "pymunk":
            from simulation_interface.pymunk_interface import PymunkSceneInterface

            self.scene = PymunkSceneInterface(config=self.simulator_config)

        else:
            raise ValueError(f"Unsupported simulator type: {self.simulator_type}")

        self.logger.info(f"{self.simulator_type} scene created successfully")

    def _setup_robots(self):
        """Add robots to scene by calling each agent's setup_robots method."""
        # Make robot configs available before setup
        for agent_name, agent in self.agents.items():
            self.robot_configs.update(agent.robot_configs)

        for agent_name, agent in self.agents.items():
            self.logger.debug(f"Setting up robots for agent: {agent_name}")
            agent.setup_robots()

        self.logger.info(f"Setup complete for {len(self.robots)} robots")

    def _setup_cameras(self):
        """Setup cameras before scene is built."""
        if not self.camera_configs:
            return

        self.logger.debug("Setting up camera manager...")
        self.scene.setup_camera_manager()

        for camera_name, camera_config in self.camera_configs.items():
            self.logger.debug(f"Adding camera: {camera_name}")
            self.scene.camera_manager.add_camera(camera_name, camera_config)

        self.logger.info(f"Setup complete for {len(self.camera_configs)} cameras")

    def _build_scene(self):
        """Build the scene after all entities including camera are added."""
        # Build the scene now that all robots and camera are added
        self.pre_build_hooks()
        self.scene.build_scene()
        self.post_build_hooks()
        self.logger.info(f"✅ {self.simulator_type} scene built successfully")

    def _setup_dp_parameters(self):
        """Setup DP parameters for all robots."""
        for agent in self.agents.values():
            for robot_name in agent.robot_names:
                agent._setup_robot_dp_parameters(robot_name)
                self.logger.debug(f"DP parameters set for robot: {robot_name}")

    def _detect_obs_spaces(self):
        """Detect observation spaces for each robot."""
        # TODO: Think about how to handle Joint Training case
        for training_cfg in self.training_configs.values():
            training_cfg.detect_obs_spaces(self)

        self.logger.info("Observation spaces detected successfully")

    def _initialize_buffers(self):
        """Initialize all data buffers with proper dimensions."""
        self.logger.info("Initializing data buffers...")

        for training_cfg in self.training_configs.values():
            if training_cfg.use_shared_obs:
                self.shared_observations[training_cfg.training_name] = torch.zeros(
                    (self.n_envs,) + training_cfg.shared_observation_space.shape, device=self.device
                )
            for i, (robot_name, robot_cfg) in enumerate(training_cfg.robot_configs.items()):
                # Action buffers are always independent for each robot
                self.action_buffers[robot_name] = torch.zeros(
                    (self.n_envs,) + robot_cfg.action_space.shape, device=self.device
                )
                self.joint_action_buffers[robot_name] = torch.zeros(
                    (self.n_envs,) + (len(robot_cfg.joint_names),), device=self.device
                )
                if i != 0 and training_cfg.is_joint:
                    # Set reference to first robot
                    first_robot_name = training_cfg.robot_names[0]
                    self.observations.add_ref(robot_name, first_robot_name)
                    self.rewards.add_ref(robot_name, first_robot_name)
                    self._cumulative_rewards.add_ref(robot_name, first_robot_name)
                    self.episode_rewards.add_ref(robot_name, first_robot_name)
                    self.truncations.add_ref(robot_name, first_robot_name)
                    self.dead_robots.add_ref(robot_name, first_robot_name)
                    self.infos.add_ref(robot_name, first_robot_name)
                    continue

                obs_shape = (self.n_envs,) + robot_cfg.observation_space.shape
                self.observations[robot_name] = torch.zeros(obs_shape, device=self.device)
                self.rewards[robot_name] = {}
                self._cumulative_rewards[robot_name] = torch.zeros(self.n_envs, device=self.device)
                self.episode_rewards[robot_name] = torch.zeros(self.n_envs, device=self.device)
                self.truncations[robot_name] = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
                self.dead_robots[robot_name] = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
                self.infos[robot_name] = {}  # Will be populated during episodes

                self.logger.debug(f"Initialized buffers for robot {robot_name} (obs_shape={obs_shape})")

        self.logger.info("✅ Data buffers initialized")

    def _create_subvecenvs(self):
        """Create subVecEnv instances for each TrainingConfigBundle."""
        for training_name, training_cfg in self.training_configs.items():
            self.subvecenvs[training_name] = training_cfg.trainer_env_factory(self)

            # Register subvecenv in agent
            for agent in self.agents.values():
                if training_cfg in agent.training_configs:
                    agent.subvecenvs[training_name] = self.subvecenvs[training_name]

            self.logger.debug(f"Created subVecEnv: {training_name} ({type(self.subvecenvs[training_name]).__name__})")

        self.logger.info(f"Created {len(self.subvecenvs)} subVecEnvs")

    """
    RL CORE
    """

    def reset(self) -> None:
        """Reset the environment (called once at the beginning).

        Following CLAUDE.md specifications:
        1. The reset only called once at the beginning
        2. call _reset_envs(all)
        3. reset scheduler
        """
        self.logger.info("🔄 Resetting AEC environment...")

        # Reset all environments
        all_env_indices = torch.arange(self.n_envs, device=self.device)
        self._reset_envs(all_env_indices)

        # Reset scheduler
        if self.scheduler:
            self.scheduler.reset()

        # Reset episode tracking
        self.episode_frame_count.zero_()
        self.global_frame_count = 0

        # Set initial agent selection
        self.agent_selection = self.scheduler.get_current_agent()

        self.logger.info(f"✅ AEC environment reset complete, starting with agent: {self.agent_selection}")

    def _reset_envs(self, env_indices: torch.Tensor) -> None:
        """Reset robots in specified environments.

        Args:
            env_indices: Tensor of environment indices to reset
        """
        self.logger.debug(f"Resetting {len(env_indices)} environments")

        self.pre_reset_hooks(env_indices)

        # Call all agents' _reset_envs method
        for agent in self.agents.values():
            agent._reset_robots(env_indices)

        # Reset episode time for these environments
        self.episode_frame_count[env_indices] = 0.0
        self.episode_count[env_indices] += 1
        for robot_name in self.robot_names:
            self.episode_rewards[robot_name][env_indices] = 0.0

        # Reset dead robot flags
        for robot_name in self.dead_robots.keys():
            self.dead_robots[robot_name][env_indices] = False

        # Reset custom fields
        target_field_names = [
            field_name
            for field_name, field_properties in self._registered_fields.items()
            if field_properties["clear_in_reset"]
        ]
        self.auto_clear_fields(target_field_names, env_indices=env_indices)

        self.post_reset_hooks(env_indices)
        self.logger.debug(f"Reset complete for {len(env_indices)} environments")

    def step(self) -> None:
        """Execute one step of the AEC environment."""
        if self.agent_selection is None:
            raise RuntimeError("No agent selected. Call reset() first.")

        self.logger.debug(f"AEC frame {self.global_frame_count}: agent {self.agent_selection}")

        # Execute agent step
        current_agent = self.agents[self.agent_selection]

        # Wait for actions from subVecEnvs
        current_agent._wait_for_actions()

        # Step hook
        self.pre_step_hooks()

        # At time point, the subVecEnvs must have finish using previous observations
        # Do cleanups
        self._clear_buffers()

        robot_actions = {}
        for training_cfg in current_agent.training_configs:
            for robot_name, robot_config in training_cfg.robot_configs.items():
                robot_actions[robot_name] = robot_config.action_preprocessing_function(self)

        # Apply local motion models and store joint actions in self.joint_action_buffers
        current_agent._apply_locomotion_models(robot_actions)

        # Post processing
        joint_actions = {}
        for robot_name, robot_config in current_agent.robot_configs.items():
            # Post processing function will use self.joint_action_buffers as input
            processed_actions = robot_config.action_postprocessing_function(self)
            joint_actions[robot_name] = processed_actions

        # Apply to simulator
        for robot_name, joint_action in joint_actions.items():
            robot_config = current_agent.robot_configs[robot_name]
            # Apply to simulator based on control mode
            current_agent._apply_robot_action(robot_name, joint_action)

            self.logger.debug(f"Applied action to robot {robot_name}: shape {joint_action.shape}")

        # Update agent selection and step simulator based on scheduler
        next_agent, frames_to_advance = self.scheduler.step()

        self.agent_selection = next_agent

        self.scene.step(frames_to_advance)

        self.episode_frame_count += frames_to_advance
        self.global_frame_count += frames_to_advance

        self.post_step_hooks()

    def last(self) -> None:
        """Send observations/rewards/terminations to current agent's subVecEnvs.

        1. Calculate obs,reward,truncation,info for each subVecEnv of current agent
        2. Check termination of environment
        3. Collect termination/truncation observations of dead robots
        4. Do auto reset for terminated envs, revoke dead robots
        5. Call current selected agent's last method
        """
        self.pre_last_hooks()
        current_agent = self.agents[self.agent_selection]

        self.logger.debug(f"AEC last(): agent {self.agent_selection}")

        # Step 1: Calculate obs,reward,truncation,info for current agent's robots
        for training_cfg in current_agent.training_configs:
            self.infos.update(training_cfg.info_function(self))
            self.observations.update(training_cfg.obs_function(self))
            self.truncations.update(training_cfg.truncation_function(self))
            self.rewards.update(training_cfg.reward_function(self))

            if training_cfg.use_shared_obs:
                self.shared_observations[training_cfg.training_name] = training_cfg.shared_obs_function(self)

        # We also calculate properties for dead robots in some envs in previous step, the obs,truncation,info are harmless, but the reward might affect other robot, we need to mask it
        for robot_name, dead_mask in self.dead_robots.items():
            reward_dict = self.rewards[robot_name]
            for target_robot_name, reward_tensor in reward_dict.items():
                reward_tensor.masked_fill_(dead_mask, 0.0)

        self._accumulate_rewards()  # Dump temporary rewards into self._cumulative_rewards

        # Step 2: Check termination conditions
        self.terminations = self._check_termination_conditions()

        # Record all the agents that dead in current step by comparing old and new dead robots, till here we haven't bring any robot back to life, so the diff is all the new dead robots
        robot_dead_in_current_step = {
            robot_name: self.truncations[robot_name] | self.terminations for robot_name in self.robots.keys()
        }

        # If one env terminate, robots_with_new_dead_env would be all robots, else it will be all the truncated robots
        robots_with_new_dead_env = {
            robot_name for robot_name, dead_idx in robot_dead_in_current_step.items() if torch.any(dead_idx)
        }

        if len(robots_with_new_dead_env) != 0:
            self.logger.debug(f"robots_with_new_dead_env: {robots_with_new_dead_env}")

        # Update current termination and truncation into dead_robots - ONLY FOR CURRENT AGENT
        # Why only for current agent?
        # - truncation death is only for current agent
        # - termination death is for all agents, but we want other agent to keep alive until its own step to notice their termination death
        for robot_name, new_dead_idx in robot_dead_in_current_step.items():
            if robot_name not in current_agent.robot_names:
                continue
            self.dead_robots[robot_name] |= new_dead_idx

        # Step 3: Collect termination/truncation observations (only for current agent's robots)

        for robot_idx, robot_name in enumerate(current_agent.robot_names):
            if robot_name not in robots_with_new_dead_env:
                continue
            dead_idx = robot_dead_in_current_step[robot_name]

            training_config = self.training_configs[self.robot_2_training_name[robot_name]]

            for r_name in training_config.robot_names:
                if r_name == robot_name or training_config.is_shared:
                    # Create the termination_obs/episode_reward tensor in info
                    # - for current robot
                    # - for all robot in this training config if it is shared parameter training, which requires all info have same keys
                    if self.record_termination_obs and "termination_obs" not in self.infos[r_name]:
                        self.infos[r_name]['termination_obs'] = torch.zeros_like(self.shared_observations)
                        self.infos[r_name]['termination_obs'].fill_(float('nan'))
                    if (
                        self.record_termination_obs
                        and training_config.use_shared_obs
                        and "termination_shared_obs" not in self.infos[r_name]
                    ):
                        self.infos[r_name]['termination_shared_obs'] = torch.zeros_like(self.shared_observations)
                        self.infos[r_name]['termination_shared_obs'].fill_(float('nan'))
                    if "episode_reward" not in self.infos[r_name]:
                        self.infos[r_name]['episode_reward'] = torch.zeros(self.n_envs, device=self.device)
                        self.infos[r_name]['episode_reward'].fill_(float('nan'))

            # Only copy those dead env's obs and episode reward
            if self.record_termination_obs:
                self.infos[robot_name]['termination_obs'][dead_idx] = self.observations[robot_name][dead_idx]
            if self.record_termination_obs and training_cfg.use_shared_obs:
                self.infos[robot_name]['termination_shared_obs'][dead_idx] = self.shared_observations[dead_idx]

            # always record episode reward
            self.infos[robot_name]['episode_reward'][dead_idx] = self.episode_rewards[robot_name][dead_idx]

            # Also record the shared obs

        # Step 4: Auto reset - ONLY FOR ENVS THAT TERMINATE AND THIS IS THE LAST ACTIVE AGENT IN THE ENV
        all_robots_dead = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        for robot_name in self.dead_robots.keys():
            all_robots_dead &= self.dead_robots[robot_name]
        should_auto_reset = self.terminations & all_robots_dead

        if torch.any(should_auto_reset):
            terminated_indices = torch.where(should_auto_reset)[0]
            self._reset_envs(terminated_indices)

        # Record the reset observation to replace the old termination observations
        for training_cfg in current_agent.training_configs:
            robot_names = set(training_cfg.robot_names)
            # Only recalculate if some robot in this training bundle has new dead env
            if not robot_names.isdisjoint(robots_with_new_dead_env):
                self.observations.update(training_cfg.obs_function(self))
                if training_cfg.use_shared_obs:
                    self.shared_observations[training_cfg.training_name] = training_cfg.shared_obs_function(self)

        # Step 5: Send everything
        self.logger.debug(f"Agent {current_agent.agent_name} last() called")

        # Move data from AEC's buffers to subVecEnv buffers
        for training_cfg in current_agent.training_configs:
            training_name = training_cfg.training_name
            subvecenv = current_agent.subvecenvs[training_name]

            subvecenv.term_buffer = self.terminations
            subvecenv.episode_length_buffer = self.episode_frame_count.clone()
            if training_cfg.use_shared_obs:
                subvecenv.shared_obs_buffer = self.shared_observations[training_name]
            for i, robot_name in enumerate(training_cfg.robot_configs.keys()):
                subvecenv.obs_buffer[robot_name] = self.observations[robot_name]
                subvecenv.rew_buffer[robot_name] = self._cumulative_rewards[robot_name]
                subvecenv.trunc_buffer[robot_name] = self.truncations[robot_name]
                subvecenv.info_buffer[robot_name] = self.infos[robot_name]
                if i == 0 and training_cfg.is_joint:
                    break  # Only first robot in bundle has valid obs/rew/trunc/info

        # Notify subVecEnvs that data is ready
        current_agent._notify_data_ready()

        self.post_last_hooks()

        self.logger.debug(f"{current_agent.agent_name} last() complete")

    def _check_termination_conditions(self) -> torch.Tensor:
        """Check which environments should terminate.

        Returns:
            Boolean tensor indicating which environments should terminate
        """
        # Condition 1: Episode time exceeds maximum
        time_exceeded = (self.episode_frame_count.float() / self.simulation_frequency) >= self.max_episode_length_s

        # Condition 2: All robots in environment are dead
        # For each environment, check if all robots are dead
        all_robots_dead = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        for robot_name in self.robots.keys():
            all_robots_dead &= self.dead_robots[robot_name] | self.truncations[robot_name]

        # Environment terminates if either condition is met
        terminated = time_exceeded | all_robots_dead

        if torch.any(terminated):
            self.logger.debug(f"Terminated environments: {torch.where(terminated)[0].tolist()}")
        return terminated

    def _accumulate_rewards(self) -> None:
        for robot_name, reward_dict in self.rewards.items():
            for target_robot_name, reward_tensor in reward_dict.items():
                self._cumulative_rewards[target_robot_name].add_(reward_tensor)
                self.episode_rewards[target_robot_name].add_(reward_tensor)

    def _clear_buffers(self) -> None:
        # Clear fields for all robots
        for robot_name in self.robots.keys():
            self.rewards[robot_name] = {}
            self.truncations[robot_name].zero_()

        # Clear fields for current agent
        for robot_name in self.agents[self.agent_selection].robot_names:
            self._cumulative_rewards[robot_name].zero_()
            self.infos[robot_name] = {}

        # Why not clear terminations? Because it will be recalculated and overwrite in each step, no need to clear it
        # Why not clear observations? Because it will not be used by anyone until the last() method of corresponding agent overwrite it
        # Why not fully clear infos? Provide a hacky way to allow agents send message to each other
        # Why not fully clear _cumulative_rewards? Because it should persist util get consumed by corresponding's agent's last()

        # Auto clear registered fields
        target_field_names_clear_all = []
        target_field_names_clear_selected = []
        for field_name, field_properties in self._registered_fields.items():
            if field_properties["clear_after_use"]:
                if field_properties["only_clear_for_selected_agents"]:
                    target_field_names_clear_selected.append(field_name)
                else:
                    target_field_names_clear_all.append(field_name)
        self.auto_clear_fields(target_field_names_clear_all)
        self.auto_clear_fields(
            target_field_names_clear_selected, robot_names=self.agents[self.agent_selection].robot_names
        )

    def register_tensor_field(
        self,
        field_name: str,
        tensor_sample: torch.Tensor,
        per_robot: bool = True,
        per_env: bool = True,
        clear_in_reset: bool = False,
        clear_after_use: bool = False,
        only_clear_for_selected_agents: bool = True,
        ref_joint_robots: bool = False,
    ) -> None:
        """
        Register a tensor field for each robot in each environment.

        # Only for runtime register

        Args:
            field_name (str): Name of the field
            tensor_sample (torch.Tensor): Sample tensor, will be expanded n_envs times
            per_robot (bool, optional): Whether to register the field for each robot. Defaults to True.
            per_env (bool, optional): Whether to register the field for each robot in each environment. Defaults to True.
            clear_in_reset (bool, optional): Whether to clear the field in reset. Defaults to False.
            clear_after_use (bool, optional): Whether to clear the field after use. Defaults to False.
            only_clear_for_selected_agents (bool, optional): Whether to clear the field only for selected agents. Defaults to True.
            ref_joint_robots (bool, optional): For joint robot, only create one tensor, others ref to it. Defaults to True.
        """
        # Validate inputs
        if not isinstance(tensor_sample, torch.Tensor):
            raise TypeError(f"tensor_sample must be a torch.Tensor, got {type(tensor_sample)}")

        if hasattr(self, field_name):
            raise ValueError(f"Name {field_name} has already been occupied")

        # Move tensor to correct device
        tensor_sample = tensor_sample.to(self.device)
        expanded_tensor_sample = tensor_sample.unsqueeze(0).expand(self.n_envs, *tensor_sample.shape)

        final_sample_tensor = expanded_tensor_sample if per_env else tensor_sample

        if not per_robot:
            setattr(self, field_name, final_sample_tensor.clone())
        else:
            dict_of_tensors = RefDict()
            for training_cfg in self.training_configs.values():
                for i, robot_cfg in enumerate(training_cfg.robot_configs.values()):
                    if ref_joint_robots and training_cfg.is_joint and i != 0:
                        # Set reference to first robot
                        first_robot_name = training_cfg.robot_names[0]
                        dict_of_tensors.add_ref(robot_cfg.name, first_robot_name)
                        continue

                    dict_of_tensors[robot_cfg.name] = final_sample_tensor.clone()

            setattr(self, field_name, dict_of_tensors)

        self._registered_fields[field_name] = {
            'per_robot': per_robot,
            'per_env': per_env,
            'clear_in_reset': clear_in_reset,
            'clear_after_use': clear_after_use,
            'only_clear_for_selected_agents': only_clear_for_selected_agents,
            'ref_joint_robots': ref_joint_robots,
            'tensor_shape': tensor_sample.shape,
            'tensor_dtype': tensor_sample.dtype,
        }
        self.logger.info(f"Registered tensor field '{field_name}' (per_robot={per_robot}, per_env={per_env})")

    def auto_clear_fields(self, field_names, robot_names=None, env_indices=None):
        """Clear registered tensor fields based on their configuration."""
        if not hasattr(self, '_registered_fields'):
            return

        if robot_names is None:
            robot_names = list(self.robots.keys())
        if env_indices is None:
            env_indices = torch.arange(self.n_envs, device=self.device)

        if isinstance(env_indices, list):
            env_indices = torch.tensor(env_indices, device=self.device)

        for field_name in field_names:
            field_info = self._registered_fields[field_name]
            field = getattr(self, field_name)

            if not field_info['per_robot']:
                if not field_info['per_env']:
                    field.zero_()
                else:
                    field[env_indices] = 0
            else:
                for robot_name in robot_names:
                    if robot_name not in field:
                        continue

                    if not field_info['per_env']:
                        field[robot_name].zero_()
                    else:
                        field[robot_name][env_indices] = 0

    """
    HELPERS
    """

    def __repr__(self) -> str:
        """String representation of the AEC environment."""
        return (
            f"VectorizedAECEnv(n_envs={self.n_envs}, "
            f"agents={len(self.agents)}, "
            f"robots={len(self.robots)}, "
            f"frame={self.global_frame_count})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def global_time(self) -> float:
        return self.global_frame_count / self.simulation_frequency

    @property
    def current_agent(self) -> Agent:
        return self.agents[self.agent_selection]

    @property
    def robots(self) -> Dict[str, RobotInterface]:
        return self.scene.robots

    @property
    def robot_names(self) -> List[str]:
        return list(self.robot_configs.keys())

    @property
    def simulator_type(self) -> str:
        return self.simulator_config.simulator_type

    # Hooks
    def pre_build_hooks(self):
        for training_cfg in self.training_configs.values():
            if training_cfg.pre_build_hook is not None:
                training_cfg.pre_build_hook(self)

    def post_build_hooks(self):
        for training_cfg in self.training_configs.values():
            if training_cfg.post_build_hook is not None:
                training_cfg.post_build_hook(self)

    def pre_reset_hooks(self, env_indices):
        for training_cfg in self.training_configs.values():
            if training_cfg.pre_reset_hook is not None:
                training_cfg.pre_reset_hook(self, env_indices)

    def post_reset_hooks(self, env_indices):
        for training_cfg in self.training_configs.values():
            if training_cfg.post_reset_hook is not None:
                training_cfg.post_reset_hook(self, env_indices)

    def pre_last_hooks(self):
        for training_cfg in self.training_configs.values():
            if training_cfg.pre_last_hook is not None:
                training_cfg.pre_last_hook(self)

    def post_last_hooks(self):
        for training_cfg in self.training_configs.values():
            if training_cfg.post_last_hook is not None:
                training_cfg.post_last_hook(self)

    def pre_step_hooks(self):
        """
        After recieving the action but before any step logic (including action processing)
        """
        for training_cfg in self.current_agent.training_configs:
            if training_cfg.pre_step_hook is not None:
                training_cfg.pre_step_hook(self)

        # Update camera system if enabled
        if self.camera_configs and self.scene.camera_manager:
            # Camera manager automatically gets robot positions from scene
            self.scene.camera_manager.step()

    def post_step_hooks(self):
        for training_cfg in self.current_agent.training_configs:
            if training_cfg.post_step_hook is not None:
                training_cfg.post_step_hook(self)


if __name__ == "__main__":
    print("VectorizedAECEnv implementation complete")
    print("This is the core orchestrator for the Genesis MARL framework")
