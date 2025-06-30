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

import genesis as gs
import torch
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple, Union

from genesis_logging import get_module_logger
from utils import RefDict
from configs import (
    TrainingConfigBundle,
    RobotConfig,
    SingleRobotTrainingBundle,
    JointTrainingBundle,
    SharedNetworkBundle,
)
from agent import Agent
from sub_vecenv import SingleRobotSubVecEnv, JointSubVecEnv, SharedNetworkSubVecEnv
from centralized_scheduler import CentralizedFrequencyScheduler, FrequencyOptimizer

logger = get_module_logger("aec_env")


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
        training_config_bundles: List[TrainingConfigBundle],
        n_envs: int = 64,
        device: str = "cuda",
        max_episode_length_s: float = 20.0,
        simulation_frequency: Optional[int] = None,
        render: bool = False,
    ):
        """Initialize the Vectorized AEC Environment.

        Args:
            training_config_bundles: List of training configurations
            n_envs: Number of parallel environments
            device: Device for computations
            max_episode_length_s: Maximum episode length in seconds
            simulation_frequency: Simulation frequency, leave None for automatic optimization
            render: Whether to enable rendering
        """
        logger.info(f"Initializing VectorizedAECEnv with {len(training_config_bundles)} bundles")
        logger.info(f"  n_envs: {n_envs}, device: {device}, max_episode_length: {max_episode_length_s}s")

        if simulation_frequency is None:
            training_frequencies = {bundle.training_name: bundle.frequency for bundle in training_config_bundles}
            simulation_frequency, frequency_info = FrequencyOptimizer.find_optimal_genesis_frequency(
                robot_frequencies=list(training_frequencies.values()), max_genesis_freq=150, tolerance=0.05
            )
            # Print frequency info as table
            logger.info("Automatic optimized frequency:")
            logger.info(f"{'Agent':<10}{'Desired':<10}{'Actual':<10}{'Period':<10}{'Error':<10}")
            for name, info in frequency_info.items():
                logger.info(
                    f"{name:<10}{info['desired_frequency']:<10.1f}Hz"
                    f"{info['actual_frequency']:<10.1f}Hz"
                    f"{info['period_frames']:<10}"
                    f"{info['frequency_error']*100:<10.1f}%"
                )
            logger.info(f"Genesis frequency: {simulation_frequency:.1f} Hz (dt = {1.0 / simulation_frequency:.4f}s)")

        # Basic properties
        self.n_envs = n_envs
        self.device = device
        self.max_episode_length_s = max_episode_length_s
        self.simulation_frequency = simulation_frequency
        self.render = render

        # Store input bundles
        self.training_config_bundles = training_config_bundles

        # Reference fields (centralized data storage)
        self.robots: Dict[str, Any] = {}  # robot_name -> robot entity
        self.robot_configs: Dict[str, RobotConfig] = {}  # robot_name -> robot config
        self.agents: Dict[str, Agent] = {}  # agent_name -> agent instance
        self.subvecenvs: Dict[str, Any] = {}  # training_name -> subVecEnv instance
        self.semaphores: Dict[str, threading.Semaphore] = {}  # training_name -> semaphore

        # AEC related fields (using RefDict for shared data)
        self.terminations: Dict[str, torch.Tensor] = RefDict()  # robot_name -> tensor[n_envs, bool]
        self.truncations: Dict[str, torch.Tensor] = RefDict()  # robot_name -> tensor[n_envs, bool]
        self.rewards: Dict[str, Dict[str, torch.Tensor]] = RefDict()  # robot_name -> Dict[str, tensor[n_envs, float]]
        self._cumulative_rewards: Dict[str, torch.Tensor] = RefDict()  # robot_name -> tensor[n_envs, float]
        self.infos: Dict[str, Dict[str, Any]] = RefDict()  # robot_name -> dict
        self.observations: Dict[str, torch.Tensor] = RefDict()  # robot_name -> tensor[n_envs, obs_dim]

        # Action recievers
        self.action_buffers: Dict[str, List[torch.Tensor]] = RefDict()  # robot_name -> list[torch.Tensor]

        # Soft death related fields
        self.dead_robots: Dict[str, torch.Tensor] = RefDict()  # robot_name -> tensor[n_envs, bool]

        # AEC state
        self.agent_selection: Optional[str] = None  # Current selected agent
        self.scheduler: Optional[CentralizedFrequencyScheduler] = None

        # Episode tracking
        self.episode_frame_count: torch.Tensor = torch.zeros(n_envs, device=device)  # Time in seconds
        self.global_frame_count: int = 0

        # Genesis scene (the only place it should exist)
        self.scene: Optional[gs.Scene] = None

        logger.info("Starting AEC initialization sequence...")
        self._initialize_aec()
        logger.info("✅ VectorizedAECEnv initialization complete")

    def _initialize_aec(self):
        """Initialize the AEC environment following CLAUDE.md specifications."""

        # Step 1: Initialize all resources (semaphores, etc)
        logger.info("Step 1: Initializing resources...")
        self._initialize_resources()

        # Step 2: Group TrainingConfigBundles by frequency and initialize agents
        logger.info("Step 3: Grouping bundles by frequency and creating agents...")
        self._create_agents()

        # Step 4: Setup scheduler
        logger.info("Step 4: Setting up scheduler...")
        self._setup_scheduler()

        # Step 5: Setup Genesis scene
        logger.info("Step 5: Setting up Genesis scene...")
        self._setup_genesis_scene()

        # Step 6: Add robots by calling agent's setup_robots method
        logger.info("Step 6: Adding robots to scene...")
        self._setup_robots()

        # Step 7: Setup DP parameters according to each RobotConfig
        logger.info("Step 7: Setting up DP parameters...")
        self._setup_dp_parameters()

        # Step 8: Detect observation dimensions
        logger.info("Step 8: Detecting observation dimensions...")
        self._detect_obs_dimensions()

        # Step 8: Initialize all buffers (now we know obs dims)
        logger.info("Step 9: Initializing data buffers...")
        self._initialize_buffers()

        # Step 9: Create subVecEnv entities based on TrainingConfigBundles
        logger.info("Step 2: Creating subVecEnv entities...")
        self._create_subvecenvs()

    def _initialize_resources(self):
        """Initialize semaphores and other resources."""
        for bundle in self.training_config_bundles:
            training_name = bundle.training_name

            # Create semaphore for this training
            # Start with 0 (subVecEnv should send first)
            semaphore = threading.Semaphore(1)
            self.semaphores[training_name] = semaphore

            logger.debug(f"Created semaphore for training: {training_name}")

        logger.info(f"Initialized {len(self.semaphores)} semaphores")

    def _create_subvecenvs(self):
        """Create subVecEnv instances for each TrainingConfigBundle."""
        for training_cfg in self.training_config_bundles:
            training_name = training_cfg.training_name

            # Create subVecEnv using factory function
            if isinstance(training_cfg, SingleRobotTrainingBundle):
                self.subvecenvs[training_name] = SingleRobotSubVecEnv(training_cfg, self)
            elif isinstance(training_cfg, JointTrainingBundle):
                self.subvecenvs[training_name] = JointSubVecEnv(training_cfg, self)
            elif isinstance(training_cfg, SharedNetworkBundle):
                self.subvecenvs[training_name] = SharedNetworkSubVecEnv(training_cfg, self)

            # Set semaphore reference
            self.subvecenvs[training_name].semaphore = self.semaphores[training_name]

            # Register subvecenv in agent
            for agent in self.agents.values():
                if training_cfg in agent.training_config_bundles:
                    agent.subvecenvs[training_name] = self.subvecenvs[training_name]

            logger.debug(f"Created subVecEnv: {training_name} ({type(self.subvecenvs[training_name]).__name__})")

        logger.info(f"Created {len(self.subvecenvs)} subVecEnvs")

    def _create_agents(self):
        """Group bundles by frequency and create agents."""
        # Group bundles by frequency
        frequency_groups = defaultdict(list)
        for training_cfg in self.training_config_bundles:
            frequency = training_cfg.frequency
            frequency_groups[frequency].append(training_cfg)

        # Create agents for each frequency group
        for frequency, bundles in frequency_groups.items():
            agent_name = f"agent_{frequency}hz"

            # Create agent with all bundles of this frequency
            agent = Agent(agent_name=agent_name, frequency=frequency, training_config_bundles=bundles, mother_env=self)

            self.agents[agent_name] = agent

            # Set subVecEnv and semaphore references in agent
            for training_cfg in bundles:
                training_name = training_cfg.training_name
                agent.semaphores[training_name] = self.semaphores[training_name]

            logger.debug(f"Created agent: {agent_name} with {bundles}")

    def _setup_scheduler(self):
        """Setup the centralized frequency scheduler."""
        agent_frequencies = {}
        for agent_name, agent in self.agents.items():
            agent_frequencies[agent_name] = agent.frequency

        self.scheduler = CentralizedFrequencyScheduler(agent_frequencies=agent_frequencies)

        logger.info(f"Scheduler setup with {len(agent_frequencies)} agents")

    def _setup_genesis_scene(self):
        """Setup the Genesis simulator scene."""
        logger.info("Initializing Genesis...")
        gs.init(backend=gs.gpu)

        # Create scene with appropriate settings
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1 / self.simulation_frequency, substeps=4),
            viewer_options=gs.options.ViewerOptions(res=(800, 600) if self.render else (1, 1)),
            vis_options=gs.options.VisOptions(n_rendered_envs=min(4, self.n_envs) if self.render else 1),
            show_viewer=self.render,
        )

        # Add ground plane
        self.scene.add_entity(gs.morphs.Plane())

        logger.info("Genesis scene created successfully")

    def _setup_robots(self):
        """Add robots to scene by calling each agent's setup_robots method."""
        for agent_name, agent in self.agents.items():
            logger.debug(f"Setting up robots for agent: {agent_name}")
            agent.setup_robots()
            self.robot_configs.update(agent.robot_configs)

        # Build the scene now that all robots are added
        self.scene.build(n_envs=self.n_envs)
        logger.info("✅ Genesis scene built successfully")

        logger.info(f"Setup complete for {len(self.robots)} robots")

    def _setup_dp_parameters(self):
        """Setup DP parameters for all robots."""
        for agent in self.agents.values():
            for robot_name in agent.robot_names:
                agent._setup_robot_dp_parameters(robot_name)
                logger.debug(f"DP parameters set for robot: {robot_name}")

    def _detect_obs_dimensions(self):
        """Detect observation dimensions for each robot."""

        for robot_name, robot_cfg in self.robot_configs.items():
            robot_cfg.detect_obs_dim(self)

        logger.info("Observation dimensions detected successfully")

    def _initialize_buffers(self):
        """Initialize all data buffers with proper dimensions."""
        logger.info("Initializing data buffers...")

        for training_cfg in self.training_config_bundles:
            isJointTrainingBundle = isinstance(training_cfg, JointTrainingBundle)
            for i, robot_cfg in enumerate(training_cfg.robot_configs):
                robot_name = robot_cfg.name
                if isJointTrainingBundle and i != 0:
                    # Set reference to first robot
                    first_robot_name = training_cfg.robot_configs[0].name
                    self.observations.add_ref(robot_name, first_robot_name)
                    self.rewards.add_ref(robot_name, first_robot_name)
                    self._cumulative_rewards.add_ref(robot_name, first_robot_name)
                    self.terminations.add_ref(robot_name, first_robot_name)
                    self.truncations.add_ref(robot_name, first_robot_name)
                    self.dead_robots.add_ref(robot_name, first_robot_name)
                    self.infos.add_ref(robot_name, first_robot_name)
                    continue
                self.observations[robot_name] = torch.zeros(self.n_envs, robot_cfg.observation_dim, device=self.device)
                self.rewards[robot_name] = {}
                self._cumulative_rewards[robot_name] = torch.zeros(self.n_envs, 1, device=self.device)
                self.terminations[robot_name] = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
                self.truncations[robot_name] = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
                self.dead_robots[robot_name] = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
                self.infos[robot_name] = {}  # Will be populated during episodes

                logger.debug(f"Initialized buffers for robot {robot_name} (obs_dim={robot_cfg.observation_dim})")

        logger.info("✅ Data buffers initialized")

    def reset(self) -> None:
        """Reset the environment (called once at the beginning).

        Following CLAUDE.md specifications:
        1. The reset only called once at the beginning
        2. call _reset_envs(all)
        3. reset scheduler
        """
        logger.info("🔄 Resetting AEC environment...")

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

        logger.info(f"✅ AEC environment reset complete, starting with agent: {self.agent_selection}")

    def _reset_envs(self, env_indices: torch.Tensor) -> None:
        """Reset robots in specified environments.

        Args:
            env_indices: Tensor of environment indices to reset
        """
        logger.debug(f"Resetting environments: {env_indices.tolist()}")

        # Call all agents' _reset_envs method
        for agent in self.agents.values():
            agent._reset_robots(env_indices)

        # Reset episode time for these environments
        self.episode_frame_count[env_indices] = 0.0

        # Reset dead robot flags
        for robot_name in self.dead_robots.keys():
            self.dead_robots[robot_name][env_indices] = False

        logger.debug(f"Reset complete for {len(env_indices)} environments")

    def step(self) -> None:
        """Execute one step of the AEC environment.

        Following CLAUDE.md specifications:
        1. Call current selected agent's step method
        """
        if self.agent_selection is None:
            raise RuntimeError("No agent selected. Call reset() first.")

        # Clear Cumulative rewards of current agent
        for robot_name in self.agents[self.agent_selection].robot_names:
            self._cumulative_rewards[robot_name].zero_()

        # Clear immediate rewards used in last
        self._clear_rewards()

        logger.debug(f"AEC frame {self.global_frame_count}: agent {self.agent_selection}")

        # Get current agent
        current_agent = self.agents[self.agent_selection]

        # Execute agent step
        current_agent.step()

        # Update agent selection and step simulator based on scheduler
        next_agent, frames_to_advance = self.scheduler.step()

        self.agent_selection = next_agent
        self.scene.step(frames_to_advance)

        self.episode_frame_count += frames_to_advance
        self.global_frame_count += frames_to_advance

    def last(self) -> None:
        """Send observations/rewards/terminations to current agent's subVecEnvs.

        Following CLAUDE.md specifications:
        1. Calculate obs,reward,truncation,info for each subVecEnv of current agent
        2. Check termination of environment - reach 20s or all robots dead
        3. Collect termination/truncation observations of dead robots
        4. Do auto reset for terminated envs, revoke dead robots
        5. Call current selected agent's last method
        """
        if self.agent_selection is None:
            raise RuntimeError("No agent selected. Call reset() first.")

        current_agent = self.agents[self.agent_selection]

        logger.debug(f"AEC last(): agent {self.agent_selection}")

        old_dead_robots = self.dead_robots

        # Step 1: Calculate obs,reward,truncation,info for current agent's robots
        for training_cfg in current_agent.training_config_bundles:
            self.observations.update(training_cfg.obs_function(self))
            self.rewards.update(training_cfg.reward_function(self))
            self.truncations.update(training_cfg.truncation_function(self))
            self.infos.update(training_cfg.info_function(self))

        # We also calculate properties for dead robots in some envs in previous step, the obs,truncation,info are harmless, but the reward might affect other robot's reward, we need to mask it
        for robot_name, dead_mask in self.dead_robots.items():
            reward_dict = self.rewards[robot_name]
            for target_robot_name, reward_tensor in reward_dict.items():
                reward_tensor[dead_mask].zero_()

        # Update the dead robots with new truncations
        for robot_name in current_agent.robots:
            self.dead_robots[robot_name] |= self.truncations[robot_name]

        self._accumulate_rewards()
        # Step 2: Check termination conditions
        terminated_envs = self._check_termination_conditions()

        # Record all the agents that dead in current step by comparing old and new dead robots, till here we haven't bring any robot back to life, so the diff is all the new dead robots
        robot_dead_in_current_step = {
            robot_name: old_dead_robots[robot_name] != self.dead_robots[robot_name] for robot_name in self.robots.keys()
        }

        # Step 3: Collect termination/truncation observations
        for robot_name, dead_idx in robot_dead_in_current_step.items():
            if torch.any(dead_idx):
                # Store current observations as termination observations
                if 'termination_obs' not in self.infos[robot_name]:
                    self.infos[robot_name]['termination_obs'] = torch.zeros_like(self.observations[robot_name])
                    self.infos[robot_name]['termination_obs'].fill_(float('nan'))

                self.infos[robot_name]['termination_obs'][dead_idx] = self.observations[robot_name][dead_idx]

        # Step 4: Auto reset terminated environments
        if torch.any(terminated_envs):
            terminated_indices = torch.where(terminated_envs)[0]
            logger.debug(f"Auto-resetting environments: {terminated_indices.tolist()}")
            self._reset_envs(terminated_indices)

        # Record the reset observation to replace the old termination observations
        for training_cfg in current_agent.training_config_bundles:
            # Just update all for code simplicity, for those who are not resetted, obs don't change
            self.observations.update(training_cfg.obs_function(self))

        # Step 5: Call current agent's last method to send everything
        current_agent.last()

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
        for robot_name in self.dead_robots.keys():
            robot_dead = self.dead_robots[robot_name]
            all_robots_dead = all_robots_dead & robot_dead

        # Environment terminates if either condition is met
        terminated = time_exceeded | all_robots_dead

        # Update termination flags for all robots in terminated environments
        for robot_name in self.terminations.keys():
            self.terminations[robot_name] = terminated
            # Set their dead flags to True
            self.dead_robots[robot_name] |= terminated

        return terminated

    def _accumulate_rewards(self) -> None:
        for robot_name, reward_dict in self.rewards.items():
            for target_robot_name, reward_tensor in reward_dict.items():
                self._cumulative_rewards[target_robot_name].add_(reward_tensor)

    def _clear_rewards(self) -> None:
        for robot_name in self.rewards.keys():
            self.rewards[robot_name] = {}

    def __repr__(self) -> str:
        """String representation of the AEC environment."""
        return (
            f"VectorizedAECEnv(n_envs={self.n_envs}, "
            f"agents={len(self.agents)}, "
            f"robots={len(self.robots)}, "
            f"frame={self.global_frame_count})"
        )


if __name__ == "__main__":
    print("VectorizedAECEnv implementation complete")
    print("This is the core orchestrator for the Genesis MARL framework")
