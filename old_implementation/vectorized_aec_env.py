"""Vectorized AEC Environment for Genesis MARL Framework V3.

This implements a vectorized version of the PettingZoo AEC API where:
- All environments step the same agent together (centralized scheduling)
- Dead agents are handled with soft removal (NaN padding)
- Auto-reset happens per environment when all agents die
- Direct tensor sharing with condition variables for coordination
"""

import torch
import numpy as np
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import traceback
import gymnasium.spaces as spaces

import genesis as gs

from .robot_config import RobotConfig, RobotConfigBundle, JointTrainingBundle, SharedNetworkBundle
from .agent import Agent
from .centralized_scheduler import CentralizedFrequencyScheduler, find_optimal_genesis_frequency
from .joint_detector import get_joint_names_from_config
from .genesis_logging import get_module_logger
# Import SubVecEnv classes
from .sub_vecenv_wrapper import SubVecEnvWrapper, create_subvecenv_v3
from .joint_training_subvecenv import JointTrainingSubVecEnv
from .shared_network_subvecenv import SharedNetworkSubVecEnv

logger = get_module_logger("vectorized_aec_env")


class BasicRobotBundle(RobotConfigBundle):
    """Basic implementation of RobotConfigBundle for automatically created bundles."""
    
    def __init__(self, bundle_name: str, frequency: float, robot_configs: List[RobotConfig]):
        super().__init__(bundle_name, frequency)
        self.robot_configs = robot_configs
    
    def get_action_dim(self) -> int:
        return sum(c.action_dim for c in self.robot_configs)
    
    def get_observation_dim(self) -> int:
        return sum(c.observation_dim for c in self.robot_configs)
    
    def get_n_envs_multiplier(self) -> int:
        return 1


class VectorizedAECEnv:
    """Vectorized AEC Environment for multi-robot MARL with direct tensor sharing.
    
    Key design principles (V3):
    - Direct tensor sharing: SubVecEnvs access mother env data directly (no copying)
    - Condition variable coordination: Simple synchronization instead of queues
    - Centralized agent selection: all environments step the same agent
    - Soft dead agent handling: dead agents stay in data structures with NaN
    - Vectorized operations: all methods work on [n_envs, ...] tensors
    - Auto-reset: individual environments reset when all agents die
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "genesis_marl_vecenv_v3",
    }
    
    def __init__(
        self,
        robot_configs: Union[List[RobotConfig], List[RobotConfigBundle]],
        n_envs: int,
        scene_config: Optional[Dict[str, Any]] = None,
        max_episode_length: int = 1000,
        genesis_freq: Optional[float] = None,
        show_viewer: bool = False,
        device: Optional[str] = None,
        log_level: str = "INFO",
    ):
        """Initialize vectorized AEC environment.
        
        Args:
            robot_configs: List of robot configurations or bundles (mixed input supported)
            n_envs: Number of parallel environments
            scene_config: Optional Genesis scene configuration
            max_episode_length: Maximum episode length in steps
            genesis_freq: Genesis simulation frequency (auto-calculated if None)
            show_viewer: Whether to show Genesis viewer
            device: PyTorch device (auto-detected if None)
            log_level: Logging level
        """
        
        # Setup Genesis-compatible logging
        from .genesis_logging import setup_genesis_logging
        self.genesis_logger = setup_genesis_logging(level=log_level)
        
        logger.info("="*80)
        logger.info("Initializing VectorizedAECEnv V3 (Direct Tensor Sharing)")
        logger.info("="*80)
        logger.info(f"Number of robots: {len(robot_configs)}")
        logger.info(f"Number of environments: {n_envs}")
        logger.info(f"Max episode length: {max_episode_length}")
        
        # Store original inputs
        self.original_robot_configs = robot_configs
        self.n_envs = n_envs
        self.max_episode_length = max_episode_length
        self.show_viewer = show_viewer
        
        # IMPORTANT CHANGE 0: Extract and reorganize robot configs into bundles
        self._extract_and_reorganize_robot_configs()
        
        # Initialize Genesis if needed
        if not hasattr(gs, 'device'):
            logger.info("Initializing Genesis...")
            gs.init(seed=0, precision="32", logging_level="warning")
        self.device = gs.device if device is None else device
        logger.info(f"Using device: {self.device}")
        
        # Setup agents (AEC API) - now using agent bundles instead of individual robots
        self.possible_agents = [agent.agent_name for agent in self.agents]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}
        self.n_agents = len(self.possible_agents)
        
        logger.info(f"Agent bundles created: {[agent.agent_name for agent in self.agents]}")
        
        # Calculate optimal Genesis frequency using all basic robot configs
        robot_frequencies = [config.frequency for config in self.all_basic_robot_configs]
        if genesis_freq is None:
            self.genesis_freq, freq_info = find_optimal_genesis_frequency(robot_frequencies)
        else:
            self.genesis_freq = genesis_freq
        self.dt = 1.0 / self.genesis_freq
        
        logger.info(f"Genesis frequency: {self.genesis_freq:.1f} Hz (dt={self.dt:.6f}s)")
        
        # Create centralized scheduler using agent bundle names (not individual robot names)
        agent_bundle_names = [agent.agent_name for agent in self.agents]
        agent_bundle_frequencies = [agent.bundle.frequency for agent in self.agents]
        
        logger.info(f"🗓️ Scheduler will track bundle names: {agent_bundle_names}")
        logger.info(f"📊 Bundle frequencies: {dict(zip(agent_bundle_names, agent_bundle_frequencies))}")
        
        self.scheduler = CentralizedFrequencyScheduler(
            robot_names=agent_bundle_names,  # Actually agent bundle names now
            robot_frequencies=agent_bundle_frequencies,
            genesis_freq=self.genesis_freq,
            device=str(self.device)
        )
        
        # Create scene and setup robots via Agent classes
        self._create_scene(scene_config)
        self._setup_robots_via_agents()
        
        # Initialize all buffers
        self._init_buffers()
        
        # Auto-spawn SubVecEnv threads (DESIGN.md requirement)
        self._spawn_subvecenvs()
        
        # Track simulation state
        self.total_frames = 0
        self._initialized = True
        
        logger.info("VectorizedAECEnv V3 initialized successfully!")
        logger.info("="*80)
    
    def _extract_and_reorganize_robot_configs(self):
        """Reorganize original user bundles by frequency while preserving structure.
        
        This implements IMPORTANT CHANGE 0 from DESIGN.md:
        1. Preserve original user bundle structure (don't extract)
        2. Reorganize original bundles by frequency into Agent instances
        3. Each Agent knows which original bundles it manages
        4. Initialize bidirectional coordination flags per Agent
        """
        logger.info("🔄 Reorganizing robot configs by frequency (preserving original bundles)...")
        
        # Step 1: Group original bundles/configs by frequency
        frequency_groups = defaultdict(list)
        
        for item in self.original_robot_configs:
            if isinstance(item, RobotConfigBundle):
                frequency = item.frequency
                frequency_groups[frequency].append(item)
                logger.debug(f"Added bundle '{item.bundle_name}' to {frequency}Hz group")
            elif isinstance(item, RobotConfig):
                frequency = item.frequency
                frequency_groups[frequency].append(item)
                logger.debug(f"Added robot config '{item.name}' to {frequency}Hz group")
            else:
                raise ValueError(f"Invalid robot config type: {type(item)}")
        
        logger.info(f"🔀 Reorganized into {len(frequency_groups)} frequency groups")
        for freq, items in frequency_groups.items():
            item_names = []
            for item in items:
                if isinstance(item, RobotConfigBundle):
                    item_names.append(f"Bundle({item.bundle_name})")
                else:
                    item_names.append(f"Robot({item.name})")
            logger.info(f"  {freq}Hz: {item_names}")
        
        # Step 2: Create Agent instances for each frequency group
        self.agents : List[Agent] = []
        
        # IMPORTANT CHANGE 1: Initialize bidirectional coordination flags per Agent
        self.coordination_flags = {}  # {agent_name: threading.Condition}
        self.flag_states = {}  # {agent_name: "AEC_SIDE" | "SUBVECENV_SIDE"}
        self.action_cache = {}  # {agent_name: tensor}
        
        for i, (frequency, original_bundles) in enumerate(frequency_groups.items()):
            # Create Agent name
            agent_name = f"agent_{i}_{frequency}Hz"
            
            # IMPORTANT CHANGE 2: Create Agent instance with original bundles
            agent = Agent(
                original_bundles=original_bundles,
                frequency=frequency,
                agent_name=agent_name,
                mother_env=self
            )
            self.agents.append(agent)
            
            # IMPORTANT CHANGE 1: Initialize coordination flags for this agent
            self.coordination_flags[agent_name] = threading.Condition()
            self.flag_states[agent_name] = "AEC_SIDE"  # Initially AEC should send
            self.action_cache[agent_name] = None
            
            logger.debug(f"🔧 Initialized coordination for '{agent_name}': flag='AEC_SIDE', condition_var=created")
            
            # Set coordination references in Agent
            agent.set_coordination_references(
                self.coordination_flags[agent_name],
                self.flag_states,
                self.action_cache
            )
            
            # Count total robots for logging
            total_robots = len(agent.flattened_robot_configs)
            logger.info(f"✅ Created Agent '{agent_name}' with {len(original_bundles)} original bundles, {total_robots} total robots")
        
        # Step 3: Create flattened robot configs for compatibility
        self.all_basic_robot_configs = []
        for agent in self.agents:
            self.all_basic_robot_configs.extend(agent.flattened_robot_configs)
        
        logger.info(f"🤖 Created {len(self.agents)} Agent instances with coordination flags")
        logger.info(f"📦 Total flattened robot configs: {len(self.all_basic_robot_configs)}")
    
    def _create_scene(self, scene_config: Optional[Dict[str, Any]] = None):
        """Create Genesis scene."""
        logger.debug("Creating Genesis scene...")
        
        # Default configuration - always include viewer_options and vis_options
        # Genesis handles them appropriately when show_viewer=False
        default_config = {
            "sim_options": gs.options.SimOptions(dt=self.dt, substeps=2),
            "viewer_options": gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            "vis_options": gs.options.VisOptions(rendered_envs_idx=list(range(min(4, self.n_envs)))),
            "rigid_options": gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            "show_viewer": self.show_viewer,
        }
        
        if scene_config:
            default_config.update(scene_config)
        
        self.scene = gs.Scene(**default_config)
        
        # Add ground plane
        self.scene.add_entity(gs.morphs.Plane())
        
        logger.debug(f"Scene created (show_viewer={self.show_viewer})")
    
    def _setup_robots_via_agents(self):
        """Setup robots via Agent classes (IMPORTANT CHANGE 2).
        
        This implements the Agent-based robot setup from DESIGN.md:
        - Agent classes handle robot creation in Genesis scene
        - Proper robot indexing for control parameter setup
        """
        logger.debug("🤖 Setting up robots via Agent classes...")
        
        # Collect all robots from all agents
        self.robots = []
        self.robot_to_agent_mapping = {}  # {robot_index: agent_index}
        
        robot_index = 0
        for agent_idx, agent in enumerate(self.agents):
            logger.debug(f"Setting up robots for Agent '{agent.agent_name}'")
            
            # Agent sets up its robots in the scene
            agent_robots = agent.setup_robots(self.scene)
            
            # Track robot indices for this agent
            agent_robot_indices = list(range(robot_index, robot_index + len(agent_robots)))
            agent.set_robot_indices(agent_robot_indices)
            
            # Add to global robot list and mapping
            self.robots.extend(agent_robots)
            for i in range(len(agent_robots)):
                self.robot_to_agent_mapping[robot_index + i] = agent_idx
            
            robot_index += len(agent_robots)
            logger.debug(f"  Added {len(agent_robots)} robots for Agent '{agent.agent_name}'")
        
        # Build scene with parallel environments
        self.scene.build(n_envs=self.n_envs)
        logger.debug(f"🏗️ Scene built with {self.n_envs} parallel environments")
        
        # Setup control parameters via Agent classes
        for agent in self.agents:
            agent.setup_control_parameters()
        
        logger.info(f"🚀 Robot setup complete: {len(self.robots)} robots across {len(self.agents)} agents")
    
    def _spawn_subvecenvs(self):
        """Auto-spawn SubVecEnv instances in threads with OnPolicyRunner integration.
        
        This implements DESIGN.md section 236-238:
        1. Create SubVecEnv instances based on bundle types (basic/joint/shared)
        2. Wrap in OnPolicyRunner and start training threads automatically
        3. Setup logging redirection per SubVecEnv
        """
        logger.info("🚀 Auto-spawning SubVecEnv threads...")
        
        self.subvecenv_threads = {}
        self.subvecenvs = {}
        
        for agent in self.agents:
            logger.info(f"📱 Spawning SubVecEnv threads for Agent '{agent.agent_name}'")
            
            # Get SubVecEnv specifications from the Agent
            subvecenv_specs = agent.get_subvecenv_specs()
            logger.info(f"  Agent manages {len(subvecenv_specs)} SubVecEnvs")
            
            for spec in subvecenv_specs:
                subvecenv_name = spec["agent_name"]
                logger.info(f"  📱 Creating SubVecEnv '{subvecenv_name}' (type: {spec['type']})")
                
                try:
                    # Create SubVecEnv based on specification
                    if spec["type"] == "joint_training":
                        subvecenv = JointTrainingSubVecEnv(
                            bundle=spec["bundle"],
                            mother_env=self,
                            n_envs=self.n_envs,
                            device=str(self.device),
                            timeout=30.0
                        )
                    elif spec["type"] == "shared_network":
                        subvecenv = SharedNetworkSubVecEnv(
                            bundle=spec["bundle"],
                            mother_env=self,
                            n_envs=self.n_envs,
                            device=str(self.device),
                            timeout=30.0
                        )
                    elif spec["type"] == "basic": 
                        subvecenv = SubVecEnvWrapper(
                            agent_name=subvecenv_name,
                            mother_env=self,
                            n_envs=self.n_envs,
                            obs_dim=spec["obs_dim"],
                            action_dim=spec["action_dim"],
                            device=str(self.device),
                            timeout=30.0
                        )
                    elif spec["type"] == "single_robot":
                        subvecenv = SubVecEnvWrapper(
                            agent_name=subvecenv_name,
                            mother_env=self,
                            n_envs=self.n_envs,
                            obs_dim=spec["obs_dim"],
                            action_dim=spec["action_dim"],
                            device=str(self.device),
                            timeout=30.0
                        )
                    else:
                        logger.error(f"Unknown SubVecEnv type: {spec['type']}")
                        continue
                    
                    self.subvecenvs[subvecenv_name] = subvecenv
                
                    # DESIGN.md requirement: spawn training thread with OnPolicyRunner
                    logger.info(f"    🔧 Spawning training thread for '{subvecenv_name}'")
                    
                    def training_thread_function(subvecenv, subvecenv_name):
                        """Training thread function that runs OnPolicyRunner."""
                        try:
                            # Import rsl_rl
                            from rsl_rl.runners.on_policy_runner import OnPolicyRunner
                            
                            logger.info(f"🏃 Training thread started for '{subvecenv_name}'")
                            
                            # Initialize SubVecEnv
                            obs, extras = subvecenv.reset()
                            logger.info(f"  SubVecEnv '{subvecenv_name}' reset completed, obs shape: {obs.shape}")
                            
                            # Create minimal runner config for the agent
                            runner_cfg = {
                                'algorithm': {
                                    'entropy_coef': 0.01,
                                    'gamma': 0.99,
                                    'lam': 0.95,
                                    'num_learning_epochs': 5,
                                    'num_mini_batches': 4,
                                    'value_loss_coef': 1.0,
                                    'clip_param': 0.2,
                                    'desired_kl': 0.01,
                                    'max_grad_norm': 1.0
                                },
                                'runner': {
                                    'max_iterations': 100000,
                                    'num_steps_per_env': 24,
                                    'save_interval': 100,
                                    'experiment_name': f'{subvecenv_name}_training'
                                }
                            }
                            
                            # Create and start OnPolicyRunner
                            runner = OnPolicyRunner(subvecenv, runner_cfg, device=str(self.device))
                            
                            logger.info(f"🎯 OnPolicyRunner created for '{subvecenv_name}', starting training...")
                            runner.learn(num_learning_iterations=runner_cfg['runner']['max_iterations'])
                            
                        except Exception as e:
                            logger.error(f"❌ Training thread failed for '{subvecenv_name}': {e}")
                            logger.error(traceback.format_exc())
                    
                    # Start training thread
                    thread = threading.Thread(
                        target=training_thread_function,
                        args=(subvecenv, subvecenv_name),
                        name=f"Training-{subvecenv_name}",
                        daemon=True
                    )
                    thread.start()
                    self.subvecenv_threads[subvecenv_name] = thread
                    
                    logger.info(f"    ✅ Created SubVecEnv and started training thread for '{subvecenv_name}'")
                
                except Exception as e:
                    logger.error(f"    ❌ Failed to create SubVecEnv for '{subvecenv_name}': {e}")
                    continue
        
        logger.info(f"🏁 SubVecEnv spawning complete: {len(self.subvecenvs)} SubVecEnvs created")
        logger.info(f"🧵 Training threads started: {len(self.subvecenv_threads)} threads running")
        
        # Give threads a moment to initialize
        time.sleep(1.0)
        logger.info("✅ All SubVecEnv training threads are now running!")
    
    def _init_buffers(self):
        """Initialize all vectorized buffers."""
        logger.debug("Initializing buffers...")
        
        # AEC API: agents list (names, constant for soft removal)
        self.agent_names = self.possible_agents[:]
        
        # Centralized agent selection
        agent_name = self.scheduler.get_current_agent()
        self.agent_selection = agent_name
        
        # Agent alive tracking (soft removal)
        self.agent_alive_mask = torch.ones((self.n_envs, self.n_agents), 
                                          device=self.device, dtype=torch.bool)
        
        # AEC state tracking (all vectorized)
        self.terminations = {
            agent: torch.zeros(self.n_envs, device=self.device, dtype=torch.bool)
            for agent in self.possible_agents
        }
        self.truncations = {
            agent: torch.zeros(self.n_envs, device=self.device, dtype=torch.bool)
            for agent in self.possible_agents
        }
        self.rewards = {
            agent: torch.zeros(self.n_envs, device=self.device, dtype=torch.float32)
            for agent in self.possible_agents
        }
        self._cumulative_rewards = {
            agent: torch.zeros(self.n_envs, device=self.device, dtype=torch.float32)
            for agent in self.possible_agents
        }
        self.infos = {
            agent: [{} for _ in range(self.n_envs)]
            for agent in self.possible_agents
        }
        
        # Observation and action spaces
        self.observation_spaces = {}
        self.action_spaces = {}
        
        # Create spaces for agent bundles (not individual robots)
        for agent in self.agents:
            agent_name = agent.agent_name
            bundle = agent.bundle
            
            # Observation space shape depends on bundle type
            if agent.is_shared_network:
                # Shared network: multiple environments per bundle
                n_envs_multiplier = bundle.get_n_envs_multiplier()
                obs_shape = (bundle.get_observation_dim(),)  # Shape per robot
            else:
                # Joint training or single robot: concatenated or single obs
                obs_shape = (bundle.get_observation_dim(),)
            
            self.observation_spaces[agent_name] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=obs_shape,
                dtype=np.float32
            )
            
            # Action space shape depends on bundle type
            if agent.is_shared_network:
                action_shape = (bundle.get_action_dim(),)  # Shape per robot
            else:
                action_shape = (bundle.get_action_dim(),)  # Concatenated or single
                
            self.action_spaces[agent_name] = spaces.Box(
                low=-1.0, high=1.0,
                shape=action_shape,
                dtype=np.float32
            )
        
        # Note: Observation buffers removed in V3 - using Agent.calculate_observations() instead
        
        # Episode tracking
        self.episode_lengths = torch.zeros(self.n_envs, device=self.device, dtype=torch.int32)
        self.episode_returns = {
            agent: torch.zeros(self.n_envs, device=self.device, dtype=torch.float32)
            for agent in self.possible_agents
        }
        
        # Environment done tracking
        self.env_done_mask = torch.zeros(self.n_envs, device=self.device, dtype=torch.bool)
        
        # V3: Direct tensor sharing coordination structures
        self._init_coordination_structures()
        
        logger.debug("Buffers initialized")
    
    def _init_coordination_structures(self):
        """Initialize coordination structures for direct tensor sharing (V3)."""
        logger.debug("Initializing V3 coordination structures...")
        
        # Condition variables for each agent (for synchronization)
        self.agent_conditions = {
            agent: threading.Condition() for agent in self.possible_agents
        }
        
        # Action storage for each agent (direct tensor sharing)
        self.agent_actions = {
            agent: None for agent in self.possible_agents
        }
        
        # Action ready flags for coordination
        self.agent_action_ready = {
            agent: False for agent in self.possible_agents
        }
        
        # SubVecEnv references (will be populated when SubVecEnvs connect)
        self.connected_subvecenvs = {}
        
        logger.debug(f"Created coordination structures for {len(self.possible_agents)} agents")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset all environments (AEC API)."""
        logger.info("Resetting all environments...")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Reset scheduler
        self.scheduler.reset()
        agent_name = self.scheduler.get_current_agent()
        self.agent_selection = agent_name
        
        # Reset simulation state
        self.total_frames = 0
        
        # Reset all environments
        all_env_indices = torch.arange(self.n_envs, device=self.device)
        self._reset_envs(all_env_indices)
        
        logger.info("Reset complete")
    
    def _reset_envs(self, env_indices: torch.Tensor):
        """Reset specific environments."""
        if len(env_indices) == 0:
            return
        
        logger.debug(f"Resetting {len(env_indices)} environments: {env_indices.tolist()}")
        
        # Reset episode tracking
        self.episode_lengths[env_indices] = 0
        for agent in self.possible_agents:
            self.episode_returns[agent][env_indices] = 0.0
        
        # Reset agent states (soft removal - agents stay alive)
        self.agent_alive_mask[env_indices] = True
        self.env_done_mask[env_indices] = False
        
        # Reset AEC states
        for agent in self.possible_agents:
            self.terminations[agent][env_indices] = False
            self.truncations[agent][env_indices] = False
            self.rewards[agent][env_indices] = 0.0
            self._cumulative_rewards[agent][env_indices] = 0.0
            for env_idx in env_indices:
                self.infos[agent][env_idx.item()] = {}
        
        # Reset robot positions via Agent classes (V3)
        for agent in self.agents:
            agent.reset_robots(env_indices)
        
        # Note: Observations now computed on-demand via Agent.calculate_observations() in V3
        
        logger.debug(f"Reset complete for envs {env_indices.tolist()}")
    
    def step(self, action: Optional[Union[np.ndarray, torch.Tensor]] = None):
        """Step with action for current agent across all environments (AEC API - V3 with Flag Coordination).
        
        IMPORTANT CHANGE 1: Implements flag coordination from DESIGN.md section 195:
        - Wait for flag to flip to AEC side (indicating action is ready)
        - Fetch action from cache (avoids race conditions)
        - Apply action and continue AEC cycle
        
        Args:
            action: Optional action tensor (for compatibility, usually None in V3)
        """
        # In V3, agent_selection is now a bundle name, not robot name
        bundle_name = self.agent_selection
        current_agent_bundle = None
        for agent in self.agents:
            if agent.agent_name == bundle_name:
                current_agent_bundle = agent
                break
        
        if not current_agent_bundle:
            logger.error(f"No Agent bundle found for bundle name: {bundle_name}")
            return
        
        agent_name = bundle_name
        agent = self.agent_selection
        
        logger.debug(f"Step called for agent {agent} (bundle: {agent_name})")
        
        # Clear accumulated reward since last step
        if agent in self._cumulative_rewards:
            self._cumulative_rewards[agent].zero_()
        
        # Clear immediate rewards
        self._clear_rewards()
        
        if action is None:
            timeout = 30.0
            start_time = time.time()
            
            with agent.coordination_flag:
                # Wait for flag to be AEC_SIDE (indicating action is ready)
                while self.flag_states[agent_name] != "AEC_SIDE":
                    elapsed = time.time() - start_time
                    remaining = timeout - elapsed
                    
                    if remaining <= 0:
                        logger.error(f"Timeout waiting for {agent_name} action")
                        action = torch.zeros((self.n_envs, agent.bundle.get_action_dim()), 
                                        device=self.device, dtype=torch.float32)
                        break
                    
                    notified = agent.coordination_flag.wait(timeout=remaining)
                    if not notified:
                        logger.warning(f"Wait timeout for {agent_name} action")
                
                # If we didn't timeout, get action from cache
                if action is None:  # Only if we didn't create zero action due to timeout
                    action = self.action_cache.get(agent_name)
                    if action is None:
                        logger.error(f"No cached action for {agent_name}")
                        action = torch.zeros((self.n_envs, agent.bundle.get_action_dim()), 
                                        device=self.device, dtype=torch.float32)
        else:
            # Legacy mode: Direct action provided (for compatibility)
            logger.debug(f"Using provided action for {agent} (legacy mode)")
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).to(self.device).float()
        
        # Validate and apply actions via Agent class
        if action is not None:
            expected_shape = (self.n_envs, current_agent_bundle.bundle.get_action_dim())
            if action.shape != expected_shape:
                logger.warning(f"Action shape mismatch for {agent_name}: {action.shape} != {expected_shape}")
                # Attempt to fix common shape issues
                if action.numel() == expected_shape[0] * expected_shape[1]:
                    action = action.view(expected_shape)
                else:
                    logger.error(f"Cannot fix action shape mismatch for {agent_name}")
                    raise ValueError(f"Expected action shape {expected_shape}, got {action.shape}")
            
            # Apply actions via Agent class (handles bundle logic)
            current_agent_bundle.apply_actions(action)
            logger.debug(f"Applied actions via Agent {agent_name}")
        
        # Get next agent and frames to advance
        next_agent_name, frames_to_advance = self.scheduler.step()
        
        logger.debug(f"Scheduler: next_agent={next_agent_name}, frames_to_advance={frames_to_advance}")
        
        # Step Genesis simulation
        if frames_to_advance > 0:
            logger.debug(f"Stepping Genesis {frames_to_advance} frames")
            for _ in range(frames_to_advance):
                self.scene.step()
                self.total_frames += 1
            
            # Update environment state after physics steps
            self._update_environment_state()
        
        # Compute rewards and check terminations
        # Note: Rewards computed on-demand via Agent.calculate_rewards() in V3
        self._accumulate_rewards()
        self._check_terminations()
        
        # Handle auto-reset for dead environments
        self._handle_auto_reset()
        
        # Automatically update agent selection in the end of each step
        self.agent_selection = next_agent_name
        
        # BUG FIX: Don't set next agent's flag prematurely to avoid race conditions
        # According to DESIGN.md, flags should only be set when agent is actually ready
        # The flag will be set to AEC_SIDE in last() when this agent is selected
        logger.debug(f"Next agent scheduled: {next_agent_name} (flag will be set when selected)")
        
        logger.debug(f"Step complete, new agent selection: {self.agent_selection}")
    
    def last(self, observe: bool = True) -> Tuple[Optional[torch.Tensor], torch.Tensor, 
                                                  torch.Tensor, torch.Tensor, List[dict]]:
        """Get vectorized last step info for current agent (AEC API - V3 with Flag Coordination).
        
        IMPORTANT CHANGE 1: Implements bidirectional flag coordination from DESIGN.md section 194:
        - Check flag of current selected agent
        - If AEC should send: serve obs/rew and set flag to SubVecEnv side
        - Else: sleep and wait for flag
        
        Returns:
            observation: Tensor[n_envs, obs_dim] or None if observe=False
            reward: Tensor[n_envs] - cumulative rewards
            terminated: Tensor[n_envs] - termination flags
            truncated: Tensor[n_envs] - truncation flags
            info: List[dict] - info for each environment
        """
        # In V3, agent_selection is now a bundle name, not robot name
        bundle_name = self.agent_selection
        current_agent_bundle = None
        for agent in self.agents:
            if agent.agent_name == bundle_name:
                current_agent_bundle = agent
                break
        
        if not current_agent_bundle:
            logger.error(f"No Agent bundle found for bundle name: {bundle_name}")
            return None, torch.zeros(0), torch.zeros(0), torch.zeros(0), []
        
        agent_name = bundle_name
        
        # BUG FIX: Atomic flag operations with proper locking
        with current_agent_bundle.coordination_flag:
            flag_state = self.flag_states.get(agent_name)
            logger.debug(f"📋 last() {agent_name}: Checking flag state '{flag_state}'")
            
            if flag_state == "AEC_SIDE":
                # AEC should send observations - serve data and flip flag
                logger.debug(f"✨ last() {agent_name}: Serving observations (flag: AEC_SIDE -> SUBVECENV_SIDE)")
                
                # Route obs/reward computation through Agent class
                observation = current_agent_bundle.calculate_observations() if observe else None
                reward = current_agent_bundle.calculate_rewards()
                
                # Get termination/truncation for this agent bundle
                terminated = self.terminations.get(agent_name)
                truncated = self.truncations.get(agent_name)
                info = self.infos.get(agent_name)
                
                # Atomically set flag to SubVecEnv side and notify (inside the lock!)
                logger.debug(f"🔄 last() {agent_name}: Setting flag AEC_SIDE -> SUBVECENV_SIDE and notifying")
                self.flag_states[agent_name] = "SUBVECENV_SIDE"
                current_agent_bundle.coordination_flag.notify_all()
                
                logger.debug(f"✅ last() {agent_name}: Completed - "
                            f"obs_shape={observation.shape if observation is not None else None}, "
                            f"terminated={terminated.sum().item()}/{self.n_envs}, "
                            f"truncated={truncated.sum().item()}/{self.n_envs}")
                
                return observation, reward, terminated, truncated, info
            else:
                # Flag is on SubVecEnv side - wait for it to flip back to AEC side
                logger.debug(f"⏸️ last() {agent_name}: Flag is '{flag_state}', need to wait for AEC_SIDE")
        
        # BUG FIX: Wait for flag change outside the lock to avoid deadlock
        logger.debug(f"🔄 last() {agent_name}: Waiting for flag change to AEC_SIDE...")
        success = current_agent_bundle.wait_for_flag_change(expected_state="AEC_SIDE", timeout=30.0)
        if not success:
            logger.warning(f"⏰ last() {agent_name}: TIMEOUT waiting for flag change")
            return None, torch.zeros(0), torch.zeros(0), torch.zeros(0), []
        
        # After flag change, recursively call last() to serve observations
        logger.debug(f"🔄 last() {agent_name}: Flag changed, recursively calling last()")
        return self.last(observe=observe)
    
# Method removed - no longer needed since agent_selection is now bundle names
    
# Method removed - scheduler now returns bundle names directly
    
    def observe(self, agent: str) -> torch.Tensor:
        """Get observations for agent bundle across all environments (AEC API - V3).
        
        Dead agents receive NaN observations.
        
        Args:
            agent: Agent bundle name (e.g., "agent_bundle_0_50Hz")
        
        Returns:
            Observations tensor with shape depending on bundle type
        """
        if agent not in self.possible_agents:
            logger.warning(f"observe() called with unknown agent: {agent}")
            return None
        
        # Find agent bundle and route through Agent class
        current_agent_bundle = None
        for agent_bundle in self.agents:
            if agent_bundle.agent_name == agent:
                current_agent_bundle = agent_bundle
                break
        
        if not current_agent_bundle:
            logger.error(f"No Agent bundle found for name: {agent}")
            return None
        
        # Get observations via Agent's compute method
        obs = current_agent_bundle.calculate_observations()
        
        # Mark dead agents with NaN
        agent_idx = self.agent_name_mapping[agent]
        dead_mask = ~self.agent_alive_mask[:, agent_idx]
        
        if current_agent_bundle.is_shared_network:
            # For shared network, expand dead mask to match observation shape
            n_robots = len(current_agent_bundle.robots)
            expanded_dead_mask = dead_mask.unsqueeze(0).repeat(n_robots, 1).flatten()
            obs[expanded_dead_mask] = float('nan')
        else:
            # For joint training and single robot bundles
            obs[dead_mask] = float('nan')
        
        num_dead = dead_mask.sum().item()
        if num_dead > 0:
            logger.debug(f"observe({agent}): {num_dead}/{self.n_envs} environments have dead agent")
        
        return obs
    
# V2 method removed - actions now applied via Agent.apply_actions() in V3
    
    def _update_environment_state(self):
        """Update state after physics simulation."""
        self.episode_lengths += 1
        # Note: Observations computed on-demand via Agent.calculate_observations() in V3
    
# V2 methods removed - using Agent.calculate_observations() and Agent.calculate_rewards() in V3
    
    def _check_terminations(self):
        """Check termination conditions for all agent bundles."""
        for i, agent_bundle in enumerate(self.agents):
            agent_name = agent_bundle.agent_name
            
            # Compute termination for each robot in the bundle
            robot_terminations = []
            for j, (robot, config) in enumerate(zip(agent_bundle.robots, agent_bundle.bundle.robot_configs)):
                pos = robot.get_pos()
                
                # Termination: fallen well below ground or too far
                fallen = pos[:, 2] < -0.5
                too_far = torch.norm(pos[:, :2], dim=-1) > 10.0
                robot_terminated = fallen | too_far
                robot_terminations.append(robot_terminated)
            
            # Bundle termination logic based on bundle type
            if agent_bundle.is_joint_training:
                # Joint training: OR logic - bundle terminates if ANY robot terminates
                bundle_terminated = torch.zeros(self.n_envs, device=self.device, dtype=torch.bool)
                for robot_term in robot_terminations:
                    bundle_terminated |= robot_term
            elif agent_bundle.is_shared_network:
                # Shared network: each robot tracked separately, then reshaped
                # Each robot appears as separate environments
                bundle_terminated = torch.cat(robot_terminations, dim=0)
            else:
                # Single robot bundle
                bundle_terminated = robot_terminations[0] if robot_terminations else torch.zeros(self.n_envs, device=self.device, dtype=torch.bool)
            
            # Only terminate if agent was alive
            alive_mask = self.agent_alive_mask[:, i]
            if agent_bundle.is_shared_network:
                # For shared network, alive mask needs to be expanded
                alive_mask = alive_mask.unsqueeze(0).repeat(len(agent_bundle.robots), 1).flatten()
            
            final_terminated = bundle_terminated & alive_mask
            self.terminations[agent_name] = final_terminated
            
            # Update alive mask
            if agent_bundle.is_shared_network:
                # Reshape back for alive mask update
                reshaped_alive = (alive_mask & ~final_terminated).view(len(agent_bundle.robots), self.n_envs)
                self.agent_alive_mask[:, i] = reshaped_alive.any(dim=0)  # Agent bundle alive if any robot alive
            else:
                self.agent_alive_mask[:, i] = alive_mask & ~final_terminated
            
            # Truncation (timeout)
            if agent_bundle.is_shared_network:
                # Expand episode lengths for shared network
                expanded_lengths = self.episode_lengths.unsqueeze(0).repeat(len(agent_bundle.robots), 1).flatten()
                self.truncations[agent_name] = expanded_lengths >= self.max_episode_length
            else:
                self.truncations[agent_name] = self.episode_lengths >= self.max_episode_length
            
            # Log terminations
            num_terminated = final_terminated.sum().item()
            if num_terminated > 0:
                bundle_type = "joint" if agent_bundle.is_joint_training else "shared" if agent_bundle.is_shared_network else "single"
                logger.info(f"Agent bundle {agent_name} ({bundle_type}) terminated in {num_terminated} environments")
    
    def _handle_auto_reset(self):
        """Auto-reset environments where all agents are dead."""
        # Check which environments have all agents dead
        any_alive = self.agent_alive_mask.any(dim=1)
        env_done_mask = ~any_alive
        
        # Find newly done environments
        newly_done = env_done_mask & ~self.env_done_mask
        newly_done_envs = newly_done.nonzero(as_tuple=False).squeeze(-1)
        
        if len(newly_done_envs) > 0:
            logger.info(f"Auto-resetting {len(newly_done_envs)} environments where all agents died")
            self._reset_envs(newly_done_envs)
        
        # Update done mask
        self.env_done_mask = env_done_mask
    
    def _clear_rewards(self):
        """Clear immediate rewards (AEC API)."""
        for agent in self.possible_agents:
            self.rewards[agent].zero_()
    
    def _accumulate_rewards(self):
        """Accumulate rewards (AEC API)."""
        for agent in self.possible_agents:
            self._cumulative_rewards[agent] += self.rewards[agent]
    
    def _was_dead_step(self, action):
        """Handle dead agent step (AEC API) - not used in vectorized version."""
        # In vectorized setting, dead agents are tracked but not removed, but their actions should be None
        self._clear_rewards()
    
    def render(self):
        """Render - handled by Genesis viewer."""
        pass
    
    def close(self):
        """Clean up resources."""
        logger.info("Closing VectorizedAECEnv...")
        # Genesis handles scene cleanup
        logger.info("VectorizedAECEnv closed")