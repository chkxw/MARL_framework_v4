#!/usr/bin/env python3
"""
Configuration Classes for Genesis MARL Framework V4
"""

from typing import Any, Dict, List, Tuple, Callable, Optional, Union
from dataclasses import asdict, dataclass, field
import torch
from genesis_logging import get_module_logger
from joint_detector import URDFJointDetector
from utils import transform_quat_by_quat, quat_to_xyz, inv_quat
import genesis as gs
from genesis.engine.entities.base_entity import Entity
from tensordict import TensorDict

logger = get_module_logger("configs")


class AECConfig:
    """Configuration for AEC training"""

    @dataclass
    class GenesisConfig:
        class_name: str = "PPO"
        clip_param: float = 0.2
        desired_kl: float = 0.01
        entropy_coef: float = 0.01
        gamma: float = 0.99
        lam: float = 0.95
        learning_rate: float = 0.001
        max_grad_norm: float = 1.0
        num_learning_epochs: int = 5
        num_mini_batches: int = 4
        schedule: str = "adaptive"
        use_clipped_value_loss: bool = True
        value_loss_coef: float = 1.0

    @dataclass
    class RunnerConfig:
        checkpoint: int = -1


class RSL_RLConfig:
    """Configuration for RSL_RL training"""

    @dataclass
    class AlgorithmConfig:
        class_name: str = "PPO"
        clip_param: float = 0.2
        desired_kl: float = 0.01
        entropy_coef: float = 0.01
        gamma: float = 0.99
        lam: float = 0.95
        learning_rate: float = 0.001
        max_grad_norm: float = 1.0
        num_learning_epochs: int = 5
        num_mini_batches: int = 4
        schedule: str = "adaptive"
        use_clipped_value_loss: bool = True
        value_loss_coef: float = 1.0

    @dataclass
    class PolicyConfig:
        activation: str = "elu"
        actor_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
        critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
        init_noise_std: float = 1.0
        class_name: str = "ActorCritic"

    @dataclass
    class RunnerConfig:
        checkpoint: int = -1
        experiment_name: str = ""
        load_run: int = -1
        log_interval: int = 1
        max_iterations: int = 0
        record_interval: int = -1
        resume: bool = False
        resume_path: Optional[str] = None
        run_name: str = ""

    @dataclass
    class TrainConfig:
        algorithm: "RSL_RLConfig.AlgorithmConfig" = field(default_factory=lambda: RSL_RLConfig.AlgorithmConfig())
        init_member_classes: Dict[str, Any] = field(default_factory=dict)
        policy: "RSL_RLConfig.PolicyConfig" = field(default_factory=lambda: RSL_RLConfig.PolicyConfig())
        runner: "RSL_RLConfig.RunnerConfig" = field(default_factory=lambda: RSL_RLConfig.RunnerConfig())
        runner_class_name: str = "OnPolicyRunner"
        num_steps_per_env: int = 24
        save_interval: int = 100
        empirical_normalization: Optional[Any] = None
        seed: int = 1

    def __new__(cls, exp_name: str, max_iterations: int):
        # This is unusual but valid
        runner = cls.RunnerConfig(experiment_name=exp_name, max_iterations=max_iterations)
        return cls.TrainConfig(runner=runner)


@dataclass
class RobotConfig:
    """Configuration for a robot in the MARL environment (V3 with functional computation)"""

    name: str
    urdf_path: str
    frequency: float  # Hz
    initial_position: List[float]  # [x, y, z]
    initial_orientation: List[float]  # [x, y, z, w] quaternion
    action_dim: Optional[int] = field(default=None)  # None means auto-detect
    observation_dim: Optional[int] = field(default=None)  # None means auto-detect through sample run of obs_function
    control_mode: str = "position"  # "position", "velocity", "force"; "force" is for both torque and force
    joint_names: List[str] = field(default_factory=list)  # Empty means auto-detect

    # Setup Function, if you need to attach something to one robot / make custom setup, do it here
    setup_function: Optional[Callable] = None

    # Functional observation and reward computation
    obs_function: Optional[Callable] = None  # Function(mother_env) -> obs_tensor
    reward_function: Optional[Callable] = None  # Function(mother_env) -> dict of reward_tensor
    truncation_function: Optional[Callable] = None  # Function(mother_env) -> truncation_tensor
    info_function: Callable = lambda AEC_env: {}  # Function(mother_env) -> info_dict

    localmotion_path: Optional[str] = field(default=None)  # localmotion path, none means action is joint values

    # Tuneable parameters
    action_scale: float = 1.0
    DP_kp: Union[float, List[float]] = 20.0  # Scalar apply to all joints, List apply to each joint
    DP_kd: Union[float, List[float]] = 0.5  # Scalar apply to all joints, List apply to each joint

    def __post_init__(self):
        # Deduce those field that don't need AEC instance
        if self.joint_names == []:
            joint_detector = URDFJointDetector()
            joint_detector.parse_urdf(self.urdf_path)
            movable_joints = joint_detector.get_movable_joints()
            self.joint_names = [joint["name"] for joint in movable_joints]

        if self.action_dim is None:
            if self.localmotion_path is None:
                self.action_dim = len(self.joint_names)
            else:
                raise ValueError("action_dim is not set and localmotion model is used. Cannot auto-detect action_dim.")

        if isinstance(self.DP_kp, float):
            self.DP_kp = [self.DP_kp] * len(self.joint_names)
        if isinstance(self.DP_kd, float):
            self.DP_kd = [self.DP_kd] * len(self.joint_names)

        # Setup Function
        def register_robot(AEC_env, robot):
            AEC_env.robots[self.name] = robot

        if self.setup_function is None:
            logger.debug(f"Using default setup for robot {self.name}")

            def setup_function(AEC_env):
                import genesis as gs

                robot = AEC_env.scene.add_entity(
                    gs.morphs.URDF(
                        file=self.urdf_path,
                        pos=self.initial_position,
                        quat=self.initial_orientation,
                        fixed=False,
                    )
                )
                return robot

            self.setup_function = lambda AEC_env: register_robot(AEC_env, setup_function(AEC_env))

        if self.obs_function is None:
            logger.debug(f"Using default obs for robot {self.name}")

            def obs_function(AEC_env):
                robot = AEC_env.robots[self.name]

                # Get robot state
                pos = robot.get_pos()  # [n_envs, 3]
                quat = robot.get_quat()  # [n_envs, 4]
                vel = robot.get_vel()  # [n_envs, 3]
                ang_vel = robot.get_ang()  # [n_envs, 3]

                obs_components = [pos, quat, vel, ang_vel]

                # Add joint states if available
                if hasattr(robot, "get_dofs_position"):
                    joint_pos = robot.get_dofs_position()
                    joint_vel = robot.get_dofs_velocity()
                    obs_components.extend([joint_pos, joint_vel])

                return torch.cat(obs_components, dim=-1)

            self.obs_function = obs_function

        if self.reward_function is None:
            logger.warning(f"Using dummy reward for robot {self.name}")

            def reward_function(AEC_env):
                reward_per_sec = 1
                reward = 1 / self.frequency * reward_per_sec

                return {self.name: torch.zeros((AEC_env.n_envs, 1), device=AEC_env.device).fill_(reward)}

            self.reward_function = reward_function

        if self.truncation_function is None:
            logger.warning(f"Using dummy truncation for robot {self.name}")

            def truncation_function(AEC_env):
                max_episode_length_s = 20  # in seconds
                termination_if_pitch_greater_than = 10
                termination_if_roll_greater_than = 10

                robot = AEC_env.robots[self.name]
                base_init_quat = torch.tensor(self.initial_orientation, device=AEC_env.device)
                base_quat = robot.get_quat()
                base_euler = quat_to_xyz(
                    transform_quat_by_quat(
                        torch.ones_like(base_quat) * inv_quat(base_init_quat),
                        base_quat,
                    ),
                    rpy=True,
                    degrees=True,
                )

                truncations = (AEC_env.episode_frame_count / AEC_env.simulation_frequency) > max_episode_length_s
                truncations |= torch.abs(base_euler[:, 1]) > termination_if_pitch_greater_than
                truncations |= torch.abs(base_euler[:, 0]) > termination_if_roll_greater_than
                return truncations

            self.truncation_function = truncation_function

    def detect_obs_dim(self, AEC_env):
        if self.observation_dim is None:
            obs = self.obs_function(AEC_env)
            self.observation_dim = obs.shape[-1]

    @property
    def n_dofs(self):
        return len(self.joint_names)


class TrainingConfigBundle:
    """Base class for configs for each RSL_RL thraining thread
    One to one mapping to subVecEnv and RSL_RL onPolicyRunner

    Bundles serveral important config files together
    """

    def __init__(
        self,
        training_name: str,
        frequency: float,
        setup_function: Optional[Callable],
        obs_function: Optional[Callable],
        reward_function: Optional[Callable],
        truncation_function: Optional[Callable],
        info_function: Callable = lambda AEC_env: {},
        robot_configs: List[RobotConfig] = [],
        rsl_rl_config: RSL_RLConfig.TrainConfig = RSL_RLConfig("test", 10000),
    ):
        """Initialize bundle.

        Args:
            bundle_name: Name of the bundle
            frequency: Control frequency for all robots in bundle (must be same)
        """
        self.training_name = training_name
        self.frequency = frequency
        self.robot_configs: List[RobotConfig] = []
        self.rsl_rl_config: RSL_RLConfig.TrainConfig = rsl_rl_config

        for config in robot_configs:
            self.add_robot(config)

        self.setup_function: Optional[Callable] = setup_function
        self.obs_function: Optional[Callable] = obs_function  # Function(mother_env) -> obs_tensor
        self.reward_function: Optional[Callable] = reward_function  # Function(mother_env) -> reward_tensor
        self.truncation_function: Optional[Callable] = truncation_function  # Function(mother_env) -> truncation_tensor
        self.info_function: Callable = info_function

        # Call post-initialization as if this is a dataclass
        self.__post_init__()

    def __post_init__(self):
        # If we have obs/reward function on bundle level,fill the sub_robot_config's obs_function and reward_function to indicate we should not use robot level functions
        if self.obs_function is not None:
            for config in self.robot_configs:
                config.obs_function = lambda AEC_env: None
        if self.reward_function is not None:
            for config in self.robot_configs:
                config.reward_function = lambda AEC_env: None
        if self.truncation_function is not None:
            for config in self.robot_configs:
                config.truncation_function = lambda AEC_env: None

        if self.setup_function is None:

            def setup_function(AEC_env):
                for config in self.robot_configs:
                    config.setup_function(AEC_env)

            self.setup_function = setup_function

        if self.obs_function is None:
            self.obs_function = lambda AEC_env: {
                config.name: config.obs_function(AEC_env) for config in self.robot_configs
            }

        if self.reward_function is None:
            self.reward_function = lambda AEC_env: {
                config.name: config.reward_function(AEC_env) for config in self.robot_configs
            }

        if self.truncation_function is None:
            self.truncation_function = lambda AEC_env: {
                config.name: config.truncation_function(AEC_env) for config in self.robot_configs
            }

    def add_robot(self, config: RobotConfig) -> None:
        """Add a robot to this bundle."""
        if config.frequency != self.frequency:
            raise ValueError(f"Robot {config.name} frequency {config.frequency} != bundle frequency {self.frequency}")
        if config.name in [robot_config.name for robot_config in self.robot_configs]:
            raise ValueError(f"Robot {config.name} already in bundle {self.training_name} or have a name conflict")
        self.robot_configs.append(config)

    def detect_obs_dim(self, AEC_env):
        for config in self.robot_configs:
            config.detect_obs_dim(AEC_env)

    @property
    def n_robots(self) -> int:
        """Get number of robots in this bundle."""
        return len(self.robot_configs)

    @property
    def action_dim(self) -> int:
        """Get total action dimension for this bundle."""
        return sum(config.action_dim for config in self.robot_configs)

    @property
    def observation_dim(self) -> Optional[int]:
        """Get total observation dimension for this bundle."""
        # Aviable after auto detect
        if any(config.observation_dim is None for config in self.robot_configs):
            return None
        else:
            return sum(config.observation_dim for config in self.robot_configs)


class SingleRobotTrainingBundle(TrainingConfigBundle):
    def __init__(
        self,
        robot_config: RobotConfig,
        rsl_rl_config: RSL_RLConfig.TrainConfig = RSL_RLConfig("test", 10000),
    ):
        super().__init__(
            robot_config.name,
            robot_config.frequency,
            setup_function=None,
            obs_function=None,
            reward_function=None,
            truncation_function=None,
            # info_function=lambda AEC_env: {},
            robot_configs=[robot_config],
            rsl_rl_config=rsl_rl_config,
        )


class JointTrainingBundle(TrainingConfigBundle):
    """Bundle for joint training - robots share concatenated observations/actions.

    In joint training:
    - Observations from all robots are concatenated per environment
    - Actions comes from model as concatenated and then split to all robots
    - Single policy controls all robots as one entity
    - n_envs stays the same, but obs/action dims increase
    """

    def __init__(
        self,
        training_name: str,
        setup_function: Optional[Callable] = None,
        obs_function: Optional[Callable] = None,  # Ok to have no specified obs function, will auto deduce
        reward_function: Optional[Callable] = None,  # Ok to have no specified obs function, will auto deduce
        truncation_function: Optional[Callable] = None,  # Ok to have no specified obs function, will auto deduce
        info_function: Callable = lambda AEC_env: {},
        robot_configs: List[RobotConfig] = [],
        rsl_rl_config: RSL_RLConfig.TrainConfig = RSL_RLConfig("test", 10000),
    ):
        tailored_obs_function = obs_function is not None
        tailored_reward_function = reward_function is not None
        tailored_truncation_function = truncation_function is not None

        frequency = robot_configs[0].frequency
        super().__init__(
            training_name,
            frequency,
            setup_function,
            obs_function,
            reward_function,
            truncation_function,
            info_function,
            robot_configs,
            rsl_rl_config,
        )

        # If use the default obs function which returns per robot obs/reward/trunc, Merge on the last dim to make it looks like a it's the obs of the first robot

        if not tailored_obs_function:
            self.obs_function = lambda AEC_env: {
                self.robot_configs[0].name: torch.cat(self.setup_function(AEC_env).values(), dim=-1)
            }

        if not tailored_reward_function:

            def reward_function(AEC_env):
                result = {}
                for config in self.robot_configs:
                    reward_dict = config.reward_function(AEC_env)
                    for robot_name, rewards in reward_dict.items():
                        if robot_name not in result:
                            result[robot_name] = rewards.clone()
                        else:
                            result[robot_name].add_(rewards)
                return result

            self.reward_function = reward_function

        if not tailored_truncation_function:
            self.truncation_function = lambda AEC_env: {
                self.robot_configs[0].name: torch.any(self.setup_function(AEC_env).values(), dim=-1)
            }


class SharedNetworkBundle(TrainingConfigBundle):
    """Bundle for shared network training - same network for multiple robot instances.

    In shared network training:
    - Multiple robots of the same type use the same policy network
    - Each robot appears as a separate environment to the policy
    - n_envs gets multiplied by number of robots
    - obs/action dims stay the same as individual robot
    """

    def __init__(
        self,
        training_name: str,
        base_config: RobotConfig,
        initial_positions: List[List[float]],
        initial_orientations: Optional[List[List[float]]] = None,
        setup_function: Optional[Callable] = None,
        obs_function: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
        truncation_function: Optional[Callable] = None,
        info_function: Callable = lambda AEC_env: {},
        rsl_rl_config: RSL_RLConfig.TrainConfig = RSL_RLConfig("test", 10000),
    ):
        """Initialize shared network bundle. Number of robots is inferred from initial position list.

        Args:
            bundle_name: Name of the bundle
            base_config: Base robot configuration to replicate
            positions: List of initial positions for each robot instance
            orientations: List of initial orientations (uses base_config orientation if None)
        """
        self.training_name = training_name
        self.frequency = base_config.frequency
        self.robot_configs: List[RobotConfig] = []
        self.rsl_rl_config: RSL_RLConfig.TrainConfig = rsl_rl_config

        if len(initial_positions) != len(initial_orientations):
            raise ValueError("Number of positions must match number of orientations")

        if initial_orientations is None:
            initial_orientations = [base_config.initial_orientation] * len(initial_positions)

        # Create individual robot configs with different names and positions
        for i, (pos, orn) in enumerate(zip(initial_positions, initial_orientations)):
            robot_config = RobotConfig(
                name=f"{training_name}_{base_config.name}_{i}",
                urdf_path=base_config.urdf_path,
                frequency=base_config.frequency,
                initial_position=pos,
                initial_orientation=orn,
                action_dim=base_config.action_dim,
                observation_dim=base_config.observation_dim,
                control_mode=base_config.control_mode,
                joint_names=base_config.joint_names.copy(),
                action_scale=base_config.action_scale,
                obs_function=base_config.obs_function,
                reward_function=base_config.reward_function,
            )
            self.add_robot(robot_config)

        self.setup_function: Optional[Callable] = setup_function
        self.obs_function: Optional[Callable] = obs_function  # Function(mother_env) -> obs_tensor
        self.reward_function: Optional[Callable] = reward_function  # Function(mother_env) -> reward_tensor
        self.truncation_function: Optional[Callable] = truncation_function  # Function(mother_env) -> truncation_tensor
        self.info_function: Callable = info_function

        # The obs/reward/truncation functions are the same as base class, the reshape happens in subVecEnv
        self.__post_init__()


if __name__ == "__main__":
    config = RSL_RLConfig("test", 10000)
    print(asdict(config))
