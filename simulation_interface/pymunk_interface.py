#!/usr/bin/env python3
"""
Pymunk Simulator Interface Implementation

This module implements the SceneInterface and RobotInterface for the Pymunk
2D physics simulator, providing a 3D API wrapper around 2D physics with
multiprocessing support for vectorized environments.
"""

import multiprocessing as mp
from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import time

import numpy as np
import pymunk
import torch

from simulation_interface.base_interfaces import EntityInterface, PrimitiveInterface, RobotInterface, SceneInterface
from utils import ReferenceFrames, ReferenceFramesInput

if TYPE_CHECKING:
    from configs.SimulatorConfig import SimulatorConfig
    from configs.CameraConfig import CameraConfig


# Multiprocessing worker function
def pymunk_worker(env_idx: int, command_queue: mp.Queue, result_queue: mp.Queue, config_dict: dict):
    """Worker process for a single Pymunk environment."""
    print(f"Worker {env_idx} starting...")
    
    # Create space
    space = pymunk.Space()
    space.gravity = config_dict['gravity'][0], config_dict['gravity'][1]

    dt = config_dict['dt']
    substeps = config_dict['substeps']

    # Global storage lists
    bodies = []  # All pymunk.Body instances
    shapes = []  # All pymunk.Shape instances
    joints = []  # All pymunk.Constraint instances (for future use)
    motors = []  # All motor constraints (for future use)
    
    # Entity name to indices mapping
    entity_indices = {}  # name -> {'body_indices': [...], 'shape_indices': [...]}

    while True:
        try:
            command = command_queue.get(timeout=0.1)

            if command['type'] == 'stop':
                break

            elif command['type'] == 'step':
                n_steps = command['n_steps']
                for _ in range(n_steps):
                    for _ in range(substeps):
                        space.step(dt / substeps)
                result_queue.put({'env_idx': env_idx, 'type': 'step_done'})

            elif command['type'] == 'add_entity':
                # Add bodies and shapes to space
                entity_name = command['name']
                entity_body_indices = []
                entity_shape_indices = []
                
                # Create bodies and allocate indices
                for body_data in command['bodies']:
                    body = pymunk.Body(body_data['mass'], body_data['moment'])
                    body.position = body_data['position']
                    body.angle = body_data['angle']
                    
                    # Add to global list and track index
                    body_idx = len(bodies)
                    bodies.append(body)
                    entity_body_indices.append(body_idx)
                    
                    # Add to space if dynamic
                    if body.body_type == pymunk.Body.DYNAMIC:
                        space.add(body)

                # Create shapes and allocate indices
                for shape_data in command['shapes']:
                    local_body_idx = shape_data['body_idx']
                    if local_body_idx >= 0:
                        # Use entity's body
                        global_body_idx = entity_body_indices[local_body_idx]
                        body = bodies[global_body_idx]
                    else:
                        # Use static body
                        body = space.static_body

                    if shape_data['type'] == 'circle':
                        shape = pymunk.Circle(body, shape_data['radius'])
                    elif shape_data['type'] == 'box':
                        shape = pymunk.Poly.create_box(body, shape_data['size'])
                    elif shape_data['type'] == 'segment':
                        shape = pymunk.Segment(body, shape_data['a'], shape_data['b'], shape_data['radius'])

                    shape.friction = shape_data.get('friction', 0.5)
                    
                    # Add to global list and track index
                    shape_idx = len(shapes)
                    shapes.append(shape)
                    entity_shape_indices.append(shape_idx)
                    
                    # Add to space
                    space.add(shape)

                # Store entity indices mapping
                entity_indices[entity_name] = {
                    'body_indices': entity_body_indices,
                    'shape_indices': entity_shape_indices
                }
                
                result_queue.put({
                    'env_idx': env_idx, 
                    'type': 'entity_added', 
                    'name': entity_name,
                    'body_indices': entity_body_indices,
                    'shape_indices': entity_shape_indices
                })

            elif command['type'] == 'get_state':
                body_idx = command['body_idx']
                if 0 <= body_idx < len(bodies):
                    body = bodies[body_idx]
                    state = {
                        'position': (body.position.x, body.position.y),
                        'angle': body.angle,
                        'velocity': (body.velocity.x, body.velocity.y),
                        'angular_velocity': body.angular_velocity,
                    }
                else:
                    state = None

                result_queue.put({'env_idx': env_idx, 'type': 'state', 'body_idx': body_idx, 'state': state})

            elif command['type'] == 'set_state':
                body_idx = command['body_idx']
                if 0 <= body_idx < len(bodies) and command['state']:
                    body = bodies[body_idx]
                    state = command['state']
                    if 'position' in state:
                        body.position = state['position']
                    if 'angle' in state:
                        body.angle = state['angle']
                    if 'velocity' in state:
                        body.velocity = state['velocity']
                    if 'angular_velocity' in state:
                        body.angular_velocity = state['angular_velocity']

                result_queue.put({'env_idx': env_idx, 'type': 'state_set'})

            elif command['type'] == 'apply_force':
                body_idx = command['body_idx']
                if 0 <= body_idx < len(bodies):
                    body = bodies[body_idx]
                    force = command['force']
                    torque = command.get('torque', 0)
                    body.apply_force_at_world_point(force, body.position)
                    body.torque += torque

                result_queue.put({'env_idx': env_idx, 'type': 'force_applied'})

        except Empty:
            continue
        except Exception as e:
            result_queue.put({'env_idx': env_idx, 'type': 'error', 'error': str(e)})


@dataclass
class PymunkEntityData:
    """Container for all Pymunk entity data across environments."""

    name: str
    entity_type: str  # "robot" or "primitive"
    n_envs: int

    # Global indices for each environment
    body_indices: List[List[int]] = field(default_factory=list)  # [n_envs][n_bodies] - indices into worker's bodies list
    shape_indices: List[List[int]] = field(default_factory=list)  # [n_envs][n_shapes] - indices into worker's shapes list

    # Initial configuration for workers (used only during add_entity)
    body_configs: List[List[Dict]] = field(default_factory=list)  # [n_envs][n_bodies]
    shape_configs: List[List[Dict]] = field(default_factory=list)  # [n_envs][n_shapes]

    # Mapping from body index to list of shape indices
    body_to_shapes: Dict[int, List[int]] = field(default_factory=dict)

    # Joint and motor storage (empty for 2D)
    joints: List[Any] = field(default_factory=list)
    motors: List[Any] = field(default_factory=list)

    # Body relationship table (empty for 2D)
    body_relationships: Dict[Tuple[int, int], Tuple[Optional[int], Optional[int]]] = field(default_factory=dict)

    # PD control parameters (shape: [n_envs, 3 + n_motors])
    # First 3: base control (x, y, yaw), rest: motor control
    pd_kp: Optional[torch.Tensor] = None
    pd_kd: Optional[torch.Tensor] = None
    pd_targets: Optional[torch.Tensor] = None  # Target positions/velocities
    pd_last_error: Optional[torch.Tensor] = None  # For derivative term

    # Initial states for reset
    initial_positions: Optional[torch.Tensor] = None  # [n_envs, 3]
    initial_orientations: Optional[torch.Tensor] = None  # [n_envs, 4] quaternion
    initial_velocities: Optional[torch.Tensor] = None  # [n_envs, 3]
    initial_angular_velocities: Optional[torch.Tensor] = None  # [n_envs, 3]

    # Cached states from workers
    cached_states: Optional[Dict[int, Dict]] = field(default_factory=dict)  # env_idx -> state

    # Reference to scene (not spaces directly)
    scene: Optional['PymunkSceneInterface'] = None

    @property
    def n_bodies(self) -> int:
        """Number of bodies per environment."""
        return len(self.body_configs[0]) if self.body_configs and self.body_configs[0] else 0

    @property
    def n_shapes(self) -> int:
        """Number of shapes per environment."""
        return len(self.shape_configs[0]) if self.shape_configs and self.shape_configs[0] else 0

    @property
    def n_motors(self) -> int:
        """Number of motors."""
        return len(self.motors)


class PymunkEntityInterface(EntityInterface):
    """Pymunk implementation of EntityInterface with multiprocessing support."""

    def __init__(self, name: str, entity_data: PymunkEntityData):
        super().__init__(name)
        self._entity_data = entity_data

    @property
    def scene(self) -> 'PymunkSceneInterface':
        return self._entity_data.scene

    @property
    def device(self) -> str:
        return self.scene.device

    @property
    def _n_envs(self) -> int:
        return self._entity_data.n_envs

    def _update_cached_states(self, env_indices: Optional[torch.Tensor] = None):
        """Request state updates from workers."""
        if env_indices is None:
            env_indices = range(self._n_envs)
        else:
            env_indices = env_indices.cpu().numpy()

        # Send get_state commands for root body (index 0)
        for i, env_idx in enumerate(env_indices):
            env_idx = int(env_idx)
            # Get root body index (first body)
            if self._entity_data.body_indices and env_idx < len(self._entity_data.body_indices):
                if self._entity_data.body_indices[env_idx]:
                    body_idx = self._entity_data.body_indices[env_idx][0]
                    self.scene._send_command(env_idx, {'type': 'get_state', 'body_idx': body_idx})

        # Collect results
        results = self.scene._collect_results(len(env_indices), timeout=1.0)

        # Update cache
        for result in results:
            if result['type'] == 'state':
                self._entity_data.cached_states[result['env_idx']] = result['state']

    # Raw state getters
    def _get_pos_raw(self, env_indices: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Get position from cached states."""
        self._update_cached_states(env_indices)

        if env_indices is None:
            env_indices = torch.arange(self._n_envs, device=self.device)

        positions = torch.zeros((len(env_indices), 3), device=self.device, dtype=torch.float32)

        for i, env_idx in enumerate(env_indices):
            env_idx = int(env_idx)
            state = self._entity_data.cached_states.get(env_idx)
            if state and state.get('position'):
                positions[i, 0] = state['position'][0]
                positions[i, 1] = state['position'][1]
                # z is always 0 in 2D

        return positions

    def _get_orientation_raw(self, env_indices: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Get orientation as quaternion from cached states."""
        self._update_cached_states(env_indices)

        if env_indices is None:
            env_indices = torch.arange(self._n_envs, device=self.device)

        orientations = torch.zeros((len(env_indices), 4), device=self.device, dtype=torch.float32)

        for i, env_idx in enumerate(env_indices):
            env_idx = int(env_idx)
            state = self._entity_data.cached_states.get(env_idx)
            if state and 'angle' in state:
                angle = state['angle']
                # Convert 2D rotation to quaternion (rotation around Z axis)
                orientations[i, 0] = np.cos(angle / 2)  # w
                orientations[i, 1] = 0  # x
                orientations[i, 2] = 0  # y
                orientations[i, 3] = np.sin(angle / 2)  # z

        return orientations

    def _get_lin_vel_raw(self, env_indices: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Get linear velocity from cached states."""
        self._update_cached_states(env_indices)

        if env_indices is None:
            env_indices = torch.arange(self._n_envs, device=self.device)

        velocities = torch.zeros((len(env_indices), 3), device=self.device, dtype=torch.float32)

        for i, env_idx in enumerate(env_indices):
            env_idx = int(env_idx)
            state = self._entity_data.cached_states.get(env_idx)
            if state and state.get('velocity'):
                velocities[i, 0] = state['velocity'][0]
                velocities[i, 1] = state['velocity'][1]
                # vz is always 0 in 2D

        return velocities

    def _get_ang_vel_raw(self, env_indices: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Get angular velocity from cached states."""
        self._update_cached_states(env_indices)

        if env_indices is None:
            env_indices = torch.arange(self._n_envs, device=self.device)

        angular_velocities = torch.zeros((len(env_indices), 3), device=self.device, dtype=torch.float32)

        for i, env_idx in enumerate(env_indices):
            env_idx = int(env_idx)
            state = self._entity_data.cached_states.get(env_idx)
            if state and 'angular_velocity' in state:
                angular_velocities[i, 2] = state['angular_velocity']  # Only Z rotation in 2D

        return angular_velocities

    # Raw state setters
    def _set_pos_raw(
        self, pos: torch.Tensor, env_indices: Optional[torch.Tensor] = None, zero_velocity: bool = True, **kwargs
    ) -> None:
        """Set position via worker commands."""
        if env_indices is None:
            env_indices = torch.arange(self._n_envs, device=self.device)

        pos_cpu = pos.cpu().numpy()

        for i, env_idx in enumerate(env_indices):
            env_idx = int(env_idx)
            # Get root body index
            if self._entity_data.body_indices and env_idx < len(self._entity_data.body_indices):
                if self._entity_data.body_indices[env_idx]:
                    body_idx = self._entity_data.body_indices[env_idx][0]
                    state = {'position': (pos_cpu[i, 0], pos_cpu[i, 1])}
                    if zero_velocity:
                        state['velocity'] = (0, 0)
                        state['angular_velocity'] = 0

                    self.scene._send_command(env_idx, {'type': 'set_state', 'body_idx': body_idx, 'state': state})

        # Wait for confirmation
        self.scene._collect_results(len(env_indices), timeout=1.0)

    def _set_orientation_raw(
        self, quat: torch.Tensor, env_indices: Optional[torch.Tensor] = None, zero_velocity: bool = True, **kwargs
    ) -> None:
        """Set orientation from quaternion."""
        if env_indices is None:
            env_indices = torch.arange(self._n_envs, device=self.device)

        quat_cpu = quat.cpu().numpy()

        for i, env_idx in enumerate(env_indices):
            env_idx = int(env_idx)
            # Get root body index
            if self._entity_data.body_indices and env_idx < len(self._entity_data.body_indices):
                if self._entity_data.body_indices[env_idx]:
                    body_idx = self._entity_data.body_indices[env_idx][0]
                    # Convert quaternion to 2D angle
                    w, x, y, z = quat_cpu[i]
                    angle = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

                    state = {'angle': angle}
                    if zero_velocity:
                        state['angular_velocity'] = 0

                    self.scene._send_command(env_idx, {'type': 'set_state', 'body_idx': body_idx, 'state': state})

        self.scene._collect_results(len(env_indices), timeout=1.0)

    def _set_lin_vel_raw(self, lin_vel: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        """Set linear velocity."""
        if env_indices is None:
            env_indices = torch.arange(self._n_envs, device=self.device)

        vel_cpu = lin_vel.cpu().numpy()

        for i, env_idx in enumerate(env_indices):
            env_idx = int(env_idx)
            # Get root body index
            if self._entity_data.body_indices and env_idx < len(self._entity_data.body_indices):
                if self._entity_data.body_indices[env_idx]:
                    body_idx = self._entity_data.body_indices[env_idx][0]
                    self.scene._send_command(
                        env_idx,
                        {
                            'type': 'set_state',
                            'body_idx': body_idx,
                            'state': {'velocity': (vel_cpu[i, 0], vel_cpu[i, 1])},
                        },
                    )

        self.scene._collect_results(len(env_indices), timeout=1.0)

    def _set_ang_vel_raw(self, ang_vel: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        """Set angular velocity."""
        if env_indices is None:
            env_indices = torch.arange(self._n_envs, device=self.device)

        ang_vel_cpu = ang_vel.cpu().numpy()

        for i, env_idx in enumerate(env_indices):
            env_idx = int(env_idx)
            # Get root body index
            if self._entity_data.body_indices and env_idx < len(self._entity_data.body_indices):
                if self._entity_data.body_indices[env_idx]:
                    body_idx = self._entity_data.body_indices[env_idx][0]
                    self.scene._send_command(
                        env_idx,
                        {'type': 'set_state', 'body_idx': body_idx, 'state': {'angular_velocity': ang_vel_cpu[i, 2]}},
                    )

        self.scene._collect_results(len(env_indices), timeout=1.0)

    # Control methods with PD control
    def _control_pos_raw(self, pos: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        """Control position using PD control."""
        if env_indices is None:
            env_indices = torch.arange(self._n_envs, device=self.device)

        # Update PD targets
        if self._entity_data.pd_targets is not None:
            for i, env_idx in enumerate(env_indices):
                env_idx = int(env_idx)
                self._entity_data.pd_targets[env_idx, 0:2] = pos[i, 0:2]

    def _control_orientation_raw(
        self, quat: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs
    ) -> None:
        """Control orientation using PD control."""
        if env_indices is None:
            env_indices = torch.arange(self._n_envs, device=self.device)

        # Convert quaternion to 2D angle and update PD target
        if self._entity_data.pd_targets is not None:
            quat_cpu = quat.cpu().numpy()
            for i, env_idx in enumerate(env_indices):
                env_idx = int(env_idx)
                w, x, y, z = quat_cpu[i]
                angle = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                self._entity_data.pd_targets[env_idx, 2] = angle

    def _control_lin_vel_raw(self, lin_vel: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        """Direct velocity control."""
        self._set_lin_vel_raw(lin_vel, env_indices, **kwargs)

    def _control_ang_vel_raw(self, ang_vel: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        """Direct angular velocity control."""
        self._set_ang_vel_raw(ang_vel, env_indices, **kwargs)


class PymunkPrimitiveInterface(PymunkEntityInterface, PrimitiveInterface):
    """Pymunk implementation of PrimitiveInterface."""

    def __init__(self, name: str, entity_data: PymunkEntityData, primitive_type: str):
        PrimitiveInterface.__init__(self, name, primitive_type)
        PymunkEntityInterface.__init__(self, name, entity_data)
        self._primitive_type = primitive_type


class PymunkRobotInterface(PymunkEntityInterface, RobotInterface):
    """Pymunk implementation of RobotInterface."""

    def __init__(self, name: str, entity_data: PymunkEntityData):
        RobotInterface.__init__(self, name)
        PymunkEntityInterface.__init__(self, name, entity_data)

    # Joint methods - all raise NotImplementedError
    def _get_joint_pos_raw(
        self, joint_indices: Optional[torch.Tensor] = None, env_indices: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError("Joint operations not supported in 2D Pymunk simulation")

    def _get_joint_vel_raw(
        self, joint_indices: Optional[torch.Tensor] = None, env_indices: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError("Joint operations not supported in 2D Pymunk simulation")

    def _set_joint_pos_raw(
        self,
        pos: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        zero_velocity: bool = False,
        **kwargs,
    ) -> None:
        raise NotImplementedError("Joint operations not supported in 2D Pymunk simulation")

    def _set_joint_vel_raw(
        self,
        vel: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError("Joint operations not supported in 2D Pymunk simulation")

    def _control_joint_pos_raw(
        self,
        target_pos: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError("Joint operations not supported in 2D Pymunk simulation")

    def _control_joint_vel_raw(
        self,
        target_vel: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError("Joint operations not supported in 2D Pymunk simulation")

    def _control_joint_force_raw(
        self,
        forces: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError("Joint operations not supported in 2D Pymunk simulation")

    def set_kp_gains(
        self,
        kp: Union[float, torch.Tensor],
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        # Set base PD gains (only first 3 for x, y, angle)
        if isinstance(kp, float):
            kp_tensor = torch.full((3,), kp, device=self.device, dtype=torch.float32)
        else:
            kp_tensor = kp[:3] if len(kp) >= 3 else kp

        if self._entity_data.pd_kp is None:
            self._entity_data.pd_kp = torch.zeros((self._n_envs, 3), device=self.device, dtype=torch.float32)

        if env_indices is None:
            self._entity_data.pd_kp[:, :] = kp_tensor
        else:
            for env_idx in env_indices:
                env_idx = int(env_idx)
                self._entity_data.pd_kp[env_idx, :] = kp_tensor

    def set_kd_gains(
        self,
        kd: Union[float, torch.Tensor],
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        # Set base PD gains (only first 3 for x, y, angle)
        if isinstance(kd, float):
            kd_tensor = torch.full((3,), kd, device=self.device, dtype=torch.float32)
        else:
            kd_tensor = kd[:3] if len(kd) >= 3 else kd

        if self._entity_data.pd_kd is None:
            self._entity_data.pd_kd = torch.zeros((self._n_envs, 3), device=self.device, dtype=torch.float32)

        if env_indices is None:
            self._entity_data.pd_kd[:, :] = kd_tensor
        else:
            for env_idx in env_indices:
                env_idx = int(env_idx)
                self._entity_data.pd_kd[env_idx, :] = kd_tensor

    def get_joint(self, joint_name: str) -> Any:
        raise NotImplementedError("Joint operations not supported in 2D Pymunk simulation")


class PymunkSceneInterface(SceneInterface):
    """Pymunk implementation of SceneInterface with multiprocessing for vectorization."""

    def __init__(self, config: 'SimulatorConfig'):
        super().__init__()
        self._config = config
        self._entities: Dict[str, PymunkEntityData] = {}
        self._entity_interfaces: Dict[str, EntityInterface] = {}

        # Multiprocessing setup
        self._processes = []
        self._command_queues = []
        self._result_queues = []

        # Initialize multiprocessing - use fork on Linux for better compatibility
        import sys
        if sys.platform == 'linux':
            mp_context = mp.get_context('fork')
        else:
            mp_context = mp.get_context('spawn')

        # Create config dict for workers
        self._worker_config = {'gravity': self._config.gravity, 'dt': self.dt, 'substeps': self._config.substeps}

        # Start worker processes
        for env_idx in range(self.n_envs):
            command_queue = mp_context.Queue()
            result_queue = mp_context.Queue()

            process = mp_context.Process(
                target=pymunk_worker, args=(env_idx, command_queue, result_queue, self._worker_config)
            )
            process.start()

            self._command_queues.append(command_queue)
            self._result_queues.append(result_queue)
            self._processes.append(process)

        self._sim_step = 0

    def __del__(self):
        """Cleanup worker processes."""
        self._shutdown_workers()

    def _shutdown_workers(self):
        """Shutdown all worker processes."""
        for queue in self._command_queues:
            queue.put({'type': 'stop'})

        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()

    @property
    def n_envs(self) -> int:
        return self._config.n_envs

    @property
    def device(self) -> str:
        return self._config.device

    @property
    def dt(self) -> float:
        return self._config.dt

    @property
    def sim_step(self) -> int:
        return self._sim_step

    @property
    def entities(self) -> Dict[str, EntityInterface]:
        return self._entity_interfaces

    @property
    def primitives(self) -> Dict[str, PrimitiveInterface]:
        return {name: iface for name, iface in self._entity_interfaces.items() if isinstance(iface, PrimitiveInterface)}

    @property
    def robots(self) -> Dict[str, RobotInterface]:
        return {name: iface for name, iface in self._entity_interfaces.items() if isinstance(iface, RobotInterface)}

    def _send_command(self, env_idx: int, command: dict):
        """Send command to specific worker."""
        self._command_queues[env_idx].put(command)

    def _broadcast_command(self, command: dict):
        """Send command to all workers."""
        for queue in self._command_queues:
            queue.put(command)

    def _collect_results(self, expected_count: int, timeout: float = 5.0) -> List[dict]:
        """Collect results from workers."""
        results = []
        start_time = time.time()

        while len(results) < expected_count:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for {expected_count} results, got {len(results)}")

            for queue in self._result_queues:
                try:
                    result = queue.get(timeout=0.01)
                    results.append(result)
                except Empty:
                    continue

        return results

    def build_scene(self) -> None:
        """Build the scene - send all entities to workers."""
        # Send entity configurations to all workers
        for name, entity_data in self._entities.items():
            # Initialize index storage for this entity
            entity_data.body_indices = []
            entity_data.shape_indices = []
            
            for env_idx in range(self.n_envs):
                command = {
                    'type': 'add_entity',
                    'name': name,
                    'bodies': entity_data.body_configs[env_idx],
                    'shapes': entity_data.shape_configs[env_idx],
                }
                self._send_command(env_idx, command)

        # Wait for all entities to be added and collect indices
        expected = len(self._entities) * self.n_envs
        results = self._collect_results(expected, timeout=10.0)
        
        # Store returned indices in entity data
        for result in results:
            if result['type'] == 'entity_added':
                entity_name = result['name']
                env_idx = result['env_idx']
                if entity_name in self._entities:
                    entity_data = self._entities[entity_name]
                    # Ensure we have enough environments
                    while len(entity_data.body_indices) <= env_idx:
                        entity_data.body_indices.append([])
                        entity_data.shape_indices.append([])
                    
                    entity_data.body_indices[env_idx] = result['body_indices']
                    entity_data.shape_indices[env_idx] = result['shape_indices']

        self._sim_step = 0

    def step(self, n_steps: int = 1) -> None:
        """Step simulation using multiprocessing."""
        for _ in range(n_steps):
            # Apply PD control forces
            self._apply_pd_control()

            # Send step command to all workers
            self._broadcast_command({'type': 'step', 'n_steps': 1})

            # Wait for all workers to complete
            self._collect_results(self.n_envs, timeout=5.0)

            self._sim_step += 1

    def _apply_pd_control(self):
        """Apply PD control forces to all entities."""
        for entity_data in self._entities.values():
            if entity_data.pd_targets is None or entity_data.pd_kp is None or entity_data.pd_kd is None:
                continue

            # Update states first
            entity_interface = self._entity_interfaces[entity_data.name]
            entity_interface._update_cached_states()

            # Calculate PD control
            positions = torch.zeros((self.n_envs, 3), device=self.device, dtype=torch.float32)
            angles = torch.zeros((self.n_envs,), device=self.device, dtype=torch.float32)
            velocities = torch.zeros((self.n_envs, 2), device=self.device, dtype=torch.float32)
            angular_velocities = torch.zeros((self.n_envs,), device=self.device, dtype=torch.float32)

            for env_idx in range(self.n_envs):
                state = entity_data.cached_states.get(env_idx)
                if state:
                    if state.get('position'):
                        positions[env_idx, 0] = state['position'][0]
                        positions[env_idx, 1] = state['position'][1]
                    if 'angle' in state:
                        angles[env_idx] = state['angle']
                    if state.get('velocity'):
                        velocities[env_idx, 0] = state['velocity'][0]
                        velocities[env_idx, 1] = state['velocity'][1]
                    if 'angular_velocity' in state:
                        angular_velocities[env_idx] = state['angular_velocity']

            # Calculate errors
            pos_error = entity_data.pd_targets[:, 0:2] - positions[:, 0:2]
            angle_error = entity_data.pd_targets[:, 2] - angles

            # Normalize angle error to [-pi, pi]
            angle_error = torch.atan2(torch.sin(angle_error), torch.cos(angle_error))

            # Calculate derivatives
            if entity_data.pd_last_error is None:
                entity_data.pd_last_error = torch.zeros_like(entity_data.pd_targets)
                d_pos_error = -velocities  # Use negative velocity as derivative of error
                d_angle_error = -angular_velocities
            else:
                d_pos_error = (pos_error - entity_data.pd_last_error[:, 0:2]) / self.dt
                d_angle_error = (angle_error - entity_data.pd_last_error[:, 2]) / self.dt
                entity_data.pd_last_error[:, 0:2] = pos_error
                entity_data.pd_last_error[:, 2] = angle_error

            # Calculate forces
            forces = entity_data.pd_kp[:, 0:2] * pos_error + entity_data.pd_kd[:, 0:2] * d_pos_error
            torques = entity_data.pd_kp[:, 2] * angle_error + entity_data.pd_kd[:, 2] * d_angle_error

            # Send force commands to workers
            forces_cpu = forces.cpu().numpy()
            torques_cpu = torques.cpu().numpy()

            for env_idx in range(self.n_envs):
                # Get root body index for this environment
                if entity_data.body_indices and env_idx < len(entity_data.body_indices):
                    if entity_data.body_indices[env_idx]:
                        body_idx = entity_data.body_indices[env_idx][0]
                        self._send_command(
                            env_idx,
                            {
                                'type': 'apply_force',
                                'body_idx': body_idx,
                                'force': (forces_cpu[env_idx, 0], forces_cpu[env_idx, 1]),
                                'torque': torques_cpu[env_idx],
                            },
                        )

            # Collect confirmations
            self._collect_results(self.n_envs, timeout=1.0)

    def add_robot(self, name: str, urdf_path: str, pos: torch.Tensor, quat: torch.Tensor, **kwargs) -> RobotInterface:
        """Add robot using bounding box from URDF."""
        # Try to extract bounding box from URDF
        try:
            from urdf_metadata_extractor import URDFMetadataExtractor
            extractor = URDFMetadataExtractor()
            extractor.parse_urdf(urdf_path)
            bbox = extractor.get_overall_bounding_box()
            
            if bbox is None:
                # Default box if no bounding box found
                size = [1.0, 1.0]
            else:
                size = [bbox['max'][0] - bbox['min'][0], bbox['max'][1] - bbox['min'][1]]
        except Exception:
            # If URDF parsing fails, use default size
            size = [1.0, 0.5]

        # Create entity data
        entity_data = PymunkEntityData(name=name, entity_type="robot", n_envs=self.n_envs, scene=self)

        # Store initial states
        entity_data.initial_positions = pos
        entity_data.initial_orientations = quat

        # Create body and shape configs for each environment
        pos_cpu = pos.cpu().numpy()
        quat_cpu = quat.cpu().numpy()

        for env_idx in range(self.n_envs):
            # Convert quaternion to 2D angle
            if env_idx < len(quat_cpu):
                w, x, y, z = quat_cpu[env_idx]
            else:
                w, x, y, z = quat_cpu[0]  # Use first quaternion for all if not enough
            angle = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

            # Body config
            body_config = {
                'mass': kwargs.get('mass', 1.0),
                'moment': pymunk.moment_for_box(kwargs.get('mass', 1.0), size),
                'position': (
                    pos_cpu[0] if env_idx >= len(pos_cpu) else pos_cpu[env_idx, 0],
                    pos_cpu[1] if env_idx >= len(pos_cpu) else pos_cpu[env_idx, 1],
                ),
                'angle': angle,
            }

            # Shape config
            shape_config = {'type': 'box', 'body_idx': 0, 'size': size, 'friction': kwargs.get('friction', 0.5)}

            entity_data.body_configs.append([body_config])
            entity_data.shape_configs.append([shape_config])

        entity_data.body_to_shapes[0] = [0]

        # Initialize PD control
        entity_data.pd_targets = torch.zeros((self.n_envs, 3), device=self.device, dtype=torch.float32)
        entity_data.pd_kp = torch.zeros((self.n_envs, 3), device=self.device, dtype=torch.float32)
        entity_data.pd_kd = torch.zeros((self.n_envs, 3), device=self.device, dtype=torch.float32)

        # Store entity data
        self._entities[name] = entity_data

        # Create and store interface
        robot_interface = PymunkRobotInterface(name, entity_data)
        self._entity_interfaces[name] = robot_interface

        return robot_interface

    # Primitive creation methods
    def add_sphere(
        self, name: str, radius: float, pos: torch.Tensor, quat: Optional[torch.Tensor] = None, **kwargs
    ) -> PrimitiveInterface:
        """Add sphere (circle in 2D)."""
        entity_data = PymunkEntityData(name=name, entity_type="primitive", n_envs=self.n_envs, scene=self)

        pos_cpu = pos.cpu().numpy()

        for env_idx in range(self.n_envs):
            body_config = {
                'mass': kwargs.get('mass', 1.0),
                'moment': pymunk.moment_for_circle(kwargs.get('mass', 1.0), 0, radius),
                'position': (pos_cpu[0], pos_cpu[1]),
                'angle': 0,
            }

            shape_config = {'type': 'circle', 'body_idx': 0, 'radius': radius, 'friction': kwargs.get('friction', 0.5)}

            entity_data.body_configs.append([body_config])
            entity_data.shape_configs.append([shape_config])

        entity_data.body_to_shapes[0] = [0]

        self._entities[name] = entity_data
        primitive_interface = PymunkPrimitiveInterface(name, entity_data, "sphere")
        self._entity_interfaces[name] = primitive_interface

        return primitive_interface

    def add_box(
        self, name: str, size: torch.Tensor, pos: torch.Tensor, quat: Optional[torch.Tensor] = None, **kwargs
    ) -> PrimitiveInterface:
        """Add box primitive."""
        entity_data = PymunkEntityData(name=name, entity_type="primitive", n_envs=self.n_envs, scene=self)

        pos_cpu = pos.cpu().numpy()
        size_cpu = size.cpu().numpy() if isinstance(size, torch.Tensor) else size

        # Use only x and y dimensions for 2D box
        box_size = (size_cpu[0], size_cpu[1])

        for env_idx in range(self.n_envs):
            # Get position for this environment
            if len(pos_cpu.shape) == 1:
                position = (pos_cpu[0], pos_cpu[1])
            else:
                position = (pos_cpu[env_idx, 0], pos_cpu[env_idx, 1])

            # Get angle from quaternion if provided
            angle = 0
            if quat is not None:
                quat_cpu = quat.cpu().numpy()
                if env_idx < len(quat_cpu):
                    w, x, y, z = quat_cpu[env_idx]
                else:
                    w, x, y, z = quat_cpu[0]
                angle = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

            body_config = {
                'mass': kwargs.get('mass', 1.0),
                'moment': pymunk.moment_for_box(kwargs.get('mass', 1.0), box_size),
                'position': position,
                'angle': angle,
            }

            shape_config = {'type': 'box', 'body_idx': 0, 'size': box_size, 'friction': kwargs.get('friction', 0.5)}

            entity_data.body_configs.append([body_config])
            entity_data.shape_configs.append([shape_config])

        entity_data.body_to_shapes[0] = [0]

        self._entities[name] = entity_data
        primitive_interface = PymunkPrimitiveInterface(name, entity_data, "box")
        self._entity_interfaces[name] = primitive_interface

        return primitive_interface

    def add_cylinder(
        self, name: str, radius: float, height: float, pos: torch.Tensor, quat: Optional[torch.Tensor] = None, **kwargs
    ) -> PrimitiveInterface:
        """Add cylinder (treated as circle in 2D)."""
        # In 2D, cylinder becomes a circle - height is ignored
        return self.add_sphere(name, radius, pos, quat, **kwargs)

    def add_mesh(
        self,
        name: str,
        mesh_path: str,
        pos: torch.Tensor,
        quat: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> PrimitiveInterface:
        """Add mesh (approximated as box using bounding box)."""
        # For 2D, we approximate mesh as a box based on its bounding box
        # Default to unit box if we can't load the mesh
        size = torch.tensor([1.0, 1.0, 1.0], device=self.device)

        # You could implement mesh loading and bounding box calculation here
        # For now, use the scale if provided
        if scale is not None:
            size = size * scale

        return self.add_box(name, size, pos, quat, **kwargs)

    def add_plane(self, name: str, **kwargs) -> PrimitiveInterface:
        """Add ground plane (static line segment in 2D)."""
        entity_data = PymunkEntityData(name=name, entity_type="primitive", n_envs=self.n_envs, scene=self)

        # Create a long horizontal line segment for ground
        ground_length = kwargs.get('length', 1000.0)
        ground_y = kwargs.get('y', 0.0)

        for env_idx in range(self.n_envs):
            # No body config for static shapes
            entity_data.body_configs.append([])

            # Static line segment
            shape_config = {
                'type': 'segment',
                'body_idx': -1,  # -1 indicates static body
                'a': (-ground_length / 2, ground_y),
                'b': (ground_length / 2, ground_y),
                'radius': kwargs.get('thickness', 0.1),
                'friction': kwargs.get('friction', 0.5),
            }

            entity_data.shape_configs.append([shape_config])

        self._entities[name] = entity_data
        primitive_interface = PymunkPrimitiveInterface(name, entity_data, "plane")
        self._entity_interfaces[name] = primitive_interface

        return primitive_interface

    # Camera methods - not supported in 2D
    def create_camera(self, name: str, config: 'CameraConfig') -> None:
        """Cameras not supported in 2D Pymunk simulation."""
        raise NotImplementedError("Camera operations not supported in 2D Pymunk simulation")

    def set_camera_pose(
        self,
        name: str,
        pos: Tuple[float, float, float],
        lookat: Tuple[float, float, float],
        up: Tuple[float, float, float],
    ) -> None:
        """Cameras not supported in 2D Pymunk simulation."""
        raise NotImplementedError("Camera operations not supported in 2D Pymunk simulation")

    def render_camera(self, name: str, rgb: bool = True) -> None:
        """Cameras not supported in 2D Pymunk simulation."""
        raise NotImplementedError("Camera operations not supported in 2D Pymunk simulation")

    def start_camera_recording(self, name: str) -> None:
        """Cameras not supported in 2D Pymunk simulation."""
        raise NotImplementedError("Camera operations not supported in 2D Pymunk simulation")

    def stop_camera_recording(self, name: str, save_path: str, fps: int) -> None:
        """Cameras not supported in 2D Pymunk simulation."""
        raise NotImplementedError("Camera operations not supported in 2D Pymunk simulation")

    # Debug visualization methods - not supported in headless 2D
    def _draw_debug_arrow(self, name, pos, vec=(0, 0, 1), radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        """Debug visualization not supported in headless 2D Pymunk simulation."""
        raise NotImplementedError("Debug visualization not supported in headless 2D Pymunk simulation")

    def _draw_debug_line(self, name, start, end, radius=0.002, color=(1.0, 0.0, 0.0, 0.5)):
        """Debug visualization not supported in headless 2D Pymunk simulation."""
        raise NotImplementedError("Debug visualization not supported in headless 2D Pymunk simulation")

    def _draw_debug_frame(self, name, T, axis_length=1.0, origin_size=0.015, axis_radius=0.01):
        """Debug visualization not supported in headless 2D Pymunk simulation."""
        raise NotImplementedError("Debug visualization not supported in headless 2D Pymunk simulation")

    def _draw_debug_mesh(self, name, mesh, pos, T=None):
        """Debug visualization not supported in headless 2D Pymunk simulation."""
        raise NotImplementedError("Debug visualization not supported in headless 2D Pymunk simulation")

    def _draw_debug_sphere(self, name, pos, radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        """Debug visualization not supported in headless 2D Pymunk simulation."""
        raise NotImplementedError("Debug visualization not supported in headless 2D Pymunk simulation")

    def _draw_debug_box(self, name, bounds, color=(1.0, 0.0, 0.0, 1.0), wireframe=True, wireframe_radius=0.0015):
        """Debug visualization not supported in headless 2D Pymunk simulation."""
        raise NotImplementedError("Debug visualization not supported in headless 2D Pymunk simulation")

    def _draw_debug_points(self, name, poss, colors=(1.0, 0.0, 0.0, 0.5)):
        """Debug visualization not supported in headless 2D Pymunk simulation."""
        raise NotImplementedError("Debug visualization not supported in headless 2D Pymunk simulation")

    def _draw_debug_path(self, name, qposs, entity, link_idx=-1, density=0.3, frame_scaling=1.0):
        """Debug visualization not supported in headless 2D Pymunk simulation."""
        raise NotImplementedError("Debug visualization not supported in headless 2D Pymunk simulation")

    def _clear_debug_marks(self, names: Optional[List[str]] = None) -> None:
        """Debug visualization not supported in headless 2D Pymunk simulation."""
        raise NotImplementedError("Debug visualization not supported in headless 2D Pymunk simulation")
