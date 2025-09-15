#!/usr/bin/env python3
"""
Genesis Simulator Interface Implementation

This module implements the SceneInterface and RobotInterface for the Genesis
physics simulator, providing a 1:1 mapping to the current Genesis functionality.
"""

import inspect
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING

import genesis as gs
import numpy as np
import torch

from simulation_interface.base_interfaces import EntityInterface, PrimitiveInterface, RobotInterface, SceneInterface
from simulation_interface.debug_mark_manager import DebugMark, DebugMarkManager
from utils import ReferenceFrames, ReferenceFramesInput
from utils.helpers import (
    convert_orientation,
    apply_relative_transform,
    apply_relative_transform_inverse,
    apply_simple_offset,
    apply_simple_offset_inverse,
)

if TYPE_CHECKING:
    from configs.SimulatorConfig import SimulatorConfig
    from configs.CameraConfig import CameraConfig


class GenesisEntityInterface(EntityInterface):
    def __init__(self, name: str, entity: Any):
        EntityInterface.__init__(self, name)
        self._entity = entity

    # ==============================================
    # EntityInterface Abstract Methods Implementation
    # ==============================================

    def _get_pos_raw(self, env_indices: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Get raw position from Genesis simulator."""
        return self._entity.get_pos(envs_idx=env_indices, **kwargs)

    def _get_orientation_raw(self, env_indices: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Get raw orientation from Genesis simulator as quaternion."""
        return self._entity.get_quat(envs_idx=env_indices, **kwargs)

    def _get_lin_vel_raw(self, env_indices: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Get raw linear velocity from Genesis simulator."""
        return self._entity.get_vel(envs_idx=env_indices, **kwargs)

    def _get_ang_vel_raw(self, env_indices: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Get raw angular velocity from Genesis simulator."""
        return self._entity.get_ang(envs_idx=env_indices, **kwargs)

    def _set_pos_raw(
        self, pos: torch.Tensor, env_indices: Optional[torch.Tensor] = None, zero_velocity: bool = True, **kwargs
    ) -> None:
        """Set position for specified environments (raw implementation)."""
        self._entity.set_pos(pos, envs_idx=env_indices, zero_velocity=zero_velocity, **kwargs)

    def _set_orientation_raw(
        self, quat: torch.Tensor, env_indices: Optional[torch.Tensor] = None, zero_velocity: bool = True, **kwargs
    ) -> None:
        """Set orientation for specified environments (raw implementation)."""
        self._entity.set_quat(quat, envs_idx=env_indices, zero_velocity=zero_velocity, **kwargs)

    def _set_lin_vel_raw(self, lin_vel: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        raise NotImplementedError("Genesis does not support setting linear velocity directly")

    def _set_ang_vel_raw(self, ang_vel: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        raise NotImplementedError("Genesis does not support setting angular velocity directly")

    def _control_pos_raw(self, pos: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        raise NotImplementedError("Genesis does not support controlling  position directly")

    def _control_orientation_raw(
        self, quat: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs
    ) -> None:
        raise NotImplementedError("Genesis does not support controlling orientation directly")

    def _control_lin_vel_raw(self, lin_vel: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        raise NotImplementedError("Genesis does not support controlling linear velocity directly")

    def _control_ang_vel_raw(self, ang_vel: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        raise NotImplementedError("Genesis does not support controlling angular velocity directly")


class GenesisPrimitiveInterface(GenesisEntityInterface, PrimitiveInterface):
    def __init__(self, name: str, entity: Any, primitive_type: str):
        PrimitiveInterface.__init__(self, name, primitive_type)
        GenesisEntityInterface.__init__(self, name, entity)

    def _set_lin_vel_raw(self, lin_vel: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        """Set linear velocity for specified environments (raw implementation)."""
        if len(self._entity.joints) ==1 and len(self._entity.joints[0].dofs_idx_local) == 6:
            root_joint = self._entity.joints[0]
            root_index = root_joint.dofs_idx_local
            self._entity.set_dofs_velocity(lin_vel, dofs_idx_local=root_index[0:3], envs_idx=env_indices, **kwargs)
        else:
            raise ValueError(f"Primitive does not have a root joint, cannot set linear velocity")

    def _set_ang_vel_raw(self, ang_vel: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        """Set angular velocity for specified environments (raw implementation)."""
        # The only way to set robot angular velocity in Genesis is through root joint velocity
        if len(self._entity.joints) ==1 and len(self._entity.joints[0].dofs_idx_local) == 6:
            root_joint = self._entity.joints[0]
            root_index = root_joint.dofs_idx_local
            self._entity.set_dofs_velocity(ang_vel, dofs_idx_local=root_index[3:6], envs_idx=env_indices, **kwargs)
        else:
            raise ValueError(f"Primitive does not have a root joint, cannot set angular velocity")
        # TODO: Implement override for control methods, they can all be done through root joint
class GenesisRobotInterface(GenesisEntityInterface, RobotInterface):
    """Genesis implementation of RobotInterface.

    This class wraps a Genesis entity and provides the standardized interface
    for robot control and observation.
    """

    def __init__(self, name: str, entity: Any):
        """Initialize the Genesis robot interface.

        Args:
            name: Unique name for this robot
            entity: Genesis robot entity (RigidEntity)
        """
        RobotInterface.__init__(self, name)
        GenesisEntityInterface.__init__(self, name, entity)

    # ==============================================
    # EntityInterface Abstract Methods Override
    # ==============================================

    def _set_lin_vel_raw(self, lin_vel: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        """Set linear velocity for specified environments (raw implementation)."""
        # The only way to set robot velocity in Genesis is through root joint velocity
        if "root_joint" in [joint.name for joint in self._entity.joints]:
            root_joint = self._entity.get_joint(name="root_joint")
            root_index = root_joint.dofs_idx_local
            self._entity.set_dofs_velocity(lin_vel, dofs_idx_local=root_index[0:3], envs_idx=env_indices, **kwargs)
        else:
            raise ValueError(f"Robot does not have a root joint, cannot set linear velocity")

    def _set_ang_vel_raw(self, ang_vel: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        """Set angular velocity for specified environments (raw implementation)."""
        # The only way to set robot angular velocity in Genesis is through root joint velocity
        if "root_joint" in [joint.name for joint in self._entity.joints]:
            root_joint = self._entity.get_joint(name="root_joint")
            root_index = root_joint.dofs_idx_local
            self._entity.set_dofs_velocity(ang_vel, dofs_idx_local=root_index[3:6], envs_idx=env_indices, **kwargs)
        else:
            raise ValueError(f"Robot does not have a root joint, cannot set angular velocity")

    # TODO: Implement override for control methods, they can all be done through root joint

    # ==============================================
    # RobotInterface Abstract Methods Implementation
    # ==============================================

    def _get_joint_pos_raw(
        self, joint_indices: Optional[torch.Tensor] = None, env_indices: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """Get raw joint positions from Genesis simulator."""
        return self._entity.get_dofs_position(dofs_idx_local=joint_indices, envs_idx=env_indices, **kwargs)

    def _get_joint_vel_raw(
        self, joint_indices: Optional[torch.Tensor] = None, env_indices: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """Get raw joint velocities from Genesis simulator."""
        return self._entity.get_dofs_velocity(dofs_idx_local=joint_indices, envs_idx=env_indices, **kwargs)

    def _set_joint_pos_raw(
        self,
        pos: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        zero_velocity: bool = False,
        **kwargs,
    ) -> None:
        """Set joint positions for specified joints and environments (raw implementation)."""
        self._entity.set_dofs_position(
            pos, dofs_idx_local=joint_indices, envs_idx=env_indices, zero_velocity=zero_velocity, **kwargs
        )

    def _set_joint_vel_raw(
        self,
        vel: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Set joint velocities for specified joints and environments (raw implementation)."""
        self._entity.set_dofs_velocity(vel, dofs_idx_local=joint_indices, envs_idx=env_indices, **kwargs)

    def _control_joint_pos_raw(
        self,
        target_pos: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Apply position control to specified joints (raw implementation)."""
        self._entity.control_dofs_position(
            position=target_pos, dofs_idx_local=joint_indices, envs_idx=env_indices, **kwargs
        )

    def _control_joint_vel_raw(
        self,
        target_vel: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Apply velocity control to specified joints (raw implementation)."""
        self._entity.control_dofs_velocity(
            velocity=target_vel, dofs_idx_local=joint_indices, envs_idx=env_indices, **kwargs
        )

    def _control_joint_force_raw(
        self,
        forces: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Apply force/torque control to specified joints."""
        self._entity.control_dofs_force(force=forces, dofs_idx_local=joint_indices, envs_idx=env_indices, **kwargs)

    """
    Set PD parameters
    """

    def set_kp_gains(
        self,
        kp: Union[float, torch.Tensor],
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Set proportional gains for specified joints."""
        self._entity.set_dofs_kp(kp, dofs_idx_local=joint_indices, envs_idx=env_indices, **kwargs)

    def set_kd_gains(
        self,
        kd: Union[float, torch.Tensor],
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Set derivative gains for specified joints."""
        self._entity.set_dofs_kv(kd, dofs_idx_local=joint_indices, envs_idx=env_indices, **kwargs)

    def get_joint(self, joint_name: str) -> Any:
        """Get joint object by name for accessing joint-specific properties.

        Args:
            joint_name: Name of the joint

        Returns:
            Any: Simulator-specific joint object
        """
        return self._entity.get_joint(joint_name)


class GenesisSceneInterface(SceneInterface):
    """Genesis implementation of SceneInterface.

    This class wraps a Genesis Scene and provides the standardized interface
    for scene management and simulation control.
    """

    def __init__(self, config: 'SimulatorConfig'):
        """Initialize the Genesis scene interface from configuration.

        Args:
            config: Simulator configuration containing device, n_envs, and all setup parameters
        """
        super().__init__()
        self._device = config.device
        self._n_envs = config.n_envs
        self._robots = {}
        self._primitives = {}
        self._debug_objects = {}
        self._cameras = {}  # Store cameras by name

        # Initialize Genesis and create scene (merged from _setup_from_general_config)
        self._initialize_genesis_and_scene(config)

    def _initialize_genesis_and_scene(self, config: 'SimulatorConfig'):
        """Initialize Genesis and create the scene from configuration."""
        # Initialize Genesis if not already done
        try:
            # Check if Genesis is already initialized
            context = gs.get_context()
            # If we get here, Genesis is already initialized
        except Exception:
            # Genesis not initialized, initialize it
            backend = gs.gpu if self._device == 'cuda' else gs.cpu
            try:
                gs.init(backend=backend)
            except Exception as e:
                # Handle case where Genesis is already initialized but get_context failed
                if "already initialized" in str(e).lower():
                    pass  # Genesis is initialized, continue
                else:
                    raise e

        # Convert general config to Genesis-specific options
        genesis_kwargs = self._translate_config_to_genesis(config)

        # Create Genesis scene
        self._scene = gs.Scene(**genesis_kwargs)

        # Add ground plane by default
        self._scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        return self._n_envs

    @property
    def device(self) -> str:
        """Device used for computations."""
        return self._device

    @property
    def dt(self) -> float:
        """Get simulation timestep."""
        return self._scene.dt

    @property
    def sim_step(self) -> int:
        """Get current simulation step."""
        return self._scene.t

    # Provide access to underlying Genesis scene for advanced users
    @property
    def genesis_scene(self) -> gs.Scene:
        """Get the underlying Genesis scene object.

        This allows advanced users to access Genesis-specific functionality
        that isn't exposed through the interface.
        """
        return self._scene

    def _add_entity(self, morph_func, material_func=gs.materials.Rigid, surface_func=gs.surfaces.Default, **kwargs):
        """Remap args to genesis add_entity"""
        entity_kwargs = set(inspect.signature(gs.Scene.add_entity).parameters.keys())
        to_remove = {'morph', 'material', 'surface', 'self'}
        entity_kwargs.difference_update(to_remove)
        morph_kwargs = set(inspect.signature(morph_func).parameters.keys())
        material_kwargs = set(inspect.signature(material_func).parameters.keys())
        surface_kwargs = set(inspect.signature(surface_func).parameters.keys())

        # Check for kwargs collison
        for k in morph_kwargs & material_kwargs:
            raise ValueError(f"Collision between morph and material kwargs: {k}")
        for k in morph_kwargs & surface_kwargs:
            raise ValueError(f"Collision between morph and surface kwargs: {k}")
        for k in material_kwargs & surface_kwargs:
            raise ValueError(f"Collision between material and surface kwargs: {k}")

        entity_func_kwargs = {}
        morph_func_kwargs = {}
        material_func_kwargs = {}
        surface_func_kwargs = {}

        for k, v in kwargs.items():
            if k == "morph" and isinstance(v, dict):
                morph_func_kwargs.update(v)
            elif k == "material" and isinstance(v, dict):
                morph_func_kwargs.update(v)
            elif k == "surface" and isinstance(v, dict):
                morph_func_kwargs.update(v)
            elif k in entity_kwargs:
                entity_func_kwargs[k] = v
            elif k in morph_kwargs:
                morph_func_kwargs[k] = v
            elif k in material_kwargs:
                material_func_kwargs[k] = v
            elif k in surface_kwargs:
                surface_func_kwargs[k] = v
            else:
                raise ValueError(f"Invalid argument: {k}")

        kwargs = {
            **entity_func_kwargs,
            "morph": morph_func(**morph_func_kwargs),
            "material": material_func(**material_func_kwargs),
            "surface": surface_func(**surface_func_kwargs),
        }

        return self._scene.add_entity(**kwargs)

    def _translate_config_to_genesis(self, config: 'SimulatorConfig') -> dict:
        """Translate general config to Genesis-specific options."""
        # Map constraint solver names
        solver_map = {
            "newton": gs.constraint_solver.Newton,
            "cg": gs.constraint_solver.CG,
        }

        constraint_solver = solver_map.get(config.constraint_solver.lower(), gs.constraint_solver.Newton)

        return {
            "sim_options": gs.options.SimOptions(
                dt=config.dt,
                substeps=config.substeps,
                gravity=config.gravity,
            ),
            "viewer_options": gs.options.ViewerOptions(
                res=config.camera_config.res,  # (width, height)
                max_FPS=config.max_fps,
                camera_pos=config.camera_config.offset,  # Use offset as camera position
                camera_lookat=(0.0, 0.0, 0.0),  # Default lookat at origin
                camera_up=(0.0, 0.0, 1.0),  # Standard up vector
                camera_fov=config.camera_config.fov,  # Use camera config FOV
            ),
            "vis_options": gs.options.VisOptions(rendered_envs_idx=config.rendered_envs_idx),
            "rigid_options": gs.options.RigidOptions(
                dt=config.dt,
                constraint_solver=constraint_solver,
                enable_collision=config.enable_collision,
                enable_joint_limit=config.enable_joint_limit,
            ),
            "show_viewer": config.show_viewer,
            **config.simulator_specific.get('genesis', {}),  # Genesis-specific overrides
        }

    def add_robot(self, name: str, urdf_path: str, pos: torch.Tensor, quat: torch.Tensor, **kwargs) -> RobotInterface:
        """Add a robot to each environment and return a RobotInterface."""
        # Validate unique name
        self._validate_unique_name(name)

        # Create Genesis robot entity
        robot_entity = self._add_entity(gs.morphs.URDF, file=urdf_path, pos=pos, quat=quat, **kwargs)

        # Create Genesis robot interface
        robot_interface = GenesisRobotInterface(name, robot_entity)
        self._robots[name] = robot_interface

        # Return wrapped interface
        return robot_interface

    # ==============================================
    # Primitive Creation Methods
    # ==============================================

    def add_sphere(
        self, name: str, radius: float, pos: torch.Tensor, quat: Optional[torch.Tensor] = None, **kwargs
    ) -> PrimitiveInterface:
        """Add sphere primitive."""
        # Validate unique name
        self._validate_unique_name(name)

        # Create Genesis sphere entity
        sphere_entity = self._add_entity(gs.morphs.Sphere, radius=radius, pos=pos, quat=quat, **kwargs)

        # Create primitive interface
        primitive_interface = GenesisPrimitiveInterface(name, sphere_entity, "sphere")
        self._primitives[name] = primitive_interface

        return primitive_interface

    def add_box(
        self, name: str, size: torch.Tensor, pos: torch.Tensor, quat: Optional[torch.Tensor] = None, **kwargs
    ) -> PrimitiveInterface:
        """Add box primitive."""
        # Validate unique name
        self._validate_unique_name(name)

        # Create Genesis box entity
        box_entity = self._add_entity(gs.morphs.Box, size=size, pos=pos, quat=quat, **kwargs)

        # Create primitive interface
        primitive_interface = GenesisPrimitiveInterface(name, box_entity, "box")
        self._primitives[name] = primitive_interface

        return primitive_interface

    def add_cylinder(
        self, name: str, radius: float, height: float, pos: torch.Tensor, quat: Optional[torch.Tensor] = None, **kwargs
    ) -> PrimitiveInterface:
        """Add cylinder primitive."""
        # Validate unique name
        self._validate_unique_name(name)

        # Create Genesis cylinder entity
        cylinder_entity = self._add_entity(
            gs.morphs.Cylinder, radius=radius, height=height, pos=pos, quat=quat, **kwargs
        )

        # Create primitive interface
        primitive_interface = GenesisPrimitiveInterface(name, cylinder_entity, "cylinder")
        self._primitives[name] = primitive_interface

        return primitive_interface

    def add_mesh(
        self,
        name: str,
        mesh_path: str,
        pos: torch.Tensor,
        quat: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> PrimitiveInterface:
        """Add mesh primitive."""
        # Validate unique name
        self._validate_unique_name(name)

        # Create Genesis mesh entity
        mesh_entity = self._add_entity(gs.morphs.Mesh, file=mesh_path, pos=pos, quat=quat, scale=scale, **kwargs)

        # Create primitive interface
        primitive_interface = GenesisPrimitiveInterface(name, mesh_entity, "mesh")
        self._primitives[name] = primitive_interface

        return primitive_interface

    def add_plane(self, name: str, **kwargs) -> PrimitiveInterface:
        """Add ground plane primitive."""
        # Validate unique name
        self._validate_unique_name(name)

        # Create Genesis plane entity
        plane_entity = self._add_entity(gs.morphs.Plane, **kwargs)

        # Create primitive interface
        primitive_interface = GenesisPrimitiveInterface(name, plane_entity, "plane")
        self._primitives[name] = primitive_interface

        return primitive_interface

    def build_scene(self) -> None:
        """Finalize scene construction after all robots/bodies are added."""
        self._scene.build(n_envs=self.n_envs)

    def step(self, n_steps: int = 1) -> None:
        """Advance the simulation by n_steps."""
        for _ in range(n_steps):
            self._scene.step()

    # ==============================================
    # Entity Management
    # ==============================================

    @property
    def entities(self) -> Dict[str, GenesisEntityInterface]:
        return {**self._primitives, **self._robots}

    @property
    def primitives(self) -> Dict[str, GenesisPrimitiveInterface]:
        return self._primitives

    @property
    def robots(self) -> Dict[str, GenesisRobotInterface]:
        return self._robots

    # ==============================================
    # Camera Methods Implementation
    # ==============================================

    def create_camera(self, name: str, config: 'CameraConfig') -> None:
        """Create a camera in the simulation.

        Args:
            name: Unique name for the camera
            config: Camera configuration object
        """
        if name in self._cameras:
            raise ValueError(f"Camera '{name}' already exists")

        # Extract configuration from CameraConfig
        camera = self._scene.add_camera(
            res=config.res,
            pos=list(config.offset),  # Initial position
            lookat=list(config.lookat),  # Will be updated during tracking
            up=list(config.up),
            fov=config.fov,
            aperture=0.0,
            focus_dist=0.0,
            GUI=False,  # Headless by default
            spp=64,
            denoise=True,
        )

        self._cameras[name] = camera

    def set_camera_pose(
        self,
        name: str,
        pos: Tuple[float, float, float],
        lookat: Tuple[float, float, float],
        up: Tuple[float, float, float],
    ) -> None:
        """Set camera position and orientation.

        Args:
            name: Camera name
            pos: Camera position (x, y, z)
            lookat: Look-at target position (x, y, z)
            up: Up vector (x, y, z)
        """
        if name not in self._cameras:
            raise ValueError(f"Camera '{name}' not found")

        camera = self._cameras[name]
        camera.set_pose(pos=pos, lookat=lookat, up=up)

    def render_camera(self, name: str, rgb: bool = True) -> None:
        """Render a frame from the camera.

        Args:
            name: Camera name
            rgb: Whether to render RGB (vs depth/segmentation)
        """
        if name not in self._cameras:
            raise ValueError(f"Camera '{name}' not found")

        camera = self._cameras[name]
        return camera.render(rgb=rgb)[0]

    def start_camera_recording(self, name: str) -> None:
        """Start recording from a camera.

        Args:
            name: Camera name
        """
        if name not in self._cameras:
            raise ValueError(f"Camera '{name}' not found")

        camera = self._cameras[name]
        camera.start_recording()

    def stop_camera_recording(self, name: str, save_path: str, fps: int) -> None:
        """Stop recording and save video.

        Args:
            name: Camera name
            save_path: Path to save video file
            fps: Frames per second for video
        """
        if name not in self._cameras:
            raise ValueError(f"Camera '{name}' not found")

        camera = self._cameras[name]
        camera.stop_recording(save_to_filename=save_path, fps=fps)

    # ==============================================
    # Debug Visualization Methods Implementation
    # ==============================================
    def _draw_debug_line(self, name, start, end, radius=0.002, color=(1.0, 0.0, 0.0, 0.5)):
        debug_line = self._scene.draw_debug_line(start, end, radius=radius, color=color)
        self._debug_objects[name] = debug_line
        return debug_line

    def _draw_debug_arrow(self, name, pos, vec=(0, 0, 1), radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        debug_arrow = self._scene.draw_debug_arrow(pos, vec=vec, radius=radius, color=color)
        self._debug_objects[name] = debug_arrow
        return debug_arrow

    def _draw_debug_frame(self, name, T, axis_length=1.0, origin_size=0.015, axis_radius=0.01):
        debug_frame = self._scene.draw_debug_frame(
            T, axis_length=axis_length, origin_size=origin_size, axis_radius=axis_radius
        )
        self._debug_objects[name] = debug_frame
        return debug_frame

    def _draw_debug_mesh(self, name, mesh, pos=np.zeros(3), T=None):
        debug_mesh = self._scene.draw_debug_mesh(mesh, pos=pos, T=T)
        self._debug_objects[name] = debug_mesh
        return debug_mesh

    def _draw_debug_sphere(self, name, pos, radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        debug_sphere = self._scene.draw_debug_sphere(pos, radius=radius, color=color)
        self._debug_objects[name] = debug_sphere
        return debug_sphere

    def _draw_debug_box(self, name, bounds, color=(1.0, 0.0, 0.0, 1.0), wireframe=True, wireframe_radius=0.0015):
        debug_box = self._scene.draw_debug_box(
            bounds, color=color, wireframe=wireframe, wireframe_radius=wireframe_radius
        )
        self._debug_objects[name] = debug_box
        return debug_box

    def _draw_debug_points(self, name, poss, colors=(1.0, 0.0, 0.0, 0.5)):
        debug_points = self._scene.draw_debug_points(poss, colors=colors)
        self._debug_objects[name] = debug_points
        return debug_points

    def _draw_debug_path(self, name, qposs, entity, link_idx=-1, density=0.3, frame_scaling=1.0):
        debug_path = self._scene.draw_debug_path(
            qposs, entity, link_idx=link_idx, density=density, frame_scaling=frame_scaling
        )
        self._debug_objects[name] = debug_path
        return debug_path

    def _clear_debug_marks(self, names: Optional[List[str]] = None):
        if names is None:
            names = list(self._debug_objects.keys())
        for name in names:
            debug_object = self._debug_objects[name]
            if debug_object is None:
                continue
            self._scene.clear_debug_object(debug_object)

    """
    Helpers
    """

    def _validate_unique_name(self, name: str) -> None:
        if name in self.primitives:
            raise ValueError(f"Primitive with name '{name}' already exists")

        if name in self.robots:
            raise ValueError(f"Robot with name '{name}' already exists")

        if name in self._cameras:
            raise ValueError(f"Camera with name '{name}' already exists")
