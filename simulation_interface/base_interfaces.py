#!/usr/bin/env python3
"""
Abstract Base Interfaces for Simulation Backends

This module defines the abstract base classes that all simulator implementations
must inherit from to be compatible with the Genesis MARL Framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING
from utils import transform_by_quat, inv_quat, xyz_to_quat, quat_to_xyz

import torch

# Type alias for flexible reference frame input
from utils import ReferenceFrames, ReferenceFramesInput
from simulation_interface.camera_manager import CameraManager
from simulation_interface.debug_mark_manager import DebugMarkManager, DebugMark
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


# ==============================================
# Core Interfaces
# ==============================================


class EntityInterface(ABC):
    """Base interface for all simulation entities (robots, primitives).

    This interface provides common state access and manipulation methods
    that apply to any physical entity in the simulation, including:
    - Position and orientation
    - Linear and angular velocity
    - Support for relative transformations

    All methods accept torch.Tensor inputs shaped [n_envs, ...] and optional
    env_indices to support vectorized environments.
    """

    def __init__(self, name: str):
        """Initialize entity with unique name.

        Args:
            name: Unique identifier for this entity
        """
        self._name = name

    @property
    def name(self) -> str:
        """Get entity's unique name."""
        return self._name

    # ==============================================
    # Raw State Getters (Abstract)
    # ==============================================

    @abstractmethod
    def _get_pos_raw(self, env_indices: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Get raw position from simulator.

        Args:
            env_indices: Environment indices to query. None means all environments.

        Returns:
            torch.Tensor: Positions [n_envs, 3] or [len(env_indices), 3]
        """
        pass

    @abstractmethod
    def _get_orientation_raw(self, env_indices: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Get raw orientation from simulator as quaternion.

        Args:
            env_indices: Environment indices to query. None means all environments.

        Returns:
            torch.Tensor: Orientations as quaternions [n_envs, 4] or [len(env_indices), 4]
        """
        pass

    @abstractmethod
    def _get_lin_vel_raw(self, env_indices: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Get raw linear velocity from simulator.

        Args:
            env_indices: Environment indices to query. None means all environments.

        Returns:
            torch.Tensor: Linear velocities [n_envs, 3] or [len(env_indices), 3]
        """
        pass

    @abstractmethod
    def _get_ang_vel_raw(self, env_indices: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Get raw angular velocity from simulator.

        Args:
            env_indices: Environment indices to query. None means all environments.

        Returns:
            torch.Tensor: Angular velocities [n_envs, 3] or [len(env_indices), 3]
        """
        pass

    # ==============================================
    # Public State Getters (with relative_to support)
    # ==============================================

    def get_pos(
        self, env_indices: Optional[torch.Tensor] = None, relative_to: ReferenceFramesInput = None, **kwargs
    ) -> torch.Tensor:
        """Get position for each environment.

        Args:
            relative_to: Optional reference frame or flexible input to compute relative coordinates.
                        Accepts: ReferenceFrames, [n_envs,3], [n_envs,6], [n_envs,7],
                        [n_envs,2,3], etc.
            env_indices: Environment indices to query. None means all environments.

        Returns:
            torch.Tensor: Positions [n_envs, 3] or [len(env_indices), 3]
        """
        # Get raw position data
        pos = self._get_pos_raw(env_indices=env_indices, **kwargs)

        # Convert flexible input to ReferenceFrames
        ref_frame = ReferenceFrames.from_input(relative_to, device=pos.device)

        # Apply relative transformation if requested
        return pos if ref_frame is None else apply_relative_transform(pos, ref_frame)

    def get_orientation(
        self,
        format: Literal["quat", "euler", "xyz", "rpy"] = "quat",
        env_indices: Optional[torch.Tensor] = None,
        relative_to: ReferenceFramesInput = None,
        **kwargs,
    ) -> torch.Tensor:
        """Get orientation for each environment.

        Args:
            format: Orientation format ("quat", "euler", "xyz", "rpy")
            relative_to: Optional reference frame or flexible input to compute relative orientation
            env_indices: Environment indices to query. None means all environments.

        Returns:
            torch.Tensor: Orientations [n_envs, format_dim]
        """
        # Get raw orientation data (always quaternion)
        quat = self._get_orientation_raw(env_indices=env_indices, **kwargs)

        # Convert flexible input to ReferenceFrames
        ref_frame = ReferenceFrames.from_input(relative_to, device=quat.device)

        # Apply relative transformation if requested
        quat = apply_relative_transform(quat, ref_frame)

        # Convert to target format
        return convert_orientation(quat, target_format=format)

    def get_lin_vel(
        self, env_indices: Optional[torch.Tensor] = None, relative_to: ReferenceFramesInput = None, **kwargs
    ) -> torch.Tensor:
        """Get linear velocity for each environment.

        Args:
            relative_to: Optional reference frame or flexible input to transform velocity
            env_indices: Environment indices to query. None means all environments.

        Returns:
            torch.Tensor: Linear velocities [n_envs, 3] or [len(env_indices), 3]
        """
        # Get raw velocity data
        vel = self._get_lin_vel_raw(env_indices=env_indices, **kwargs)

        # Convert flexible input to ReferenceFrames
        ref_frame = ReferenceFrames.from_input(relative_to, device=vel.device)

        # Apply relative transformation if requested
        return vel if ref_frame is None else apply_relative_transform(vel, ref_frame)

    def get_ang_vel(
        self, env_indices: Optional[torch.Tensor] = None, relative_to: ReferenceFramesInput = None, **kwargs
    ) -> torch.Tensor:
        """Get angular velocity for each environment.

        Args:
            relative_to: Optional reference frame or flexible input to transform velocity
            env_indices: Environment indices to query. None means all environments.

        Returns:
            torch.Tensor: Angular velocities [n_envs, 3] or [len(env_indices), 3]
        """
        # Get raw velocity data
        vel = self._get_ang_vel_raw(env_indices=env_indices, **kwargs)

        # Convert flexible input to ReferenceFrames
        ref_frame = ReferenceFrames.from_input(relative_to, device=vel.device)

        # Apply relative transformation if requested
        return vel if ref_frame is None else apply_relative_transform(vel, ref_frame)

    # ==============================================
    # Raw State Setters (Abstract)
    # ==============================================

    @abstractmethod
    def _set_pos_raw(
        self, pos: torch.Tensor, env_indices: Optional[torch.Tensor] = None, zero_velocity: bool = True, **kwargs
    ) -> None:
        """Set position for specified environments (raw implementation).

        Args:
            pos: New positions [n_envs, 3] or [len(env_indices), 3] (absolute coordinates)
            env_indices: Environment indices to update. None means all environments.
            zero_velocity: Whether to reset velocities after setting position
        """
        pass

    @abstractmethod
    def _set_orientation_raw(
        self, quat: torch.Tensor, env_indices: Optional[torch.Tensor] = None, zero_velocity: bool = True, **kwargs
    ) -> None:
        """Set orientation for specified environments (raw implementation).

        Args:
            quat: New orientations as quaternions [n_envs, 4] or [len(env_indices), 4] (absolute coordinates)
            env_indices: Environment indices to update. None means all environments.
            zero_velocity: Whether to reset velocities after setting orientation
        """
        pass

    @abstractmethod
    def _set_lin_vel_raw(self, lin_vel: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        """Set linear velocity for specified environments (raw implementation).

        Args:
            lin_vel: New linear velocities [n_envs, 3] or [len(env_indices), 3] (absolute coordinates)
            env_indices: Environment indices to update. None means all environments.
        """
        pass

    @abstractmethod
    def _set_ang_vel_raw(self, ang_vel: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        """Set angular velocity for specified environments (raw implementation).

        Args:
            ang_vel: New angular velocities [n_envs, 3] or [len(env_indices), 3] (absolute coordinates)
            env_indices: Environment indices to update. None means all environments.
        """
        pass

    @abstractmethod
    def _control_pos_raw(self, pos: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        pass

    @abstractmethod
    def _control_orientation_raw(
        self, quat: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def _control_lin_vel_raw(self, lin_vel: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        pass

    @abstractmethod
    def _control_ang_vel_raw(self, ang_vel: torch.Tensor, env_indices: Optional[torch.Tensor] = None, **kwargs) -> None:
        pass

    # ==============================================
    # Public State Setters (with relative_to support)
    # ==============================================

    def set_pos(
        self,
        pos: torch.Tensor,
        env_indices: Optional[torch.Tensor] = None,
        zero_velocity: bool = True,
        relative_to: ReferenceFramesInput = None,
        **kwargs,
    ) -> None:
        """Set position for specified environments.

        Args:
            pos: New positions [n_envs, 3] or [len(env_indices), 3] (in relative coordinates if relative_to is provided)
            env_indices: Environment indices to update. None means all environments.
            zero_velocity: Whether to reset velocities after setting position
            relative_to: Optional reference frame - if provided, pos is treated as relative coordinates
        """
        # Convert relative coordinates to absolute if needed
        if relative_to is not None:
            ref_frame = ReferenceFrames.from_input(relative_to, device=pos.device)
            pos = apply_relative_transform_inverse(pos, ref_frame)

        # Call the raw implementation
        self._set_pos_raw(pos, env_indices=env_indices, zero_velocity=zero_velocity, **kwargs)

    def set_orientation(
        self,
        angle: torch.Tensor,
        format: Literal["quat", "euler", "xyz", "rpy"] = "quat",
        env_indices: Optional[torch.Tensor] = None,
        zero_velocity: bool = True,
        relative_to: ReferenceFramesInput = None,
        **kwargs,
    ) -> None:
        """Set orientation for specified environments.

        Args:
            quat: New orientations as quaternions [n_envs, 4] or [len(env_indices), 4] (in relative coordinates if relative_to is provided)
            env_indices: Environment indices to update. None means all environments.
            zero_velocity: Whether to reset velocities after setting orientation
            relative_to: Optional reference frame - if provided, quat is treated as relative coordinates
        """
        # Convert relative coordinates to absolute if needed
        quat = convert_orientation(angle, source_format=format, target_format="quat")

        if relative_to is not None:
            ref_frame = ReferenceFrames.from_input(relative_to, device=angle.device)
            quat = apply_relative_transform_inverse(quat, ref_frame)

        # Call the raw implementation
        self._set_orientation_raw(quat, env_indices=env_indices, zero_velocity=zero_velocity, **kwargs)

    def set_lin_vel(
        self,
        lin_vel: torch.Tensor,
        env_indices: Optional[torch.Tensor] = None,
        relative_to: ReferenceFramesInput = None,
        **kwargs,
    ) -> None:
        """Set linear velocity for specified environments.

        Args:
            lin_vel: New linear velocities [n_envs, 3] or [len(env_indices), 3] (in relative coordinates if relative_to is provided)
            env_indices: Environment indices to update. None means all environments.
            relative_to: Optional reference frame - if provided, lin_vel is treated as relative coordinates
        """
        # Convert relative coordinates to absolute if needed
        if relative_to is not None:
            ref_frame = ReferenceFrames.from_input(relative_to, device=lin_vel.device)
            lin_vel = apply_relative_transform_inverse(lin_vel, ref_frame)

        # Call the raw implementation
        self._set_lin_vel_raw(lin_vel, env_indices=env_indices, **kwargs)

    def set_ang_vel(
        self,
        ang_vel: torch.Tensor,
        env_indices: Optional[torch.Tensor] = None,
        relative_to: ReferenceFramesInput = None,
        **kwargs,
    ) -> None:
        """Set angular velocity for specified environments.

        Args:
            ang_vel: New angular velocities [n_envs, 3] or [len(env_indices), 3] (in relative coordinates if relative_to is provided)
            env_indices: Environment indices to update. None means all environments.
            relative_to: Optional reference frame - if provided, ang_vel is treated as relative coordinates
        """
        # Convert relative coordinates to absolute if needed
        if relative_to is not None:
            ref_frame = ReferenceFrames.from_input(relative_to, device=ang_vel.device)
            ang_vel = apply_relative_transform_inverse(ang_vel, ref_frame)

        # Call the raw implementation
        self._set_ang_vel_raw(ang_vel, env_indices=env_indices, **kwargs)

    def control_pos(
        self,
        pos: torch.Tensor,
        env_indices: Optional[torch.Tensor] = None,
        relative_to: ReferenceFramesInput = None,
        **kwargs,
    ) -> None:

        # Convert relative coordinates to absolute if needed
        if relative_to is not None:
            ref_frame = ReferenceFrames.from_input(relative_to, device=pos.device)
            pos = apply_relative_transform_inverse(pos, ref_frame)

        # Call the raw implementation
        self._control_pos_raw(pos, env_indices=env_indices, **kwargs)

    def control_orientation(
        self,
        quat: torch.Tensor,
        env_indices: Optional[torch.Tensor] = None,
        relative_to: ReferenceFramesInput = None,
        **kwargs,
    ) -> None:
        # Convert relative coordinates to absolute if needed
        if relative_to is not None:
            ref_frame = ReferenceFrames.from_input(relative_to, device=quat.device)
            quat = apply_relative_transform_inverse(quat, ref_frame)

        # Call the raw implementation
        self._control_orientation_raw(quat, env_indices=env_indices, **kwargs)

    def control_lin_vel(
        self,
        lin_vel: torch.Tensor,
        env_indices: Optional[torch.Tensor] = None,
        relative_to: ReferenceFramesInput = None,
        **kwargs,
    ) -> None:

        # Convert relative coordinates to absolute if needed
        if relative_to is not None:
            ref_frame = ReferenceFrames.from_input(relative_to, device=lin_vel.device)
            lin_vel = apply_relative_transform_inverse(lin_vel, ref_frame)

        # Call the raw implementation
        self._control_lin_vel_raw(lin_vel, env_indices=env_indices, **kwargs)

    def control_ang_vel(
        self,
        ang_vel: torch.Tensor,
        env_indices: Optional[torch.Tensor] = None,
        relative_to: ReferenceFramesInput = None,
        **kwargs,
    ) -> None:
        # Convert relative coordinates to absolute if needed
        if relative_to is not None:
            ref_frame = ReferenceFrames.from_input(relative_to, device=ang_vel.device)
            ang_vel = apply_relative_transform_inverse(ang_vel, ref_frame)

        # Call the raw implementation
        self._control_ang_vel_raw(ang_vel, env_indices=env_indices, **kwargs)


class RobotInterface(EntityInterface):
    """Abstract interface for robot control and observation.

    This interface extends EntityInterface with robot-specific operations including:
    - Getting/setting joint state
    - Applying controls (position, velocity, force)
    - PD gain tuning
    - Utility methods for joint information

    All methods accept torch.Tensor inputs shaped [n_envs, ...] and optional
    joint_indices and env_indices to support vectorized environments.
    """

    # ==============================================
    # Raw State Getters (Robot Base - Inherited from EntityInterface)
    # ==============================================
    # The following methods are inherited from EntityInterface and provide
    # base state access. Robot implementations should implement the
    # _get_pos_raw, _get_orientation_raw, _get_lin_vel_raw, _get_ang_vel_raw
    # methods from EntityInterface.

    @abstractmethod
    def _get_joint_pos_raw(
        self, joint_indices: Optional[torch.Tensor] = None, env_indices: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """Get raw joint positions from simulator.

        Args:
            joint_indices: Indices of joints to query. None means all joints.
            env_indices: Environment indices to query. None means all environments.

        Returns:
            torch.Tensor: Joint positions [n_envs, n_joints] or [len(env_indices), len(joint_indices)]
        """
        pass

    @abstractmethod
    def _get_joint_vel_raw(
        self, joint_indices: Optional[torch.Tensor] = None, env_indices: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """Get raw joint velocities from simulator.

        Args:
            joint_indices: Indices of joints to query. None means all joints.
            env_indices: Environment indices to query. None means all environments.

        Returns:
            torch.Tensor: Joint velocities [n_envs, n_joints] or [len(env_indices), len(joint_indices)]
        """
        pass

    # ==============================================
    # Public State Getters (Base State - Inherited from EntityInterface)
    # ==============================================

    def get_joint_pos(
        self,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        relative_to: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Get joint positions for selected joints.

        Args:
            joint_indices: Indices of joints to query. None means all joints.
            env_indices: Environment indices to query. None means all environments.
            relative_to: Optional reference joint positions to subtract (simple offset)

        Returns:
            torch.Tensor: Joint positions [n_envs, n_joints] or [len(env_indices), len(joint_indices)]
        """
        pos = self._get_joint_pos_raw(joint_indices=joint_indices, env_indices=env_indices, **kwargs)

        # Apply simple offset if requested
        return apply_simple_offset(pos, relative_to)

    def get_joint_vel(
        self,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        relative_to: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Get joint velocities for selected joints.

        Args:
            joint_indices: Indices of joints to query. None means all joints.
            env_indices: Environment indices to query. None means all environments.
            relative_to: Optional reference joint velocities to subtract (simple offset)

        Returns:
            torch.Tensor: Joint velocities [n_envs, n_joints] or [len(env_indices), len(joint_indices)]
        """
        vel = self._get_joint_vel_raw(joint_indices=joint_indices, env_indices=env_indices, **kwargs)

        # Apply simple offset if requested
        return apply_simple_offset(vel, relative_to)

    # ==============================================
    # Public State Setters (Base State - Inherited from EntityInterface)
    # ==============================================
    # The following methods are inherited from EntityInterface:
    # - set_pos() -> set_base_pos()
    # - set_orientation() -> set_base_orientation()
    # - set_lin_vel() -> set_base_lin_vel()
    # - set_ang_vel() -> set_base_ang_vel()

    def set_joint_pos(
        self,
        pos: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        zero_velocity: bool = False,
        relative_to: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Set joint positions for specified joints and environments.

        Args:
            pos: New joint positions [n_envs, n_joints] or [len(env_indices), len(joint_indices)] (in relative coordinates if relative_to is provided)
            joint_indices: Joint indices to update. None means all joints.
            env_indices: Environment indices to update. None means all environments.
            zero_velocity: Whether to reset joint velocities after setting positions
            relative_to: Optional reference joint positions - if provided, pos is treated as relative coordinates
        """
        # Convert relative coordinates to absolute if needed
        pos = apply_simple_offset_inverse(pos, relative_to)

        # Call the raw implementation
        self._set_joint_pos_raw(
            pos, joint_indices=joint_indices, env_indices=env_indices, zero_velocity=zero_velocity, **kwargs
        )

    def set_joint_vel(
        self,
        vel: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        relative_to: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Set joint velocities for specified joints and environments.

        Args:
            vel: New joint velocities [n_envs, n_joints] or [len(env_indices), len(joint_indices)] (in relative coordinates if relative_to is provided)
            joint_indices: Joint indices to update. None means all joints.
            env_indices: Environment indices to update. None means all environments.
            relative_to: Optional reference joint velocities - if provided, vel is treated as relative coordinates
        """
        # Convert relative coordinates to absolute if needed
        vel = apply_simple_offset_inverse(vel, relative_to)

        # Call the raw implementation
        self._set_joint_vel_raw(vel, joint_indices=joint_indices, env_indices=env_indices, **kwargs)

    # ==============================================
    # Raw State Setters (Robot Base - Inherited from EntityInterface)
    # ==============================================
    # The following methods are inherited from EntityInterface and provide
    # base state setting. Robot implementations should implement the
    # _set_pos_raw, _set_orientation_raw, _set_lin_vel_raw, _set_ang_vel_raw
    # methods from EntityInterface.
    @abstractmethod
    def _set_joint_pos_raw(
        self,
        pos: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        zero_velocity: bool = False,
        **kwargs,
    ) -> None:
        """Set joint positions for specified joints and environments (raw implementation).

        Args:
            pos: New joint positions [n_envs, n_joints] or [len(env_indices), len(joint_indices)] (absolute coordinates)
            joint_indices: Joint indices to update. None means all joints.
            env_indices: Environment indices to update. None means all environments.
            zero_velocity: Whether to reset joint velocities after setting positions
        """
        pass

    @abstractmethod
    def _set_joint_vel_raw(
        self,
        vel: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Set joint velocities for specified joints and environments (raw implementation).

        Args:
            vel: New joint velocities [n_envs, n_joints] or [len(env_indices), len(joint_indices)] (absolute coordinates)
            joint_indices: Joint indices to update. None means all joints.
            env_indices: Environment indices to update. None means all environments.
        """
        pass

    # ==============================================
    # Public Control Methods (Unified Interface)
    # ==============================================

    def control_joint_pos(
        self,
        target_pos: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        relative_to: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Apply position control to specified joints.

        Args:
            target_pos: Target joint positions [n_envs, n_joints] (in relative coordinates if relative_to is provided)
            joint_indices: Joint indices to control. None means all joints.
            env_indices: Environment indices to control. None means all environments.
            relative_to: Optional reference joint positions - if provided, target_pos is treated as relative coordinates
        """
        # Convert relative coordinates to absolute if needed
        target_pos = apply_simple_offset_inverse(target_pos, relative_to)

        # Call the raw implementation
        self._control_joint_pos_raw(target_pos, joint_indices=joint_indices, env_indices=env_indices, **kwargs)

    def control_joint_vel(
        self,
        target_vel: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        relative_to: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Apply velocity control to specified joints.

        Args:
            target_vel: Target joint velocities [n_envs, n_joints] (in relative coordinates if relative_to is provided)
            joint_indices: Joint indices to control. None means all joints.
            env_indices: Environment indices to control. None means all environments.
            relative_to: Optional reference joint velocities - if provided, target_vel is treated as relative coordinates
        """
        # Convert relative coordinates to absolute if needed
        target_vel = apply_simple_offset_inverse(target_vel, relative_to)

        # Call the raw implementation
        self._control_joint_vel_raw(target_vel, joint_indices=joint_indices, env_indices=env_indices, **kwargs)

    def control_joint_force(
        self,
        forces: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        relative_to: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Apply force/torque control to specified joints.

        Args:
            forces: Target forces/torques [n_envs, n_joints] (absolute coordinates)
            joint_indices: Joint indices to control. None means all joints.
            env_indices: Environment indices to control. None means all environments.
        """
        target_force = apply_simple_offset_inverse(forces, relative_to)
        self._control_joint_force_raw(target_force, joint_indices=joint_indices, env_indices=env_indices, **kwargs)

    # ==============================================
    # Abstract Control Methods (Raw Implementation)
    # ==============================================

    @abstractmethod
    def _control_joint_pos_raw(
        self,
        target_pos: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Apply position control to specified joints (raw implementation).

        Args:
            target_pos: Target joint positions [n_envs, n_joints] (absolute coordinates)
            joint_indices: Joint indices to control. None means all joints.
            env_indices: Environment indices to control. None means all environments.
        """
        pass

    @abstractmethod
    def _control_joint_vel_raw(
        self,
        target_vel: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Apply velocity control to specified joints (raw implementation).

        Args:
            target_vel: Target joint velocities [n_envs, n_joints] (absolute coordinates)
            joint_indices: Joint indices to control. None means all joints.
            env_indices: Environment indices to control. None means all environments.
        """
        pass

    @abstractmethod
    def _control_joint_force_raw(
        self,
        forces: torch.Tensor,
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Apply force/torque control to specified joints (raw implementation).

        Args:
            forces: Target forces/torques [n_envs, n_joints] (absolute coordinates)
            joint_indices: Joint indices to control. None means all joints.
            env_indices: Environment indices to control. None means all environments.
        """
        pass

    # ==============================================
    # PD Control
    # ==============================================

    @abstractmethod
    def set_kp_gains(
        self,
        kp: Union[float, torch.Tensor],
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Set proportional gains for specified joints.

        Args:
            kp: Proportional gains [n_envs, n_joints] or scalar
            joint_indices: Joint indices to update. None means all joints.
            env_indices: Environment indices to update. None means all environments.
        """
        pass

    @abstractmethod
    def set_kd_gains(
        self,
        kd: Union[float, torch.Tensor],
        joint_indices: Optional[torch.Tensor] = None,
        env_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Set derivative gains for specified joints.

        Args:
            kd: Derivative gains [n_envs, n_joints] or scalar
            joint_indices: Joint indices to update. None means all joints.
            env_indices: Environment indices to update. None means all environments.
        """
        pass

    # ==============================================
    # Utility Methods
    # ==============================================

    # @abstractmethod
    # def get_joint_names(self) -> List[str]:
    #     """Get list of joint names.

    #     Returns:
    #         List[str]: Joint names in order
    #     """
    #     pass

    # @abstractmethod
    # def get_joint_indices(self, joint_names: List[str]) -> torch.Tensor:
    #     """Convert joint names to indices.

    #     Args:
    #         joint_names: List of joint names

    #     Returns:
    #         torch.Tensor: Joint indices [len(joint_names)]
    #     """
    #     pass

    # @abstractmethod
    @abstractmethod
    def get_joint(self, joint_name: str) -> Any:
        """Get joint object by name for accessing joint-specific properties.

        Args:
            joint_name: Name of the joint

        Returns:
            Any: Simulator-specific joint object
        """
        pass

    # @property
    # @abstractmethod
    # def dof_names(self) -> List[str]:
    #     """Get list of DOF names in order.

    #     Returns:
    #         List[str]: DOF names
    #     """
    #     pass

    # @abstractmethod
    # def get_dof_name_idx(self, name: str) -> int:
    #     """Get local DOF index by name.

    #     Args:
    #         name: DOF name

    #     Returns:
    #         int: Local DOF index
    #     """
    #     pass


class PrimitiveInterface(EntityInterface):
    """Interface for primitive shapes and objects.

    Provides the same state access as robots but without joint control.
    All primitive-specific operations are handled through the base EntityInterface.
    """

    def __init__(self, name: str, primitive_type: str):
        """Initialize primitive interface.

        Args:
            name: Unique identifier for this primitive
            primitive_type: Type of primitive (sphere, box, cylinder, mesh, plane)
        """
        super().__init__(name)
        self._primitive_type = primitive_type

    @property
    def primitive_type(self) -> str:
        """Get the type of primitive."""
        return self._primitive_type


class SceneInterface(ABC):
    """Abstract interface for simulation scene management.

    This interface manages the simulation world and provides common operations
    needed by the framework including:
    - Building environments and stepping simulation
    - Creating robots and cameras
    - Managing simulation parameters

    It hides simulator-specific details while providing access to underlying
    simulation objects for advanced users.
    """

    def __init__(self):
        """Initialize the scene interface.

        Note: Concrete implementations should store the underlying scene object
        as self._scene and implement properties for n_envs and device.
        """
        self._scene = None
        self._camera_manager = None
        self._debug_manager = None

    @property
    @abstractmethod
    def n_envs(self) -> int:
        """Number of parallel environments.

        Returns:
            int: Number of environments
        """
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Device used for computations.

        Returns:
            str: Device string ("cuda" or "cpu")
        """
        pass

    @abstractmethod
    def build_scene(self) -> None:
        """Finalize scene construction after all robots/bodies are added.

        This method must be called after adding all entities and before
        starting simulation or querying robot states.
        """
        pass

    @abstractmethod
    def step(self, n_steps: int = 1) -> None:
        """Advance the simulation by n_steps.

        Args:
            n_steps: Number of simulation steps to advance
        """
        pass

    @property
    @abstractmethod
    def dt(self) -> float:
        """Get simulation timestep.

        Returns:
            float: Simulation timestep in seconds
        """
        pass

    @property
    def simulation_frequency(self) -> float:
        """Return the simulation frequency (Hz).

        Returns:
            float: Simulation frequency in Hz
        """
        return 1.0 / self.dt

    @property
    @abstractmethod
    def sim_step(self) -> int:
        """Get current simulation step.

        Returns:
            int: Current simulation step
        """
        pass

    @property
    def sim_time(self) -> float:
        """Get current simulation time."""
        return self.sim_step * self.dt

    # ==============================================
    # Entity Management
    # ==============================================

    @property
    @abstractmethod
    def entities(self) -> Dict[str, EntityInterface]:
        """All entities including robots and primitives."""
        pass

    @property
    @abstractmethod
    def primitives(self) -> Dict[str, PrimitiveInterface]:
        """Only primitive entities."""
        pass

    @property
    @abstractmethod
    def robots(self) -> Dict[str, RobotInterface]:
        """Only robot entities."""
        pass

    # ==============================================
    # Primitive Creation Methods
    # ==============================================

    @abstractmethod
    def add_sphere(
        self, name: str, radius: float, pos: torch.Tensor, quat: Optional[torch.Tensor] = None, **kwargs
    ) -> PrimitiveInterface:
        """Add sphere primitive.

        Args:
            name: Unique name for the sphere
            radius: Sphere radius
            pos: Position tensor [n_envs, 3]
            quat: Optional orientation quaternion [n_envs, 4]
            **kwargs: Additional primitive-specific options

        Returns:
            PrimitiveInterface: Interface for the sphere
        """
        pass

    @abstractmethod
    def add_box(
        self, name: str, size: torch.Tensor, pos: torch.Tensor, quat: Optional[torch.Tensor] = None, **kwargs
    ) -> PrimitiveInterface:
        """Add box primitive.

        Args:
            name: Unique name for the box
            size: Box dimensions [x, y, z]
            pos: Position tensor [n_envs, 3]
            quat: Optional orientation quaternion [n_envs, 4]
            **kwargs: Additional primitive-specific options

        Returns:
            PrimitiveInterface: Interface for the box
        """
        pass

    @abstractmethod
    def add_cylinder(
        self, name: str, radius: float, height: float, pos: torch.Tensor, quat: Optional[torch.Tensor] = None, **kwargs
    ) -> PrimitiveInterface:
        """Add cylinder primitive.

        Args:
            name: Unique name for the cylinder
            radius: Cylinder radius
            height: Cylinder height
            pos: Position tensor [n_envs, 3]
            quat: Optional orientation quaternion [n_envs, 4]
            **kwargs: Additional primitive-specific options

        Returns:
            PrimitiveInterface: Interface for the cylinder
        """
        pass

    @abstractmethod
    def add_mesh(
        self,
        name: str,
        mesh_path: str,
        pos: torch.Tensor,
        quat: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> PrimitiveInterface:
        """Add mesh primitive.

        Args:
            name: Unique name for the mesh
            mesh_path: Path to mesh file
            pos: Position tensor [n_envs, 3]
            quat: Optional orientation quaternion [n_envs, 4]
            scale: Optional scale factors [x, y, z]
            **kwargs: Additional primitive-specific options

        Returns:
            PrimitiveInterface: Interface for the mesh
        """
        pass

    @abstractmethod
    def add_plane(self, name: str, **kwargs) -> PrimitiveInterface:
        """Add ground plane primitive.

        Args:
            name: Unique name for the plane
            **kwargs: Additional primitive-specific options

        Returns:
            PrimitiveInterface: Interface for the plane
        """
        pass

    # ==============================================
    # Robot Creation Methods
    # ==============================================
    @abstractmethod
    def add_robot(self, name: str, urdf_path: str, pos: torch.Tensor, quat: torch.Tensor, **kwargs) -> RobotInterface:
        """Add a robot to each environment and return a RobotInterface.

        Args:
            urdf_path: Path to the robot URDF file
            pos: Initial positions
            quat: Initial orientations as quaternions
            **kwargs: Additional robot configuration options

        Returns:
            RobotInterface: Interface for controlling the robot
        """
        pass

    # ==============================================
    # Low-Level Camera Methods (Abstract)
    # ==============================================

    @abstractmethod
    def create_camera(self, name: str, config: 'CameraConfig') -> None:
        """Create a camera in the simulation.

        Args:
            name: Unique name for the camera
            config: Camera configuration object
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def render_camera(self, name: str, rgb: bool = True) -> None:
        """Render a frame from the camera.

        Args:
            name: Camera name
            rgb: Whether to render RGB (vs depth/segmentation)
        """
        pass

    @abstractmethod
    def start_camera_recording(self, name: str) -> None:
        """Start recording from a camera.

        Args:
            name: Camera name
        """
        pass

    @abstractmethod
    def stop_camera_recording(self, name: str, save_path: str, fps: int) -> None:
        """Stop recording and save video.

        Args:
            name: Camera name
            save_path: Path to save video file
            fps: Frames per second for video
        """
        pass

    # ==============================================
    # High-Level Camera Management
    # ==============================================

    @property
    def camera_manager(self) -> Optional[CameraManager]:
        """Get camera manager instance."""
        self.setup_camera_manager()
        return getattr(self, '_camera_manager', None)

    def setup_camera_manager(self) -> None:
        """Setup camera manager for tracking and periodic recording."""
        if not hasattr(self, '_camera_manager') or self._camera_manager is None:
            self._camera_manager = CameraManager(self)

    def step_with_cameras(self, n_steps: int = 1, target_positions: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """Step simulation and update camera tracking.

        Args:
            n_steps: Number of simulation steps
            target_positions: Optional dict of target positions for tracking
        """
        self.step(n_steps)
        if self.camera_manager:
            self.camera_manager.step(target_positions)

    # ==============================================
    # Low Level Debug Visualization Methods (abstract)
    # ==============================================

    @abstractmethod
    def _draw_debug_arrow(self, name, pos, vec=(0, 0, 1), radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        """Simulator-specific arrow drawing implementation."""
        pass

    @abstractmethod
    def _draw_debug_line(self, name, start, end, radius=0.002, color=(1.0, 0.0, 0.0, 0.5)):
        """Simulator-specific line drawing implementation."""
        pass

    @abstractmethod
    def _draw_debug_frame(self, name, T, axis_length=1.0, origin_size=0.015, axis_radius=0.01):
        """Simulator-specific frame drawing implementation."""
        pass

    @abstractmethod
    def _draw_debug_mesh(self, name, mesh, pos, T=None):
        """Simulator-specific mesh drawing implementation."""
        pass

    @abstractmethod
    def _draw_debug_sphere(self, name, pos, radius=0.01, color=(1.0, 0.0, 0.0, 0.5)):
        """Simulator-specific sphere drawing implementation."""
        pass

    @abstractmethod
    def _draw_debug_box(self, name, bounds, color=(1.0, 0.0, 0.0, 1.0), wireframe=True, wireframe_radius=0.0015):
        """Simulator-specific box drawing implementation."""
        pass

    @abstractmethod
    def _draw_debug_points(self, name, poss, colors=(1.0, 0.0, 0.0, 0.5)):
        """Simulator-specific points drawing implementation."""
        pass

    @abstractmethod
    def _draw_debug_path(self, name, qposs, entity, link_idx=-1, density=0.3, frame_scaling=1.0):
        """Simulator-specific path drawing implementation."""
        pass

    @abstractmethod
    def _clear_debug_marks(self, names: Optional[List[str]] = None) -> None:
        """Simulator-specific clear all marks implementation."""
        pass

    # ==============================================
    # High Level Debug Visualization
    # ==============================================
    @property
    def debug_manager(self) -> Optional["DebugMarkManager"]:
        """Get debug manager if initialized."""
        self.setup_debug_manager()
        return self._debug_manager

    def setup_debug_manager(self) -> 'DebugMarkManager':
        """Setup debug visualization manager.

        Returns:
            DebugMarkManager: Debug manager instance
        """
        if self._debug_manager is None:
            self._debug_manager = DebugMarkManager(self)
        return self._debug_manager
