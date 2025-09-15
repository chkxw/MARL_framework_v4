from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from .geom import xyz_to_quat


ReferenceFramesInput = Union[None, torch.Tensor, List, Tuple, 'ReferenceFrames']


@dataclass
class ReferenceFrames:
    """Reference frames for relative transformations.

    Defines both position and orientation references for computing relative states.
    All transformations use these as the reference coordinate systems.
    Handles batched data for multiple environments.
    """

    position: torch.Tensor  # [n_envs, 3] - reference positions
    orientation: torch.Tensor  # [n_envs, 4] - reference orientations (quaternions)

    @classmethod
    def from_input(cls, data: ReferenceFramesInput, device: str = None) -> Optional['ReferenceFrames']:
        """Create ReferenceFrames from flexible input formats.

        Creates reference frames with positions [n_envs, 3] and orientations [n_envs, 4].
        Always assumes the first dimension is n_envs (number of environments).

        Supported formats:
        - None: returns None (no reference frames)
        - ReferenceFrames: returns as-is
        - [n_envs, 3]: position array -> positions only, orientations = identity quats
        - [n_envs, 4]: quaternion array -> quaternions only, positions = zeros
        - [n_envs, 6]: position + euler array -> positions + euler angles
        - [n_envs, 7]: position + quaternion array -> positions + quaternions
        - [n_envs, 2, 3]: position + euler reshaped

        Args:
            data: Input data in various formats
            device: Device for tensor creation

        Returns:
            ReferenceFrames or None if input is None
        """
        if data is None:
            return None

        if isinstance(data, cls):
            return data

        # Convert to tensor if needed
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32, device=device)
        else:
            if device is None:
                device = data.device
            data = data.to(device=device, dtype=torch.float32)

        # Handle different input shapes - always assume first dim is n_envs
        if data.dim() == 2:
            return cls._from_2d_tensor(data, device)
        elif data.dim() == 3:
            return cls._from_3d_tensor(data, device)
        else:
            raise ValueError(f"Unsupported input shape: {data.shape}. Expected 2D or 3D tensor with first dimension as n_envs.")

    @classmethod
    def _from_2d_tensor(cls, data: torch.Tensor, device: str) -> 'ReferenceFrames':
        """Handle 2D tensor inputs [n_envs, ...]."""
        n_envs = data.shape[0]
        
        if data.shape[1] == 3:
            # [n_envs, 3] -> positions only
            position = data
            orientation = torch.zeros((n_envs, 4), device=device)
            orientation[:, 0] = 1.0  # identity quaternions [qw=1, qx=0, qy=0, qz=0]
        elif data.shape[1] == 4:
            # [n_envs, 4] -> quaternions only
            position = torch.zeros((n_envs, 3), device=device)
            orientation = data
        elif data.shape[1] == 6:
            # [n_envs, 6] -> positions + euler angles
            position = data[:, :3]
            euler = data[:, 3:]
            orientation = xyz_to_quat(euler)
        elif data.shape[1] == 7:
            # [n_envs, 7] -> positions + quaternions
            position = data[:, :3]
            orientation = data[:, 3:]
        else:
            raise ValueError(f"Unsupported 2D tensor shape: {data.shape}. Second dimension must be 3, 4, 6, or 7.")

        return cls(position=position, orientation=orientation)

    @classmethod
    def _from_3d_tensor(cls, data: torch.Tensor, device: str) -> 'ReferenceFrames':
        """Handle 3D tensor inputs [n_envs, 2, 3]."""
        if data.shape[1] == 2 and data.shape[2] == 3:
            # [n_envs, 2, 3] -> positions + euler angles reshaped
            position = data[:, 0, :]  # [n_envs, 3]
            euler = data[:, 1, :]  # [n_envs, 3]
            # Ensure tensors are on the correct device
            position = position.to(device=device)
            euler = euler.to(device=device)
            orientation = xyz_to_quat(euler)
            return cls(position=position, orientation=orientation)
        else:
            raise ValueError(f"Unsupported 3D tensor shape: {data.shape}. Expected [n_envs, 2, 3] for positions + euler angles.")