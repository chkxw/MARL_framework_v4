import inspect
from typing import Literal, Optional

import torch
from utils.geom import quat_to_xyz, xyz_to_quat, transform_by_quat, inv_quat
from utils.reference_frame import ReferenceFrames


def gs_rand_float(lower, upper, shape, device):
    """Random float generator helper function."""
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def check_boolean_tensor(termination_or_truncation):
    if isinstance(termination_or_truncation, torch.Tensor):
        if termination_or_truncation.dtype != torch.bool:
            raise ValueError(f"Expected boolean tensor, got tensor of{termination_or_truncation.dtype}")
        return termination_or_truncation
    else:
        raise ValueError(f"Expected boolean tensor, got {type(termination_or_truncation)}")


def wrap_normal_function_as_closure(name, user_func):
    if user_func is None:
        return None
    sig = inspect.signature(user_func)
    if len(sig.parameters) == 1:
        return user_func
    elif len(sig.parameters) == 2:

        def name_wrapper(aec_env):
            return user_func(name, aec_env)

        return name_wrapper
    else:
        raise ValueError(f"Expected function with 1 or 2 parameters, got {len(sig.parameters)}")


def wrap_reset_function_as_closure(name, user_func):
    if user_func is None:
        return None
    sig = inspect.signature(user_func)
    if len(sig.parameters) == 2:
        return user_func
    elif len(sig.parameters) == 3:

        def name_wrapper(aec_env, env_indices):
            return user_func(name, aec_env, env_indices)

        return name_wrapper
    else:
        raise ValueError(f"Expected function with 2 or 3 parameters, got {len(sig.parameters)}")


# def wrapped_as_closure(name, user_func, expected_args=1):
#     """
#     In case the user function is not a closure, provide the name of its caller object as a work around for "self"

#     But when use create the function, they actually have all information they need to get the "self" instance from AECEnv
#     """
#     if user_func is None:
#         return None
#     sig = inspect.signature(user_func)
#     if len(sig.parameters) == expected_args:
#         return user_func
#     else:

#         def name_wrapper(*args, **kwargs):
#             return user_func(name, *args, **kwargs)

#         return name_wrapper


def convert_orientation(
    orientation: torch.Tensor, 
    source_format: Literal["quat", "euler", "xyz", "rpy"] = "quat",
    target_format: Literal["quat", "euler", "xyz", "rpy"] = "quat"
) -> torch.Tensor:
    """Convert orientation between different formats.

    Args:
        orientation: Input orientation tensor 
                    [n_envs, 4] for quat, [n_envs, 3] for euler/xyz/rpy
        source_format: Source format ("quat", "euler", "xyz", "rpy")
        target_format: Target format ("quat", "euler", "xyz", "rpy")

    Returns:
        torch.Tensor: Converted orientation in target format
    """
    # If source and target are the same, return as-is
    if source_format == target_format:
        return orientation
    
    # Convert source to quaternion first (if not already)
    if source_format == "quat":
        quat = orientation
    elif source_format in ["euler", "xyz"]:
        quat = xyz_to_quat(orientation, rpy=False)  
    elif source_format == "rpy":
        quat = xyz_to_quat(orientation, rpy=True)
    else:
        raise ValueError(f"Unsupported source format: {source_format}")
    
    # Convert quaternion to target format
    if target_format == "quat":
        return quat
    elif target_format in ["euler", "xyz"]:
        return quat_to_xyz(quat, rpy=False)
    elif target_format == "rpy":
        return quat_to_xyz(quat, rpy=True)
    else:
        raise ValueError(f"Unsupported target format: {target_format}")


def apply_simple_offset(data: torch.Tensor, relative_to: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply simple offset subtraction for scalar data like joint positions/velocities.

    Args:
        data: Input data tensor
        relative_to: Optional reference values to subtract

    Returns:
        torch.Tensor: Data with offset applied (data - relative_to)
    """
    if relative_to is None:
        return data

    if not isinstance(relative_to, torch.Tensor):
        relative_to = torch.tensor(relative_to, device=data.device, dtype=data.dtype)

    return data - relative_to


def apply_simple_offset_inverse(data: torch.Tensor, relative_to: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply simple offset addition for converting relative to absolute coordinates.

    Args:
        data: Input data tensor (relative coordinates)
        relative_to: Optional reference values to add

    Returns:
        torch.Tensor: Data with offset applied (data + relative_to)
    """
    if relative_to is None:
        return data

    if not isinstance(relative_to, torch.Tensor):
        relative_to = torch.tensor(relative_to, device=data.device, dtype=data.dtype)

    return data + relative_to


def apply_relative_transform(data: torch.Tensor, reference: Optional[ReferenceFrames]) -> torch.Tensor:
    """Apply relative transformation to data using reference frame.

    Automatically detects data type based on last dimension:
    - Last dim 3: Position/velocity data - apply position offset and rotation
    - Last dim 4: Quaternion data - apply orientation transformation
    - Last dim 7: Position + quaternion data - apply both transformations

    Args:
        data: Input data tensor [..., 3], [..., 4], or [..., 7]
        reference: Reference frame for transformation (None = no transformation)

    Returns:
        torch.Tensor: Transformed data relative to reference frame
    """
    if reference is None:
        return data

    last_dim = data.shape[-1]

    if last_dim == 3:
        # Position or velocity data [n_envs, 3] - apply offset and rotation
        ref_pos = reference.position  # Already [n_envs, 3]
        offset_data = data - ref_pos

        # Apply rotation transformation
        inv_ref_quat = inv_quat(reference.orientation)  # Already [n_envs, 4]
        return transform_by_quat(offset_data, inv_ref_quat)

    elif last_dim == 4:
        # Quaternion data [n_envs, 4] - apply orientation transformation
        inv_ref_quat = inv_quat(reference.orientation)  # Already [n_envs, 4]
        return transform_by_quat(data, inv_ref_quat)

    elif last_dim == 7:
        # Position + quaternion data [n_envs, 7] - apply both transformations
        pos_data = data[..., :3]  # Extract position
        quat_data = data[..., 3:]  # Extract quaternion

        # Transform position
        transformed_pos = apply_relative_transform(pos_data, reference)
        # Transform quaternion
        transformed_quat = apply_relative_transform(quat_data, reference)

        # Concatenate back
        return torch.cat([transformed_pos, transformed_quat], dim=-1)

    else:
        raise ValueError(f"Unsupported data shape: {data.shape}. Last dimension must be 3, 4, or 7.")


def apply_relative_transform_inverse(data: torch.Tensor, reference: Optional[ReferenceFrames]) -> torch.Tensor:
    """Apply inverse relative transformation to convert from relative back to absolute coordinates.

    Automatically detects data type based on last dimension:
    - Last dim 3: Position/velocity data - apply rotation and position offset
    - Last dim 4: Quaternion data - apply orientation transformation
    - Last dim 7: Position + quaternion data - apply both transformations

    Args:
        data: Input data tensor [..., 3], [..., 4], or [..., 7] (relative coordinates)
        reference: Reference frame for transformation (None = no transformation)

    Returns:
        torch.Tensor: Transformed data in absolute coordinates
    """
    if reference is None:
        return data

    last_dim = data.shape[-1]

    if last_dim == 3:
        # Position or velocity data [n_envs, 3] - apply rotation then position offset
        # Apply inverse rotation transformation first
        ref_quat = reference.orientation  # Already [n_envs, 4]
        rotated_data = transform_by_quat(data, ref_quat)

        # Apply position offset
        ref_pos = reference.position  # Already [n_envs, 3]
        return rotated_data + ref_pos

    elif last_dim == 4:
        # Quaternion data [n_envs, 4] - apply orientation transformation
        ref_quat = reference.orientation  # Already [n_envs, 4]
        return transform_by_quat(data, ref_quat)

    elif last_dim == 7:
        # Position + quaternion data [n_envs, 7] - apply both transformations
        pos_data = data[..., :3]  # Extract position
        quat_data = data[..., 3:]  # Extract quaternion

        # Transform position and quaternion
        transformed_pos = apply_relative_transform_inverse(pos_data, reference)
        transformed_quat = apply_relative_transform_inverse(quat_data, reference)

        # Concatenate back
        return torch.cat([transformed_pos, transformed_quat], dim=-1)

    else:
        raise ValueError(f"Unsupported data shape: {data.shape}. Last dimension must be 3, 4, or 7.")
