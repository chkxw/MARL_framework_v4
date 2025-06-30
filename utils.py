import numpy as np
import torch
from scipy.spatial.transform import Rotation


class RefDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._refs = {}  # Store reference mappings

    def add_ref(self, ref_key, target_key):
        """Make ref_key always return the value of target_key."""
        self._refs[ref_key] = target_key

    def __getitem__(self, key):
        if key in self._refs:
            # Follow the reference
            return super().__getitem__(self._refs[key])
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key in self._refs:
            # Set value for the target key
            super().__setitem__(self._refs[key], value)
        else:
            super().__setitem__(key, value)

    def __contains__(self, key):
        return key in self._refs or super().__contains__(key)
        
    def keys(self):
        """Return all keys including reference keys."""
        return list(super().keys()) + list(self._refs.keys())
        
    def actual_keys(self):
        """Return keys that have actual storage (not references)."""
        return list(super().keys())
    
    def ref_keys(self):
        """Return keys that are references."""
        return list(self._refs.keys())


def inv_quat(quat):
    if isinstance(quat, torch.Tensor):
        scaling = torch.tensor([1, -1, -1, -1], device=quat.device)
        return quat * scaling
    elif isinstance(quat, np.ndarray):
        scaling = np.array([1, -1, -1, -1], dtype=quat.dtype)
        return quat * scaling
    else:
        raise TypeError("Unsupported type for quat: {}".format(type(quat)))


def quat_to_xyz(quat, rpy=False, degrees=False):
    if isinstance(quat, torch.Tensor):
        # Extract quaternion components
        qw, qx, qy, qz = quat.unbind(-1)

        # Roll (x-axis rotation)
        if rpy:
            sinr_cosp = 2 * (qw * qx + qy * qz)
        else:
            sinr_cosp = -2 * (qy * qz - qw * qx)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        if rpy:
            sinp = 2 * (qw * qy - qz * qx)
        else:
            sinp = 2 * (qx * qz + qw * qy)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * torch.tensor(torch.pi / 2),
            torch.asin(sinp),
        )

        # Yaw (z-axis rotation)
        if rpy:
            siny_cosp = 2 * (qw * qz + qx * qy)
        else:
            siny_cosp = -2 * (qx * qy - qw * qz)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        rpy = torch.stack([roll, pitch, yaw], dim=-1)
        if degrees:
            rpy *= 180.0 / torch.pi
        return rpy
    elif isinstance(quat, np.ndarray):
        rot = Rotation.from_quat(quat, scalar_first=True)
        if rpy:
            return rot.as_euler("xyz", degrees=degrees)
        return rot.as_euler("zyx", degrees=degrees)[::-1]
    else:
        raise TypeError("Unsupported type for quat: {}".format(type(quat)))


def transform_quat_by_quat(v, u):
    if isinstance(v, torch.Tensor) and isinstance(u, torch.Tensor):
        assert v.shape == u.shape, f"{v.shape} != {u.shape}"
        w1, x1, y1, z1 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        w2, x2, y2, z2 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)
        quat = torch.stack([w, x, y, z], dim=-1)
        return quat
    elif isinstance(v, np.ndarray) and isinstance(u, np.ndarray):
        assert v.shape == u.shape, f"{v.shape} != {u.shape}"
        w1, x1, y1, z1 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        w2, x2, y2, z2 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
        # This method transforms quat_v by quat_u
        # This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)
        quat = np.stack([w, x, y, z], axis=-1)
        return quat
    else:
        raise TypeError("Unsupported type for quat: {} and {}".format(type(v), type(u)))
