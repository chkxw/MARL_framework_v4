"""
Geometry utility functions for quaternions, transformations, and rotations.

This module provides pure numpy/torch implementations without any
simulator-specific dependencies. Extracted and cleaned from Genesis geom.py.
"""

import math
import numpy as np
import numba as nb
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

# Constants
EPS = 1e-12

# ------------------------------------------------------------------------------------
# -------------------------------- torch and numpy -----------------------------------
# ------------------------------------------------------------------------------------


def inv_quat(quat):
    if isinstance(quat, torch.Tensor):
        _quat = quat.clone()
        _quat[..., 1:].neg_()
    elif isinstance(quat, np.ndarray):
        _quat = quat.copy()
        _quat[..., 1:] *= -1
    else:
        raise TypeError(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")
    return _quat


def inv_T(T):
    if isinstance(T, torch.Tensor):
        T_inv = torch.zeros_like(T)
    elif isinstance(T, np.ndarray):
        T_inv = np.zeros_like(T)
    else:
        raise TypeError(f"the input must be torch.Tensor or np.ndarray. got: {type(T)=}")

    trans, R = T[..., :3, 3], T[..., :3, :3]
    T_inv[..., 3, 3] = 1.0
    T_inv[..., :3, 3] = -R.T @ trans
    T_inv[..., :3, :3] = R.T

    return T_inv


def normalize(x, eps: float = 1e-12):
    if isinstance(x, torch.Tensor):
        return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)
    elif isinstance(x, np.ndarray):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)
    else:
        raise TypeError(f"the input must be either torch.Tensor or np.ndarray. got: {type(x)=}")


def rot6d_to_R(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] http://arxiv.org/abs/1812.07035
    """
    if isinstance(d6, torch.Tensor):
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)
    elif isinstance(d6, np.ndarray):
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
        dot = np.sum(b1 * a2, axis=-1, keepdims=True)
        b2 = a2 - dot * b1
        b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
        b3 = np.cross(b1, b2, axis=-1)
        return np.stack((b1, b2, b3), axis=-2)
    else:
        raise TypeError(f"the input must be either torch.Tensor or np.ndarray. got: {type(d6)=}")


def R_to_rot6d(R):
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        R: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] http://arxiv.org/abs/1812.07035
    """
    if isinstance(R, torch.Tensor):
        return R[..., :2, :].flatten(start_dim=-2)
    elif isinstance(R, np.ndarray):
        return R[..., :2, :].reshape((*R.shape[:-2], 6))
    else:
        raise TypeError(f"the input must be either torch.Tensor or np.ndarray. got: {type(R)=}")


@nb.jit(nopython=True, cache=True)
def _np_xyz_to_quat(xyz: np.ndarray, rpy: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the (qw, qx, qy, qz) Quaternion representation of a single or a
    batch of Yaw-Pitch-Roll Euler angles.

    :param xyz: N-dimensional array whose first dimension gathers the 3
                Yaw-Pitch-Roll Euler angles [Roll, Pitch, Yaw].
    :param out: Pre-allocated array in which to store the result. If not
                provided, a new array is freshly-allocated and returned, which
                is slower.
    """
    assert xyz.ndim >= 1
    if out is None:
        out_ = np.empty((*xyz.shape[:-1], 4))
    else:
        assert out.shape == (*xyz.shape[:-1], 4)
        out_ = out

    rpy2 = 0.5 * xyz
    roll2, pitch2, yaw2 = rpy2[..., 0], rpy2[..., 1], rpy2[..., 2]
    cosr, sinr = np.cos(roll2), np.sin(roll2)
    cosp, sinp = np.cos(pitch2), np.sin(pitch2)
    cosy, siny = np.cos(yaw2), np.sin(yaw2)
    sign = 1.0 if rpy else -1.0

    out_[..., 0] = cosr * cosp * cosy + sign * sinr * sinp * siny
    out_[..., 1] = sinr * cosp * cosy - sign * cosr * sinp * siny
    out_[..., 2] = cosr * sinp * cosy + sign * sinr * cosp * siny
    out_[..., 3] = cosr * cosp * siny - sign * sinr * sinp * cosy

    return out_


def _tc_xyz_to_quat(xyz: torch.Tensor, rpy: bool = False, *, out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None:
        out = torch.empty((*xyz.shape[:-1], 4), dtype=xyz.dtype, device=xyz.device)

    roll2, pitch2, yaw2 = (0.5 * xyz).unbind(-1)
    cosr, sinr = roll2.cos(), roll2.sin()
    cosp, sinp = pitch2.cos(), pitch2.sin()
    cosy, siny = yaw2.cos(), yaw2.sin()
    sign = 1.0 if rpy else -1.0

    out[..., 0] = cosr * cosp * cosy + sign * sinr * sinp * siny
    out[..., 1] = sinr * cosp * cosy - sign * cosr * sinp * siny
    out[..., 2] = cosr * sinp * cosy + sign * sinr * cosp * siny
    out[..., 3] = cosr * cosp * siny - sign * sinr * sinp * cosy

    return out


def xyz_to_quat(xyz, rpy=False, degrees=False):
    if degrees:
        xyz = xyz * (math.pi / 180.0)

    if isinstance(xyz, torch.Tensor):
        return _tc_xyz_to_quat(xyz, rpy)
    elif isinstance(xyz, np.ndarray):
        return _np_xyz_to_quat(xyz, rpy)
    else:
        raise TypeError(f"the input must be either torch.Tensor or np.ndarray. got: {type(xyz)=}")


@nb.jit(nopython=True, cache=True)
def _np_quat_to_R(quat: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the Rotation Matrix representation of a single or a batch of quaternions.

    :param quat: N-dimensional array whose last dimension gathers the 4 quaternion coordinates (qw, qx, qy, qz).
    :param out: Pre-allocated array in which to store the result. If not provided, a new array is freshly-allocated and
                returned, which is slower.
    """
    assert quat.ndim >= 1
    if out is None:
        out_ = np.empty((*quat.shape[:-1], 3, 3), dtype=quat.dtype)
    else:
        assert out.shape == (*quat.shape[:-1], 3, 3)
        out_ = out

    s = 2.0 / np.sum(np.square(quat), -1)
    q_w, q_x, q_y, q_z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    q_xs, q_ys, q_zs = q_x * s, q_y * s, q_z * s
    q_wx, q_wy, q_wz = q_w * q_xs, q_w * q_ys, q_w * q_zs
    q_xx, q_xy, q_xz = q_x * q_xs, q_x * q_ys, q_x * q_zs
    q_yy, q_yz = q_y * q_ys, q_y * q_zs
    q_zz = q_z * q_zs

    out_[..., 0, 0] = 1.0 - (q_yy + q_zz)
    out_[..., 0, 1] = q_xy - q_wz
    out_[..., 0, 2] = q_xz + q_wy
    out_[..., 1, 0] = q_xy + q_wz
    out_[..., 1, 1] = 1.0 - (q_xx + q_zz)
    out_[..., 1, 2] = q_yz - q_wx
    out_[..., 2, 0] = q_xz - q_wy
    out_[..., 2, 1] = q_yz + q_wx
    out_[..., 2, 2] = 1.0 - (q_xx + q_yy)

    return out_


def _tc_quat_to_R(quat, out=None):
    if out is None:
        out = torch.empty((*quat.shape[:-1], 3, 3), dtype=quat.dtype, device=quat.device)

    q_w, q_x, q_y, q_z = torch.tensor_split(quat, 4, dim=-1)

    s = 2.0 / (quat**2).sum(dim=-1, keepdim=True)
    q_vec_s = s * quat[..., 1:]
    q_wx, q_wy, q_wz = torch.unbind(q_w * q_vec_s, -1)
    q_xx, q_xy, q_xz = torch.unbind(q_x * q_vec_s, -1)
    q_yy, q_yz = torch.unbind(q_y * q_vec_s[..., 1:], -1)
    q_zz = q_z[..., 0] * q_vec_s[..., -1]

    out[..., 0, 0] = 1.0 - (q_yy + q_zz)
    out[..., 0, 1] = q_xy - q_wz
    out[..., 0, 2] = q_xz + q_wy
    out[..., 1, 0] = q_xy + q_wz
    out[..., 1, 1] = 1.0 - (q_xx + q_zz)
    out[..., 1, 2] = q_yz - q_wx
    out[..., 2, 0] = q_xz - q_wy
    out[..., 2, 1] = q_yz + q_wx
    out[..., 2, 2] = 1.0 - (q_xx + q_yy)

    return out


def quat_to_R(quat, *, out=None):
    # NOTE: Ignore zero-norm quaternion for efficiency

    if all(isinstance(e, torch.Tensor) for e in (quat, out) if e is not None):
        return _tc_quat_to_R(quat, out=out)
    elif all(isinstance(e, np.ndarray) for e in (quat, out) if e is not None):
        return _np_quat_to_R(quat, out=out)
    else:
        raise TypeError(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")


@nb.jit(nopython=True, cache=True)
def _np_quat_to_xyz(quat, rpy=False, out=None):
    """Compute the Yaw-Pitch-Roll Euler angles representation of a single or a batch of quaternions.

    The Roll, Pitch and Yaw angles are guaranteed to be within range [-pi,pi], [-pi/2,pi/2], [-pi,pi], respectively.

    :param quat: N-dimensional array whose last dimension gathers the 4 quaternion coordinates (qw, qx, qy, qz).
    :param out: Pre-allocated array in which to store the result. If not provided, a new array is freshly-allocated
                and returned, which is slower.
    """
    assert quat.ndim >= 1
    if out is None:
        out_ = np.empty((*quat.shape[:-1], 3), dtype=quat.dtype)
    else:
        assert out.shape == (*quat.shape[:-1], 3)
        out_ = out

    s = 2.0 / np.sum(np.square(quat), -1)
    q_w, q_x, q_y, q_z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    q_xs, q_ys, q_zs = q_x * s, q_y * s, q_z * s
    q_wx, q_wy, q_wz = q_w * q_xs, q_w * q_ys, q_w * q_zs
    q_xx, q_xy, q_xz = q_x * q_xs, q_x * q_ys, q_x * q_zs
    q_yy, q_yz, q_zz = q_y * q_ys, q_y * q_zs, q_z * q_zs

    if rpy:
        sinr_cosp = q_wx + q_yz
    else:
        sinr_cosp = q_wx - q_yz
    out_[..., 0] = np.arctan2(sinr_cosp, 1.0 - (q_xx + q_yy))

    if rpy:
        sinp = q_wy - q_xz
    else:
        sinp = q_xz + q_wy
    out_[..., 1] = -0.5 * math.pi + 2.0 * np.arctan2(np.sqrt(1.0 + sinp), np.sqrt(1.0 - sinp))

    if rpy:
        siny_cosp = q_wz + q_xy
    else:
        siny_cosp = q_wz - q_xy
    out_[..., 2] = np.arctan2(siny_cosp, 1.0 - (q_yy + q_zz))

    return out_


def _tc_quat_to_xyz(quat, rpy=False, out=None):
    if out is None:
        out = torch.empty((*quat.shape[:-1], 3), dtype=quat.dtype, device=quat.device)

    # Extract quaternion components
    q_w, q_x, q_y, q_z = torch.tensor_split(quat, 4, dim=-1)

    s = 2.0 / (quat**2).sum(dim=-1, keepdim=True)
    q_vec_s = s * quat[..., 1:]
    q_wx, q_wy, q_wz = torch.unbind(q_w * q_vec_s, -1)
    q_xx, q_xy, q_xz = torch.unbind(q_x * q_vec_s, -1)
    q_yy, q_yz = torch.unbind(q_y * q_vec_s[..., 1:], -1)
    q_zz = q_z[..., 0] * q_vec_s[..., 2]

    # Roll (x-axis rotation)
    if rpy:
        sinr_cosp = q_wx + q_yz
    else:
        sinr_cosp = q_wx - q_yz
    cosr_cosp = 1.0 - (q_xx + q_yy)
    out[..., 0] = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    if rpy:
        sinp = q_wy - q_xz
    else:
        sinp = q_xz + q_wy
    out[..., 1] = -0.5 * math.pi + 2.0 * torch.atan2(torch.sqrt(1.0 + sinp), torch.sqrt(1.0 - sinp))

    # Yaw (z-axis rotation)
    if rpy:
        siny_cosp = q_wz + q_xy
    else:
        siny_cosp = q_wz - q_xy
    cosy_cosp = 1.0 - (q_yy + q_zz)
    out[..., 2] = torch.atan2(siny_cosp, cosy_cosp)

    return out


def quat_to_xyz(quat, rpy=False, degrees=False):
    if isinstance(quat, torch.Tensor):
        rpy_result = _tc_quat_to_xyz(quat, rpy)
    elif isinstance(quat, np.ndarray):
        rpy_result = _np_quat_to_xyz(quat, rpy)
    else:
        raise TypeError(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")
    if degrees:
        rpy_result *= 180.0 / math.pi
    return rpy_result


@nb.jit(nopython=True, cache=True)
def _np_R_to_quat(R, out=None):
    assert R.ndim >= 2

    if out is None:
        out_ = np.empty((*R.shape[:-2], 4), dtype=R.dtype)
    else:
        assert out.shape == (*R.shape[:-2], 4)
        out_ = out

    for i in np.ndindex(R.shape[:-2]):
        R_i = R[i]
        quat_i = out_[i]

        if R_i[2, 2] < 0.0:
            if R_i[0, 0] > R_i[1, 1]:
                t = 1.0 + R_i[0, 0] - R_i[1, 1] - R_i[2, 2]
                quat_i[0] = R_i[2, 1] - R_i[1, 2]
                quat_i[1] = t
                quat_i[2] = R_i[1, 0] + R_i[0, 1]
                quat_i[3] = R_i[0, 2] + R_i[2, 0]
            else:
                t = 1.0 - R_i[0, 0] + R_i[1, 1] - R_i[2, 2]
                quat_i[0] = R_i[0, 2] - R_i[2, 0]
                quat_i[1] = R_i[1, 0] + R_i[0, 1]
                quat_i[2] = t
                quat_i[3] = R_i[2, 1] + R_i[1, 2]
        else:
            if R_i[0, 0] < -R_i[1, 1]:
                t = 1.0 - R_i[0, 0] - R_i[1, 1] + R_i[2, 2]
                quat_i[0] = R_i[1, 0] - R_i[0, 1]
                quat_i[1] = R_i[0, 2] + R_i[2, 0]
                quat_i[2] = R_i[2, 1] + R_i[1, 2]
                quat_i[3] = t
            else:
                t = 1.0 + R_i[0, 0] + R_i[1, 1] + R_i[2, 2]
                quat_i[0] = t
                quat_i[1] = R_i[2, 1] - R_i[1, 2]
                quat_i[2] = R_i[0, 2] - R_i[2, 0]
                quat_i[3] = R_i[1, 0] - R_i[0, 1]
        quat_i /= 2.0 * np.sqrt(t)

    return out_


def _tc_R_to_quat(R, out=None):
    if out is None:
        out = torch.zeros((*R.shape[:-2], 4), dtype=R.dtype, device=R.device)

    # Flattening batch dimensions because multi-dimensional masking is acting weird
    out_ = out.reshape((-1, 4))
    R = R.reshape((-1, 3, 3))

    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    trace = diag.sum(-1)

    # Compute quaternion based on the trace of the matrix
    mask1 = trace > 0.0
    mask2 = ~mask1 & (diag[:, 0] >= diag[:, 1]) & (diag[:, 0] >= diag[:, 2])
    mask3 = ~mask1 & ~mask2 & (diag[:, 1] >= diag[:, 2])
    mask4 = ~mask1 & ~mask2 & ~mask3

    S = 2.0 * torch.sqrt(trace[mask1] + 1.0)
    out_[mask1, 0] = 0.25 * S
    out_[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / S
    out_[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / S
    out_[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / S

    S = 2.0 * torch.sqrt(1.0 + diag[mask2, 0] - diag[mask2, 1] - diag[mask2, 2])
    out_[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / S
    out_[mask2, 1] = 0.25 * S
    out_[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / S
    out_[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / S

    S = 2.0 * torch.sqrt(1.0 + diag[mask3, 1] - diag[mask3, 0] - diag[mask3, 2])
    out_[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / S
    out_[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / S
    out_[mask3, 2] = 0.25 * S
    out_[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / S

    S = 2.0 * torch.sqrt(1.0 + diag[mask4, 2] - diag[mask4, 0] - diag[mask4, 1])
    out_[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / S
    out_[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / S
    out_[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / S
    out_[mask4, 3] = 0.25 * S

    return out


def R_to_quat(R, *, out=None):
    if all(isinstance(e, torch.Tensor) for e in (R, out) if e is not None):
        return _tc_R_to_quat(R, out=out)
    elif all(isinstance(e, np.ndarray) for e in (R, out) if e is not None):
        return _np_R_to_quat(R, out=out)
    else:
        raise TypeError(f"the input must be either torch.Tensor or np.ndarray. got: {type(R)=}")


def trans_R_to_T(trans=None, R=None, *, out=None):
    is_torch = all(isinstance(e, torch.Tensor) for e in (trans, R) if e is not None)
    is_numpy = not is_torch and all(isinstance(e, np.ndarray) for e in (trans, R) if e is not None)

    T = out
    B = () if trans is None and R is None else R.shape[:-2] if trans is None else trans.shape[:-1]
    dtype = R.dtype if trans is None else trans.dtype
    if is_torch:
        if T is None:
            device = R.device if trans is None else trans.device
            T = torch.zeros((*B, 4, 4), dtype=dtype, device=device)
    elif is_numpy:
        if T is None:
            T = np.zeros((*B, 4, 4), dtype=dtype)
    else:
        raise TypeError(f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(trans)=} and {type(R)=}")
    if B:
        assert T.shape == (*B, 4, 4)

    T[..., 3, 3] = 1.0
    if trans is not None:
        T[..., :3, 3] = trans
    if R is None:
        if is_torch:
            torch.diagonal(T, dim1=-2, dim2=-1).fill_(1.0)
        else:
            T[..., [0, 1, 2], [0, 1, 2]] = 1.0
    else:
        T[..., :3, :3] = R

    return T


def R_to_T(R, *, out=None):
    return trans_R_to_T(None, R, out=out)


def trans_to_T(trans, *, out=None):
    return trans_R_to_T(trans, None, out=out)


def trans_quat_to_T(trans=None, quat=None, *, out=None):
    is_torch = all(isinstance(e, torch.Tensor) for e in (trans, quat) if e is not None)
    is_numpy = not is_torch and all(isinstance(e, np.ndarray) for e in (trans, quat) if e is not None)

    T = out
    B = () if trans is None and quat is None else quat.shape[:-2] if trans is None else trans.shape[:-1]
    if is_torch:
        if T is None:
            T = torch.zeros((*B, 4, 4), dtype=trans.dtype, device=trans.device)
    elif is_numpy:
        if T is None:
            T = np.zeros((*B, 4, 4), dtype=trans.dtype)
    else:
        raise TypeError(f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(trans)=} and {type(quat)=}")
    if B:
        assert T.shape == (*B, 4, 4)

    T[..., 3, 3] = 1.0
    if trans is not None:
        T[..., :3, 3] = trans
    if quat is not None:
        quat_to_R(quat, out=T[..., :3, :3])

    return T


@nb.jit(nopython=True, cache=True)
def _np_quat_mul(u, v, out=None):
    if out is None:
        out_ = np.empty(u.shape, dtype=u.dtype)
    else:
        assert out.shape == u.shape
        out_ = out

    w1, x1, y1, z1 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    w2, x2, y2, z2 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))

    out_[..., 0] = qq - ww + (z1 - y1) * (y2 - z2)
    out_[..., 1] = qq - xx + (x1 + w1) * (x2 + w2)
    out_[..., 2] = qq - yy + (w1 - x1) * (y2 + z2)
    out_[..., 3] = qq - zz + (z1 + y1) * (w2 - x2)

    return out_


def _tc_quat_mul(u, v, out=None):
    if out is None:
        out = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    w1, x1, y1, z1 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    w2, x2, y2, z2 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))

    out[..., 0] = qq - ww + (z1 - y1) * (y2 - z2)
    out[..., 1] = qq - xx + (x1 + w1) * (x2 + w2)
    out[..., 2] = qq - yy + (w1 - x1) * (y2 + z2)
    out[..., 3] = qq - zz + (z1 + y1) * (w2 - x2)

    return out


def transform_quat_by_quat(v, u):
    """
    This method transforms quat_v by quat_u.

    This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
    """
    assert u.shape == v.shape
    assert u.ndim >= 1

    if all(isinstance(e, torch.Tensor) for e in (u, v)):
        quat = _tc_quat_mul(u, v)
    elif all(isinstance(e, np.ndarray) for e in (u, v)):
        quat = _np_quat_mul(u, v, out=None)
    else:
        raise TypeError(f"The inputs must all be torch.Tensor or np.ndarray. got: {type(v)=} and {type(u)=}")

    return normalize(quat)


@nb.jit(nopython=True, cache=True)
def _np_transform_by_quat(v, quat, out=None):
    if out is None:
        out_ = np.empty(v.shape, dtype=v.dtype)
    else:
        assert out.shape == v.shape
        out_ = out

    v_T, quat_T, out_T = v.T, quat.T, out_.T
    v_x, v_y, v_z = v_T
    q_ww, q_wx, q_wy, q_wz = quat_T[0] * quat_T
    q_xx, q_xy, q_xz = quat_T[1] * quat_T[1:]
    q_yy, q_yz = quat_T[2] * quat_T[2:]
    q_zz = quat_T[3] * quat_T[3]

    out_T[0] = v_x * (q_xx + q_ww - q_yy - q_zz) + v_y * (2.0 * q_xy - 2.0 * q_wz) + v_z * (2.0 * q_xz + 2.0 * q_wy)
    out_T[1] = v_x * (2.0 * q_wz + 2.0 * q_xy) + v_y * (q_ww - q_xx + q_yy - q_zz) + v_z * (2.0 * q_yz - 2.0 * q_wx)
    out_T[2] = v_x * (2.0 * q_xz - 2.0 * q_wy) + v_y * (2.0 * q_wx + 2.0 * q_yz) + v_z * (q_ww - q_xx - q_yy + q_zz)

    out_T /= q_ww + q_xx + q_yy + q_zz

    return out_


def _tc_transform_by_quat(v, quat, out=None):
    if out is None:
        out = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    v_x, v_y, v_z = torch.unbind(v, dim=-1)
    q_w, q_x, q_y, q_z = torch.tensor_split(quat, 4, dim=-1)
    q_ww, q_wx, q_wy, q_wz = torch.unbind(q_w * quat, -1)
    q_xx, q_xy, q_xz = torch.unbind(q_x * quat[..., 1:], -1)
    q_yy, q_yz = torch.unbind(q_y * quat[..., 2:], -1)
    q_zz = q_z[..., 0] * quat[..., 3]

    out[..., 0] = v_x * (q_xx + q_ww - q_yy - q_zz) + v_y * (2.0 * q_xy - 2.0 * q_wz) + v_z * (2.0 * q_xz + 2.0 * q_wy)
    out[..., 1] = v_x * (2.0 * q_wz + 2.0 * q_xy) + v_y * (q_ww - q_xx + q_yy - q_zz) + v_z * (2.0 * q_yz - 2.0 * q_wx)
    out[..., 2] = v_x * (2.0 * q_xz - 2.0 * q_wy) + v_y * (2.0 * q_wx + 2.0 * q_yz) + v_z * (q_ww - q_xx - q_yy + q_zz)

    out /= (q_ww + q_xx + q_yy + q_zz)[..., None]

    return out


def transform_by_quat(v, quat):
    """
    This method transforms quat_v by quat_u.

    This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
    """
    assert v.ndim >= 1 and quat.ndim >= 1

    if all(isinstance(e, torch.Tensor) for e in (v, quat)):
        return _tc_transform_by_quat(v, quat)
    elif all(isinstance(e, np.ndarray) for e in (v, quat)):
        return _np_transform_by_quat(v, quat, out=None)
    else:
        raise TypeError(f"The inputs must all be torch.Tensor or np.ndarray. got: {type(v)=} and {type(quat)=}")


def axis_angle_to_quat(angle, axis):
    if isinstance(angle, torch.Tensor) and isinstance(axis, torch.Tensor):
        theta = (0.5 * angle).unsqueeze(-1)
        xyz = normalize(axis) * theta.sin()
        w = theta.cos()
        return normalize(torch.cat([w, xyz], dim=-1))
    elif isinstance(angle, np.ndarray) and isinstance(axis, np.ndarray):
        theta = (0.5 * angle)[..., None]
        xyz = normalize(axis) * np.sin(theta)
        w = np.cos(theta)
        return normalize(np.concatenate([w, xyz], axis=-1))
    else:
        raise TypeError(f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(angle)=} and {type(axis)=}")


def transform_by_xyz(pos, xyz):
    return transform_by_quat(pos, xyz_to_quat(xyz))


def transform_by_trans_quat(pos, trans, quat):
    return transform_by_quat(pos, quat) + trans


def inv_transform_by_quat(pos, quat):
    return transform_by_quat(pos, inv_quat(quat))


def inv_transform_by_trans_quat(pos, trans, quat):
    return inv_transform_by_quat(pos - trans, quat)


def transform_pos_quat_by_trans_quat(pos, quat, t_trans, t_quat):
    new_pos = t_trans + transform_by_quat(pos, t_quat)
    new_quat = transform_quat_by_quat(quat, t_quat)
    return new_pos, new_quat


def transform_by_R(pos, R):
    """
    Transforms 3D points by a 3x3 rotation matrix or a batch of matrices, supporting both NumPy arrays and PyTorch
    tensors.

    Parameters
    ----------
    pos: np.ndarray | torch.Tensor
        A numpy array or torch tensor of 3D points. Can be a single point (3,), a batch of points (B, 3), or a batched
        batch of points (B, N, 3).
    R: np.ndarray | torch.Tensor
        The 3x3 rotation matrix or a batch of B rotation matrices of shape (B, 3, 3). Must be of the same type as `pos`.

    Returns
    -------
        The transformed points in a shape corresponding to the input dimensions.
    """
    assert pos.shape[-1] == 3 and R.shape[-2:] == (3, 3)
    assert R.ndim - pos.ndim in (0, 1)

    B = R.shape[:-2]
    N = pos.shape[-2] if pos.ndim == R.ndim else 1
    R = R.reshape((-1, 3, 3))
    pos_ = pos.reshape((math.prod(B), N, 3))

    if all(isinstance(e, torch.Tensor) for e in (pos, R) if e is not None):
        new_pos = torch.bmm(R, pos_.swapaxes(1, 2)).swapaxes(1, 2)
    elif all(isinstance(e, np.ndarray) for e in (pos, R) if e is not None):
        new_pos = np.matmul(R, pos_.swapaxes(1, 2)).swapaxes(1, 2)
    else:
        raise TypeError(f"both of the inputs must be torch.Tensor or np.ndarray. got: {type(pos)=} and {type(R)=}")

    new_pos = new_pos.reshape(pos.shape)

    return new_pos


def transform_by_trans_R(pos, trans, R):
    assert trans.shape[:-1] == R.shape[:-2]

    B = trans.shape[:-1]
    if trans.ndim < pos.ndim:
        trans = trans[..., None, :]

    new_pos = transform_by_R(pos, R)
    new_pos += trans

    return new_pos


def transform_by_T(pos, T):
    """
    Transforms 3D points by a 4x4 transformation matrix or a batch of matrices, supporting both NumPy arrays and
    PyTorch tensors.

    Parameters
    ----------
    pos: np.ndarray | torch.Tensor
        A numpy array or torch tensor of 3D points. Can be a single point (3,), a batch of points (B, 3), or a
        batched batch of points (B, N, 3).
    T: np.ndarray | torch.Tensor
        The 4x4 transformation matrix or a batch of B transformation matrices of shape (B, 4, 4). Must be of the
        same type as `pos`.

    Returns
    -------
        The transformed points in a shape corresponding to the input dimensions.
    """
    return transform_by_trans_R(pos, T[..., :3, 3], T[..., :3, :3])


def inv_transform_by_T(pos, T):
    trans, R = T[..., :3, 3], T[..., :3, :3]

    R_inv = R.swapaxes(-1, -2)
    if pos.ndim == T.ndim:
        trans = trans.reshape((-1, 1, 3))

    return transform_by_R(pos - trans, R_inv)


# ------------------------------------------------------------------------------------
# ------------------------------------- numpy ----------------------------------------
# ------------------------------------------------------------------------------------


def scale_to_T(scale):
    T = np.eye(4, dtype=scale.dtype)
    T[[0, 1, 2], [0, 1, 2]] = scale
    return T


@nb.jit(nopython=True, cache=True)
def z_up_to_R(z, up=None, out=None):
    B = z.shape[:-1]
    if out is None:
        out_ = np.empty((*B, 3, 3), dtype=z.dtype)
    else:
        assert out.shape == (*B, 3, 3)
        out_ = out

    z_norm = np.sqrt(np.sum(np.square(z.reshape((-1, 3))), -1)).reshape(B)

    out_[..., 2] = z
    for i in np.ndindex(B):
        z_norm_i = z_norm[i]
        R = out_[i]
        x, y, z = R.T

        if z_norm_i > EPS:
            z /= z_norm_i
        else:
            z[:] = 0.0, 1.0, 0.0

        if up is not None:
            x[:] = np.cross(up, z)
        else:
            x[0] = z[1]
            x[1] = -z[0]
            x[2] = 0.0
        x_norm = np.linalg.norm(x)
        if x_norm > EPS:
            x /= x_norm
            y[:] = np.cross(z, x)
        else:
            R[:] = np.eye(3, dtype=R.dtype)

    return out_


def pos_lookat_up_to_T(pos, lookat, up, *, dtype=np.float32):
    pos = np.asarray(pos, dtype=dtype)
    lookat = np.asarray(lookat, dtype=dtype)
    up = np.asarray(up, dtype=dtype)

    T = np.zeros((4, 4), dtype=dtype)
    T[3, 3] = 1.0
    T[:3, 3] = pos

    z = pos - lookat
    z_norm = np.linalg.norm(z)
    if z_norm < EPS:
        z = np.array([0.0, 1.0, 0.0], dtype=dtype)
    z_up_to_R(z, up=up, out=T[:3, :3])

    return T


def T_to_pos_lookat_up(T):
    pos = T[:3, 3]
    lookat = T[:3, 3] - T[:3, 2]
    up = T[:3, 1]
    return pos, lookat, up


def euler_to_quat(euler_xyz):
    return xyz_to_quat(np.asarray(euler_xyz), rpy=True, degrees=True)


@nb.jit(nopython=True, cache=True)
def _np_euler_to_R(rpy: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Compute the Rotation Matrix representation of a single or a batch of Yaw-Pitch-Roll Euler angles.

    :param rpy: N-dimensional array whose last dimension gathers the 3 Yaw-Pitch-Roll Euler angles [Roll, Pitch, Yaw].
    :param out: Pre-allocated array in which to store the result. If not provided, a new array is freshly-allocated and
                returned, which is slower.
    """
    assert rpy.ndim >= 1
    if out is None:
        out_ = np.empty((*rpy.shape[1:], 3, 3), dtype=rpy.dtype)
    else:
        assert out.shape == (*rpy.shape[1:], 3, 3)
        out_ = out

    cos_rpy, sin_rpy = np.cos(rpy), np.sin(rpy)
    cos_roll, cos_pitch, cos_yaw = cos_rpy[..., 0], cos_rpy[..., 1], cos_rpy[..., 2]
    sin_roll, sin_pitch, sin_yaw = sin_rpy[..., 0], sin_rpy[..., 1], sin_rpy[..., 2]

    out_[..., 0, 0] = cos_pitch * cos_yaw
    out_[..., 0, 1] = -cos_roll * sin_yaw + sin_roll * sin_pitch * cos_yaw
    out_[..., 0, 2] = sin_roll * sin_yaw + cos_roll * sin_pitch * cos_yaw
    out_[..., 1, 0] = cos_pitch * sin_yaw
    out_[..., 1, 1] = cos_roll * cos_yaw + sin_roll * sin_pitch * sin_yaw
    out_[..., 1, 2] = -sin_roll * cos_yaw + cos_roll * sin_pitch * sin_yaw
    out_[..., 2, 0] = -sin_pitch
    out_[..., 2, 1] = sin_roll * cos_pitch
    out_[..., 2, 2] = cos_roll * cos_pitch

    return out_


def euler_to_R(euler_xyz):
    return _np_euler_to_R(np.asarray(euler_xyz) * (math.pi / 180.0))


def slerp(q0, q1, t):
    """
    Perform spherical linear interpolation between two quaternions.

    Parameters:
    q0 : numpy.array
        The start quaternion (4-dimensional vector).
    q1 : numpy.array
        The end quaternion (4-dimensional vector).
    t : float
        The interpolation parameter between 0 and 1.

    Returns:
    numpy.array
        The interpolated quaternion (4-dimensional vector).
    """
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot_product = np.dot(q0, q1)

    if dot_product < 0.0:
        q1 = -q1
        dot_product = -dot_product

    dot_product = np.clip(dot_product, -1.0, 1.0)

    theta_0 = np.arccos(dot_product)
    sin_theta_0 = np.sin(theta_0)

    if sin_theta_0 < 1e-6:
        return (1.0 - t) * q0 + t * q1

    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot_product * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return s0 * q0 + s1 * q1


def random_quaternion(batch_size):
    # Generate three uniform random numbers for each quaternion in the batch
    u1, u2, u3 = np.random.rand(3, batch_size)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.stack((q1, q2, q3, q4), axis=1)


# ------------------------------------------------------------------------------------
# ------------------------------------- misc ----------------------------------------
# ------------------------------------------------------------------------------------


def zero_pos():
    return np.zeros(3, dtype=np.float32)


def identity_quat():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def tc_zero_pos():
    return torch.zeros(3, dtype=torch.float32)


def tc_identity_quat():
    return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)


def nowhere():
    # let's inject a bit of humor here
    return np.array([2333333, 6666666, 5201314], dtype=np.float32)


def default_solver_params():
    """
    Default solver parameters (timeconst, dampratio, dmin, dmax, width, mid, power).

    Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
    """
    return np.array([0.0, 1.0e00, 9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0e00])


def default_friction():
    return 1.0


def default_dofs_kp(n=6):
    return np.full((n,), fill_value=100.0, dtype=np.float32)


def default_dofs_kv(n=6):
    return np.full((n,), fill_value=10.0, dtype=np.float32)
