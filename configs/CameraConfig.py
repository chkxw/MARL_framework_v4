#!/usr/bin/env python3
"""
Camera Configuration for Genesis MARL Framework

This module defines the CameraConfig class for configuring camera tracking
and periodic recording in simulation environments.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
import time


@dataclass
class CameraConfig:
    """Configuration for camera tracking and periodic recording.

    This class defines all parameters needed for automatic camera tracking
    and time-based periodic recording during simulation.
    """

    # Camera visual settings
    res: Tuple[int, int] = (1280, 720)  # Camera resolution (width, height)
    fov: float = 45.0  # Vertical field of view in degrees
    offset: Tuple[float, float, float] = (3.0, 3.0, 2.0)  # Camera offset from tracked position (x, y, z)
    lookat: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Look-at target (x, y, z), may change during tracking
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)  # Up vector (x, y, z)

    # Recording settings
    enable_recording: bool = False  # Whether to enable video recording
    record_interval_s: float = 60.0  # Time between recording sessions (seconds)
    record_duration_s: float = 10.0  # Duration of each recording session (seconds)
    fps: Optional[int] = None  # Recording framerate (None = use simulation frequency)

    # File naming and storage
    video_prefix: str = "training"  # Prefix for video filenames
    save_directory: str = "."  # Directory to save videos
    timestamp_format: str = "%Y%m%d_%H%M%S"  # Timestamp format for filenames

    # Tracking behavior
    track_targets: List[Any] = field(
        default_factory=list
    )  # List of targets to track
    smooth_tracking: bool = True  # Whether to smooth camera movement
    smoothing_factor: float = 0.1  # Smoothing factor for camera movement (0.0 = no smoothing, 1.0 = instant)

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.res[0] <= 0 or self.res[1] <= 0:
            raise ValueError("Camera resolution must be positive")

        if self.fov <= 0 or self.fov >= 180:
            raise ValueError("Camera FOV must be between 0 and 180 degrees")

        if self.record_interval_s <= 0:
            raise ValueError("Recording interval must be positive")

        if self.record_duration_s <= 0:
            raise ValueError("Recording duration must be positive")

        if self.record_duration_s >= self.record_interval_s:
            raise ValueError("Recording duration must be less than recording interval")

        if not (0.0 <= self.smoothing_factor <= 1.0):
            raise ValueError("Smoothing factor must be between 0.0 and 1.0")

    def get_video_filename(self, timestamp: Optional[float] = None) -> str:
        """Generate a video filename with timestamp.

        Args:
            timestamp: Unix timestamp. If None, uses current time.

        Returns:
            Generated filename with timestamp
        """
        if timestamp is None:
            timestamp = time.time()

        time_str = time.strftime(self.timestamp_format, time.localtime(timestamp))
        return f"{self.video_prefix}_{time_str}.mp4"

    def get_full_video_path(self, timestamp: Optional[float] = None) -> str:
        """Get full path for video file.

        Args:
            timestamp: Unix timestamp. If None, uses current time.

        Returns:
            Full path to video file
        """
        import os

        filename = self.get_video_filename(timestamp)
        return os.path.join(self.save_directory, filename)
