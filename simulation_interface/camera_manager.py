#!/usr/bin/env python3
"""
Camera Manager for Genesis MARL Framework

This module provides high-level camera management functionality that is
simulator-agnostic. It manages multiple cameras by name and provides
unified APIs for camera tracking, recording, and periodic capture.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import torch
from marl_logging import get_class_logger

if TYPE_CHECKING:
    from configs.CameraConfig import CameraConfig
    from simulation_interface.base_interfaces import SceneInterface


@dataclass
class CameraState:
    """Internal state for a managed camera."""

    config: 'CameraConfig'
    recording_active: bool = False
    recording_start_time: float = 0.0
    last_recording_time: float = float('-inf')  # negative infinity
    last_camera_pos: Optional[Tuple[float, float, float]] = None


class CameraManager:
    """High-level camera manager for tracking and recording.

    This class provides a simulator-agnostic interface for managing multiple
    cameras, automatic tracking, and periodic recording based on simulation time.
    All interaction with actual camera objects is done through the scene interface
    using camera names.
    """

    def __init__(self, scene_interface: 'SceneInterface'):
        """Initialize camera manager.

        Args:
            scene_interface: The scene interface that provides low-level camera operations
        """
        self.scene = scene_interface
        self.camera_states: Dict[str, CameraState] = {}
        self.logger = get_class_logger(self.__class__.__name__, "main")

    @property
    def simulation_frequency(self) -> int:
        """Get simulation frequency from scene interface."""
        return int(self.scene.simulation_frequency)

    def add_camera(self, name: str, config: 'CameraConfig') -> None:
        """Add a camera to be managed.

        Args:
            name: Unique name for the camera
            config: Camera configuration
        """
        if name in self.camera_states:
            self.logger.warning(f"Camera '{name}' already exists, replacing it")
            self.remove_camera(name)

        # Create camera state
        state = CameraState(config=config)

        # Create simulator camera through scene interface
        self.scene.create_camera(name, config)
        self.camera_states[name] = state
        self.logger.info(f"Added camera '{name}' with resolution {config.res}")

    def remove_camera(self, name: str) -> None:
        """Remove a camera from management.

        Args:
            name: Name of camera to remove
        """
        if name not in self.camera_states:
            self.logger.warning(f"Camera '{name}' not found")
            return

        state = self.camera_states[name]

        # Stop recording if active
        if state.recording_active:
            self.stop_recording(name)

        # Remove from tracking
        del self.camera_states[name]
        self.logger.info(f"Removed camera '{name}' from management")

    def add_tracking_target(self, camera_name: str, target_name: str) -> None:
        """Add a tracking target for a camera.

        Args:
            camera_name: Name of camera
            target_name: Name of target to track (e.g., robot name)
        """
        if camera_name not in self.camera_states:
            self.logger.error(f"Camera '{camera_name}' not found")
            return

        # Note: tracking targets are stored in the camera config
        self.logger.debug(f"Registered tracking target '{target_name}' for camera '{camera_name}'")

    def update_tracking(self, camera_name: str, target_positions: Dict[str, torch.Tensor]) -> None:
        """Update camera position to track targets.

        Args:
            camera_name: Name of camera to update
            target_positions: Dict mapping target names to positions [3]
        """

        state = self.camera_states[camera_name]

        # Track all provided targets
        positions = list(target_positions.values())
        if not positions:
            center = torch.tensor([0.0, 0.0, 0.5], device=self.scene.device)
        else:
            center = torch.stack(positions).mean(dim=0)

        # Apply offset from config
        offset = state.config.offset
        center_np = center.cpu().numpy()
        camera_pos = (center_np[0] + offset[0], center_np[1] + offset[1], center_np[2] + offset[2])

        # Apply smoothing if enabled
        if state.config.smooth_tracking and state.last_camera_pos is not None:
            alpha = state.config.smoothing_factor
            camera_pos = tuple(alpha * new + (1 - alpha) * old for new, old in zip(camera_pos, state.last_camera_pos))

        state.last_camera_pos = camera_pos

        # Update camera pose through scene interface
        self.scene.set_camera_pose(camera_name, camera_pos, tuple(center_np), (0, 0, 1))

    def start_recording(self, camera_name: str) -> bool:
        """Start recording for a camera.

        Args:
            camera_name: Name of camera

        Returns:
            True if recording started successfully
        """
        if camera_name not in self.camera_states:
            self.logger.error(f"Camera '{camera_name}' not found")
            return False

        state = self.camera_states[camera_name]
        if state.recording_active:
            self.logger.warning(f"Camera '{camera_name}' is already recording")
            return False

        self.scene.start_camera_recording(camera_name)
        state.recording_active = True
        state.recording_start_time = self.scene.sim_time
        self.logger.info(f"Started recording on camera '{camera_name}'")
        return True

    def stop_recording(self, camera_name: str, save_path: Optional[str] = None) -> bool:
        """Stop recording for a camera.

        Args:
            camera_name: Name of camera
            save_path: Optional custom save path

        Returns:
            True if recording stopped successfully
        """
        if camera_name not in self.camera_states:
            self.logger.error(f"Camera '{camera_name}' not found")
            return False

        state = self.camera_states[camera_name]
        if not state.recording_active:
            self.logger.warning(f"Camera '{camera_name}' is not recording")
            return False

        # Generate save path if not provided
        if save_path is None:
            save_path = state.config.get_full_video_path(state.recording_start_time)

        fps = state.config.fps or self.simulation_frequency

        self.scene.stop_camera_recording(camera_name, save_path, fps)
        state.recording_active = False
        duration = self.scene.sim_time - state.recording_start_time
        self.logger.info(
            f"Stopped recording on camera '{camera_name}', duration: {duration:.1f}s, saved to: {save_path}"
        )
        return True

    def step(self) -> None:
        """Update all cameras - handle tracking and periodic recording.

        This should be called after each simulation step.
        Camera manager automatically gets robot positions from scene.
        """
        current_time = self.scene.sim_time

        # Get all robot positions automatically from scene
        target_positions = {}

        for camera_name, state in self.camera_states.items():
            # Update tracking if targets are configured
            for track_target in state.config.track_targets:
                robot_interface = self.scene.robots[track_target]
                pos = robot_interface.get_pos()[0]
                target_positions[track_target] = pos
                self.update_tracking(camera_name, target_positions)

            # Handle periodic recording
            if state.config.enable_recording:
                # Check if we should start recording
                if not state.recording_active:
                    time_since_last = current_time - state.last_recording_time
                    if time_since_last >= state.config.record_interval_s:
                        # Start recording at exact intervals
                        if abs(current_time % state.config.record_interval_s) < (1.0 / self.simulation_frequency):
                            self.start_recording(camera_name)

                # Check if we should stop recording
                elif state.recording_active:
                    recording_duration = current_time - state.recording_start_time
                    if recording_duration >= state.config.record_duration_s:
                        self.stop_recording(camera_name)
                        state.last_recording_time = current_time

            # Render frame if recording
            if state.recording_active:
                self.scene.render_camera(camera_name, rgb=True)

    @property
    def camera_names(self) -> List[str]:
        """Get list of managed camera names."""
        return list(self.camera_states.keys())

    def get_camera_config(self, camera_name: str) -> Optional['CameraConfig']:
        """Get configuration for a camera.

        Args:
            camera_name: Name of camera

        Returns:
            Camera configuration or None if not found
        """
        if camera_name in self.camera_states:
            return self.camera_states[camera_name].config
        return None

    def is_recording(self, camera_name: str) -> bool:
        """Check if a camera is currently recording.

        Args:
            camera_name: Name of camera

        Returns:
            True if camera is recording
        """
        if camera_name in self.camera_states:
            return self.camera_states[camera_name].recording_active
        return False

    @property
    def status(self) -> Dict[str, Any]:
        """Get status of all cameras.

        Returns:
            Dictionary with camera status information
        """
        status = {'num_cameras': len(self.camera_states), 'cameras': {}}

        for name, state in self.camera_states.items():
            camera_status = {
                'recording': state.recording_active,
                'resolution': state.config.res,
                'fps': state.config.fps or self.simulation_frequency,
                'track_targets': state.config.track_targets,
            }

            if state.recording_active:
                camera_status['recording_duration'] = self.scene.sim_time - state.recording_start_time

            status['cameras'][name] = camera_status

        return status
