"""
Simulation Interface Package

This package provides abstract interfaces for different physics simulators,
enabling the Genesis MARL Framework to work with multiple simulation backends.

Supported simulators:
- Genesis (fully implemented)
- MuJoCo (skeleton implementation)
- Isaac Gym (skeleton implementation)
"""

from .base_interfaces import RobotInterface, SceneInterface

__all__ = ["SceneInterface", "RobotInterface"]