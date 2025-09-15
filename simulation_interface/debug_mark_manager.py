#!/usr/bin/env python3
"""
Debug Mark Manager for Simulation Visualization

This module provides a centralized interface for managing debug visualization
marks in the simulation, including arrows, lines, points, spheres, and text.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .base_interfaces import SceneInterface


@dataclass
class DebugMark:
    """Represents a single debug visualization mark."""

    name: str
    mark_type: str  # arrow, line, point, sphere, text
    tag: Optional[str] = None  # Optional tag for grouping marks
    data: Dict[str, Any] = None  # Mark-specific data


class DebugMarkManager(ABC):
    """Abstract base class for debug visualization management.

    This class provides a unified interface for creating and managing debug
    visualization marks across different simulator backends.
    """

    def __init__(self, scene: 'SceneInterface'):
        """Initialize the debug mark manager.

        Args:
            scene: The scene interface to use for visualization
        """
        self._scene = scene
        self._marks_counter: Dict[str, int] = {
            "arrow": 0,
            "line": 0,
            "point": 0,
            "sphere": 0,
            "text": 0,
            "frame": 0,
            "frames": 0,
            "mesh": 0,
            "spheres": 0,
            "box": 0,
            "points": 0,
            "path": 0,
        }
        self._marks: Dict[str, DebugMark] = {}
        self._marks_by_tag: Dict[str, List[DebugMark]] = {}

    # ==============================================
    # Mark Creation Methods
    # ==============================================
    def draw_debug_line(self, start, end, radius=0.002, color=(1.0, 0.0, 0.0, 0.5), name=None, tag=None):
        if name is None:
            name = self._auto_fill_name("line")

        self._add_mark(
            DebugMark(name, "line", tag=tag, data={"start": start, "end": end, "radius": radius, "color": color})
        )
        self._scene._draw_debug_line(name,**self._marks[name].data)

    def draw_debug_arrow(self, pos, vec=(0, 0, 1), radius=0.01, color=(1.0, 0.0, 0.0, 0.5), name=None, tag=None):
        if name is None:
            name = self._auto_fill_name("arrow")

        self._add_mark(
            DebugMark(name, "arrow", tag=tag, data={"pos": pos, "vec": vec, "radius": radius, "color": color})
        )
        self._scene._draw_debug_arrow(name,**self._marks[name].data)

    def draw_debug_frame(self, T, axis_length=1.0, origin_size=0.015, axis_radius=0.01, name=None, tag=None):
        if name is None:
            name = self._auto_fill_name("frame")

        self._add_mark(
            DebugMark(
                name,
                "frame",
                tag=tag,
                data={"T": T, "axis_length": axis_length, "origin_size": origin_size, "axis_radius": axis_radius},
            )
        )
        self._scene._draw_debug_frame(name,**self._marks[name].data)

    def draw_debug_mesh(self, mesh, pos, T=None, name=None, tag=None):
        if name is None:
            name = self._auto_fill_name("mesh")

        self._add_mark(DebugMark(name, "mesh", tag=tag, data={"mesh": mesh, "pos": pos, "T": T}))
        self._scene._draw_debug_mesh(name,**self._marks[name].data)

    def draw_debug_sphere(self, pos, radius=0.01, color=(1.0, 0.0, 0.0, 0.5), name=None, tag=None):
        if name is None:
            name = self._auto_fill_name("sphere")

        self._add_mark(DebugMark(name, "sphere", tag=tag, data={"pos": pos, "radius": radius, "color": color}))
        self._scene._draw_debug_sphere(name,**self._marks[name].data)

    def draw_debug_box(
        self, bounds, color=(1.0, 0.0, 0.0, 1.0), wireframe=True, wireframe_radius=0.0015, name=None, tag=None
    ):
        if name is None:
            name = self._auto_fill_name("box")

        self._add_mark(
            DebugMark(
                name,
                "box",
                tag=tag,
                data={"bounds": bounds, "color": color, "wireframe": wireframe, "wireframe_radius": wireframe_radius},
            )
        )
        self._scene._draw_debug_box(name,**self._marks[name].data)

    def draw_debug_points(self, poss, colors=(1.0, 0.0, 0.0, 0.5), name=None, tag=None):
        if name is None:
            name = self._auto_fill_name("points")

        self._add_mark(DebugMark(name, "points", tag=tag, data={"poss": poss, "colors": colors}))
        self._scene._draw_debug_points(name,**self._marks[name].data)

    def draw_debug_path(self, qposs, entity, link_idx=-1, density=0.3, frame_scaling=1.0, name=None, tag=None):
        if name is None:
            name = self._auto_fill_name("path")

        self._add_mark(
            DebugMark(
                name,
                "path",
                tag=tag,
                data={
                    "qposs": qposs,
                    "entity": entity,
                    "link_idx": link_idx,
                    "density": density,
                    "frame_scaling": frame_scaling,
                },
            )
        )
        self._scene._draw_debug_path(name,**self._marks[name].data)

    # ==============================================
    # Mark Management Methods
    # ==============================================

    def clear(self, names: Optional[List[str]] = None, tag: Optional[str] = None) -> None:
        """Clear all debug marks."""
        victimes = []
        if names is None and tag is None:
            victimes = [mark.name for mark in self._marks]

        if names is not None:
            victimes.extend(names)

        if tag is not None and tag in self._marks_by_tag:
            victimes.extend([mark.name for mark in self._marks_by_tag[tag]])

        # Remove from internal tracking
        for name in victimes:
            if self._marks[name].tag in self._marks_by_tag:
                self._marks_by_tag[self._marks[name].tag].remove(self._marks[name])
            del self._marks[name]

        self._scene._clear_debug_marks(victimes)

    # ==============================================
    # Helper Methods
    # ==============================================

    def _add_mark(self, mark: DebugMark) -> None:
        """Add a mark to internal tracking.

        Args:
            mark: Mark to add
        """
        self._marks[mark.name] = mark
        if mark.tag:
            if mark.tag not in self._marks_by_tag:
                self._marks_by_tag[mark.tag] = []
            self._marks_by_tag[mark.tag].append(mark)

    def _auto_fill_name(self, mark_type: str) -> str:
        name = f"{mark_type}_{self._marks_counter[mark_type]}"
        self._marks_counter[mark_type] += 1
        return name
