#!/usr/bin/env python3
"""
Simulator Configuration System

This module provides a uniform configuration system that works across all simulators.
Users can either provide a generic SimulatorConfig or a custom setup function.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from configs.CameraConfig import CameraConfig

@dataclass
class SimulatorConfig:
    """Universal simulator configuration that works across all physics simulators.
    
    This dataclass provides a common interface for configuring different simulators.
    Each simulator interface will translate these settings to simulator-specific options.
    """
    simulator_type: str = "genesis"
    
    # === Core Simulation Parameters ===
    dt: float = 0.02  # Simulation timestep in seconds
    substeps: int = 4  # Number of substeps per simulation step
    
    # === Physics Parameters ===
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)  # Gravity vector
    enable_collision: bool = True  # Enable collision detection
    enable_joint_limit: bool = True  # Enable joint limits
    
    # === Rendering/Visualization ===
    show_viewer: bool = False  # Show GUI viewer
    max_fps: Optional[int] = None  # Maximum FPS for viewer (None = unlimited)
    rendered_envs_idx: List[int] = field(default_factory=lambda: [0])  # Which envs to render
    
    # === Viewer Camera Settings ===
    camera_config: CameraConfig = field(default_factory=CameraConfig)
    
    # === Solver Settings ===
    constraint_solver: str = "newton"  # Constraint solver type
    solver_iterations: int = 20  # Number of solver iterations
    
    # === Device Settings ===
    device: str = "cuda"
    
    # === Parallel Settings ===
    n_envs: int = 1  # Number of parallel environments

    # === Simulator-Specific Overrides ===
    simulator_specific: Dict[str, Any] = field(default_factory=dict)  # Simulator-specific options
    
    # === Custom Setup Function ===
    custom_setup_function: Optional[Callable] = None  # Custom scene setup function
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.dt <= 0:
            raise ValueError("Timestep (dt) must be positive")
        if self.substeps < 1:
            raise ValueError("Substeps must be at least 1")
        if self.max_fps is not None and self.max_fps <= 0:
            raise ValueError("Max FPS must be positive or None")
            
    @property
    def frequency(self) -> float:
        """Get frequency of this config."""
        return 1.0 / self.dt
        
    def merge_with_defaults(self, defaults: 'SimulatorConfig') -> 'SimulatorConfig':
        """Merge this config with default values, prioritizing this config's values.
        
        Args:
            defaults: Default configuration to use for missing values
            
        Returns:
            SimulatorConfig: Merged configuration
        """
        merged_dict = {}
        
        for field_info in fields(self):
            field_name = field_info.name
            current_value = getattr(self, field_name)
            default_value = getattr(defaults, field_name)
            
            # Special handling for simulator_specific (merge dicts)
            if field_name == 'simulator_specific':
                merged_dict[field_name] = {**default_value, **current_value}
            # Special handling for custom_setup_function (check for None)
            elif field_name == 'custom_setup_function':
                merged_dict[field_name] = current_value if current_value is not None else default_value
            # For all other fields, use current if different from default
            else:
                merged_dict[field_name] = current_value if current_value != default_value else default_value
        
        return SimulatorConfig(**merged_dict)


def create_default_config(
    dt: float = 0.02,
    show_viewer: bool = False,
    **kwargs
) -> SimulatorConfig:
    """Create a default simulator configuration.
    
    Args:
        dt: Simulation timestep
        show_viewer: Whether to show GUI viewer
        **kwargs: Additional configuration options
        
    Returns:
        SimulatorConfig: Default configuration
    """
    return SimulatorConfig(
        dt=dt,
        show_viewer=show_viewer,
        **kwargs
    )