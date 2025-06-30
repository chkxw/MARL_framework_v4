"""Joint detection utilities for Genesis robots.

Based on the approach used in generic_interactive_explorer.py
"""

import os
import logging
from typing import List, Tuple, Dict, Optional
import xml.etree.ElementTree as ET

from .genesis_logging import get_module_logger


logger = get_module_logger("joint_detector")


def discover_controllable_joints(robot, urdf_path: str) -> Tuple[List[str], List[int]]:
    """Discover controllable joints from a robot and URDF file.
    
    Args:
        robot: Genesis robot entity (after scene.build())
        urdf_path: Path to URDF file
    
    Returns:
        joint_names: List of controllable joint names
        motor_dof_indices: List of corresponding DOF indices
    """
    logger.debug(f"Discovering joints from URDF: {urdf_path}")
    
    joint_names = []
    motor_dof_indices = []
    
    if os.path.exists(urdf_path):
        try:
            # Parse URDF to find controllable joints (exclude fixed joints)
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            
            all_joints = root.findall('.//joint')
            controllable_count = 0
            fixed_count = 0
            
            # Get scene-wide DOF count for validation (not just this robot's DOFs)
            try:
                robot_dofs = robot.get_dofs_position()
                if robot_dofs.ndim > 1:
                    robot_dof_count = robot_dofs.shape[-1]
                else:
                    robot_dof_count = len(robot_dofs)
                
                # For multi-robot scenes, DOF indices are scene-wide
                # Try to get scene-wide DOF count from robot's scene
                try:
                    # Access the scene through the robot to get total DOF count
                    scene_dof_count = robot_dof_count * 10  # Conservative estimate for validation
                    max_dofs = scene_dof_count
                    logger.debug(f"  Robot has {robot_dof_count} DOFs, using scene-wide validation limit: {max_dofs}")
                except:
                    # Fallback to large number for multi-robot scenes
                    max_dofs = 1000  # Large fallback for multi-robot validation
                    logger.debug(f"  Robot has {robot_dof_count} DOFs, using large validation limit: {max_dofs}")
                    
            except Exception as e:
                logger.warning(f"Could not get robot DOF count: {e}")
                max_dofs = 1000  # Large fallback for multi-robot scenes
            
            for joint in all_joints:
                joint_name = joint.get('name')
                joint_type = joint.get('type', 'unknown')
                
                # Skip fixed joints - they are not controllable
                if joint_type == 'fixed':
                    fixed_count += 1
                    logger.debug(f"  Skipping fixed joint: {joint_name}")
                    continue
                
                # Only process movable joints
                if joint_name and joint_type in ['revolute', 'continuous', 'prismatic']:
                    try:
                        # Try to get the joint from robot
                        joint_obj = robot.get_joint(joint_name)
                        if hasattr(joint_obj, 'dof_start'):
                            dof_idx = joint_obj.dof_start
                            
                            # Validate DOF index is within range
                            if 0 <= dof_idx < max_dofs:
                                joint_names.append(joint_name)
                                motor_dof_indices.append(dof_idx)
                                controllable_count += 1
                                logger.debug(f"  Found controllable joint: {joint_name} ({joint_type}) -> DOF {dof_idx}")
                            else:
                                logger.debug(f"  Joint {joint_name} DOF index {dof_idx} out of range (max: {max_dofs})")
                    except Exception as e:
                        logger.debug(f"  Joint {joint_name} not accessible in robot: {e}")
                else:
                    logger.debug(f"  Skipping unsupported joint type: {joint_name} ({joint_type})")
            
            logger.info(f"URDF analysis: {fixed_count} fixed joints (skipped), {controllable_count} controllable joints (detected)")
            
        except Exception as e:
            logger.warning(f"Could not parse URDF: {e}, falling back to direct DOF detection")
            return _fallback_joint_discovery(robot)
    else:
        logger.warning(f"URDF file not found: {urdf_path}, falling back to direct DOF detection")
        return _fallback_joint_discovery(robot)
    
    return joint_names, motor_dof_indices


def _fallback_joint_discovery(robot) -> Tuple[List[str], List[int]]:
    """Fallback joint discovery by examining robot DOFs directly."""
    logger.info("Using fallback DOF-based joint discovery...")
    
    try:
        # Get all DOF positions to determine how many DOFs we have
        all_positions = robot.get_dofs_position()
        if hasattr(all_positions, 'shape'):
            # Multi-environment tensor - get single environment
            if all_positions.ndim > 1:
                num_dofs = all_positions.shape[-1]
            else:
                num_dofs = len(all_positions)
        else:
            num_dofs = len(all_positions)
        
        # Create generic joint names and DOF indices
        joint_names = [f"joint_{i}" for i in range(num_dofs)]
        motor_dof_indices = list(range(num_dofs))
        
        logger.info(f"Fallback discovered {num_dofs} DOFs")
        return joint_names, motor_dof_indices
        
    except Exception as e:
        logger.error(f"Fallback joint discovery failed: {e}")
        return [], []


def get_joint_names_from_config(robot, config) -> Tuple[List[str], List[int]]:
    """Get joint names and DOF indices, with auto-detection if needed.
    
    Args:
        robot: Genesis robot entity (after scene.build())
        config: RobotConfig with optional joint_names
    
    Returns:
        joint_names: List of joint names
        motor_dof_indices: List of DOF indices
    """
    if config.joint_names:
        # Use provided joint names
        logger.debug(f"Using provided joint names: {config.joint_names}")
        joint_names = config.joint_names
        motor_dof_indices = []
        
        for name in joint_names:
            try:
                joint = robot.get_joint(name)
                motor_dof_indices.append(joint.dof_start)
            except Exception as e:
                logger.error(f"Joint {name} not found: {e}")
                motor_dof_indices.append(-1)
        
        return joint_names, motor_dof_indices
    
    else:
        # Auto-detect joints
        logger.debug("Auto-detecting joints...")
        return discover_controllable_joints(robot, config.urdf_path)