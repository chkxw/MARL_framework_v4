#!/usr/bin/env python3
"""
HARL Patcher for Custom Environment Support

This module provides runtime patching for HARL to support custom environments
without modifying the original HARL library files.
"""

import functools
from typing import Any, Dict

# Global registry for custom environments
_custom_env_registry: Dict[str, Any] = {}

# Store original functions for unpatching
_original_functions: Dict[str, Any] = {}


def register_custom_env(env_type: str, env_instance: Any):
    """Register a custom environment instance with the patcher.
    
    Args:
        env_type: String identifier for the environment type
        env_instance: Pre-built environment instance that implements HARL interface
    """
    _custom_env_registry[env_type] = env_instance


def patch_harl_envs():
    """Apply patches to HARL environment functions to support custom environments."""
    
    # Import HARL modules
    try:
        from harl.utils import envs_tools
        from harl.utils import configs_tools
        from harl.envs import LOGGER_REGISTRY
        from harl.common.base_logger import BaseLogger
    except ImportError as e:
        raise ImportError(f"Failed to import HARL modules for patching: {e}")
    
    # Create a custom logger class for custom environments
    class CustomLogger(BaseLogger):
        def get_task_name(self):
            return self.env_args.get("custom_type", "custom")
    
    # Save original functions for unpatching
    _original_functions['make_train_env'] = envs_tools.make_train_env
    _original_functions['get_task_name'] = configs_tools.get_task_name  
    _original_functions['get_num_agents'] = envs_tools.get_num_agents
    
    # Get references for local use
    original_make_train_env = _original_functions['make_train_env']
    original_get_task_name = _original_functions['get_task_name']
    original_get_num_agents = _original_functions['get_num_agents']
    
    def patched_make_train_env(env_name, seed, n_threads, env_args):
        if env_name == "custom":
            custom_type = env_args.get("custom_type", "default")
            if custom_type not in _custom_env_registry:
                raise ValueError(f"Custom environment type '{custom_type}' not registered")

            # Check if this is an evaluation call vs training call
            if env_args.get("is_eval", False):
                raise NotImplementedError("Custom environments don't support evaluation")

            # Return the environment only once
            return _custom_env_registry[custom_type]
        # Prevent using n_threads to create subproc envs
    
    def patched_get_task_name(env, env_args):
        """Patched version of get_task_name that supports custom environments."""
        if env == "custom":
            # Use custom_type as task name, fallback to "custom"
            return env_args.get("custom_type", "custom")
        else:
            # Use original function for standard environments
            return original_get_task_name(env, env_args)
    
    def patched_get_num_agents(env, env_args, envs):
        """Patched version of get_num_agents that supports custom environments."""
        if env == "custom":
            # Get agent count from registered environment
            custom_type = env_args.get("custom_type", "default")
            if custom_type not in _custom_env_registry:
                raise ValueError(f"Custom environment type '{custom_type}' not registered")
            
            custom_env = _custom_env_registry[custom_type]
            return custom_env.num_agents
        else:
            # Use original function for standard environments  
            return original_get_num_agents(env, env_args, envs)
    
    # Apply patches
    envs_tools.make_train_env = patched_make_train_env
    configs_tools.get_task_name = patched_get_task_name
    envs_tools.get_num_agents = patched_get_num_agents
    
    # Add custom logger to LOGGER_REGISTRY
    if "custom" not in LOGGER_REGISTRY:
        LOGGER_REGISTRY["custom"] = CustomLogger
        _original_functions['logger_registry_had_custom'] = False
    else:
        _original_functions['logger_registry_had_custom'] = True
        _original_functions['original_custom_logger'] = LOGGER_REGISTRY["custom"]
    
    print("✅ HARL patches applied successfully - custom environment support enabled")


def unpatch_harl_envs():
    """Remove patches from HARL environment functions (for cleanup)."""
    
    try:
        from harl.utils import envs_tools
        from harl.utils import configs_tools
        from harl.envs import LOGGER_REGISTRY
        
        # Restore original functions if they were saved
        if 'make_train_env' in _original_functions:
            envs_tools.make_train_env = _original_functions['make_train_env']
        if 'get_task_name' in _original_functions:
            configs_tools.get_task_name = _original_functions['get_task_name']
        if 'get_num_agents' in _original_functions:
            envs_tools.get_num_agents = _original_functions['get_num_agents']
        
        # Restore logger registry
        if 'logger_registry_had_custom' in _original_functions:
            if not _original_functions['logger_registry_had_custom']:
                # Remove custom logger if it wasn't there originally
                LOGGER_REGISTRY.pop("custom", None)
            else:
                # Restore original custom logger
                LOGGER_REGISTRY["custom"] = _original_functions['original_custom_logger']
            
        # Clear stored functions
        _original_functions.clear()
        
        print("✅ HARL patches removed successfully - original functions restored")
        
    except ImportError as e:
        print(f"⚠️ Failed to unpatch HARL: {e}")


def list_registered_envs():
    """List all registered custom environments."""
    return list(_custom_env_registry.keys())


def clear_registry():
    """Clear all registered custom environments."""
    global _custom_env_registry
    _custom_env_registry.clear()