#!/usr/bin/env python3
"""
MARL Framework Logging Utilities

This module provides consistent logging for the MARL VecEnv Framework:
[MARL] [HH:MM:SS] [LEVEL] message

With ANSI color codes and professional styling.
"""

import logging
import sys
from typing import Dict, List, Optional


class MARLFormatter(logging.Formatter):
    """Custom formatter for MARL framework logging."""

    def __init__(self, verbose_time: bool = False, theme: str = "dark"):
        super().__init__()

        self.theme = theme

        # Color mapping (matches Genesis colors exactly)
        if theme == "dark":
            self.mapping = {
                logging.DEBUG: "\x1b[38;5;119m",  # GREEN
                logging.INFO: "\x1b[38;5;159m",  # BLUE
                logging.WARNING: "\x1b[38;5;226m",  # YELLOW
                logging.ERROR: "\x1b[38;5;9m",  # RED
                logging.CRITICAL: "\x1b[38;5;9m",  # RED
            }
        elif theme == "light":
            self.mapping = {
                logging.DEBUG: "\x1b[38;5;2m",  # GREEN
                logging.INFO: "\x1b[38;5;17m",  # BLUE
                logging.WARNING: "\x1b[38;5;3m",  # YELLOW
                logging.ERROR: "\x1b[38;5;1m",  # RED
                logging.CRITICAL: "\x1b[38;5;1m",  # RED
            }
        else:  # dumb theme (no colors)
            self.mapping = {
                logging.DEBUG: "",
                logging.INFO: "",
                logging.WARNING: "",
                logging.ERROR: "",
                logging.CRITICAL: "",
            }

        # Time format (matches Genesis)
        if verbose_time:
            self.TIME = "%(asctime)s.%(msecs)03d"
            self.DATE_FORMAT = "%y-%m-%d %H:%M:%S"
        else:
            self.TIME = "%(asctime)s"
            self.DATE_FORMAT = "%H:%M:%S"

        self.LEVEL = "%(levelname)s"
        self.MESSAGE = "%(message)s"
        self.RESET = "\x1b[0m" if theme != "dumb" else ""

        self.last_color = ""

    def colored_fmt(self, color: str) -> str:
        """Create the colored format string with MARL framework title and logger name."""
        self.last_color = color
        return f"{color}[MARL:%(name)s] [{self.TIME}] [{self.LEVEL}] {self.MESSAGE}{self.RESET}"

    def format(self, record) -> str:
        """Format the log record with MARL framework formatting."""
        color = self.mapping.get(record.levelno, "")
        log_fmt = self.colored_fmt(color)
        formatter = logging.Formatter(log_fmt, datefmt=self.DATE_FORMAT)
        return formatter.format(record)


class MARLLogger:
    """Logger for MARL framework with consistent formatting."""

    def __init__(self, name: str = "marl", level: str = "INFO", verbose_time: bool = False, theme: str = "dark"):
        """
        Initialize MARL framework logger.

        Args:
            name: Logger name (default: "marl")
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            verbose_time: If True, use verbose timestamp format
            theme: Color theme ("dark", "light", "dumb")
        """

        if isinstance(level, str):
            level = level.upper()

        # Create logger with specified name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level))

        # Remove existing handlers to avoid duplicates
        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)

        # Create MARL framework formatter
        self._formatter = MARLFormatter(verbose_time=verbose_time, theme=theme)

        # Create handler that writes to stdout
        self._handler = logging.StreamHandler(sys.stdout)
        self._handler.setLevel(getattr(logging, level))
        self._handler.setFormatter(self._formatter)

        # Add handler to logger
        self._logger.addHandler(self._handler)

        # Prevent propagation to avoid duplicate messages
        self._logger.propagate = False

    def debug(self, message: str):
        """Log debug message."""
        self._logger.debug(message)

    def info(self, message: str):
        """Log info message."""
        self._logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self._logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self._logger.error(message)

    def critical(self, message: str):
        """Log critical message."""
        self._logger.critical(message)

    def setLevel(self, level: str):
        """Change logging level for this logger and all its child loggers."""
        if isinstance(level, str):
            level = level.upper()
        log_level = getattr(logging, level)

        # Set the main logger and handler
        self._logger.setLevel(log_level)
        self._handler.setLevel(log_level)

        # Set all existing child loggers to the same level
        logger_name = self._logger.name
        for name in logging.Logger.manager.loggerDict:
            if name.startswith(f"{logger_name}."):
                child_logger = logging.getLogger(name)
                child_logger.setLevel(log_level)


def setup_logging(
    name: str = "marl", level: str = "INFO", verbose_time: bool = False, theme: str = "dark"
) -> MARLLogger:
    """
    Setup logging for the MARL framework.

    Args:
        name: Logger name (default: "marl")
        level: Logging level
        verbose_time: Use verbose timestamp format
        theme: Color theme

    Returns:
        MARLLogger instance
    """
    return MARLLogger(name=name, level=level, verbose_time=verbose_time, theme=theme)


def get_module_logger(
    module_name: str, parent_logger_name: str = "marl", level: Optional[str] = None
) -> logging.Logger:
    """
    Get a module-specific logger that inherits MARL formatting.

    Args:
        module_name: Name of the module (e.g., "vectorized_aec_env")
        parent_logger: Parent logger name (default: "marl")
        level: Optional level override

    Returns:
        Logger instance with MARL formatting
    """
    logger_name = f"{parent_logger_name}.{module_name}"
    logger = logging.getLogger(logger_name)

    if level:
        logger.setLevel(getattr(logging, level.upper()))
    else:
        parent_logger = logging.getLogger(parent_logger_name)
        logger.setLevel(parent_logger.level)

    # Logger will inherit formatter from parent if no handlers are set
    return logger


def get_class_logger(
    class_name: str, instance_id: str, level: Optional[str] = None, verbose_time: bool = False, theme: str = "dark"
) -> MARLLogger:
    """
    Get a hierarchical class-specific logger with MARL formatting under AEC parent.
    
    Creates hierarchical loggers with format: AEC.{class_name}.{instance_id}
    Example: AEC.Training.go2_locomotion, AEC.Robot.go2_robot_1, AEC.main

    Args:
        class_name: Name of the class (e.g., "Training", "Robot", "Agent", "VectorizedAECEnv")
        instance_id: Unique identifier for this instance (e.g., training_name, robot name, "main")
        level: Logging level (default: "INFO")
        verbose_time: Use verbose timestamp format
        theme: Color theme ("dark", "light", "dumb")

    Returns:
        Hierarchical MARLLogger instance with full MARL formatting under AEC parent
    """
    # Sanitize instance_id to ensure it's logger-name safe
    safe_instance_id = str(instance_id).replace(" ", "_").replace(".", "_")
    
    # Create hierarchical logger name under AEC parent
    if class_name == "VectorizedAECEnv":
        # Special case: AEC is the root, so instance_id becomes the direct child
        logger_name = f"AEC.{safe_instance_id}"
    else:
        # All other components are children of AEC: AEC.ClassName.instance_id
        logger_name = f"AEC.{class_name}.{safe_instance_id}"
    
    # Set default level if not provided
    if level is None:
        level = "INFO"
    
    # Create hierarchical MARL-compatible logger
    return MARLLogger(name=logger_name, level=level, verbose_time=verbose_time, theme=theme)


def cleanup_class_loggers(class_name: str) -> None:
    """
    Clean up all hierarchical loggers for a specific class to prevent memory leaks.
    
    This is useful for long-running applications that create many class instances.

    Args:
        class_name: Name of the class whose loggers should be cleaned up
    """
    if class_name == "VectorizedAECEnv":
        # Clean up AEC root loggers
        prefix = "AEC."
    else:
        # Clean up specific class loggers under AEC
        prefix = f"AEC.{class_name}."
    
    loggers_to_remove = []
    
    for logger_name in logging.Logger.manager.loggerDict:
        if isinstance(logger_name, str) and logger_name.startswith(prefix):
            loggers_to_remove.append(logger_name)
    
    for logger_name in loggers_to_remove:
        logger = logging.getLogger(logger_name)
        # Remove all handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        # Remove from logger manager
        if logger_name in logging.Logger.manager.loggerDict:
            del logging.Logger.manager.loggerDict[logger_name]


def get_active_class_loggers() -> Dict[str, List[str]]:
    """
    Get a summary of all active hierarchical class loggers grouped by class name.
    
    Useful for debugging and monitoring logger instances under AEC hierarchy.
        
    Returns:
        Dictionary mapping class names to lists of instance IDs
    """
    class_loggers = {}
    
    for logger_name in logging.Logger.manager.loggerDict:
        if isinstance(logger_name, str) and logger_name.startswith("AEC."):
            # Extract class name and instance ID from hierarchical format
            parts = logger_name.split(".")
            
            if len(parts) == 2:
                # Format: AEC.instance_id (VectorizedAECEnv case)
                # But only count loggers that actually have handlers (not intermediate parents)
                logger_obj = logging.getLogger(logger_name)
                if logger_obj.handlers:  # Only count loggers with actual handlers
                    class_name = "VectorizedAECEnv"
                    instance_id = parts[1]
                    if class_name not in class_loggers:
                        class_loggers[class_name] = []
                    class_loggers[class_name].append(instance_id)
            elif len(parts) == 3:
                # Format: AEC.ClassName.instance_id
                # Only count loggers that actually have handlers (not intermediate parents)
                logger_obj = logging.getLogger(logger_name)
                if logger_obj.handlers:  # Only count loggers with actual handlers
                    class_name = parts[1]
                    instance_id = parts[2]
                    if class_name not in class_loggers:
                        class_loggers[class_name] = []
                    class_loggers[class_name].append(instance_id)
            # Skip intermediate parent loggers (no handlers) and malformed names
    
    return class_loggers


def set_all_class_loggers_level(level: str) -> None:
    """
    Set logging level for all existing hierarchical class loggers under AEC.
    
    This is useful when you want to change the logging level after loggers have been created.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level_upper = level.upper()
    log_level = getattr(logging, level_upper)
    
    # Update all hierarchical class loggers under AEC
    for logger_name in logging.Logger.manager.loggerDict:
        if isinstance(logger_name, str) and logger_name.startswith("AEC."):
            logger_obj = logging.getLogger(logger_name)
            logger_obj.setLevel(log_level)
            # Also update handlers if it's a MARLLogger
            for handler in logger_obj.handlers:
                handler.setLevel(log_level)


def set_aec_hierarchy_level(level: str) -> None:
    """
    Set logging level for the entire AEC hierarchy starting from the root.
    
    This sets the AEC root logger level and all child loggers will inherit it
    through standard Python logging hierarchy behavior.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level_upper = level.upper()
    log_level = getattr(logging, level_upper)
    
    # Get the AEC root logger and set its level
    aec_root = logging.getLogger("AEC")
    aec_root.setLevel(log_level)
    
    # Also update all existing child loggers explicitly
    set_all_class_loggers_level(level)


# Test function to demonstrate the logging
def test_logging():
    """Test function to show MARL logging in action."""
    print("=== Testing MARL Framework Logging ===\n")

    # Create MARL framework logger
    logger = setup_logging(level="DEBUG")

    # Test all log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    print("\n=== Testing Module Logger ===\n")

    # Test module-specific logger
    module_logger = get_module_logger("test_module")
    module_logger.info("Message from test_module")
    module_logger.warning("Warning from test_module")

    print("\n=== Testing Hierarchical Class-Specific Loggers ===\n")

    # Test hierarchical class-specific loggers under AEC
    aec_logger = get_class_logger("VectorizedAECEnv", "main")
    robot_logger1 = get_class_logger("Robot", "go2_robot_1")
    robot_logger2 = get_class_logger("Robot", "nao_robot_1") 
    training_logger = get_class_logger("Training", "go2_locomotion")
    agent_logger = get_class_logger("Agent", "agent_50Hz")
    
    aec_logger.info("AEC Environment initialized")
    robot_logger1.info("Robot 1 initialized successfully")
    robot_logger2.warning("Robot 2 calibration needed")
    training_logger.info("Training configuration loaded")
    agent_logger.info("Agent running at 50Hz")
    
    print("\n=== Active Hierarchical Class Loggers ===\n")
    active_loggers = get_active_class_loggers()
    for class_name, instances in active_loggers.items():
        print(f"{class_name}: {instances}")
        
    print("\n=== Testing Logger Level Changes ===\n")
    print("Setting all class loggers to DEBUG level...")
    set_all_class_loggers_level("DEBUG")
    
    robot_logger1.debug("Robot 1 DEBUG: This should now be visible")
    training_logger.debug("Training DEBUG: This should now be visible")

    print("\n=== Testing Verbose Time Format ===\n")

    # Test with verbose time
    verbose_logger = setup_logging(level="INFO", verbose_time=True)
    verbose_logger.info("Message with verbose timestamp")

    print("\n=== Testing Light Theme ===\n")

    # Test light theme
    light_logger = setup_logging(level="INFO", theme="light")
    light_logger.info("Message with light theme")
    light_logger.warning("Warning with light theme")


if __name__ == "__main__":
    test_logging()
