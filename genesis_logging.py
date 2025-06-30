#!/usr/bin/env python3
"""
Genesis-compatible logging for MARL VecEnv Framework

This module provides logging that matches Genesis's exact format:
[Genesis] [HH:MM:SS] [LEVEL] message

With the same ANSI color codes and styling as Genesis.
"""

import logging
import sys
from typing import Optional


class GenesisFormatter(logging.Formatter):
    """Custom formatter that matches Genesis's logging format exactly."""
    
    def __init__(self, verbose_time: bool = False, theme: str = "dark"):
        super().__init__()
        
        self.theme = theme
        
        # Color mapping (matches Genesis colors exactly)
        if theme == "dark":
            self.mapping = {
                logging.DEBUG: "\x1b[38;5;119m",    # GREEN
                logging.INFO: "\x1b[38;5;159m",     # BLUE  
                logging.WARNING: "\x1b[38;5;226m",  # YELLOW
                logging.ERROR: "\x1b[38;5;9m",      # RED
                logging.CRITICAL: "\x1b[38;5;9m",   # RED
            }
        elif theme == "light":
            self.mapping = {
                logging.DEBUG: "\x1b[38;5;2m",      # GREEN
                logging.INFO: "\x1b[38;5;17m",      # BLUE
                logging.WARNING: "\x1b[38;5;3m",    # YELLOW
                logging.ERROR: "\x1b[38;5;1m",      # RED
                logging.CRITICAL: "\x1b[38;5;1m",   # RED
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
        """Create the colored format string (matches Genesis format with GenesisMARL title)."""
        self.last_color = color
        return f"{color}[GenesisMARL] [{self.TIME}] [{self.LEVEL}] {self.MESSAGE}{self.RESET}"
    
    def format(self, record) -> str:
        """Format the log record (matches Genesis format)."""
        color = self.mapping.get(record.levelno, "")
        log_fmt = self.colored_fmt(color)
        formatter = logging.Formatter(log_fmt, datefmt=self.DATE_FORMAT)
        return formatter.format(record)


class GenesisCompatibleLogger:
    """Logger that mimics Genesis logging behavior exactly."""
    
    def __init__(self, 
                 name: str = "genesis", 
                 level: str = "INFO",
                 verbose_time: bool = False,
                 theme: str = "dark"):
        """
        Initialize Genesis-compatible logger.
        
        Args:
            name: Logger name (default: "genesis" to match Genesis)
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
        
        # Create Genesis-compatible formatter
        self._formatter = GenesisFormatter(verbose_time=verbose_time, theme=theme)
        
        # Create handler that writes to stdout (like Genesis)
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
        """Change logging level."""
        if isinstance(level, str):
            level = level.upper()
        log_level = getattr(logging, level)
        self._logger.setLevel(log_level)
        self._handler.setLevel(log_level)


def setup_genesis_logging(name: str = "genesis_marl", 
                         level: str = "INFO",
                         verbose_time: bool = False,
                         theme: str = "dark") -> GenesisCompatibleLogger:
    """
    Setup Genesis-compatible logging for the MARL framework.
    
    Args:
        name: Logger name (default: "genesis_marl")
        level: Logging level 
        verbose_time: Use verbose timestamp format
        theme: Color theme
    
    Returns:
        GenesisCompatibleLogger instance
    """
    return GenesisCompatibleLogger(name=name, level=level, 
                                  verbose_time=verbose_time, theme=theme)


def get_module_logger(module_name: str, 
                     parent_logger: str = "genesis_marl",
                     level: Optional[str] = None) -> logging.Logger:
    """
    Get a module-specific logger that inherits GenesisMARL formatting.
    
    Args:
        module_name: Name of the module (e.g., "vectorized_aec_env")
        parent_logger: Parent logger name (default: "genesis_marl")
        level: Optional level override
    
    Returns:
        Logger instance with GenesisMARL formatting
    """
    logger_name = f"{parent_logger}.{module_name}"
    logger = logging.getLogger(logger_name)
    
    if level:
        logger.setLevel(getattr(logging, level.upper()))
    
    # Logger will inherit formatter from parent if no handlers are set
    return logger


# Test function to demonstrate the logging
def test_genesis_logging():
    """Test function to show GenesisMARL logging in action."""
    print("=== Testing GenesisMARL-Compatible Logging ===\n")
    
    # Create GenesisMARL-compatible logger
    logger = setup_genesis_logging(level="DEBUG")
    
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
    
    print("\n=== Testing Verbose Time Format ===\n")
    
    # Test with verbose time
    verbose_logger = setup_genesis_logging(level="INFO", verbose_time=True)
    verbose_logger.info("Message with verbose timestamp")
    
    print("\n=== Testing Light Theme ===\n")
    
    # Test light theme
    light_logger = setup_genesis_logging(level="INFO", theme="light")
    light_logger.info("Message with light theme")
    light_logger.warning("Warning with light theme")


if __name__ == "__main__":
    test_genesis_logging()