"""Centralized scheduler for Genesis MARL VecEnv V2.

This module implements centralized agent scheduling where all environments
step the same agent together, maintaining vectorized efficiency.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
import heapq
from dataclasses import dataclass
import math

from .genesis_logging import get_module_logger


logger = get_module_logger("centralized_scheduler")


@dataclass
class RobotSchedule:
    """Robot scheduling information for fixed-frequency control"""
    agent_id: str
    frequency: float
    period_frames: int
    last_frame: int
    next_frame: int
    
    def __lt__(self, other):
        return self.next_frame < other.next_frame


class CentralizedFrequencyScheduler:
    """Centralized scheduler for multi-robot MARL with different frequencies.
    
    Key principle: All environments step the same agent together.
    This maintains vectorized execution while supporting different robot frequencies.
    """
    
    def __init__(self, 
                 robot_names: List[str], 
                 robot_frequencies: List[float], 
                 genesis_freq: float,
                 device: str = "cuda"):
        """Initialize centralized scheduler.
        
        Args:
            robot_names: List of robot/agent names
            robot_frequencies: List of robot control frequencies (Hz)
            genesis_freq: Genesis simulation frequency (Hz)
            device: PyTorch device
        """
        self.robot_names = robot_names
        self.robot_frequencies = robot_frequencies
        self.genesis_freq = genesis_freq
        self.dt = 1.0 / genesis_freq
        self.device = device
        self.n_agents = len(robot_names)
        
        logger.info(f"Initializing CentralizedFrequencyScheduler")
        logger.info(f"  Genesis frequency: {genesis_freq:.1f} Hz (dt={self.dt:.6f}s)")
        logger.info(f"  Number of agents: {self.n_agents}")
        logger.info(f"  Robot frequencies: {robot_frequencies}")
        
        # Calculate period frames for each robot
        self.period_frames = []
        self.actual_frequencies = []
        self.frequency_errors = []
        
        for i, (name, freq) in enumerate(zip(robot_names, robot_frequencies)):
            period_seconds = 1.0 / freq
            period_frames = max(1, round(period_seconds / self.dt))
            actual_period = period_frames * self.dt
            actual_freq = 1.0 / actual_period
            freq_error = abs(actual_freq - freq) / freq
            
            self.period_frames.append(period_frames)
            self.actual_frequencies.append(actual_freq)
            self.frequency_errors.append(freq_error)
            
            logger.debug(f"  Agent {name}: desired={freq:.1f}Hz, actual={actual_freq:.1f}Hz, "
                        f"period={period_frames} frames, error={freq_error*100:.1f}%")
        
        # Initialize scheduling state
        self.reset()
        self._log_frequency_analysis()
    
    def _log_frequency_analysis(self):
        """Log detailed frequency analysis."""
        logger.info("=" * 70)
        logger.info("Centralized Scheduler - Frequency Analysis")
        logger.info("=" * 70)
        logger.info(f"Genesis Frequency: {self.genesis_freq:.1f} Hz (dt = {self.dt:.4f}s)")
        logger.info("")
        logger.info("Agent Frequency Mapping:")
        logger.info("Agent Name     | Desired | Actual  | Frames | Error  | Status")
        logger.info("-" * 70)
        
        for i, name in enumerate(self.robot_names):
            desired_freq = self.robot_frequencies[i]
            actual_freq = self.actual_frequencies[i]
            period_frames = self.period_frames[i]
            freq_error = self.frequency_errors[i]
            status = "✅ OK" if freq_error <= 0.05 else "⚠️  High"
            
            logger.info(f"{name:14} | {desired_freq:6.1f}Hz | {actual_freq:6.1f}Hz | "
                       f"{period_frames:6} | {freq_error*100:5.1f}% | {status}")
        
        high_error_count = sum(1 for e in self.frequency_errors if e > 0.05)
        if high_error_count > 0:
            logger.warning(f"⚠️  {high_error_count} agent(s) have frequency error > 5%")
        else:
            logger.info("✅ All agents within 5% frequency tolerance")
        logger.info("=" * 70)
    
    def reset(self):
        """Reset scheduler to initial state."""
        logger.debug("Resetting centralized scheduler")
        
        self.current_frame = 0
        self.current_agent_name = None
        
        # Initialize priority queue with robot schedules
        self.schedules = []
        for i, name in enumerate(self.robot_names):
            schedule = RobotSchedule(
                agent_id=name,
                frequency=self.actual_frequencies[i],
                period_frames=self.period_frames[i],
                last_frame=0,
                next_frame=0  # All agents can act at frame 0
            )
            heapq.heappush(self.schedules, schedule)
        
        # Agents ready to act at current frame
        self._agents_ready = []
        
        # Get first agent
        self._advance_to_next_agent()
    
    def _advance_to_next_agent(self):
        """Internal method to select next agent."""
        # If we have agents ready, use them
        if self._agents_ready:
            agent_id = self._agents_ready.pop(0)
            self.current_agent_name = agent_id
            logger.debug(f"Selected ready agent: {agent_id}")
            return
        
        # Otherwise, advance to next scheduled frame
        if not self.schedules:
            self.current_agent_name = None
            logger.debug("No more scheduled agents")
            return
        
        # Get next scheduled agent
        next_schedule = self.schedules[0]
        target_frame = next_schedule.next_frame
        
        # Collect all agents that should act at this frame
        ready_agents = []
        while self.schedules and self.schedules[0].next_frame <= target_frame:
            schedule = heapq.heappop(self.schedules)
            ready_agents.append(schedule.agent_id)
            
            # Reschedule for next period
            schedule.last_frame = target_frame
            schedule.next_frame = target_frame + schedule.period_frames
            heapq.heappush(self.schedules, schedule)
        
        # Set current agent
        if ready_agents:
            self.current_agent_name = ready_agents[0]
            self._agents_ready = ready_agents[1:]
            
            logger.debug(f"Advanced to frame {target_frame}, selected agent: {self.current_agent_name}, "
                        f"{len(self._agents_ready)} more agents ready")
    
    def get_current_agent(self) -> Optional[str]:
        """Get current agent that should act.
        
        Returns:
            agent_name: Name of current agent or None
        """
        return self.current_agent_name
    
    def step(self) -> Tuple[Optional[str], int]:
        """Step to next agent and return simulation frames needed.
        
        Returns:
            next_agent_name: Name of next agent or None
            frames_to_advance: Number of Genesis simulation frames to advance
        """
        # Calculate frames to advance before moving to next agent
        frames_to_advance = 0
        if self.schedules and self.current_agent_name is not None:
            next_action_frame = self.schedules[0].next_frame
            frames_to_advance = max(0, next_action_frame - self.current_frame)
            self.current_frame = next_action_frame
        
        # Move to next agent
        self._advance_to_next_agent()
        
        logger.debug(f"Scheduler step: agent={self.current_agent_name}, "
                    f"frames_to_advance={frames_to_advance}, "
                    f"current_frame={self.current_frame}")
        
        return self.current_agent_name, frames_to_advance
    
    def get_frames_until_next_agent(self) -> int:
        """Get number of frames until next agent should act."""
        if not self.schedules:
            return 0
        
        next_frame = self.schedules[0].next_frame
        return max(0, next_frame - self.current_frame)


def find_optimal_genesis_frequency(robot_frequencies: List[float], 
                                 max_genesis_freq: float = 1000.0,
                                 tolerance: float = 0.05) -> Tuple[float, Dict]:
    """Find optimal Genesis frequency for given robot frequencies.
    
    Args:
        robot_frequencies: List of robot control frequencies (Hz)
        max_genesis_freq: Maximum allowed Genesis frequency (Hz)
        tolerance: Maximum acceptable frequency error (fraction)
    
    Returns:
        optimal_freq: Optimal Genesis frequency
        freq_info: Dictionary with frequency analysis info
    """
    logger.info(f"Finding optimal Genesis frequency for robots: {robot_frequencies}")
    
    candidates = []
    
    # Strategy 1: Multiples of highest frequency
    max_robot_freq = max(robot_frequencies)
    for mult in [1, 2, 4, 5, 8, 10, 20]:
        candidate = max_robot_freq * mult
        if candidate <= max_genesis_freq:
            candidates.append(candidate)
    
    # Strategy 2: Common simulation frequencies
    common_freqs = [50, 100, 200, 250, 500, 1000]
    candidates.extend([f for f in common_freqs if f <= max_genesis_freq])
    
    # Strategy 3: LCM-based frequency
    try:
        from math import gcd
        from functools import reduce
        
        # Scale to integers to find LCM
        scale = 10
        scaled_freqs = [int(f * scale) for f in robot_frequencies]
        lcm = reduce(lambda a, b: a * b // gcd(a, b), scaled_freqs)
        lcm_freq = float(lcm) / scale
        
        if lcm_freq <= max_genesis_freq:
            candidates.append(lcm_freq)
    except:
        pass
    
    # Evaluate all candidates
    best_freq = None
    best_score = float('inf')
    best_info = None
    
    for genesis_freq in sorted(set(candidates)):
        dt = 1.0 / genesis_freq
        info = {"genesis_freq": genesis_freq, "dt": dt, "robots": {}}
        total_error = 0.0
        
        for i, robot_freq in enumerate(robot_frequencies):
            period_seconds = 1.0 / robot_freq
            period_frames = max(1, round(period_seconds / dt))
            actual_period = period_frames * dt
            actual_freq = 1.0 / actual_period
            freq_error = abs(actual_freq - robot_freq) / robot_freq
            
            info["robots"][f"robot_{i}"] = {
                "desired_freq": robot_freq,
                "actual_freq": actual_freq,
                "period_frames": period_frames,
                "freq_error": freq_error,
                "within_tolerance": freq_error <= tolerance
            }
            
            total_error += freq_error
        
        # Score based on total error and Genesis frequency (prefer lower frequencies)
        score = total_error + genesis_freq / max_genesis_freq * 0.1
        
        if score < best_score:
            best_score = score
            best_freq = genesis_freq
            best_info = info
    
    if best_freq is None:
        # Fallback to reasonable default
        best_freq = min(max_robot_freq * 4, max_genesis_freq)
        logger.warning(f"Using fallback Genesis frequency: {best_freq} Hz")
    
    logger.info(f"Selected Genesis frequency: {best_freq} Hz")
    return best_freq, best_info