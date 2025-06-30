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
from scipy.optimize import minimize_scalar

from genesis_logging import get_module_logger


logger = get_module_logger("centralized_scheduler")


@dataclass
class AgentSchedule:
    """Agent scheduling information for fixed-frequency control"""

    agent_id: str
    frequency: float
    period_frames: int
    last_frame: int
    next_frame: int

    def __lt__(self, other):
        return self.next_frame < other.next_frame


class FrequencyOptimizer:
    """Optimizes Genesis simulation frequency for multi-robot scenarios"""

    @staticmethod
    def find_optimal_genesis_frequency(
        robot_frequencies: List[float], max_genesis_freq: float = 1000.0, tolerance: float = 0.05
    ) -> Tuple[float, Dict[str, Dict]]:
        """
        Find optimal Genesis frequency that minimizes mean error while ensuring
        all robots have < tolerance error from their desired frequency.

        This is a constrained optimization problem:
        - Minimize: mean frequency error across all robots
        - Subject to: max error < tolerance AND genesis_freq <= max_genesis_freq
        """

        def compute_error(genesis_freq: float) -> Tuple[float, float]:
            """Compute mean and max error for a given genesis frequency"""
            errors = []
            for robot_freq in robot_frequencies:
                # Calculate how many genesis timesteps per robot control step
                period_frames = round(genesis_freq / robot_freq)
                period_frames = max(1, period_frames)

                # Actual frequency the robot will run at
                actual_freq = genesis_freq / period_frames

                # Relative error
                error = abs(actual_freq - robot_freq) / robot_freq
                errors.append(error)

            return np.mean(errors), max(errors)

        def objective(genesis_freq: float) -> float:
            """Objective function: returns large value if constraints violated"""
            mean_error, max_error = compute_error(genesis_freq)

            # Penalty for violating tolerance constraint
            if max_error > tolerance:
                return 1e6 + max_error

            # When errors are equal (or very close), prefer lower frequencies
            # Add a small penalty proportional to the frequency
            return mean_error + (genesis_freq / max_genesis_freq) * 1e-6

        # Find minimum frequency that could possibly work
        # Genesis freq must be at least as high as any robot freq
        min_genesis_freq = max(robot_frequencies)

        # First, try to find the absolute minimum frequency that works
        # by checking multiples of the robot frequencies
        candidate_freqs = []

        # For each robot frequency, find multiples that could work
        for robot_freq in robot_frequencies:
            # Try multiples of this frequency
            multiple = 1
            while robot_freq * multiple <= max_genesis_freq:
                genesis_freq = robot_freq * multiple
                _, max_error = compute_error(genesis_freq)
                if max_error <= tolerance:
                    candidate_freqs.append(genesis_freq)
                multiple += 1

        # If we found candidates, start with the minimum
        if candidate_freqs:
            min_genesis_freq = min(candidate_freqs)

            # Check if this minimum already satisfies all constraints
            mean_error, max_error = compute_error(min_genesis_freq)
            if max_error <= tolerance:
                # This is our optimal solution
                optimal_freq = min_genesis_freq
            else:
                # Need to optimize further
                result = minimize_scalar(
                    objective,
                    bounds=(min_genesis_freq, max_genesis_freq),
                    method='bounded',
                    options={'xatol': 0.1},  # 0.1 Hz precision is enough
                )
                optimal_freq = result.x
        else:
            # No simple multiples work, need full optimization
            result = minimize_scalar(
                objective,
                bounds=(min_genesis_freq, max_genesis_freq),
                method='bounded',
                options={'xatol': 0.1},
            )
            optimal_freq = result.x

        # Verify the solution
        mean_error, max_error = compute_error(optimal_freq)

        if max_error > tolerance:
            logger.warning(
                f"Cannot achieve {tolerance*100}% tolerance for all robots. " f"Max error: {max_error*100:.1f}%"
            )

        # Generate detailed info
        robot_info = FrequencyOptimizer._generate_info(optimal_freq, robot_frequencies)

        logger.info(
            f"Optimal Genesis frequency: {optimal_freq:.1f} Hz "
            f"(mean error: {mean_error*100:.2f}%, max error: {max_error*100:.2f}%)"
        )

        return optimal_freq, robot_info

    @staticmethod
    def _generate_info(genesis_freq: float, robot_frequencies: List[float]) -> Dict[str, Dict]:
        """Generate detailed information about frequency mapping"""
        robot_info = {}

        for i, robot_freq in enumerate(robot_frequencies):
            period_frames = round(genesis_freq / robot_freq)
            period_frames = max(1, period_frames)
            actual_freq = genesis_freq / period_frames
            error = abs(actual_freq - robot_freq) / robot_freq

            robot_info[f"robot_{i}"] = {
                "desired_frequency": robot_freq,
                "actual_frequency": actual_freq,
                "period_frames": period_frames,
                "frequency_error": error,
                "within_tolerance": error <= 0.05,
            }

        return robot_info


class CentralizedFrequencyScheduler:
    """Centralized scheduler for multi-agent MARL with different frequencies.

    Key principle: All environments step the same agent together.
    This maintains vectorized execution while supporting different agent frequencies.
    """

    def __init__(
        self,
        agent_frequencies: Dict[str, float],
        genesis_freq: Optional[float] = None,
        max_genesis_freq: float = 150.0,
        device: str = "cuda",
    ):
        """Initialize centralized scheduler.

        Args:
            agent_frequencies: Dictionary mapping agent_name -> frequency (Hz)
            base_dt: Base simulation timestep (seconds)
            device: PyTorch device
        """
        if genesis_freq is None:
            genesis_freq, frequency_info = FrequencyOptimizer.find_optimal_genesis_frequency(
                robot_frequencies=list(agent_frequencies.values()), max_genesis_freq=max_genesis_freq, tolerance=0.05
            )
            # Print frequency info as table
            logger.info("Automatic optimized frequency:")
            logger.info(f"{'Agent':<10}{'Desired':<10}{'Actual':<10}{'Period':<10}{'Error':<10}")
            for name, info in frequency_info.items():
                logger.info(
                    f"{name:<10}{info['desired_frequency']:<10.1f}Hz"
                    f"{info['actual_frequency']:<10.1f}Hz"
                    f"{info['period_frames']:<10}"
                    f"{info['frequency_error']*100:<10.1f}%"
                )
            logger.info(f"Genesis frequency: {genesis_freq:.1f} Hz (dt = {1.0 / genesis_freq:.4f}s)")

        self.agent_frequencies = agent_frequencies
        self.agent_names = list(agent_frequencies.keys())
        self.frequency_values = list(agent_frequencies.values())
        self.genesis_freq = genesis_freq
        self.dt = 1.0 / self.genesis_freq
        self.device = device
        self.n_agents = len(self.agent_names)

        logger.info(f"Initializing CentralizedFrequencyScheduler")
        logger.info(f"  Genesis frequency: {self.genesis_freq:.1f} Hz (dt={self.dt:.6f}s)")
        logger.info(f"  Number of agents: {self.n_agents}")
        logger.info(f"  Agent frequencies: {self.frequency_values}")

        # Calculate period frames for each agent
        self.period_frames = []
        self.actual_frequencies = []
        self.frequency_errors = []

        for i, (name, freq) in enumerate(zip(self.agent_names, self.frequency_values)):
            period_seconds = 1.0 / freq
            period_frames = max(1, round(period_seconds / self.dt))
            actual_period = period_frames * self.dt
            actual_freq = 1.0 / actual_period
            freq_error = abs(actual_freq - freq) / freq

            self.period_frames.append(period_frames)
            self.actual_frequencies.append(actual_freq)
            self.frequency_errors.append(freq_error)

            logger.debug(
                f"  Agent {name}: desired={freq:.1f}Hz, actual={actual_freq:.1f}Hz, "
                f"period={period_frames} frames, error={freq_error*100:.1f}%"
            )

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

        for i, name in enumerate(self.agent_names):
            desired_freq = self.frequency_values[i]
            actual_freq = self.actual_frequencies[i]
            period_frames = self.period_frames[i]
            freq_error = self.frequency_errors[i]
            status = "✅ OK" if freq_error <= 0.05 else "⚠️  High"

            logger.info(
                f"{name:14} | {desired_freq:6.1f}Hz | {actual_freq:6.1f}Hz | "
                f"{period_frames:6} | {freq_error*100:5.1f}% | {status}"
            )

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

        # Initialize priority queue with agent schedules
        self.schedules = []
        for i, name in enumerate(self.agent_names):
            schedule = AgentSchedule(
                agent_id=name,
                frequency=self.actual_frequencies[i],
                period_frames=self.period_frames[i],
                last_frame=0,
                next_frame=0,  # All agents can act at frame 0
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

            logger.debug(
                f"Advanced to frame {target_frame}, selected agent: {self.current_agent_name}, "
                f"{len(self._agents_ready)} more agents ready"
            )

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

        logger.debug(
            f"Scheduler step: agent={self.current_agent_name}, "
            f"frames_to_advance={frames_to_advance}, "
            f"current_frame={self.current_frame}"
        )

        return self.current_agent_name, frames_to_advance

    def get_frames_until_next_agent(self) -> int:
        """Get number of frames until next agent should act."""
        if not self.schedules:
            return 0

        next_frame = self.schedules[0].next_frame
        return max(0, next_frame - self.current_frame)


def find_optimal_genesis_frequency(
    agent_frequencies: List[float], max_genesis_freq: float = 1000.0, tolerance: float = 0.05
) -> Tuple[float, Dict]:
    """Find optimal Genesis frequency for given agent frequencies.

    Args:
        agent_frequencies: List of agent control frequencies (Hz)
        max_genesis_freq: Maximum allowed Genesis frequency (Hz)
        tolerance: Maximum acceptable frequency error (fraction)

    Returns:
        optimal_freq: Optimal Genesis frequency
        freq_info: Dictionary with frequency analysis info
    """
    logger.info(f"Finding optimal Genesis frequency for agents: {agent_frequencies}")

    candidates = []

    # Strategy 1: Multiples of highest frequency
    max_agent_freq = max(agent_frequencies)
    for mult in [1, 2, 4, 5, 8, 10, 20]:
        candidate = max_agent_freq * mult
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
        scaled_freqs = [int(f * scale) for f in agent_frequencies]
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
        info = {"genesis_freq": genesis_freq, "dt": dt, "agents": {}}
        total_error = 0.0

        for i, agent_freq in enumerate(agent_frequencies):
            period_seconds = 1.0 / agent_freq
            period_frames = max(1, round(period_seconds / dt))
            actual_period = period_frames * dt
            actual_freq = 1.0 / actual_period
            freq_error = abs(actual_freq - agent_freq) / agent_freq

            info["agents"][f"agent_{i}"] = {
                "desired_freq": agent_freq,
                "actual_freq": actual_freq,
                "period_frames": period_frames,
                "freq_error": freq_error,
                "within_tolerance": freq_error <= tolerance,
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
        best_freq = min(max_agent_freq * 4, max_genesis_freq)
        logger.warning(f"Using fallback Genesis frequency: {best_freq} Hz")

    logger.info(f"Selected Genesis frequency: {best_freq} Hz")
    return best_freq, best_info
