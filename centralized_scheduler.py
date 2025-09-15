"""Centralized scheduler for Genesis MARL VecEnv V2.

This module implements centralized agent scheduling where all environments
step the same agent together, maintaining vectorized efficiency.
"""

import heapq
from dataclasses import dataclass
from functools import reduce
from math import gcd, lcm
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from marl_logging import get_class_logger


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
        robot_frequencies: List[float],
        max_genesis_freq: float = 1000.0,
        tolerance: float = 0.05,
        require_integer: bool = True,
    ) -> Tuple[int, Dict[str, Dict]]:  # Return type is now int
        """
        Find optimal Genesis frequency that minimizes mean error while ensuring
        all robots have < tolerance error from their desired frequency.

        Search order:
        1. Try LCM of robot frequencies
        2. Use optimizer to find a candidate
        3. If optimizer doesn't give perfect result, check multiples
        4. Return best candidate found
        """

        def compute_error(genesis_freq: int) -> Tuple[float, float]:  # Now expects int
            """Compute mean and max error for a given genesis frequency"""
            errors = []
            for robot_freq in robot_frequencies:
                period_frames = round(genesis_freq / robot_freq)
                period_frames = max(1, period_frames)
                actual_freq = genesis_freq / period_frames
                error = abs(actual_freq - robot_freq) / robot_freq
                errors.append(error)
            return np.mean(errors), max(errors)

        # Step 1: Try LCM first
        if require_integer:
            # Convert frequencies to integers for LCM calculation
            int_freqs = [int(round(f)) for f in robot_frequencies]
            lcm_freq = reduce(lcm, int_freqs)  # This is already an int

            if lcm_freq <= int(max_genesis_freq):
                mean_error, max_error = compute_error(lcm_freq)
                if max_error <= tolerance:
                    robot_info = FrequencyOptimizer._generate_info(lcm_freq, robot_frequencies)
                    return lcm_freq, robot_info
        else:
            # For non-integer case, handle differently
            raise ValueError("This function is designed for integer frequencies. Set require_integer=True")

        # Step 2: Use optimizer to find a candidate
        def objective(x):
            genesis_freq = int(x[0])  # Ensure int inside objective
            mean_error, max_error = compute_error(genesis_freq)
            if max_error > tolerance:
                return 1e6 + max_error
            return mean_error + (genesis_freq / max_genesis_freq) * 1e-4

        min_freq = int(np.ceil(max(robot_frequencies)))
        max_freq_int = int(max_genesis_freq)
        bounds = [(min_freq, max_freq_int)]

        result = differential_evolution(
            objective,
            bounds=bounds,
            integrality=[True],  # Force integer
            seed=42,
            maxiter=300,
            popsize=15,
            atol=1e-6,
            tol=1e-6,
        )

        optimizer_freq = int(result.x[0])  # Get as int
        optimizer_mean_error, optimizer_max_error = compute_error(optimizer_freq)

        # If optimizer found perfect solution (error = 0), use it
        if optimizer_mean_error == 0:
            robot_info = FrequencyOptimizer._generate_info(optimizer_freq, robot_frequencies)
            return optimizer_freq, robot_info

        # Step 3: Iterate through multiples to find better candidates
        best_freq = optimizer_freq
        best_mean_error = optimizer_mean_error
        best_max_error = optimizer_max_error

        for robot_freq in robot_frequencies:
            multiple = 1
            while robot_freq * multiple <= max_freq_int:
                candidate_freq = int(round(robot_freq * multiple))  # Convert to int immediately

                if candidate_freq >= min_freq:
                    mean_error, max_error = compute_error(candidate_freq)

                    # Check if this is better than current best
                    if max_error <= tolerance:
                        if mean_error < best_mean_error or (
                            mean_error == best_mean_error and candidate_freq < best_freq
                        ):
                            best_freq = candidate_freq
                            best_mean_error = mean_error
                            best_max_error = max_error

                multiple += 1

        # Step 4: Return best candidate found
        robot_info = FrequencyOptimizer._generate_info(best_freq, robot_frequencies)
        return best_freq, robot_info

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
        # Initialize centralized scheduler logger (independent of specific agents)
        self.logger = get_class_logger("Scheduler", "centralized", level="INFO")
        
        if genesis_freq is None:
            genesis_freq, frequency_info = FrequencyOptimizer.find_optimal_genesis_frequency(
                robot_frequencies=list(agent_frequencies.values()), max_genesis_freq=max_genesis_freq, tolerance=0.05
            )
            # Print frequency info as table
            self.logger.info("Automatic optimized frequency:")
            self.logger.info(f"{'Agent':<10}{'Desired':<10}{'Actual':<10}{'Period':<10}{'Error':<10}")
            for name, info in frequency_info.items():
                self.logger.info(
                    f"{name:<10}{info['desired_frequency']:<10.1f}Hz"
                    f"{info['actual_frequency']:<10.1f}Hz"
                    f"{info['period_frames']:<10}"
                    f"{info['frequency_error']*100:<10.1f}%"
                )
            self.logger.info(f"Genesis frequency: {genesis_freq:.1f} Hz (dt = {1.0 / genesis_freq:.4f}s)")

        self.agent_frequencies = agent_frequencies
        self.agent_names = list(agent_frequencies.keys())
        self.frequency_values = list(agent_frequencies.values())
        self.genesis_freq = genesis_freq
        self.dt = 1.0 / self.genesis_freq
        self.device = device
        self.n_agents = len(self.agent_names)

        self.logger.info(f"Initializing CentralizedFrequencyScheduler")
        self.logger.info(f"  Genesis frequency: {self.genesis_freq:.1f} Hz (dt={self.dt:.6f}s)")
        self.logger.info(f"  Number of agents: {self.n_agents}")
        self.logger.info(f"  Agent frequencies: {self.frequency_values}")

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

            self.logger.debug(
                f"  Agent {name}: desired={freq:.1f}Hz, actual={actual_freq:.1f}Hz, "
                f"period={period_frames} frames, error={freq_error*100:.1f}%"
            )

        # Initialize scheduling state
        self.reset()
        self._log_frequency_analysis()

    def _log_frequency_analysis(self):
        """Log detailed frequency analysis."""
        self.logger.info("=" * 70)
        self.logger.info("Centralized Scheduler - Frequency Analysis")
        self.logger.info("=" * 70)
        self.logger.info(f"Genesis Frequency: {self.genesis_freq:.1f} Hz (dt = {self.dt:.4f}s)")
        self.logger.info("")
        self.logger.info("Agent Frequency Mapping:")
        self.logger.info("Agent Name     | Desired | Actual  | Frames | Error  | Status")
        self.logger.info("-" * 70)

        for i, name in enumerate(self.agent_names):
            desired_freq = self.frequency_values[i]
            actual_freq = self.actual_frequencies[i]
            period_frames = self.period_frames[i]
            freq_error = self.frequency_errors[i]
            status = "✅ OK" if freq_error <= 0.05 else "⚠️  High"

            self.logger.info(
                f"{name:14} | {desired_freq:6.1f}Hz | {actual_freq:6.1f}Hz | "
                f"{period_frames:6} | {freq_error*100:5.1f}% | {status}"
            )

        high_error_count = sum(1 for e in self.frequency_errors if e > 0.05)
        if high_error_count > 0:
            self.logger.warning(f"⚠️  {high_error_count} agent(s) have frequency error > 5%")
        else:
            self.logger.info("✅ All agents within 5% frequency tolerance")
        self.logger.info("=" * 70)

    def reset(self):
        """Reset scheduler to initial state."""
        self.logger.debug("Resetting centralized scheduler")

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
            self.logger.debug(f"Selected ready agent: {agent_id}")
            return

        # Otherwise, advance to next scheduled frame
        if not self.schedules:
            self.current_agent_name = None
            self.logger.debug("No more scheduled agents")
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

            self.logger.debug(
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

        self.logger.debug(
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