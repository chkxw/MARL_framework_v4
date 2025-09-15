import multiprocessing
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Set, Union

from marl_logging import get_class_logger


class CoordinatorBase(ABC):
    """Abstract base class for multi-state process/thread coordination"""

    @abstractmethod
    def wake(self, states: Union[str, List[str]]):
        """Wake up threads/processes waiting for the specified state(s)"""

    @abstractmethod
    def wait(self, states: Union[str, List[str]]):
        """Wait until all specified states are finished"""

    @abstractmethod
    def set_finished(self, state: str):
        """Mark a state as finished (thread completed its work)"""

    @abstractmethod
    def set_unfinished(self, states: Union[str, List[str]] = None):
        """Reset finished flags for specified states (or all if None)"""

    @abstractmethod
    def get_active_states(self) -> Set[str]:
        """Get currently active (woken but not finished) states"""


class ThreadingCoordinator(CoordinatorBase):
    """Threading version of the multi-state coordinator"""

    def __init__(self, valid_states: List[str]):
        """
        Initialize coordinator with valid state names

        Args:
            valid_states: List of valid state names
        """
        # Initialize independent class-specific logger with timestamp as unique identifier
        import time
        self.logger = get_class_logger("ThreadingCoordinator", f"t{int(time.time()*1000)}", level="INFO")
        
        self._valid_states = set(valid_states)
        self._lock = threading.Lock()

        # Condition variable for each state
        self._conditions = {state: threading.Condition(self._lock) for state in valid_states}

        # Track which states are currently active (woken)
        self._active_states: Set[str] = set()

        # Track which states have finished their work
        self._finished_states: Set[str] = set()

        # Condition for waiting on multiple states to finish
        self._finish_condition = threading.Condition(self._lock)

        self.logger.info(f"ThreadingCoordinator initialized with states: {valid_states}")

    def _validate_state(self, state: str):
        """Check if state is valid"""
        if state not in self._valid_states:
            raise ValueError(f"Invalid state: '{state}'. Valid states are: {sorted(self._valid_states)}")

    def wake(self, states: Union[str, List[str]]):
        """Wake up threads waiting for the specified state(s)"""
        if not isinstance(states, list):
            states = [states]

        with self._lock:
            for state in states:
                self._validate_state(state)

                self._active_states.add(state)
                self._finished_states.discard(state)  # Reset finished flag
                self._conditions[state].notify_all()

            self.logger.debug(f"Woke up states: {states}")
            self.logger.debug(f"Active states: {sorted(self._active_states)}")

    def wait(self, states: Union[str, List[str]]):
        """Wait until all specified states are finished"""
        if not isinstance(states, list):
            states = [states]

        # Validate all states first
        for state in states:
            self._validate_state(state)

        states_set = set(states)
        self.logger.debug(f"Waiting for states to finish: {states}")

        with self._finish_condition:
            while not states_set.issubset(self._finished_states):
                self._finish_condition.wait()

        self.logger.debug(f"All states finished: {states}")

    def set_finished(self, state: str):
        """Mark a state as finished"""
        with self._lock:
            self._validate_state(state)
            self._finished_states.add(state)
            self._active_states.discard(state)
            self._finish_condition.notify_all()

            self.logger.debug(f"State '{state}' marked as finished")
            self.logger.debug(f"Finished states: {sorted(self._finished_states)}")

    def set_unfinished(self, states: Union[str, List[str]] = None):
        """Reset finished flags for specified states"""
        with self._lock:
            if states is None:
                self._finished_states.clear()
                self.logger.debug("Reset all finished states")
            else:
                if not isinstance(states, list):
                    states = [states]
                for state in states:
                    self._validate_state(state)
                    self._finished_states.discard(state)
                self.logger.debug(f"Reset finished states: {states}")

    def get_active_states(self) -> Set[str]:
        """Get currently active states"""
        with self._lock:
            return self._active_states.copy()

    def wait_for_state(self, state: str):
        """Internal method for a thread to wait for its specific state"""
        self._validate_state(state)
        with self._conditions[state]:
            while state not in self._active_states:
                self._conditions[state].wait()
        self.logger.debug(f"Thread for state '{state}' woken up")


class MultiprocessingCoordinator(CoordinatorBase):
    """Multiprocessing version of the multi-state coordinator"""

    def __init__(self, valid_states: List[str]):
        """
        Initialize coordinator with valid state names

        Args:
            valid_states: List of valid state names
        """
        # Initialize independent class-specific logger with timestamp as unique identifier
        import time
        self.logger = get_class_logger("MultiprocessingCoordinator", f"mp{int(time.time()*1000)}", level="INFO")
        
        self._valid_states = set(valid_states)
        self._manager = multiprocessing.Manager()
        self._lock = self._manager.Lock()

        # Condition variable for each state
        self._conditions = {state: self._manager.Condition(self._lock) for state in valid_states}

        # Use manager lists for shared state
        self._active_states = self._manager.list()
        self._finished_states = self._manager.list()

        # Keep a reference to valid states in shared memory
        self._valid_states_list = self._manager.list(valid_states)

        # Condition for waiting on multiple states to finish
        self._finish_condition = self._manager.Condition(self._lock)

        self.logger.info(f"MultiprocessingCoordinator initialized with states: {valid_states}")

    def _validate_state(self, state: str):
        """Check if state is valid"""
        if state not in self._valid_states_list:
            raise ValueError(f"Invalid state: '{state}'. Valid states are: {sorted(self._valid_states_list)}")

    def wake(self, states: Union[str, List[str]]):
        """Wake up processes waiting for the specified state(s)"""
        if not isinstance(states, list):
            states = [states]

        with self._lock:
            for state in states:
                self._validate_state(state)

                if state not in self._active_states:
                    self._active_states.append(state)
                if state in self._finished_states:
                    self._finished_states.remove(state)

                self._conditions[state].notify_all()

            self.logger.debug(f"Woke up states: {states}")

    def wait(self, states: Union[str, List[str]]):
        """Wait until all specified states are finished"""
        if not isinstance(states, list):
            states = [states]

        # Validate all states first
        for state in states:
            self._validate_state(state)

        self.logger.debug(f"Waiting for states to finish: {states}")

        with self._finish_condition:
            while not all(s in self._finished_states for s in states):
                self._finish_condition.wait()

        self.logger.debug(f"All states finished: {states}")

    def set_finished(self, state: str):
        """Mark a state as finished"""
        with self._lock:
            self._validate_state(state)
            if state not in self._finished_states:
                self._finished_states.append(state)
            if state in self._active_states:
                self._active_states.remove(state)

            self._finish_condition.notify_all()

            self.logger.debug(f"State '{state}' marked as finished")

    def set_unfinished(self, states: Union[str, List[str]] = None):
        """Reset finished flags for specified states"""
        with self._lock:
            if states is None:
                # Clear all
                while self._finished_states:
                    self._finished_states.pop()
                self.logger.debug("Reset all finished states")
            else:
                if not isinstance(states, list):
                    states = [states]
                for state in states:
                    self._validate_state(state)
                    if state in self._finished_states:
                        self._finished_states.remove(state)
                self.logger.debug(f"Reset finished states: {states}")

    def get_active_states(self) -> Set[str]:
        """Get currently active states"""
        with self._lock:
            return set(self._active_states)

    def wait_for_state(self, state: str):
        """Internal method for a process to wait for its specific state"""
        self._validate_state(state)
        with self._conditions[state]:
            while state not in self._active_states:
                self._conditions[state].wait()
        self.logger.debug(f"Process for state '{state}' woken up")


# Example usage
def example_threading():
    """Example with multiple coordinated threads"""

    # Define valid states
    states = ["reader", "processor", "writer", "validator"]
    coordinator = ThreadingCoordinator(states)

    def reader_worker():
        # Wait to be woken up
        coordinator.wait_for_state("reader")
        coordinator.logger.info("Reader: Starting work")
        time.sleep(0.5)
        coordinator.logger.info("Reader: Finished reading")
        coordinator.set_finished("reader")

    def processor_worker():
        # Wait to be woken up
        coordinator.wait_for_state("processor")
        coordinator.logger.info("Processor: Starting work")
        time.sleep(0.3)
        coordinator.logger.info("Processor: Finished processing")
        coordinator.set_finished("processor")

    def writer_worker():
        # Wait to be woken up
        coordinator.wait_for_state("writer")
        coordinator.logger.info("Writer: Starting work")
        time.sleep(0.2)
        coordinator.logger.info("Writer: Finished writing")
        coordinator.set_finished("writer")

    def orchestrator():
        # Wake up reader and processor in parallel
        coordinator.logger.info("Orchestrator: Starting readers and processors")
        coordinator.wake(["reader", "processor"])

        # Wait for both to finish
        coordinator.wait(["reader", "processor"])

        # Then wake up writer
        coordinator.logger.info("Orchestrator: Starting writer")
        coordinator.wake("writer")

        # Wait for writer to finish
        coordinator.wait("writer")

        coordinator.logger.info("Orchestrator: All work completed")

    # Start all workers
    threads = [
        threading.Thread(target=reader_worker),
        threading.Thread(target=processor_worker),
        threading.Thread(target=writer_worker),
        threading.Thread(target=orchestrator),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()


def example_multiprocessing():
    """Example with multiple coordinated processes"""
    import time

    def worker(coordinator, state):
        coordinator.wait_for_state(state)
        coordinator.logger.info(f"{state}: Starting work")
        time.sleep(0.3)
        coordinator.logger.info(f"{state}: Finished")
        coordinator.set_finished(state)

    def orchestrator(coordinator):
        # Wake up multiple workers
        coordinator.logger.info("Orchestrator: Waking up all workers")
        coordinator.wake(["reader", "processor", "writer"])

        # Wait for all to finish
        coordinator.wait(["reader", "processor", "writer"])
        coordinator.logger.info("Orchestrator: All workers finished")

        # Reset and run validator alone
        coordinator.set_unfinished()
        coordinator.wake("validator")
        coordinator.wait("validator")
        coordinator.logger.info("Orchestrator: Validation complete")

    if __name__ == '__main__':
        states = ["reader", "processor", "writer", "validator"]
        coordinator = MultiprocessingCoordinator(states)

        processes = [
            multiprocessing.Process(target=worker, args=(coordinator, "reader")),
            multiprocessing.Process(target=worker, args=(coordinator, "processor")),
            multiprocessing.Process(target=worker, args=(coordinator, "writer")),
            multiprocessing.Process(target=worker, args=(coordinator, "validator")),
            multiprocessing.Process(target=orchestrator, args=(coordinator,)),
        ]

        for p in processes:
            p.start()

        for p in processes:
            p.join()


# Example of two-sided coordination (like your original request)
def example_two_sided():
    """Example simulating the original two-sided coordination"""
    import time

    coordinator = ThreadingCoordinator(["side_a", "side_b"])
    def side_a_worker():
        for i in range(3):
            coordinator.wait_for_state("side_a")
            coordinator.logger.info(f"Side A: Working on iteration {i}")
            time.sleep(0.2)
            coordinator.set_finished("side_a")

            # Wake up side B
            coordinator.wake("side_b")

    def side_b_worker():
        for i in range(3):
            coordinator.wait_for_state("side_b")
            coordinator.logger.info(f"Side B: Working on iteration {i}")
            time.sleep(0.2)
            coordinator.set_finished("side_b")

            # Wake up side A for next iteration
            if i < 2:  # Don't wake on last iteration
                coordinator.set_unfinished("side_a")
                coordinator.wake("side_a")

    # Start with side A
    coordinator.wake("side_a")

    threads = [threading.Thread(target=side_a_worker), threading.Thread(target=side_b_worker)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()


if __name__ == '__main__':
    print("=== Coordinator Examples with Independent Logging ===")
    print("(No main logger setup required - each coordinator has independent logging)")
    
    print("\n=== Threading Example ===")
    example_threading()

    print("\n=== Two-Sided Example ===")
    example_two_sided()

    # Note: Multiprocessing example needs to be run directly
    print("\n=== Multiprocessing Example ===")
    print("Note: Multiprocessing example needs to be run directly")
    # example_multiprocessing()
