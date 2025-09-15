import cProfile
import io
import pstats
import threading
from contextlib import contextmanager


class PausableProfiler:
    def __init__(self):
        self.profiler = None
        self.is_profiling = False
        self._lock = threading.Lock()
        self._has_data = False

    def _ensure_profiler(self):
        """Create profiler on first use to avoid conflicts"""
        if self.profiler is None:
            self.profiler = cProfile.Profile()

    def start(self):
        with self._lock:
            if not self.is_profiling:
                self._ensure_profiler()
                try:
                    self.profiler.enable()
                    self.is_profiling = True
                    self._has_data = True
                except ValueError as e:
                    # Another profiler is active, skip
                    print(f"Warning: {e}")

    def pause(self):
        with self._lock:
            if self.is_profiling and self.profiler is not None:
                try:
                    self.profiler.disable()
                    self.is_profiling = False
                except ValueError:
                    # Already disabled, ignore
                    pass

    def get_stats(self):
        """Get stats object, handling edge cases"""
        with self._lock:
            if self.profiler is None or not self._has_data:
                # Return empty stats
                return pstats.Stats()

            # Make sure profiler is disabled before creating stats
            if self.is_profiling:
                self.profiler.disable()
                self.is_profiling = False

            # Create stats from the profiler's data
            # Use a StringIO buffer to avoid direct construction issues
            s = io.StringIO()
            self.profiler.dump_stats(s)
            s.seek(0)
            return pstats.Stats(s)

    def save_stats(self, filename):
        """Save stats to file for later analysis"""
        with self._lock:
            if self.profiler is not None and self._has_data:
                # Ensure profiler is disabled
                if self.is_profiling:
                    self.profiler.disable()
                    self.is_profiling = False

                # Dump stats directly to file
                self.profiler.dump_stats(filename)
                print(f"Saved profiling stats to {filename}")
            else:
                print(f"No profiling data to save for {filename}")

    def print_stats(self, sort_by='cumulative', top_n=30):
        """Print formatted statistics"""
        stats = self.get_stats()
        if stats.total_calls > 0:  # Only print if we have data
            stats.sort_stats(sort_by)
            stats.print_stats(top_n)
            return stats
        else:
            print("No profiling data available")
            return None


# Thread-local storage for profilers
_thread_local = threading.local()


def get_profiler():
    """Get thread-local profiler instance"""
    if not hasattr(_thread_local, 'profiler'):
        _thread_local.profiler = PausableProfiler()
    return _thread_local.profiler


@contextmanager
def profile_active():
    """Profile only active computation, not waiting"""
    profiler = get_profiler()
    profiler.start()
    try:
        yield
    finally:
        profiler.pause()


@contextmanager
def pause_profiling():
    """Temporarily pause profiling during waits"""
    profiler = get_profiler()
    was_profiling = profiler.is_profiling
    if was_profiling:
        profiler.pause()
    try:
        yield
    finally:
        if was_profiling:
            profiler.start()
