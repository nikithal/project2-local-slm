"""
timer.py - Timing utilities for benchmarking.

Analogy: Like a stopwatch with multiple lap buttons.
You can measure total time, time to first response,
and time for the full response separately.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class TimingResult:
    """
    Stores all timing measurements for one model call.

    time_to_first_token: how long before model starts responding
    total_time:          how long the entire response took
    tokens_generated:    how many tokens the model produced
    tokens_per_second:   generation speed
    """
    time_to_first_token: float   # seconds
    total_time: float            # seconds
    tokens_generated: int
    tokens_per_second: float

    def display(self):
        """Print a formatted summary of the timing results."""
        print(f"  Time to first token: {self.time_to_first_token * 1000:.0f}ms")
        print(f"  Total time:          {self.total_time:.2f}s")
        print(f"  Tokens generated:    {self.tokens_generated}")
        print(f"  Tokens per second:   {self.tokens_per_second:.1f}")


@contextmanager
def timer():
    """
    Context manager to measure elapsed time in milliseconds.

    Usage:
        with timer() as get_elapsed:
            do_something()
        ms = get_elapsed()
    """
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000


def measure_call(model_response_generator) -> TimingResult:
    """
    Measures timing for a streaming Ollama response.

    The model streams tokens one at a time.
    We measure when the FIRST token arrives (time_to_first_token)
    and when the LAST token arrives (total_time).

    Analogy: Like measuring how long before a tap starts running
    (time to first token) vs how long to fill a glass (total time).
    """
    first_token_time = None
    start_time       = time.perf_counter()
    tokens_generated = 0
    full_response    = ""

    for chunk in model_response_generator:
        # Record when first token arrives
        if first_token_time is None:
            first_token_time = time.perf_counter() - start_time

        tokens_generated += 1
        full_response    += chunk.get("response", "")

    total_time = time.perf_counter() - start_time

    return TimingResult(
        time_to_first_token = first_token_time or 0,
        total_time          = total_time,
        tokens_generated    = tokens_generated,
        tokens_per_second   = tokens_generated / total_time if total_time > 0 else 0,
    ), full_response