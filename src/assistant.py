"""
assistant.py - Phase 1
Local AI assistant using Ollama.

Analogy: This is the expert who now lives in your house.
Instead of calling an API over the internet, you call
a function that talks to a model running on your own chip.

No internet. No API key. No cost. No privacy risk.
"""

import time
import ollama
import psutil
import os
from src.utils.timer import TimingResult


# ── Config ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "llama3.2:3b"
# ───────────────────────────────────────────────────────────────────────────


def get_memory_usage_mb() -> float:
    """
    Returns current RAM usage of this Python process in MB.
    We use this to measure how much memory each model consumes.

    Analogy: Like checking how much of your house the expert
    is taking up while they are working.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def chat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0,
    stream: bool = True,
) -> dict:
    """
    Send a prompt to a local Ollama model and measure performance.

    Returns a dict with:
    - response:            the full text response
    - time_to_first_token: seconds before first token arrived
    - total_time:          total seconds for full response
    - tokens_per_second:   generation speed
    - model:               which model was used

    temperature=0 means deterministic output (same answer every time)
    temperature=0.7 means creative/varied output
    """
    memory_before = get_memory_usage_mb()
    start_time    = time.perf_counter()

    first_token_time = None
    full_response    = ""
    token_count      = 0

    if stream:
        # Stream mode: tokens arrive one by one
        # This lets us measure time to FIRST token precisely
        for chunk in ollama.generate(
            model    = model,
            prompt   = prompt,
            options  = {"temperature": temperature},
            stream   = True,
        ):
            # Record when first token arrives
            if first_token_time is None:
                first_token_time = time.perf_counter() - start_time

            token       = chunk.get("response", "")
            full_response += token
            token_count  += 1

            # Print as it streams (like watching ChatGPT type)
            print(token, end="", flush=True)

            if chunk.get("done"):
                break

        print()  # new line after streaming finishes

    else:
        # Non-stream mode: wait for complete response
        response         = ollama.generate(
            model   = model,
            prompt  = prompt,
            options = {"temperature": temperature},
        )
        first_token_time = time.perf_counter() - start_time
        full_response    = response["response"]
        token_count      = response.get("eval_count", 0)

    total_time    = time.perf_counter() - start_time
    memory_after  = get_memory_usage_mb()

    return {
        "response":            full_response,
        "model":               model,
        "temperature":         temperature,
        "time_to_first_token": first_token_time or 0,
        "total_time":          total_time,
        "tokens_generated":    token_count,
        "tokens_per_second":   token_count / total_time if total_time > 0 else 0,
        "memory_used_mb":      memory_after - memory_before,
    }


def test_model(model: str = DEFAULT_MODEL):
    """
    Quick test to verify a model is working correctly.
    Sends a simple prompt and prints timing results.
    """
    print(f"\n{'='*50}")
    print(f"Testing model: {model}")
    print(f"{'='*50}")

    test_prompt = "In one sentence, what is machine learning?"
    print(f"\nPrompt: {test_prompt}")
    print(f"\nResponse: ", end="")

    result = chat(test_prompt, model=model)

    print(f"\n--- Timing Results ---")
    print(f"Time to first token: {result['time_to_first_token']*1000:.0f}ms")
    print(f"Total time:          {result['total_time']:.2f}s")
    print(f"Tokens per second:   {result['tokens_per_second']:.1f}")
    print(f"Memory used:         {result['memory_used_mb']:.1f}MB")

    return result


if __name__ == "__main__":
    # Run a quick test of the default model
    test_model(DEFAULT_MODEL)