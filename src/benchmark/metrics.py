import statistics
from dataclasses import dataclass, field
from typing import Optional
from src.utils.ollama_client import GenerationResult


@dataclass
class BenchmarkStats:
    """
    Aggregated statistics from multiple runs of the same model and prompt.
    
    Instead of reporting one number, we report mean ± std across N runs.
    This is honest — it shows both the average performance AND how consistent it is.
    
    High std dev = inconsistent (bad — could be thermal throttling or memory pressure)
    Low std dev  = consistent  (good — reliable performance)
    """
    model: str
    prompt_category: str
    num_runs: int           # how many times we ran the test total
    successful_runs: int    # how many succeeded (failed runs are excluded from stats)

    # Tokens per second — higher is better
    tps_mean: float         # average tokens/sec across all successful runs
    tps_std: float          # standard deviation — how much it varied
    tps_min: float          # worst run
    tps_max: float          # best run

    # Time to first token — lower is better
    ttft_mean: float
    ttft_std: float
    ttft_min: float
    ttft_max: float

    # Total end-to-end latency — lower is better
    latency_mean: float
    latency_std: float
    latency_min: float
    latency_max: float

    # Token counts
    avg_tokens_generated: float
    avg_prompt_tokens: float

    def to_dict(self) -> dict:
        """
        Converts this object to a plain dictionary.
        We need this to save results to CSV and JSON later.
        """
        return {
            "model": self.model,
            "prompt_category": self.prompt_category,
            "num_runs": self.num_runs,
            "successful_runs": self.successful_runs,
            "success_rate": round(self.successful_runs / self.num_runs, 3),
            "tps_mean": round(self.tps_mean, 2),
            "tps_std": round(self.tps_std, 2),
            "tps_min": round(self.tps_min, 2),
            "tps_max": round(self.tps_max, 2),
            "ttft_mean": round(self.ttft_mean, 4),
            "ttft_std": round(self.ttft_std, 4),
            "ttft_min": round(self.ttft_min, 4),
            "ttft_max": round(self.ttft_max, 4),
            "latency_mean": round(self.latency_mean, 4),
            "latency_std": round(self.latency_std, 4),
            "avg_tokens_generated": round(self.avg_tokens_generated, 1),
            "avg_prompt_tokens": round(self.avg_prompt_tokens, 1),
        }


def safe_stdev(values: list[float]) -> float:
    """
    Standard deviation requires at least 2 data points.
    If we only have 1 successful run, return 0 instead of crashing.
    """
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def compute_stats(
    results: list[GenerationResult],
    model: str,
    prompt_category: str,
) -> BenchmarkStats:
    """
    Takes a list of GenerationResults and computes aggregate statistics.
    
    The flow:
    1. Filter out failed runs
    2. Extract the numbers we care about into lists
    3. Compute mean, std, min, max for each metric
    4. Return a BenchmarkStats object
    """

    # Step 1 — only use runs that actually succeeded
    # A failed run has success=False and tokens_generated=0
    # Including them would drag down the averages unfairly
    successful = [r for r in results if r.success and r.tokens_generated > 0]

    num_runs = len(results)
    successful_runs = len(successful)

    # If every single run failed, return zeroes
    # This means the model couldn't load (probably out of memory)
    if not successful:
        return BenchmarkStats(
            model=model,
            prompt_category=prompt_category,
            num_runs=num_runs,
            successful_runs=0,
            tps_mean=0, tps_std=0, tps_min=0, tps_max=0,
            ttft_mean=0, ttft_std=0, ttft_min=0, ttft_max=0,
            latency_mean=0, latency_std=0, latency_min=0, latency_max=0,
            avg_tokens_generated=0,
            avg_prompt_tokens=0,
        )

    # Step 2 — extract each metric into its own list
    # This makes the statistics calculations below clean and readable
    tps_values      = [r.tokens_per_second for r in successful]
    ttft_values     = [r.time_to_first_token for r in successful]
    latency_values  = [r.total_time for r in successful]
    token_counts    = [r.tokens_generated for r in successful]
    prompt_counts   = [r.prompt_tokens for r in successful]

    # Step 3 — compute statistics for each metric
    return BenchmarkStats(
        model=model,
        prompt_category=prompt_category,
        num_runs=num_runs,
        successful_runs=successful_runs,

        tps_mean=statistics.mean(tps_values),
        tps_std=safe_stdev(tps_values),
        tps_min=min(tps_values),
        tps_max=max(tps_values),

        ttft_mean=statistics.mean(ttft_values),
        ttft_std=safe_stdev(ttft_values),
        ttft_min=min(ttft_values),
        ttft_max=max(ttft_values),

        latency_mean=statistics.mean(latency_values),
        latency_std=safe_stdev(latency_values),
        latency_min=min(latency_values),
        latency_max=max(latency_values),

        avg_tokens_generated=statistics.mean(token_counts),
        avg_prompt_tokens=statistics.mean(prompt_counts),
    )