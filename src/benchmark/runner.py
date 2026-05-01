"""
benchmark/runner.py - Phase 1
Automatically benchmarks multiple Ollama models on the same prompts.

Analogy: Like a car magazine testing Honda vs Toyota vs BMW
on the same track, same conditions, same tests.
Every model gets the same prompts. We measure the same metrics.
Results are saved to a CSV so you can compare side by side.
"""

import json
import csv
import time
import ollama
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table

console = Console()

# ── Config ─────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")

# The three models we compare (matching what you have installed)
MODELS_TO_TEST = [
    "llama3.2:3b",
    "phi4-mini:latest",
    "llama3.1:8b",
]

# Standardised test prompts — same for every model
# Mix of factual, reasoning, and creative tasks
TEST_PROMPTS = [
    "In one sentence, what is machine learning?",
    "What is the difference between RAM and storage?",
    "Write a Python function that reverses a string.",
    "Explain neural networks to a 10-year-old.",
    "What are three benefits of running AI models locally?",
]
# ───────────────────────────────────────────────────────────────────────────


def benchmark_single_model(model: str, prompts: list[str]) -> list[dict]:
    """
    Run all prompts through one model and collect timing metrics.
    Returns a list of result dicts — one per prompt.
    """
    results = []

    console.print(f"\n[bold cyan]Testing: {model}[/bold cyan]")
    console.print("─" * 50)

    # Warm up the model with one silent call first
    # This loads the model into GPU memory so our measurements
    # reflect real usage, not cold start time
    console.print("[dim]Warming up model...[/dim]")
    try:
        ollama.generate(model=model, prompt="hi", options={"num_predict": 1})
    except Exception as e:
        console.print(f"[red]Could not load {model}: {e}[/red]")
        return []

    for i, prompt in enumerate(prompts, 1):
        console.print(f"\n[dim]Prompt {i}/{len(prompts)}:[/dim] {prompt[:60]}...")

        first_token_time = None
        start_time       = time.perf_counter()
        full_response    = ""
        token_count      = 0

        try:
            for chunk in ollama.generate(
                model   = model,
                prompt  = prompt,
                options = {"temperature": 0},  # deterministic for fair comparison
                stream  = True,
            ):
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start_time

                full_response += chunk.get("response", "")
                token_count   += 1

                if chunk.get("done"):
                    break

            total_time = time.perf_counter() - start_time

            result = {
                "model":               model,
                "prompt":              prompt,
                "response":            full_response[:200],  # truncate for storage
                "time_to_first_token": round(first_token_time or 0, 3),
                "total_time":          round(total_time, 3),
                "tokens_generated":    token_count,
                "tokens_per_second":   round(token_count / total_time, 1) if total_time > 0 else 0,
                "timestamp":           datetime.now().isoformat(),
            }

            results.append(result)

            console.print(
                f"  [green]✓[/green] "
                f"{result['tokens_per_second']} tok/s | "
                f"first token: {result['time_to_first_token']*1000:.0f}ms | "
                f"total: {result['total_time']:.2f}s"
            )

        except Exception as e:
            console.print(f"  [red]✗ Error: {e}[/red]")

    return results


def save_results(all_results: list[dict]):
    """Save benchmark results to both JSON and CSV."""
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON (complete data)
    json_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save as CSV (easy to open in Excel/Sheets)
    csv_path = RESULTS_DIR / f"benchmark_{timestamp}.csv"
    if all_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    console.print(f"\n[green]Results saved:[/green]")
    console.print(f"  JSON: {json_path}")
    console.print(f"  CSV:  {csv_path}")

    return csv_path


def print_summary_table(all_results: list[dict]):
    """
    Print a formatted comparison table using Rich.
    Shows average metrics per model side by side.

    This is the table that goes in your README and portfolio report.
    """
    # Group results by model
    model_stats = {}
    for r in all_results:
        model = r["model"]
        if model not in model_stats:
            model_stats[model] = {
                "tokens_per_second":   [],
                "time_to_first_token": [],
                "total_time":          [],
            }
        model_stats[model]["tokens_per_second"].append(r["tokens_per_second"])
        model_stats[model]["time_to_first_token"].append(r["time_to_first_token"])
        model_stats[model]["total_time"].append(r["total_time"])

    # Build Rich table
    table = Table(
        title       = "Model Benchmark Results",
        show_header = True,
        header_style= "bold magenta",
    )
    table.add_column("Model",              style="cyan",  min_width=20)
    table.add_column("Avg Tokens/sec",     style="green", justify="right")
    table.add_column("Avg First Token ms", style="yellow",justify="right")
    table.add_column("Avg Total Time s",   style="blue",  justify="right")
    table.add_column("Prompts Tested",     justify="right")

    for model, stats in model_stats.items():
        avg_tps   = sum(stats["tokens_per_second"])   / len(stats["tokens_per_second"])
        avg_ttft  = sum(stats["time_to_first_token"]) / len(stats["time_to_first_token"])
        avg_total = sum(stats["total_time"])           / len(stats["total_time"])
        count     = len(stats["tokens_per_second"])

        table.add_row(
            model,
            f"{avg_tps:.1f}",
            f"{avg_ttft*1000:.0f}",
            f"{avg_total:.2f}",
            str(count),
        )

    console.print("\n")
    console.print(table)


def run_benchmark(models: list[str] = None, prompts: list[str] = None):
    """Run the full benchmark suite."""
    models  = models  or MODELS_TO_TEST
    prompts = prompts or TEST_PROMPTS

    console.print("\n[bold]Local Model Benchmark[/bold]")
    console.print(f"Models:  {', '.join(models)}")
    console.print(f"Prompts: {len(prompts)} standardised prompts")
    console.print(f"Note: Each model is warmed up before measurement\n")

    all_results = []

    for model in models:
        results = benchmark_single_model(model, prompts)
        all_results.extend(results)

    if all_results:
        print_summary_table(all_results)
        save_results(all_results)

    return all_results


if __name__ == "__main__":
    run_benchmark()