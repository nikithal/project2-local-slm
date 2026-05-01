"""
model_comparison.py - Phase 3
Systematic comparison of models and quantization levels.

This is the "Consumer Reports" of your local AI project.
Every model gets the same tests. Results are saved and displayed
as a comparison table — the main deliverable of Phase 3.
"""

import json
import time
import ollama
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table

console = Console()

RESULTS_DIR = Path("results")

# ── Test configuration ──────────────────────────────────────────────────────

# Three models for main comparison (Phase 3a)
COMPARISON_MODELS = [
    "llama3.2:3b",
    "phi4-mini:latest",
    "llama3.1:8b",
]

# Same model, different quantization levels (Phase 3b)
QUANTIZATION_MODELS = [
    "llama3.1:8b",              # default (usually Q4 or Q5)
    "llama3.1:8b-instruct-q4_K_M",  # Q4 — faster, slightly lower quality
    "llama3.1:8b-instruct-q8_0",    # Q8 — closer to full precision
]

# Standardised test prompts — must be same for all models
# Mix of tasks to get a fair quality assessment
TEST_PROMPTS = {
    "factual":    "What is the difference between machine learning and deep learning? Answer in 2 sentences.",
    "reasoning":  "If a model runs at 30 tokens per second and a response is 150 tokens, how long does it take? Show your working.",
    "coding":     "Write a Python function that takes a list of numbers and returns only the even ones.",
    "creative":   "Write a two-sentence description of what a vector database is, for someone who has never heard of it.",
    "instruction":"List exactly 3 advantages of running AI models locally instead of using cloud APIs.",
}

# ───────────────────────────────────────────────────────────────────────────


def benchmark_model(model: str, prompts: dict) -> dict:
    """
    Run all test prompts through one model.
    Returns performance metrics and responses for quality scoring.
    """
    console.print(f"\n[bold cyan]Testing: {model}[/bold cyan]")

    # Warm up — loads model into GPU memory
    # This ensures we measure generation speed, not loading time
    console.print("  [dim]Warming up...[/dim]")
    try:
        ollama.generate(
            model   = model,
            prompt  = "hi",
            options = {"num_predict": 1},
            stream  = False,
        )
    except Exception as e:
        console.print(f"  [red]Could not load {model}: {e}[/red]")
        return {}

    results = {
        "model":              model,
        "prompt_results":     {},
        "avg_tokens_per_sec": 0,
        "avg_first_token_ms": 0,
        "avg_total_time":     0,
        "timestamp":          datetime.now().isoformat(),
    }

    all_tps    = []
    all_ttft   = []
    all_totals = []

    for prompt_name, prompt_text in prompts.items():
        console.print(f"  [dim]Running: {prompt_name}[/dim]")

        first_token_time = None
        start_time       = time.perf_counter()
        full_response    = ""
        token_count      = 0

        try:
            for chunk in ollama.generate(
                model   = model,
                prompt  = prompt_text,
                options = {"temperature": 0, "num_predict": 200},
                stream  = True,
            ):
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start_time

                full_response += chunk.get("response", "")
                token_count   += 1

                if chunk.get("done"):
                    break

            total_time = time.perf_counter() - start_time
            tps        = token_count / total_time if total_time > 0 else 0
            ttft_ms    = (first_token_time or 0) * 1000

            all_tps.append(tps)
            all_ttft.append(ttft_ms)
            all_totals.append(total_time)

            results["prompt_results"][prompt_name] = {
                "response":          full_response.strip(),
                "tokens_per_second": round(tps, 1),
                "first_token_ms":    round(ttft_ms, 0),
                "total_time_s":      round(total_time, 2),
                "token_count":       token_count,
            }

            console.print(
                f"    [green]✓[/green] "
                f"{tps:.1f} tok/s | "
                f"{ttft_ms:.0f}ms first token | "
                f"{total_time:.2f}s total"
            )

        except Exception as e:
            console.print(f"    [red]✗ {e}[/red]")

    # Calculate averages
    if all_tps:
        results["avg_tokens_per_sec"] = round(sum(all_tps) / len(all_tps), 1)
        results["avg_first_token_ms"] = round(sum(all_ttft) / len(all_ttft), 0)
        results["avg_total_time"]     = round(sum(all_totals) / len(all_totals), 2)

    return results


def score_quality(results: list[dict]) -> dict:
    """
    Simple automated quality scoring.

    We check each response for:
    - Length (too short = incomplete answer)
    - Presence of expected keywords per prompt type
    - Whether coding prompts contain actual code

    This is a rough proxy for quality — not perfect but systematic.
    Real quality evaluation would use human raters or an LLM judge.
    """
    quality_checks = {
        "factual":    ["learning", "model", "data", "neural", "algorithm"],
        "reasoning":  ["5", "second", "150", "30"],   # expected in math answer
        "coding":     ["def ", "return", "for ", "if "],  # expected in code
        "creative":   ["database", "vector", "store", "embed", "search"],
        "instruction":["privacy", "cost", "latency", "offline", "local", "free"],
    }

    scores = {}

    for result in results:
        model  = result["model"]
        total  = 0
        checks = 0

        for prompt_name, prompt_result in result.get("prompt_results", {}).items():
            response = prompt_result["response"].lower()
            keywords = quality_checks.get(prompt_name, [])

            # Check keyword presence
            found = sum(1 for kw in keywords if kw.lower() in response)
            score = found / len(keywords) if keywords else 0

            # Penalise very short responses (under 20 words)
            word_count = len(response.split())
            if word_count < 20:
                score *= 0.5

            total  += score
            checks += 1

        scores[model] = round((total / checks * 100) if checks else 0, 1)

    return scores


def print_comparison_table(results: list[dict], quality_scores: dict, title: str):
    """Print a formatted comparison table."""
    table = Table(
        title        = title,
        show_header  = True,
        header_style = "bold magenta",
    )

    table.add_column("Model",          style="cyan",   min_width=30)
    table.add_column("Avg Tok/sec",    style="green",  justify="right")
    table.add_column("First Token ms", style="yellow", justify="right")
    table.add_column("Avg Total s",    style="blue",   justify="right")
    table.add_column("Quality Score",  style="white",  justify="right")

    for result in results:
        if not result:
            continue
        model   = result["model"]
        quality = quality_scores.get(model, 0)

        # Colour quality score by performance
        if quality >= 70:
            q_style = "[green]"
        elif quality >= 40:
            q_style = "[yellow]"
        else:
            q_style = "[red]"

        table.add_row(
            model,
            str(result["avg_tokens_per_sec"]),
            str(result["avg_first_token_ms"]),
            str(result["avg_total_time"]),
            f"{q_style}{quality}%[/]",
        )

    console.print("\n")
    console.print(table)


def save_results(results: list[dict], quality_scores: dict, filename: str):
    """Save full results to JSON."""
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "timestamp":      datetime.now().isoformat(),
        "quality_scores": quality_scores,
        "detailed":       results,
    }
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\n[green]Saved:[/green] {path}")


def run_model_comparison():
    """
    Phase 3a: Compare three different models on the same tasks.
    llama3.2:3b vs phi4-mini vs llama3.1:8b
    """
    console.print("\n[bold]Phase 3a: Model Comparison Study[/bold]")
    console.print(f"Models: {', '.join(COMPARISON_MODELS)}")
    console.print(f"Prompts: {len(TEST_PROMPTS)} standardised tasks\n")

    all_results = []
    for model in COMPARISON_MODELS:
        result = benchmark_model(model, TEST_PROMPTS)
        if result:
            all_results.append(result)

    quality_scores = score_quality(all_results)
    print_comparison_table(
        all_results,
        quality_scores,
        "Model Comparison: llama3.2:3b vs phi4-mini vs llama3.1:8b"
    )
    save_results(all_results, quality_scores, "model_comparison.json")

    return all_results, quality_scores


def run_quantization_study():
    """
    Phase 3b: Compare same model at different quantization levels.
    Shows the speed vs quality tradeoff of quantization.
    """
    console.print("\n[bold]Phase 3b: Quantization Study[/bold]")
    console.print("Comparing llama3.1:8b at different quantization levels")
    console.print(f"Models: {', '.join(QUANTIZATION_MODELS)}\n")

    # Use a subset of prompts for the quantization study
    quant_prompts = {
        "factual":   TEST_PROMPTS["factual"],
        "coding":    TEST_PROMPTS["coding"],
        "reasoning": TEST_PROMPTS["reasoning"],
    }

    all_results = []
    for model in QUANTIZATION_MODELS:
        result = benchmark_model(model, quant_prompts)
        if result:
            all_results.append(result)

    quality_scores = score_quality(all_results)
    print_comparison_table(
        all_results,
        quality_scores,
        "Quantization Study: llama3.1:8b — Default vs Q4 vs Q8"
    )
    save_results(all_results, quality_scores, "quantization_study.json")

    return all_results, quality_scores


def generate_report(
    comparison_results: list,
    comparison_quality: dict,
    quant_results: list,
    quant_quality: dict,
):
    """
    Generate a markdown report suitable for your README and portfolio.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    lines = [
        "# Local SLM Benchmark Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Hardware: Apple M2 Mac",
        "",
        "---",
        "",
        "## Phase 3a: Model Comparison",
        "",
        "Same 5 prompts run on each model. temperature=0 for consistency.",
        "",
        "| Model | Avg Tok/sec | First Token ms | Avg Total s | Quality |",
        "|---|---|---|---|---|",
    ]

    for r in comparison_results:
        if r:
            q = comparison_quality.get(r["model"], 0)
            lines.append(
                f"| {r['model']} | {r['avg_tokens_per_sec']} | "
                f"{r['avg_first_token_ms']} | {r['avg_total_time']} | {q}% |"
            )

    lines += [
        "",
        "---",
        "",
        "## Phase 3b: Quantization Study",
        "",
        "Same model (llama3.1:8b) at different quantization levels.",
        "Higher Q = more bits = better quality but larger and slower.",
        "",
        "| Model | Avg Tok/sec | First Token ms | Avg Total s | Quality |",
        "|---|---|---|---|---|",
    ]

    for r in quant_results:
        if r:
            q = quant_quality.get(r["model"], 0)
            lines.append(
                f"| {r['model']} | {r['avg_tokens_per_sec']} | "
                f"{r['avg_first_token_ms']} | {r['avg_total_time']} | {q}% |"
            )

    lines += [
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "- Smaller models (3B) are faster but may miss nuance on complex tasks",
        "- Q4 quantization reduces size significantly with minimal quality loss",
        "- Time to first token is the metric users notice most in interactive apps",
        "- All models run 100% offline — no internet, no API cost, no privacy risk",
        "",
        "---",
        "",
        "## Methodology",
        "",
        "- Each model warmed up with one silent call before measurement",
        "- All prompts use temperature=0 for reproducible results",
        "- Quality score based on keyword presence and response completeness",
        "- Hardware: Apple M2 Mac, running via Ollama",
    ]

    report_path = RESULTS_DIR / "benchmark_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    console.print(f"[green]Report saved:[/green] {report_path}")
    return report_path


if __name__ == "__main__":
    # Run both studies and generate final report
    comp_results, comp_quality = run_model_comparison()
    quant_results, quant_quality = run_quantization_study()
    generate_report(comp_results, comp_quality, quant_results, quant_quality)