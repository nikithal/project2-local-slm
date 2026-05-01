"""
cli.py - Command line interface for the local AI assistant.
Demonstrates Phase 1 (chat) and Phase 2 (structured output).

Run options:
  python cli.py chat       → free text chat with local model
  python cli.py sentiment  → structured sentiment analysis
  python cli.py extract    → structured information extraction
  python cli.py classify   → structured task classification
  python cli.py temperature → compare temperature settings
  python cli.py benchmark  → run full benchmark suite
"""

import sys
from rich.console import Console
from rich.panel import Panel

console = Console()


def run_chat():
    """Phase 1: Free text chat with local model."""
    from src.assistant import chat

    console.print(Panel(
        "[bold]Local AI Chat[/bold]\n"
        "Model: llama3.2:3b (running on your M2 chip)\n"
        "Type 'quit' to exit",
        style="cyan"
    ))

    while True:
        question = console.input("\n[bold green]You:[/bold green] ").strip()

        if question.lower() in ("quit", "exit", "q"):
            break

        if not question:
            continue

        console.print("[bold blue]AI:[/bold blue] ", end="")
        result = chat(question, model="llama3.2:3b")

        console.print(
            f"\n[dim]({result['tokens_per_second']:.1f} tok/s | "
            f"{result['total_time']:.2f}s total)[/dim]"
        )


def run_sentiment():
    """Phase 2: Structured sentiment analysis."""
    from src.structured_caller import call_with_schema
    from src.schemas.models import SentimentResult

    console.print(Panel("[bold]Structured Sentiment Analysis[/bold]", style="magenta"))

    test_texts = [
        "I absolutely love this product, it exceeded all my expectations!",
        "The service was terrible and the staff were rude.",
        "The meeting was scheduled for Tuesday.",
    ]

    for text in test_texts:
        console.print(f"\n[bold]Text:[/bold] {text}")
        result, meta = call_with_schema(
            prompt = f"Analyse the sentiment of this text: {text}",
            schema = SentimentResult,
        )

        if result:
            console.print(f"  Sentiment:  [bold]{result.sentiment.value}[/bold]")
            console.print(f"  Confidence: {result.confidence}")
            console.print(f"  Reasoning:  {result.reasoning}")
            console.print(f"  Attempts:   {meta['attempts']}")
        else:
            console.print(f"  [red]Failed after {meta['attempts']} attempts[/red]")


def run_extract():
    """Phase 2: Structured information extraction."""
    from src.structured_caller import call_with_schema
    from src.schemas.models import ExtractedInfo

    console.print(Panel("[bold]Structured Information Extraction[/bold]", style="magenta"))

    test_texts = [
        "Sarah Johnson, 28, recently moved to London from New York. She works as a data scientist.",
        "The package was delivered yesterday. No sender information was provided.",
        "Dr. Ahmed Al-Rashid, 45, is a cardiologist based in Dubai with 20 years of experience.",
    ]

    for text in test_texts:
        console.print(f"\n[bold]Text:[/bold] {text}")
        result, meta = call_with_schema(
            prompt = f"Extract information from this text: {text}",
            schema = ExtractedInfo,
        )

        if result:
            console.print(f"  Name:     {result.name or 'Not found'}")
            console.print(f"  Age:      {result.age or 'Not found'}")
            console.print(f"  City:     {result.city or 'Not found'}")
            console.print(f"  Job:      {result.job or 'Not found'}")
            console.print(f"  Summary:  {result.summary}")
            console.print(f"  Attempts: {meta['attempts']}")


def run_classify():
    """Phase 2: Structured task classification."""
    from src.structured_caller import call_with_schema
    from src.schemas.models import ClassifiedTask

    console.print(Panel("[bold]Structured Task Classification[/bold]", style="magenta"))

    test_inputs = [
        "What is the capital of France?",
        "Send an email to John about the meeting.",
        "The weather today is sunny.",
        "Hello, how are you?",
    ]

    for text in test_inputs:
        console.print(f"\n[bold]Input:[/bold] {text}")
        result, meta = call_with_schema(
            prompt = f"Classify this user input: {text}",
            schema = ClassifiedTask,
        )

        if result:
            console.print(f"  Type:        [bold]{result.task_type.value}[/bold]")
            console.print(f"  Intent:      {result.intent}")
            console.print(f"  Needs AI:    {result.requires_ai}")
            console.print(f"  Attempts:    {meta['attempts']}")


def run_temperature_comparison():
    """Phase 2: Compare temperature=0 vs temperature=0.7."""
    from src.structured_caller import compare_temperatures
    from src.schemas.models import SentimentResult

    compare_temperatures(
        prompt = "Analyse the sentiment: The food was okay, nothing special.",
        schema = SentimentResult,
    )


def run_benchmark():
    """Phase 1: Full model benchmark suite."""
    from src.benchmark.runner import run_benchmark
    run_benchmark()

def run_model_comparison():
    """Phase 3a: Compare three models side by side."""
    from src.comparison.model_comparison import run_model_comparison
    run_model_comparison()


def run_quantization_study():
    """Phase 3b: Compare quantization levels on same model."""
    from src.comparison.model_comparison import run_quantization_study
    run_quantization_study()


def run_full_report():
    """Phase 3: Run both studies and generate markdown report."""
    from src.comparison.model_comparison import (
        run_model_comparison,
        run_quantization_study,
        generate_report,
    )
    comp_results,  comp_quality  = run_model_comparison()
    quant_results, quant_quality = run_quantization_study()
    generate_report(comp_results, comp_quality, quant_results, quant_quality)
# ── Main entry point ────────────────────────────────────────────────────────
COMMANDS = {
    "chat":        (run_chat,       "Free text chat with local model"),
    "sentiment":   (run_sentiment,  "Structured sentiment analysis"),
    "extract":     (run_extract,    "Structured information extraction"),
    "classify":    (run_classify,   "Structured task classification"),
    "temperature": (run_temperature_comparison, "Compare temperature settings"),
    "benchmark":   (run_benchmark,  "Run full model benchmark"),
    "model_comparison": (run_model_comparison, "Compare three models side by side"),
    "quantization_study": (run_quantization_study, "Compare quantization levels on same model"),
    "full_report": (run_full_report, "Run both studies and generate markdown report"),
    "compare":     (run_model_comparison,  "Phase 3a: Compare 3 models"),        # ← ADD
    "quantize":    (run_quantization_study,"Phase 3b: Quantization study"),      # ← ADD
    "report":      (run_full_report,       "Phase 3: Full report"),              # ← ADD
}



if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        console.print("\n[bold]Local AI Assistant — Project 2[/bold]")
        console.print("Running entirely offline on your M2 Mac\n")
        console.print("Usage: python cli.py [command]\n")
        console.print("Commands:")
        for cmd, (_, description) in COMMANDS.items():
            console.print(f"  [cyan]{cmd:<12}[/cyan] {description}")
        console.print()
        sys.exit(0)

    command = sys.argv[1]
    func, _ = COMMANDS[command]
    func()