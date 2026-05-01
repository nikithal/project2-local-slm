"""
structured_caller.py - Phase 2
Forces the LLM to return valid JSON matching a Pydantic schema.
Implements retry logic when validation fails.
"""

import json
import re
import ollama
from pydantic import BaseModel, ValidationError
from typing import Type, TypeVar
from rich.console import Console

console = Console()

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "llama3.2:3b"
MAX_RETRIES   = 3


def extract_json(text: str) -> str:
    """Extract JSON from model output that might have extra text around it."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group()
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group()
    return text


def build_json_prompt(user_prompt: str, schema: Type[T]) -> str:
    """
    Build a prompt using a concrete JSON example.
    Shows the model exactly what format to return — no descriptions
    that could bleed into the values.
    """
    # Build a concrete example showing exact types expected
    example = {}
    for field_name, field in schema.model_fields.items():
        annotation = str(field.annotation)
        if "float" in annotation:
            example[field_name] = 0.95
        elif "int" in annotation:
            example[field_name] = 42
        elif "bool" in annotation:
            example[field_name] = True
        elif "list" in annotation:
            example[field_name] = ["example"]
        elif "SentimentLabel" in annotation:
            example[field_name] = "positive"
        elif "TaskType" in annotation:
            example[field_name] = "question"
        else:
            example[field_name] = "your answer here"

    example_str = json.dumps(example, indent=2)

    return f"""Respond with ONLY a JSON object. No explanation. No markdown. No extra text after values.

Task: {user_prompt}

Return exactly this structure with your answers filled in:
{example_str}

Rules:
- Use only lowercase for string values like sentiment or task type
- Numbers must be plain numbers like 0.95 not text
- No dashes, no descriptions, no extra words after values
- Your entire response must be valid JSON

JSON:"""


def call_with_schema(
    prompt: str,
    schema: Type[T],
    model: str = DEFAULT_MODEL,
    temperature: float = 0,
    max_retries: int = MAX_RETRIES,
) -> tuple:
    """
    Call the LLM and validate output against a Pydantic schema.
    Retries up to max_retries times if validation fails.
    Returns (validated_result, metadata_dict).
    """
    metadata = {
        "model":       model,
        "attempts":    0,
        "success":     False,
        "errors":      [],
        "temperature": temperature,
    }

    full_prompt    = build_json_prompt(prompt, schema)
    current_prompt = full_prompt

    for attempt in range(1, max_retries + 1):
        metadata["attempts"] = attempt
        console.print(f"  Attempt {attempt}/{max_retries}...", end=" ")

        try:
            response = ollama.generate(
                model   = model,
                prompt  = current_prompt,
                options = {
                    "temperature": temperature,
                    "num_predict": 300,
                },
                stream  = False,
            )

            raw_output = response["response"].strip()
            console.print(f"got response")
            console.print(f"  [dim]Raw: {raw_output[:150]}[/dim]")

            json_str = extract_json(raw_output)

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON: {e}"
                metadata["errors"].append(error_msg)
                console.print(f"  [yellow]JSON error: {e}[/yellow]")

                current_prompt = f"""You returned invalid JSON. Fix it.

Error: {error_msg}
Your response was: {raw_output[:150]}

Return ONLY valid JSON with no extra text:
{json.dumps({k: "value" for k in schema.model_fields.keys()}, indent=2)}"""
                continue

            try:
                # Make sentiment/enum values lowercase before validating
                for key in data:
                    if isinstance(data[key], str):
                        data[key] = data[key].lower()

                validated = schema.model_validate(data)
                metadata["success"] = True
                console.print(f"  [green]✓ Valid on attempt {attempt}[/green]")
                return validated, metadata

            except ValidationError as e:
                error_msg = str(e)
                metadata["errors"].append(error_msg)
                console.print(f"  [yellow]Validation error: {e.error_count()} field(s)[/yellow]")

                current_prompt = f"""Your JSON failed validation.
Errors: {error_msg}
Your response: {raw_output[:150]}

Return corrected JSON only:"""
                continue

        except Exception as e:
            error_msg = f"Error: {e}"
            metadata["errors"].append(error_msg)
            console.print(f"  [red]{error_msg}[/red]")

    console.print(f"  [red]✗ Failed after {max_retries} attempts[/red]")
    return None, metadata


def compare_temperatures(
    prompt: str,
    schema: Type[T],
    model: str = DEFAULT_MODEL,
):
    """Compare temperature=0 vs temperature=0.7 on the same prompt."""
    console.print(f"\n[bold]Temperature Comparison[/bold]")
    console.print(f"Model:  {model}")
    console.print(f"Prompt: {prompt[:80]}")
    console.print("─" * 50)

    results = {}

    for temp in [0, 0.7]:
        console.print(f"\n[cyan]Temperature = {temp}:[/cyan]")
        outputs = []

        for run in range(3):
            result, meta = call_with_schema(prompt, schema, model, temperature=temp)
            if result:
                outputs.append(result.model_dump())

        results[temp] = outputs

        if outputs:
            console.print(f"  Outputs across 3 runs:")
            for i, out in enumerate(outputs, 1):
                console.print(f"  Run {i}: {out}")
        else:
            console.print(f"  [red]All runs failed[/red]")

    return results