import time
import httpx
import ollama
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console

console = Console()

@dataclass
class GenerationResult:
    """
    Holds everything about one AI response.
    Think of it as a receipt — it records what happened during one generation call.
    """
    model: str
    prompt: str
    response: str
    time_to_first_token: float
    total_time: float
    tokens_per_second: float
    tokens_generated: int
    prompt_tokens: int
    raw_stats: dict = field(default_factory=dict)
    error: Optional[str] = None
    success: bool = True


class OllamaClient:
    """
    Wraps the Ollama Python library and adds:
    - Streaming so we can measure TTFT accurately
    - Health checking so we fail clearly if Ollama isn't running
    - Error handling so crashes don't take down your whole program
    """

    def __init__(self, host: str = "http://localhost:11434", timeout: int = 120):
        self.host = host
        self.timeout = timeout
        self._client = ollama.Client(host=host)

    def health_check(self) -> bool:
        """
        Checks if Ollama's server is running before we try to use it.
        If Ollama isn't running, we get a clear error instead of a confusing crash.
        """
        try:
            httpx.get(f"{self.host}/api/tags", timeout=5)
            return True
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def list_available_models(self) -> list[str]:
        """Returns the names of all models you've pulled with ollama pull."""
        try:
            response = self._client.list()
            return [model.model for model in response.models]
        except Exception:
            return []

    def generate_streaming(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> GenerationResult:
        """
        Sends a prompt and receives the response token by token.
        This lets us measure TTFT accurately — we record the exact moment
        the first token arrives, not when the full response is done.
        """

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        full_response = ""
        first_chunk_received = False
        time_to_first_token = 0.0
        final_stats = {}

        # Start clock BEFORE sending — measures from when we asked
        start_time = time.perf_counter()

        try:
            stream = self._client.chat(
                model=model,
                messages=messages,
                stream=True,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )

            for chunk in stream:
                current_time = time.perf_counter()

                # First chunk with content = record TTFT right now
                if not first_chunk_received and chunk.message.content:
                    time_to_first_token = current_time - start_time
                    first_chunk_received = True

                if chunk.message.content:
                    full_response += chunk.message.content

                # Last chunk contains Ollama's performance stats
                if chunk.done:
                    final_stats = {
                        "eval_count": chunk.eval_count,
                        "eval_duration": chunk.eval_duration,
                        "load_duration": chunk.load_duration,
                        "prompt_eval_count": chunk.prompt_eval_count,
                        "total_duration": chunk.total_duration,
                    }
                    break

            total_time = time.perf_counter() - start_time

            # eval_duration is in nanoseconds — divide by 1 billion to get seconds
            tokens_generated = final_stats.get("eval_count", 0) or 0
            eval_duration_ns = final_stats.get("eval_duration", 1) or 1
            tokens_per_second = tokens_generated / (eval_duration_ns / 1_000_000_000)

            return GenerationResult(
                model=model,
                prompt=prompt,
                response=full_response,
                time_to_first_token=time_to_first_token,
                total_time=total_time,
                tokens_per_second=tokens_per_second,
                tokens_generated=tokens_generated,
                prompt_tokens=final_stats.get("prompt_eval_count", 0) or 0,
                raw_stats=final_stats,
                success=True,
            )

        except ollama.ResponseError as e:
            console.print(f"[red]Ollama error for {model}: {e}[/red]")
            return GenerationResult(
                model=model, prompt=prompt, response="",
                time_to_first_token=0.0,
                total_time=time.perf_counter() - start_time,
                tokens_per_second=0.0, tokens_generated=0, prompt_tokens=0,
                error=str(e), success=False,
            )

        except Exception as e:
            console.print(f"[red]Unexpected error for {model}: {e}[/red]")
            return GenerationResult(
                model=model, prompt=prompt, response="",
                time_to_first_token=0.0,
                total_time=time.perf_counter() - start_time,
                tokens_per_second=0.0, tokens_generated=0, prompt_tokens=0,
                error=f"{type(e).__name__}: {str(e)}", success=False,
            )
