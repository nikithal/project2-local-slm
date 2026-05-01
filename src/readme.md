# project2-local-slm

Local AI assistant running entirely offline on Apple M2.
No internet. No API key. No cost. No privacy risk.

---

## Why local models matter

90% of enterprises cannot send data to cloud APIs due to privacy
regulations (GDPR, HIPAA). Local models solve this — the data
never leaves the machine.

---

## Stack

| Component | Tool |
|---|---|
| Model runner | Ollama |
| Models tested | Llama 3.2 3B, Phi4-mini, Llama 3.1 8B |
| Structured output | Pydantic v2 |
| CLI interface | Rich |
| Hardware | Apple M2 Mac |

---

## Phase 1: Benchmarking

All models tested on the same 5 prompts at temperature=0.

| Model | Avg Tok/sec | First Token ms | Avg Total s | Quality |
|---|---|---|---|---|
| llama3.2:3b | 35.6 | 211 | 2.97 | 81.3% |
| phi4-mini:latest | 27.1 | 239 | 4.04 | 85.3% |
| llama3.1:8b | 5.7 | 484 | 14.51 | 82.0% |

Cold start (first load from disk): ~15 seconds.
Warm latency (subsequent calls): 1-3 seconds.

---

## Phase 2: Structured Output

Enforces JSON output schema using Pydantic.
Implements retry logic with error feedback when validation fails.

```python
result, meta = call_with_schema(
    prompt = "Analyse: I love this product",
    schema = SentimentResult,
)
print(result.sentiment)    # always "positive", "negative", or "neutral"
print(result.confidence)   # always float between 0 and 1
print(meta["attempts"])    # how many retries were needed
```

Supports: sentiment analysis, information extraction, task classification.

---

## Phase 3: Quantization Study

Same model at different quantization levels.

| Model | Avg Tok/sec | First Token ms | Avg Total s | Quality |
|---|---|---|---|---|
| llama3.1:8b default | 4.7 | 10378 | 33.12 | 93.3% |
| llama3.1:8b Q4_K_M | 10.9 | 1914 | 10.62 | 93.3% |

Q4 quantization is 3x faster with identical quality.
Q8 exceeds available RAM on M2 8GB — Q4 is the practical limit
for 8B models on consumer hardware.

---

## Setup

```bash
git clone https://github.com/nikithal/project2-local-slm.git
cd project2-local-slm

python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Install Ollama from https://ollama.com and pull models:

```bash
ollama pull llama3.2:3b
ollama pull phi4-mini
ollama pull llama3.1:8b
ollama pull llama3.1:8b-instruct-q4_K_M
```

---

## Usage

```bash
python cli.py chat          # free text chat
python cli.py sentiment     # structured sentiment analysis
python cli.py extract       # structured information extraction
python cli.py classify      # structured task classification
python cli.py benchmark     # Phase 1 benchmark suite
python cli.py compare       # Phase 3a model comparison
python cli.py quantize      # Phase 3b quantization study
python cli.py report        # generate full markdown report
```

---

## Limitations

- Cold start latency ~15s when model loads from disk for first time
- Q8 quantization of 8B models exceeds 8GB M2 RAM limit
- Quality scoring uses keyword matching — not human evaluation
- Benchmark results specific to Apple M2 hardware