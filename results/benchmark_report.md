# Local SLM Benchmark Report

Generated: 2026-04-30 12:51
Hardware: Apple M2 Mac

---

## Phase 3a: Model Comparison

Same 5 prompts run on each model. temperature=0 for consistency.

| Model | Avg Tok/sec | First Token ms | Avg Total s | Quality |
|---|---|---|---|---|
| llama3.2:3b | 35.6 | 211.0 | 2.97 | 81.3% |
| phi4-mini:latest | 27.1 | 239.0 | 4.04 | 85.3% |
| llama3.1:8b | 5.7 | 20467.0 | 192.2 | 82.0% |

---

## Phase 3b: Quantization Study

Same model (llama3.1:8b) at different quantization levels.
Higher Q = more bits = better quality but larger and slower.

| Model | Avg Tok/sec | First Token ms | Avg Total s | Quality |
|---|---|---|---|---|
| llama3.1:8b | 5.7 | 1949.0 | 35.1 | 93.3% |
| llama3.1:8b-instruct-q4_K_M | 6.5 | 12295.0 | 26.28 | 93.3% |

---

## Key Findings

- Smaller models (3B) are faster but may miss nuance on complex tasks
- Q4 quantization reduces size significantly with minimal quality loss
- Time to first token is the metric users notice most in interactive apps
- All models run 100% offline — no internet, no API cost, no privacy risk

---

## Methodology

- Each model warmed up with one silent call before measurement
- All prompts use temperature=0 for reproducible results
- Quality score based on keyword presence and response completeness
- Hardware: Apple M2 Mac, running via Ollama