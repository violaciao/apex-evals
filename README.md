# APEX Evals

![APEX Evals](assets/images/title.png)

**Agentic Pipeline EXecution** is a diagnostic and evaluation framework, companion repo to the [APEX research](https://violaconseil.com/research_agentic_ai_eval_phase_1_apex.html) published by Viola Conseil. Systematic evaluation of 19 failure modes across 4 layers of the agentic tool execution pipeline — the layer existing eval frameworks don't cover.

---

## The Problem

Current eval frameworks measure what the model *says*. Nobody systematically evaluates what tools *do* — and whether the model correctly interpreted the result.

![Pipeline funnel](assets/images/funnel.png)

```
User → INTENT → Agent → [L1 Tool Selection] → [L2 Args] → TOOL CALL
                                                              ↓
Response ← [L4 Chain] ← [L3 Output Consumption] ← Tool Result

✓ MEASURED: input quality, final output
✗ NOT EVALUATED: everything in between (19 failure modes)
```

---

## Structure

```
apex-evals/
├── apex/
│   ├── config.py                        # profile switching (standard / free)
│   ├── base.py                          # EvalModule base class
│   ├── harness.py                       # LlamaIndex + Groq agent runner
│   ├── layer1_tool_selection/           # 4 failure modes
│   ├── layer2_input_construction/       # 5 failure modes  ← start here
│   │   └── semantic_arg_error.py        # ✅ implemented
│   ├── layer3_output_consumption/       # 5 failure modes
│   ├── layer4_chain_multitool/          # 5 failure modes
│   └── primitives/                      # 3 cross-layer scorers
├── fixtures/                            # schemas, traces, vcrpy cassettes
├── tests/
│   └── layer2/
│       └── test_semantic_arg_error.py   # ✅ implemented
└── reports/                             # eval run outputs
```

---

## Failure Mode Coverage

![Layer architecture](assets/images/layers.png)

| Layer | Failure Mode | Status |
|-------|-------------|--------|
| L1 | False tool trigger | 🔲 |
| L1 | Tool omission | 🔲 |
| L1 | Wrong tool selection | 🔲 |
| L1 | Ambiguous tool routing | 🔲 |
| L2 | Syntactic argument error | 🔲 |
| **L2** | **Semantic argument error** | **✅** |
| L2 | Argument injection (CVE-2025-68144) | 🔲 |
| L2 | Schema mismatch | 🔲 |
| L2 | Over/under-scoped query | 🔲 |
| **L3** | **Result hallucination completion** | **✅** |
| L3 | Stale data trust | 🔲 |
| L3 | Misinterpretation of format | 🔲 |
| L3 | Prompt injection via result | 🔲 |
| L3 | Overconfident trust | 🔲 |
| L4 | Error propagation | 🔲 |
| L4 | Privilege pivot | 🔲 |
| L4 | Infinite retry loop | 🔲 |
| L4 | State corruption | 🔲 |
| L4 | Toxic combinations | 🔲 |

---

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Set free-tier API key
export GROQ_API_KEY=your_key_here

# Run unit tests (zero API cost)
pytest tests/layer2/ -v -k "not live"

# Run live eval (Groq free tier)
APEX_PROFILE=free pytest tests/layer2/test_semantic_arg_error.py::test_live_all_scenarios -v -s
```

### Profiles

| Profile | LLM | Tools | Cost |
|---------|-----|-------|------|
| `free` (default) | Groq Llama-3.1-8b-instant | SQLite stubs | $0 |
| `anthropic` | Anthropic Claude Sonnet 4 | Testcontainers + real APIs | pay-per-call |
| `openai` | OpenAI GPT-4o | Testcontainers + real APIs | pay-per-call |
| `gemini` | Google Gemini 2.5 Pro | Testcontainers + real APIs | pay-per-call |
| `mistral` | Mistral Large | Testcontainers + real APIs | pay-per-call |

```bash
# Anthropic
export APEX_PROFILE=anthropic
export ANTHROPIC_API_KEY=your_key_here

# OpenAI
export APEX_PROFILE=openai
export OPENAI_API_KEY=your_key_here

# Google Gemini
export APEX_PROFILE=gemini
export GOOGLE_API_KEY=your_key_here

# Mistral
export APEX_PROFILE=mistral
export MISTRAL_API_KEY=your_key_here
```

---

## Three Evaluation Primitives

The 19 failure modes map to three cross-layer scoring primitives:

| Primitive | Layers | What it measures |
|-----------|--------|-----------------|
| **Tool Intent Alignment** | L1 + L2 | Does the tool call reflect user intent? |
| **Output Trust Calibration** | L3 | Is the tool output safe to build on? |
| **Chain Failure Attribution** | L4 | Which step caused the chain failure? |

---

## Research

- [Part 1 — The Blind Spot](https://violaconseil.com/research_agentic_ai_eval_phase_1.html)
- [Part 2 — APEX Framework](https://violaconseil.com/research_agentic_ai_eval_phase_1_apex.html)

## License

MIT License — see [LICENSE](LICENSE) for details.

© 2026 Viola Conseil
