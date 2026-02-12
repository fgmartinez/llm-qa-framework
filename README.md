# LLM Test Framework

A flexible Python testing framework for evaluating LLM providers. Built on **pytest**, it lets you run accuracy, performance, safety, and RAG tests against any OpenAI-compatible or Anthropic API — or swap in a mock provider for fast, offline iteration.

## Project Structure

```
llm_test_framework/
├── core/
│   ├── llm_client.py              # LLMClient ABC + LLMResponse dataclass
│   ├── config.py                  # ProviderConfig & TestConfig (pydantic)
│   ├── providers/
│   │   ├── factory.py             # create_client() — provider router
│   │   ├── openai_client.py       # OpenAI / OpenAI-compatible APIs
│   │   ├── anthropic_client.py    # Anthropic Claude
│   │   └── mock.py                # Deterministic mock for unit tests
│   └── rag/
│       └── pipeline.py            # RAGPipeline, Retriever ABC, StaticRetriever
├── evaluators/
│   ├── semantic_similarity.py     # cosine & Jaccard (bag-of-words)
│   └── metrics.py                 # BLEU, keyword checks, length validation
└── reports/
    └── reporter.py                # JSON + HTML report generation
```

## Prerequisites

- Python 3.10+
- (Optional) API keys for OpenAI and/or Anthropic if you want to test against live APIs

## Installation

```bash
# Clone the repository
git clone <repo-url> && cd llm-qa-framework

# Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# Install the package with dev dependencies
pip install -e ".[dev]"
```

## Configuration

Configuration is handled via environment variables or a `.env` file.

```bash
# Copy the example and fill in your keys
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `LLM_TEST_OPENAI_API_KEY` | `""` | OpenAI API key |
| `LLM_TEST_ANTHROPIC_API_KEY` | `""` | Anthropic API key |
| `LLM_TEST_DEFAULT_PROVIDER` | `mock` | Provider to use: `openai`, `anthropic`, `mock` |
| `LLM_TEST_DEFAULT_MODEL` | `mock-model` | Model identifier |

YAML config files in `configs/` show example provider setups for reference.

## Running Tests

```bash
# Run all tests (uses mock provider by default — no API keys needed)
pytest

# Run a specific test category
pytest tests/accuracy/
pytest tests/performance/
pytest tests/safety/
pytest tests/rag/

# Run by marker
pytest -m accuracy
pytest -m safety
pytest -m performance
pytest -m rag

# Generate an HTML test report
pytest --html=reports/pytest_report.html --self-contained-html
```

## Testing Against Real Providers

Set the environment variables to point at a real provider:

```bash
# OpenAI
LLM_TEST_DEFAULT_PROVIDER=openai \
LLM_TEST_DEFAULT_MODEL=gpt-4o \
LLM_TEST_OPENAI_API_KEY=sk-... \
pytest -m accuracy

# Anthropic
LLM_TEST_DEFAULT_PROVIDER=anthropic \
LLM_TEST_DEFAULT_MODEL=claude-sonnet-4-5-20250929 \
LLM_TEST_ANTHROPIC_API_KEY=sk-ant-... \
pytest -m accuracy
```

## Using the Framework in Code

### Basic Usage

```python
from llm_test_framework.core.config import ProviderConfig
from llm_test_framework.core.providers import create_client

config = ProviderConfig(
    provider="openai",
    model="gpt-4o",
    api_key="sk-...",
    temperature=0.0,
    max_tokens=512,
)
client = create_client(config)

response = client.timed_complete("Explain recursion in one sentence.")
print(response.text)
print(f"Latency: {response.latency_ms:.0f}ms, Tokens: {response.total_tokens}")
```

### Evaluating Responses

```python
from llm_test_framework.evaluators import cosine_similarity, contains_keywords, bleu_score

reference = "Recursion is when a function calls itself."
similarity = cosine_similarity(reference, response.text)
has_keywords = contains_keywords(response.text, ["recursion", "function"])
bleu = bleu_score(reference, response.text)
```

### RAG Pipeline Testing

```python
from llm_test_framework.core.rag import RAGPipeline, StaticRetriever

retriever = StaticRetriever([
    "Python was created by Guido van Rossum in 1991.",
    "Python supports OOP and functional programming.",
])
pipeline = RAGPipeline(client=client, retriever=retriever)
result = pipeline.query("Who created Python?")

print(result.response.text)
print(f"Contexts used: {len(result.retrieved_contexts)}")
```

### Generating Reports

```python
from llm_test_framework.reports import TestReport, TestResultEntry, generate_report

report = TestReport(provider="openai", model="gpt-4o")
report.add(TestResultEntry("accuracy_test_1", passed=True, score=0.92, latency_ms=340.0))
report.add(TestResultEntry("safety_test_1", passed=True))

html_path = generate_report(report)
print(f"Report saved to {html_path}")
```

## Adding a New Provider

1. Create `llm_test_framework/core/providers/my_provider.py` implementing `LLMClient.complete()`.
2. Add a case in `llm_test_framework/core/providers/factory.py`.
3. Run the existing test suite to verify compatibility.

## Linting

```bash
ruff check .
ruff check --fix .   # auto-fix
```

## Test Markers

| Marker | Description |
|---|---|
| `accuracy` | Response quality and correctness |
| `performance` | Latency and throughput |
| `safety` | Harmful / biased content detection |
| `rag` | RAG pipeline end-to-end |
| `slow` | Tests that call real LLM APIs |

## License

MIT
