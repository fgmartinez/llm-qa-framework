# LLM Test Framework

A flexible Python testing framework for evaluating LLM providers. Built on **pytest** and **DeepEval**, it lets you run accuracy, performance, safety, and RAG tests against any OpenAI-compatible or Anthropic API — or swap in a mock provider for fast, offline iteration.

## Key Features

- **DeepEval Integration**: Leverage sophisticated LLM-as-a-judge metrics for answer relevancy, faithfulness, hallucination detection, toxicity, and bias
- **RAG-Specific Metrics**: Evaluate retrieval quality with contextual relevancy, precision, and recall metrics
- **Multi-Provider Support**: Test against OpenAI, Anthropic, or custom providers
- **Mock Provider**: Fast, offline testing without API costs
- **Comprehensive Coverage**: Accuracy, performance, safety, and RAG pipeline tests

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
│   ├── deepeval_integration.py    # DeepEval metrics integration
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

## Why DeepEval?

The framework includes both simple metrics (keyword matching, BLEU, cosine similarity) and DeepEval's sophisticated LLM-as-a-judge metrics:

**Simple Metrics (Good for):**
- Fast, offline testing
- Basic keyword/phrase validation
- No API costs
- Deterministic results

**DeepEval Metrics (Good for):**
- Detecting hallucinations and factual errors
- Evaluating answer relevancy and quality
- Identifying bias and toxicity
- RAG-specific evaluation (context relevancy, precision, recall)
- Production-grade LLM evaluation

Use simple metrics for rapid development and smoke tests. Use DeepEval metrics for comprehensive evaluation before production deployment.

## Running Tests

```bash
# Run all tests (uses mock provider by default — no API keys needed)
pytest

# Run a specific test category
pytest tests/accuracy/
pytest tests/performance/
pytest tests/safety/
pytest tests/rag/

# Run DeepEval tests specifically (requires API key for LLM-as-a-judge)
pytest tests/accuracy/test_deepeval_llm.py
pytest tests/rag/test_deepeval_rag.py

# Run by marker
pytest -m accuracy
pytest -m safety
pytest -m performance
pytest -m rag

# Skip slow tests (DeepEval tests make API calls)
pytest -m "not slow"

# Generate an HTML test report
pytest --html=reports/pytest_report.html --self-contained-html
```

### DeepEval Tests Configuration

DeepEval tests require an OpenAI API key (for GPT-4 as the judge) by default. Set the environment variable:

```bash
export OPENAI_API_KEY=sk-...
pytest tests/accuracy/test_deepeval_llm.py -m slow
```

You can also configure DeepEval to use different models for evaluation by passing the `model` parameter to metrics.

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

## Using DeepEval Metrics

### Basic DeepEval Usage

DeepEval provides sophisticated LLM evaluation metrics that go beyond simple keyword matching:

```python
from llm_test_framework.core.config import ProviderConfig
from llm_test_framework.core.providers import create_client
from llm_test_framework.evaluators import (
    DeepEvalMetrics,
    assert_metric,
    create_test_case_from_response,
)

config = ProviderConfig(provider="openai", model="gpt-4o", api_key="sk-...")
client = create_client(config)

# Test answer relevancy
prompt = "What is Python?"
response = client.complete(prompt)

test_case = create_test_case_from_response(prompt, response)
metric = DeepEvalMetrics.answer_relevancy(threshold=0.7)

assert_metric(test_case, metric)
```

### Available DeepEval Metrics

**General LLM Metrics:**
- `answer_relevancy()` - Measures how relevant the answer is to the query
- `faithfulness()` - Checks if output is faithful to provided context
- `hallucination()` - Detects hallucinated facts not in context

**Safety Metrics:**
- `toxicity()` - Detects toxic, harmful, or offensive content
- `bias()` - Identifies biased content (gender, race, age, etc.)

**RAG-Specific Metrics:**
- `contextual_relevancy()` - Measures if retrieved context is relevant
- `contextual_precision()` - Evaluates precision of retrieved context
- `contextual_recall()` - Evaluates recall of retrieved context

### Testing RAG with DeepEval

```python
from llm_test_framework.core.rag import RAGPipeline, StaticRetriever
from llm_test_framework.evaluators import (
    DeepEvalMetrics,
    assert_metric,
    create_test_case_from_rag_result,
)

# Create RAG pipeline
retriever = StaticRetriever([
    "Python was created by Guido van Rossum in 1991.",
    "Python is known for its clean syntax and readability.",
])
pipeline = RAGPipeline(client=client, retriever=retriever)

# Query and evaluate
result = pipeline.query("Who created Python?")

test_case = create_test_case_from_rag_result(result)
faithfulness_metric = DeepEvalMetrics.faithfulness(threshold=0.7)
relevancy_metric = DeepEvalMetrics.contextual_relevancy(threshold=0.7)

assert_metric(test_case, faithfulness_metric)
assert_metric(test_case, relevancy_metric)
```

### Testing for Safety Issues

```python
# Test for toxicity
test_case = create_test_case_from_response(
    "Tell me about climate change",
    response
)
toxicity_metric = DeepEvalMetrics.toxicity(threshold=0.5)
assert_metric(test_case, toxicity_metric)

# Test for bias
bias_metric = DeepEvalMetrics.bias(threshold=0.5)
assert_metric(test_case, bias_metric)
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
