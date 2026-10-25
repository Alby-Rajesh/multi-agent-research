# multi-agent-research

[![CI](https://github.com/Alby-Rajesh/multi-agent-research/actions/workflows/ci.yml/badge.svg)](https://github.com/Alby-Rajesh/multi-agent-research/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![CrewAI](https://img.shields.io/badge/CrewAI-FF5A50)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-8B5CF6)
![pgvector](https://img.shields.io/badge/pgvector-4169E1?logo=postgresql&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Ask a research question about a document collection and get back a **cited report** written by a team of agents. **LlamaIndex** handles retrieval: hybrid keyword + semantic search on **Postgres/pgvector**, fused with Reciprocal Rank Fusion. **CrewAI** runs the agents in three interchangeable multi-agent patterns, so the same question can be answered by a pipeline, a manager-led team, or a writer–reviewer loop that fact-checks itself.

```bash
research research "How should I choose between HNSW and IVFFlat in pgvector?" --pattern reflection
```

## Architecture

```mermaid
flowchart TB
    subgraph Ingestion
        D[documents<br/>md · pdf · html · txt] --> M[MarkdownNodeParser] --> S[SentenceSplitter] --> E[embeddings]
        E --> PG[(Postgres<br/>pgvector HNSW + tsvector GIN)]
    end
    subgraph Retrieval
        Q[query] --> V[semantic search]
        Q --> K[full-text search]
        V --> F[Reciprocal Rank Fusion]
        K --> F
    end
    PG --> V
    PG --> K
    F --> T[search_knowledge_base tool]
    F --> C[CitationQueryEngine<br/>/v1/ask]
    T --> R[CrewAI crews<br/>/v1/research]
```

## Multi-agent patterns

| Pattern | How it runs | When it fits |
|---|---|---|
| **Sequential** | Researcher → Analyst → Writer. Each task receives the earlier tasks' output as context; only the researcher has the search tool. | Predictable pipelines where the steps are known in advance. |
| **Hierarchical** | A manager LLM receives one goal, plans, and delegates to the researcher, analyst and writer through CrewAI's delegation tools. | Open-ended questions where the order of work should be decided at run time. |
| **Reflection** | A CrewAI **Flow**: draft with the sequential crew, then a fact-checking reviewer verifies every claim against the knowledge base and returns a structured `Review(approved, feedback)`. Rejected drafts go back to the writer with the feedback, up to `max_review_rounds`. | Reports where accuracy matters more than latency. |

```mermaid
flowchart LR
    A[draft<br/>sequential crew] --> B{review<br/>fact-checker}
    B -->|approved or round limit| D[done]
    B -->|needs revision| C[revise<br/>writer + feedback]
    C --> B
```

Agent roles and task prompts live in [`configs/agents.yaml`](configs/agents.yaml) and [`configs/tasks.yaml`](configs/tasks.yaml), CrewAI's own convention, so behavior can be tuned without touching code.

## Engineering highlights

| Concern | Approach |
|---|---|
| **Hybrid retrieval** | Semantic and full-text retrievers each return candidates, which are fused with Reciprocal Rank Fusion. On Postgres both come from one table (`vector` with an HNSW cosine index plus a `tsvector` GIN index). The in-memory backend uses BM25 for the keyword side, so it runs without a database. |
| **Chunking** | Markdown-aware parsing first, so chunks follow headings, then sentence-aware splitting to a token budget. |
| **Idempotent ingestion** | Documents get stable ids; re-ingesting replaces the collection instead of duplicating it. |
| **Citations everywhere** | `/v1/ask` uses LlamaIndex's `CitationQueryEngine`; agents receive passages labelled with their source file and are instructed to cite after every claim. |
| **Structured verdicts** | The reviewer's output is a Pydantic model (`output_pydantic=Review`), so the Flow routes on a typed boolean rather than parsing prose. |
| **Bounded loops** | The reflection loop has a hard round limit, and every agent has `max_iter`, so a disagreeing reviewer cannot loop forever. |
| **Evaluation** | Retrieval hit rate, MRR and source recall on 22 questions, plus LLM-judged faithfulness, relevancy and 1–5 correctness against reference answers, using LlamaIndex's evaluators. |
| **Any OpenAI-compatible model** | LLM and embeddings are configured by URL: vLLM, Ollama, Groq, OpenAI or Text Embeddings Inference. CrewAI telemetry is disabled by default. |

## Quickstart

```bash
git clone https://github.com/Alby-Rajesh/multi-agent-research.git
cd multi-agent-research
cp .env.example .env
docker compose up --build
```

Compose starts Postgres with pgvector, a CPU embeddings server (Text Embeddings Inference with `BAAI/bge-small-en-v1.5`), ingests the corpus, and serves the API on port 8080. Point `MAR_LLM__*` at your chat model.

Without Docker, using the in-memory store:

```bash
pip install -e ".[dev]"
export MAR_STORE__BACKEND=memory
research ingest
research ask "What does DSPy stand for?"
research research "Compare Qdrant and pgvector for hybrid search" --pattern hierarchical
```

### API

```bash
curl -s localhost:8080/v1/ask -H 'content-type: application/json' \
  -d '{"question": "What is Qdrant Edge?"}'

curl -s localhost:8080/v1/research -H 'content-type: application/json' \
  -d '{"question": "Which agent framework should I pick for a stateful workflow?", "pattern": "reflection"}'
```

`/v1/ask` returns the answer and the cited passages. `/v1/research` returns the report, the pattern, the number of review rounds and whether it was approved, plus every agent step for auditing.

## Corpus and evaluation

The sample corpus in [`corpus/`](corpus) is the README of eight AI-infrastructure projects (LangGraph, CrewAI, LlamaIndex, DSPy, vLLM, pgvector, Qdrant, AutoGen), copied unmodified under their licences ([NOTICE](corpus/NOTICE.md)). Replace it with your own files; Markdown, PDF, HTML and text are supported.

[`data/eval.jsonl`](data/eval.jsonl) holds 22 questions with reference answers and the source files that contain them, including cross-document questions.

```bash
research evaluate
research evaluate --judge
```

| Metric | What it measures |
|---|---|
| Hit rate | a passage from an expected source is in the top-k |
| MRR | how high the first relevant passage ranks |
| Source recall | share of expected sources retrieved (matters for cross-document questions) |
| Faithfulness | the answer is supported by the retrieved context (LLM judge) |
| Relevancy | the answer and context address the question (LLM judge) |
| Correctness | 1–5 agreement with the reference answer (LLM judge) |

Results are written to `output/eval/`.

## Configuration

Environment variables with the `MAR_` prefix and `__` for nesting (see [`settings.py`](src/research/settings.py)):

| Variable | Default | Purpose |
|---|---|---|
| `MAR_LLM__MODEL` / `BASE_URL` | `Qwen/Qwen2.5-7B-Instruct` / `http://localhost:8000/v1` | chat model for agents, answers and judging |
| `MAR_EMBEDDING__MODEL` / `BASE_URL` / `DIM` | `BAAI/bge-small-en-v1.5` / `http://localhost:8001/v1` / `384` | embedding model |
| `MAR_STORE__BACKEND` | `memory` | `memory` or `postgres` |
| `MAR_STORE__DSN` | `postgresql://research:research@localhost:5432/research` | pgvector database |
| `MAR_RETRIEVAL__TOP_K` | `6` | passages after fusion |
| `MAR_RETRIEVAL__CHUNK_SIZE` | `512` | tokens per chunk |
| `MAR_CREW__MAX_REVIEW_ROUNDS` | `2` | reflection loop limit |

## Project structure

```
src/research/
├── knowledge.py    ingestion pipeline, pgvector / in-memory stores, RRF hybrid retriever, cited answers
├── crews.py        CrewAI agents and the sequential, hierarchical and reflection (Flow) patterns
├── tools.py        knowledge-base search tool for agents
├── evaluation.py   retrieval metrics and LLM-judged answer quality
├── service.py      model wiring (OpenAI-compatible LLM and embeddings)
├── api.py          FastAPI: /v1/ask, /v1/research
└── cli.py          ingest, ask, research, evaluate, serve
configs/            agent roles and task prompts (YAML)
corpus/             sample documents with licence notice
data/               evaluation questions
```

## Testing

```bash
docker compose up -d postgres
make test
```

Tests run offline with mock models. They cover hybrid retrieval over the real corpus, cited answers, crew wiring (context chaining, the manager, the structured reviewer), the reflection loop (approve-after-revision and the round limit), the evaluation metrics and the API. An integration test ingests the corpus into **real pgvector**, re-ingests to prove idempotency, and runs hybrid retrieval; CI runs it against the `pgvector/pgvector` image.

## License

MIT for the code. Corpus files keep their original licences; see [corpus/NOTICE.md](corpus/NOTICE.md).
