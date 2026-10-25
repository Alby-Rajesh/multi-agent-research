import pytest
from conftest import ROOT
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM

from research.evaluation import CaseResult, load_cases, retrieval_scores, summarize
from research.knowledge import Passage, ingest, load
from research.settings import RetrievalSettings, StoreSettings
from research.tools import KnowledgeBaseSearchTool


def test_keyword_side_of_hybrid_retrieval_finds_the_right_document(knowledge):
    assert "pgvector.md" in {p.source for p in knowledge.search("pgvector HNSW IVFFlat index build time memory")}
    assert "qdrant.md" in {p.source for p in knowledge.search("Qdrant written in Rust")}


def test_ask_returns_answer_with_sources(knowledge):
    answer = knowledge.ask("What does DSPy stand for?")
    assert answer.answer and answer.sources
    assert all(p.source.endswith(".md") for p in answer.sources)


def test_notice_file_is_not_indexed(knowledge):
    assert all(p.source != "NOTICE.md" for p in knowledge.search("licence source corpus file"))


def test_missing_index_has_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="research ingest"):
        load(StoreSettings(persist_dir=tmp_path), RetrievalSettings(), MockEmbedding(embed_dim=8), MockLLM(), 8)


def test_search_tool_formats_numbered_cited_passages():
    class KB:
        def search(self, query):
            return [Passage(source="pgvector.md", text="HNSW " * 400, score=1.0)]

    out = KnowledgeBaseSearchTool(knowledge=KB(), max_chars=50)._run("hnsw")
    assert out.startswith("[1] source: pgvector.md\n") and len(out) < 100

    class Empty:
        def search(self, query):
            return []

    assert KnowledgeBaseSearchTool(knowledge=Empty())._run("x") == "No relevant passages found."


def test_retrieval_scores_and_summary():
    passages = [Passage(source=s, text="", score=0) for s in ["qdrant.md", "pgvector.md", "vllm.md"]]
    retrieved, hit, rr, recall = retrieval_scores(passages, ["pgvector", "dspy"])
    assert retrieved == ["qdrant", "pgvector", "vllm"] and hit and rr == 0.5 and recall == 0.5
    summary = summarize(
        [
            CaseResult(
                question="a",
                retrieved=[],
                hit=True,
                reciprocal_rank=1.0,
                source_recall=1.0,
                faithful=True,
                correctness=5.0,
            ),
            CaseResult(
                question="b",
                retrieved=[],
                hit=False,
                reciprocal_rank=0.0,
                source_recall=0.0,
                faithful=False,
                correctness=None,
            ),
        ]
    )
    assert summary["hit_rate"] == 0.5 and summary["mrr"] == 0.5 and summary["faithfulness"] == 0.5
    assert summary["correctness_1_to_5"] == 5.0 and summary["relevancy"] is None


def test_eval_set_references_real_corpus_files():
    names = {p.stem for p in (ROOT / "corpus").glob("*.md")}
    cases = load_cases(ROOT / "data/eval.jsonl")
    assert len(cases) == 22
    assert all(set(c.sources) <= names for c in cases)


@pytest.mark.postgres
def test_pgvector_hybrid_store(pg_dsn, tmp_path):
    store = StoreSettings(backend="postgres", dsn=pg_dsn, table="test_knowledge")
    retrieval = RetrievalSettings(top_k=4, candidates_per_retriever=6)
    embed = MockEmbedding(embed_dim=16)
    assert ingest(ROOT / "corpus", store, retrieval, embed, 16) > 100
    assert ingest(ROOT / "corpus", store, retrieval, embed, 16) > 100
    kb = load(store, retrieval, embed, MockLLM(max_tokens=64), 16)
    assert "pgvector.md" in {p.source for p in kb.search("pgvector HNSW IVFFlat index build time memory")}
