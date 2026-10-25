import os
from pathlib import Path

import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM

from research.knowledge import ingest, load
from research.settings import RetrievalSettings, StoreSettings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def knowledge(tmp_path_factory):
    store = StoreSettings(backend="memory", persist_dir=tmp_path_factory.mktemp("storage"))
    retrieval = RetrievalSettings(top_k=4, candidates_per_retriever=6)
    embed = MockEmbedding(embed_dim=16)
    ingest(ROOT / "corpus", store, retrieval, embed, 16)
    return load(store, retrieval, embed, MockLLM(max_tokens=64), 16)


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    dsn = os.getenv("MAR_TEST_DSN")
    if not dsn:
        pytest.skip("set MAR_TEST_DSN to a Postgres database with the pgvector extension")
    return dsn
