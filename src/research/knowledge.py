import logging
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import Settings as LlamaSettings
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import LLM
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from pydantic import BaseModel

from .settings import RetrievalSettings, StoreSettings

log = logging.getLogger(__name__)

SUFFIXES = [".md", ".txt", ".pdf", ".html"]


class Passage(BaseModel):
    source: str
    text: str
    score: float


class Answer(BaseModel):
    question: str
    answer: str
    sources: list[Passage]


def source_name(node: NodeWithScore) -> str:
    return node.node.metadata.get("file_name") or node.node.metadata.get("source") or node.node.node_id


@dataclass
class KnowledgeBase:
    index: VectorStoreIndex
    retriever: BaseRetriever
    llm: LLM

    def search(self, query: str) -> list[Passage]:
        return [
            Passage(source=source_name(n), text=n.node.get_content(), score=float(n.score or 0.0))
            for n in self.retriever.retrieve(query)
        ]

    def ask(self, question: str) -> Answer:
        engine = CitationQueryEngine.from_args(
            self.index, llm=self.llm, retriever=self.retriever, citation_chunk_size=1024
        )
        response = engine.query(question)
        sources = [
            Passage(source=source_name(n), text=n.node.get_content(), score=float(n.score or 0.0))
            for n in response.source_nodes
        ]
        return Answer(question=question, answer=str(response).strip(), sources=sources)


def _pipeline(retrieval: RetrievalSettings, embed_model: BaseEmbedding) -> IngestionPipeline:
    return IngestionPipeline(
        transformations=[
            MarkdownNodeParser(),
            SentenceSplitter(chunk_size=retrieval.chunk_size, chunk_overlap=retrieval.chunk_overlap),
            embed_model,
        ]
    )


def _vector_store(store: StoreSettings, embed_dim: int):
    from llama_index.vector_stores.postgres import PGVectorStore
    from sqlalchemy.engine import make_url

    url = make_url(store.dsn.get_secret_value())
    return PGVectorStore.from_params(
        connection_string=url.set(drivername="postgresql+psycopg2").render_as_string(hide_password=False),
        async_connection_string=url.set(drivername="postgresql+asyncpg").render_as_string(hide_password=False),
        table_name=store.table,
        embed_dim=embed_dim,
        hybrid_search=True,
        text_search_config="english",
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )


def ingest(
    corpus: Path, store: StoreSettings, retrieval: RetrievalSettings, embed_model: BaseEmbedding, embed_dim: int
) -> int:
    documents = SimpleDirectoryReader(
        str(corpus), required_exts=SUFFIXES, recursive=True, exclude=["NOTICE.md"]
    ).load_data()
    seen: dict[str, int] = {}
    for doc in documents:
        name = doc.metadata["file_name"]
        seen[name] = seen.get(name, -1) + 1
        doc.id_ = f"{name}:{seen[name]}"
        doc.excluded_embed_metadata_keys = [k for k in doc.metadata if k != "file_name"]
        doc.excluded_llm_metadata_keys = [k for k in doc.metadata if k != "file_name"]
    nodes = _pipeline(retrieval, embed_model).run(documents=documents, show_progress=False)
    if store.backend == "postgres":
        context = StorageContext.from_defaults(vector_store=_vector_store(store, embed_dim))
        context.vector_store.clear()
        VectorStoreIndex(nodes, storage_context=context, embed_model=embed_model)
    else:
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        store.persist_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(store.persist_dir))
    log.info("ingested %d documents as %d chunks into %s", len(documents), len(nodes), store.backend)
    return len(nodes)


def load(
    store: StoreSettings, retrieval: RetrievalSettings, embed_model: BaseEmbedding, llm: LLM, embed_dim: int
) -> KnowledgeBase:
    LlamaSettings.llm, LlamaSettings.embed_model = llm, embed_model
    k = retrieval.candidates_per_retriever
    if store.backend == "postgres":
        index = VectorStoreIndex.from_vector_store(_vector_store(store, embed_dim), embed_model=embed_model)
        keyword = index.as_retriever(
            vector_store_query_mode=VectorStoreQueryMode.TEXT_SEARCH, sparse_top_k=k, similarity_top_k=k
        )
    else:
        if not (store.persist_dir / "docstore.json").exists():
            raise FileNotFoundError(f"no index at {store.persist_dir}; run `research ingest` first")
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(store.persist_dir)), embed_model=embed_model
        )
        from llama_index.retrievers.bm25 import BM25Retriever

        keyword = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=k)
    semantic = index.as_retriever(similarity_top_k=k)
    fused = QueryFusionRetriever(
        [semantic, keyword],
        llm=llm,
        mode="reciprocal_rerank",
        similarity_top_k=retrieval.top_k,
        num_queries=1,
        use_async=False,
    )
    return KnowledgeBase(index=index, retriever=fused, llm=llm)
