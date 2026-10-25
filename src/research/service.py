import logging
from dataclasses import dataclass

from crewai import LLM as CrewLLM
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike

from .crews import CrewFactory, Researcher
from .knowledge import KnowledgeBase, ingest, load
from .settings import Settings
from .tools import KnowledgeBaseSearchTool

log = logging.getLogger(__name__)


def llama_llm(settings: Settings, judge: bool = False) -> OpenAILike:
    s = settings.judge if judge and settings.judge else settings.llm
    return OpenAILike(
        model=s.model,
        api_base=s.base_url,
        api_key=s.api_key.get_secret_value(),
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        context_window=s.context_window,
        timeout=s.timeout,
        max_retries=s.max_retries,
        additional_kwargs=s.extra,
        is_chat_model=True,
        is_function_calling_model=False,
    )


def embedding_model(settings: Settings) -> OpenAILikeEmbedding:
    s = settings.embedding
    return OpenAILikeEmbedding(
        model_name=s.model,
        api_base=s.base_url,
        api_key=s.api_key.get_secret_value(),
        embed_batch_size=s.batch_size,
        dimensions=s.dim if s.send_dimensions else None,
    )


def crew_llm(settings: Settings) -> CrewLLM:
    s = settings.llm
    return CrewLLM(
        model=f"openai/{s.model}",
        base_url=s.base_url,
        api_key=s.api_key.get_secret_value(),
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        timeout=s.timeout,
    )


def run_ingest(settings: Settings) -> int:
    return ingest(
        settings.corpus_dir, settings.store, settings.retrieval, embedding_model(settings), settings.embedding.dim
    )


def knowledge_base(settings: Settings) -> KnowledgeBase:
    return load(
        settings.store, settings.retrieval, embedding_model(settings), llama_llm(settings), settings.embedding.dim
    )


@dataclass
class Services:
    knowledge: KnowledgeBase
    researcher: Researcher


def build_services(settings: Settings) -> Services:
    settings.configure_environment()
    kb = knowledge_base(settings)
    factory = CrewFactory(
        crew_llm(settings), KnowledgeBaseSearchTool(knowledge=kb), settings.configs_dir, settings.crew.verbose
    )
    return Services(knowledge=kb, researcher=Researcher(factory, settings.crew.max_review_rounds))
