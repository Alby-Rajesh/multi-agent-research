import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    base_url: str = "http://localhost:8000/v1"
    api_key: SecretStr = SecretStr("token-is-ignored")
    temperature: float = Field(0.1, ge=0, le=2)
    max_tokens: int = Field(2048, gt=0)
    context_window: int = Field(32768, gt=0)
    timeout: float = Field(120, gt=0)


class EmbeddingSettings(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    base_url: str = "http://localhost:8001/v1"
    api_key: SecretStr = SecretStr("token-is-ignored")
    dim: int = Field(384, gt=0)
    batch_size: int = Field(32, gt=0)


class StoreSettings(BaseModel):
    backend: Literal["memory", "postgres"] = "memory"
    persist_dir: Path = Path("storage")
    dsn: SecretStr = SecretStr("postgresql://research:research@localhost:5432/research")
    table: str = "knowledge"


class RetrievalSettings(BaseModel):
    chunk_size: int = Field(512, gt=64)
    chunk_overlap: int = Field(64, ge=0)
    top_k: int = Field(6, ge=1)
    candidates_per_retriever: int = Field(10, ge=1)


class CrewSettings(BaseModel):
    max_review_rounds: int = Field(2, ge=0)
    verbose: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MAR_", env_nested_delimiter="__", env_file=".env", extra="ignore")

    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    store: StoreSettings = StoreSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    crew: CrewSettings = CrewSettings()
    corpus_dir: Path = Path("corpus")
    configs_dir: Path = Path("configs")

    def configure_environment(self) -> None:
        os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
        os.environ.setdefault("OTEL_SDK_DISABLED", "true")


@lru_cache
def get_settings() -> Settings:
    return Settings()
