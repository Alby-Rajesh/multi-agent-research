from typing import Any, Protocol

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from .knowledge import Passage


class Searchable(Protocol):
    def search(self, query: str) -> list[Passage]: ...


class SearchInput(BaseModel):
    query: str = Field(..., description="A focused search query, e.g. 'pgvector HNSW vs IVFFlat build time'")


class KnowledgeBaseSearchTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "search_knowledge_base"
    description: str = (
        "Search the document collection with hybrid keyword and semantic retrieval. "
        "Returns numbered passages with the source file name to cite."
    )
    args_schema: type[BaseModel] = SearchInput
    knowledge: Any = Field(exclude=True, description="an object with search(query) -> list[Passage]")
    max_chars: int = 1200

    def _run(self, query: str) -> str:
        passages = self.knowledge.search(query)
        if not passages:
            return "No relevant passages found."
        return "\n\n".join(f"[{i}] source: {p.source}\n{p.text[: self.max_chars]}" for i, p in enumerate(passages, 1))
