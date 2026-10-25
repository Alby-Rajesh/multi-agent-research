import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .crews import Pattern, ResearchReport
from .knowledge import Answer
from .service import Services, build_services
from .settings import get_settings

log = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ResearchRequest(AskRequest):
    pattern: Pattern = Pattern.sequential


def create_app(services_factory: Callable[[], Services] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services_factory() if services_factory else build_services(get_settings())
        yield

    app = FastAPI(title="multi-agent-research", version="1.0.0", lifespan=lifespan)

    def guarded(fn, *args):
        try:
            return fn(*args)
        except Exception as exc:
            log.exception("request failed")
            raise HTTPException(status_code=502, detail=f"upstream failure: {type(exc).__name__}") from exc

    @app.get("/healthz")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/ask", response_model=Answer)
    def ask(body: AskRequest, request: Request) -> Answer:
        return guarded(request.app.state.services.knowledge.ask, body.question)

    @app.post("/v1/research", response_model=ResearchReport)
    def research(body: ResearchRequest, request: Request) -> ResearchReport:
        return guarded(request.app.state.services.researcher.run, body.question, body.pattern)

    return app
