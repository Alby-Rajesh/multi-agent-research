import json
import logging
from pathlib import Path

import typer

from .crews import Pattern
from .settings import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Multi-agent research over your documents.")


def _logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "openai", "LiteLLM", "urllib3", "sqlalchemy", "bm25s"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command(help="Parse, chunk, embed and index the corpus.")
def ingest(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    from .service import run_ingest

    _logging(verbose)
    typer.echo(f"indexed {run_ingest(get_settings())} chunks")


@app.command(help="Answer a question with cited RAG.")
def ask(question: str, verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    from .service import knowledge_base

    _logging(verbose)
    answer = knowledge_base(get_settings()).ask(question)
    typer.echo(answer.answer)
    for i, p in enumerate(answer.sources, 1):
        typer.secho(f"[{i}] {p.source}", fg="cyan")


@app.command(help="Run a multi-agent research crew and print the report.")
def research(
    question: str,
    pattern: Pattern = typer.Option(Pattern.sequential, help="sequential, hierarchical or reflection"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    from .service import build_services

    _logging(verbose)
    report = build_services(get_settings()).researcher.run(question, pattern)
    typer.echo(report.report)
    if report.pattern is Pattern.reflection:
        typer.secho(f"\nreview rounds: {report.review_rounds}, approved: {report.approved}", fg="cyan")


@app.command(help="Evaluate retrieval (hit rate, MRR) and, with --judge, answer quality.")
def evaluate(
    eval_set: Path = typer.Option(Path("data/eval.jsonl")),
    judge: bool = typer.Option(
        False, "--judge", help="Also score faithfulness, relevancy and correctness with the LLM."
    ),
    out: Path = typer.Option(Path("output/eval")),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    from .evaluation import load_cases, run_eval
    from .service import knowledge_base, llama_llm

    _logging(verbose)
    settings = get_settings()
    summary = run_eval(knowledge_base(settings), load_cases(eval_set), llama_llm(settings) if judge else None, out)
    typer.echo(json.dumps(summary, indent=2))


@app.command(help="Serve the HTTP API.")
def serve(host: str = "0.0.0.0", port: int = 8080) -> None:
    import uvicorn

    _logging(False)
    uvicorn.run("research.api:create_app", factory=True, host=host, port=port)


if __name__ == "__main__":
    app()
