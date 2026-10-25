import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llama_index.core.evaluation import CorrectnessEvaluator, FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.core.llms import LLM
from pydantic import BaseModel

from .knowledge import KnowledgeBase, Passage

log = logging.getLogger(__name__)


class EvalCase(BaseModel):
    question: str
    reference: str
    sources: list[str]


class CaseResult(BaseModel):
    question: str
    retrieved: list[str]
    hit: bool
    reciprocal_rank: float
    source_recall: float
    answer: str = ""
    faithful: bool | None = None
    relevant: bool | None = None
    correctness: float | None = None


def load_cases(path: Path) -> list[EvalCase]:
    return [EvalCase(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stem(name: str) -> str:
    return Path(name).stem.lower()


def retrieval_scores(passages: list[Passage], expected: list[str]) -> tuple[list[str], bool, float, float]:
    retrieved = [_stem(p.source) for p in passages]
    wanted = {_stem(s) for s in expected}
    rank = next((i for i, s in enumerate(retrieved, 1) if s in wanted), None)
    recall = len(wanted & set(retrieved)) / len(wanted)
    return retrieved, rank is not None, (1.0 / rank) if rank else 0.0, recall


def evaluate_case(kb: KnowledgeBase, case: EvalCase, judge: LLM | None) -> CaseResult:
    retrieved, hit, rr, recall = retrieval_scores(kb.search(case.question), case.sources)
    result = CaseResult(question=case.question, retrieved=retrieved, hit=hit, reciprocal_rank=rr, source_recall=recall)
    if judge is None:
        return result
    answer = kb.ask(case.question)
    contexts = [p.text for p in answer.sources]
    result.answer = answer.answer
    result.faithful = (
        FaithfulnessEvaluator(llm=judge)
        .evaluate(query=case.question, response=answer.answer, contexts=contexts)
        .passing
    )
    result.relevant = (
        RelevancyEvaluator(llm=judge).evaluate(query=case.question, response=answer.answer, contexts=contexts).passing
    )
    result.correctness = (
        CorrectnessEvaluator(llm=judge)
        .evaluate(query=case.question, response=answer.answer, reference=case.reference)
        .score
    )
    return result


def _mean(values: list) -> float | None:
    values = [float(v) for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def summarize(results: list[CaseResult]) -> dict:
    return {
        "questions": len(results),
        "hit_rate": _mean([r.hit for r in results]),
        "mrr": _mean([r.reciprocal_rank for r in results]),
        "source_recall": _mean([r.source_recall for r in results]),
        "faithfulness": _mean([r.faithful for r in results]),
        "relevancy": _mean([r.relevant for r in results]),
        "correctness_1_to_5": _mean([r.correctness for r in results]),
    }


def run_eval(kb: KnowledgeBase, cases: list[EvalCase], judge: LLM | None, out: Path, workers: int = 4) -> dict:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda c: evaluate_case(kb, c, judge), cases))
    summary = summarize(results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
