from types import SimpleNamespace

import pytest
from conftest import ROOT
from crewai import LLM, Process
from fastapi.testclient import TestClient

from research.api import create_app
from research.crews import CrewFactory, Pattern, Researcher, ResearchReport, Review
from research.knowledge import Answer, Passage
from research.tools import KnowledgeBaseSearchTool


class KB:
    def search(self, query):
        return [Passage(source="pgvector.md", text="HNSW and IVFFlat", score=1.0)]


@pytest.fixture
def factory():
    llm = LLM(model="openai/test-model", base_url="http://localhost:9/v1", api_key="x")
    return CrewFactory(llm, KnowledgeBaseSearchTool(knowledge=KB()), ROOT / "configs")


def test_sequential_crew_chains_context(factory):
    crew = factory.sequential()
    assert crew.process == Process.sequential
    assert [a.role for a in crew.agents] == ["Senior Research Analyst", "Technical Analyst", "Technical Writer"]
    research, analysis, report = crew.tasks
    assert analysis.context == [research] and report.context == [research, analysis]
    assert [t.name for t in crew.agents[0].tools] == ["search_knowledge_base"] and crew.agents[2].tools == []


def test_hierarchical_crew_has_a_manager_and_unassigned_task(factory):
    crew = factory.hierarchical()
    assert crew.process == Process.hierarchical and crew.manager_llm is not None
    assert crew.tasks[0].agent is None


def test_reviewer_returns_structured_verdict(factory):
    task = factory.reviewer().tasks[0]
    assert task.output_pydantic is Review and "{report}" in task.description


def output(raw, pydantic=None):
    return SimpleNamespace(
        raw=raw, pydantic=pydantic, tasks_output=[SimpleNamespace(agent="a", name="t", description="d", raw=raw)]
    )


class ScriptedFactory:
    def __init__(self, verdicts):
        self.verdicts, self.revisions = list(verdicts), 0

    def sequential(self):
        return SimpleNamespace(kickoff=lambda inputs: output("draft"))

    def reviewer(self):
        return SimpleNamespace(kickoff=lambda inputs: output("verdict", self.verdicts.pop(0)))

    def reviser(self):
        def kickoff(inputs):
            assert inputs["feedback"] == "add citations"
            self.revisions += 1
            return output(f"revision {self.revisions}")

        return SimpleNamespace(kickoff=kickoff)


def test_reflection_revises_until_approved():
    scripted = ScriptedFactory([Review(approved=False, feedback="add citations"), Review(approved=True, feedback="")])
    report = Researcher(scripted, max_review_rounds=3).run("q", Pattern.reflection)
    assert report.report == "revision 1" and report.review_rounds == 1 and report.approved is True
    assert len(report.steps) == 4


def test_reflection_stops_at_max_rounds():
    scripted = ScriptedFactory([Review(approved=False, feedback="add citations")] * 3)
    report = Researcher(scripted, max_review_rounds=2).run("q", Pattern.reflection)
    assert report.review_rounds == 2 and report.approved is False and report.report == "revision 2"


def test_api():
    services = SimpleNamespace(
        knowledge=SimpleNamespace(ask=lambda q: Answer(question=q, answer="HNSW [1]", sources=[])),
        researcher=SimpleNamespace(run=lambda q, p: ResearchReport(pattern=p, question=q, report="# Report")),
    )
    with TestClient(create_app(lambda: services)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.post("/v1/ask", json={"question": "hnsw?"}).json()["answer"] == "HNSW [1]"
        body = client.post("/v1/research", json={"question": "hnsw?", "pattern": "hierarchical"}).json()
        assert body["pattern"] == "hierarchical" and body["report"] == "# Report"
        assert client.post("/v1/research", json={"question": "x", "pattern": "swarm"}).status_code == 422
