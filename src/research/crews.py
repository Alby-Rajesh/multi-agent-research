import logging
from enum import StrEnum
from pathlib import Path

import yaml
from crewai import LLM, Agent, Crew, Process, Task
from crewai.flow.flow import Flow, listen, or_, router, start
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class Pattern(StrEnum):
    sequential = "sequential"
    hierarchical = "hierarchical"
    reflection = "reflection"


class Review(BaseModel):
    approved: bool = Field(description="true only if every factual claim is supported and cited")
    feedback: str = Field(description="specific problems to fix; empty when approved")


class Step(BaseModel):
    agent: str
    task: str
    output: str


class ResearchReport(BaseModel):
    pattern: Pattern
    question: str
    report: str
    review_rounds: int = 0
    approved: bool | None = None
    steps: list[Step] = []


def _steps(output) -> list[Step]:
    return [
        Step(agent=t.agent or "", task=(t.name or t.description or "")[:80], output=t.raw) for t in output.tasks_output
    ]


class CrewFactory:
    def __init__(self, llm: LLM, search_tool: BaseTool, configs_dir: Path, verbose: bool = False):
        self.llm, self.tool, self.verbose = llm, search_tool, verbose
        self.agents_cfg = yaml.safe_load((configs_dir / "agents.yaml").read_text(encoding="utf-8"))
        self.tasks_cfg = yaml.safe_load((configs_dir / "tasks.yaml").read_text(encoding="utf-8"))

    def agent(self, name: str, tools: bool = True) -> Agent:
        return Agent(
            **self.agents_cfg[name],
            llm=self.llm,
            tools=[self.tool] if tools else [],
            allow_delegation=False,
            max_iter=8,
            verbose=self.verbose,
        )

    def task(self, name: str, agent: Agent | None, **kwargs) -> Task:
        return Task(name=name, **self.tasks_cfg[name], agent=agent, **kwargs)

    def sequential(self) -> Crew:
        researcher, analyst, writer = (
            self.agent("researcher"),
            self.agent("analyst", tools=False),
            self.agent("writer", tools=False),
        )
        research = self.task("research", researcher)
        analysis = self.task("analysis", analyst, context=[research])
        report = self.task("report", writer, context=[research, analysis])
        return Crew(
            agents=[researcher, analyst, writer],
            tasks=[research, analysis, report],
            process=Process.sequential,
            verbose=self.verbose,
        )

    def hierarchical(self) -> Crew:
        agents = [self.agent("researcher"), self.agent("analyst", tools=False), self.agent("writer", tools=False)]
        return Crew(
            agents=agents,
            tasks=[self.task("managed_report", None)],
            process=Process.hierarchical,
            manager_llm=self.llm,
            verbose=self.verbose,
        )

    def reviewer(self) -> Crew:
        reviewer = self.agent("reviewer")
        return Crew(
            agents=[reviewer], tasks=[self.task("review", reviewer, output_pydantic=Review)], verbose=self.verbose
        )

    def reviser(self) -> Crew:
        writer = self.agent("writer")
        return Crew(agents=[writer], tasks=[self.task("revise", writer)], verbose=self.verbose)


class ReflectionState(BaseModel):
    question: str = ""
    report: str = ""
    feedback: str = ""
    rounds: int = 0
    approved: bool = False
    steps: list[Step] = []


class ReflectionFlow(Flow[ReflectionState]):
    def __init__(self, factory: CrewFactory, max_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.factory, self.max_rounds = factory, max_rounds

    @start()
    def draft(self):
        output = self.factory.sequential().kickoff(inputs={"question": self.state.question})
        self.state.report, self.state.steps = output.raw, _steps(output)

    @router(or_(draft, "revise"))
    def review(self):
        output = self.factory.reviewer().kickoff(inputs={"question": self.state.question, "report": self.state.report})
        verdict = (
            output.pydantic if isinstance(output.pydantic, Review) else Review(approved=False, feedback=output.raw)
        )
        self.state.approved, self.state.feedback = verdict.approved, verdict.feedback
        self.state.steps += _steps(output)
        if verdict.approved or self.state.rounds >= self.max_rounds:
            return "done"
        return "needs_revision"

    @listen("needs_revision")
    def revise(self):
        inputs = {"question": self.state.question, "report": self.state.report, "feedback": self.state.feedback}
        output = self.factory.reviser().kickoff(inputs=inputs)
        self.state.report, self.state.rounds = output.raw, self.state.rounds + 1
        self.state.steps += _steps(output)

    @listen("done")
    def finish(self):
        return self.state.report


class Researcher:
    def __init__(self, factory: CrewFactory, max_review_rounds: int):
        self.factory, self.max_review_rounds = factory, max_review_rounds

    def run(self, question: str, pattern: Pattern = Pattern.sequential) -> ResearchReport:
        log.info("research pattern=%s question=%r", pattern.value, question)
        if pattern is Pattern.reflection:
            flow = ReflectionFlow(self.factory, self.max_review_rounds)
            flow.kickoff(inputs={"question": question})
            s = flow.state
            return ResearchReport(
                pattern=pattern,
                question=question,
                report=s.report,
                review_rounds=s.rounds,
                approved=s.approved,
                steps=s.steps,
            )
        crew = self.factory.sequential() if pattern is Pattern.sequential else self.factory.hierarchical()
        output = crew.kickoff(inputs={"question": question})
        return ResearchReport(pattern=pattern, question=question, report=output.raw, steps=_steps(output))
