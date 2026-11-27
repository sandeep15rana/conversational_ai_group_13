from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)
    search_depth: Literal["basic", "advanced"] = "basic"


class ResearchSource(BaseModel):
    title: str
    url: HttpUrl
    content: str
    score: float | None = None


class ResearchResponse(BaseModel):
    query: str
    answer: str | None = None
    sources: list[ResearchSource] = Field(default_factory=list)
    provider: Literal["tavily", "duckduckgo"]
    success: bool = True
    error: str | None = None


class AnalysisRequest(BaseModel):
    operation: Literal[
        "summary",
        "describe",
        "groupby",
        "filter",
        "correlation",
        "custom"
    ]
    data: list[dict[str, Any]]
    parameters: dict[str, Any] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    operation: str
    result: Any
    row_count: int
    columns: list[str]
    success: bool = True
    error: str | None = None


class SynthesisRequest(BaseModel):
    original_query: str
    research_findings: ResearchResponse | None = None
    analytical_findings: AnalysisResponse | None = None
    memory_context: list[str] = Field(default_factory=list)


class SynthesisResponse(BaseModel):
    markdown: str
    citations: list[str] = Field(default_factory=list)
    success: bool = True
    error: str | None = None


class WorkerTask(BaseModel):
    task_id: str
    agent: Literal["research", "analytical", "synthesis"]
    instruction: str
    payload: dict[str, Any] = Field(default_factory=dict)


class WorkerResult(BaseModel):
    task_id: str
    agent: str
    output: dict[str, Any]
    success: bool
    error: str | None = None