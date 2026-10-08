# ============================================================
# IMPORTS
# ============================================================

from __future__ import annotations

import operator
import os
from typing import TypedDict, List, Optional, Literal, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# CHANGED: Tavily -> DuckDuckGo
from langchain_community.tools import DuckDuckGoSearchRun


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Make sure it is present in your .env file."
    )


# ============================================================
# LLM - Groq
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=GROQ_API_KEY,
)


# ============================================================
# Schemas
# ============================================================

class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description=(
            "One sentence describing what the reader should be "
            "able to do/understand after this section."
        ),
    )

    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description=(
            "3–6 concrete, non-overlapping subpoints "
            "to cover in this section."
        ),
    )

    target_words: int = Field(
        ...,
        description="Target word count for this section (120–550).",
    )

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str

    blog_kind: Literal[
        "explainer",
        "tutorial",
        "news_roundup",
        "comparison",
        "system_design",
    ] = "explainer"

    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool

    mode: Literal[
        "closed_book",
        "hybrid",
        "open_book",
    ]

    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(
        default_factory=list
    )


class ImageSpec(BaseModel):
    placeholder: str = Field(
        ...,
        description="e.g. [[IMAGE_1]]",
    )

    filename: str = Field(
        ...,
        description="Save under images/, e.g. qkv_flow.png",
    )

    alt: str
    caption: str

    prompt: str = Field(
        ...,
        description="Prompt to send to the image model.",
    )

    size: Literal[
        "1024x1024",
        "1024x1536",
        "1536x1024",
    ] = "1024x1024"

    quality: Literal[
        "low",
        "medium",
        "high",
    ] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str

    images: List[ImageSpec] = Field(
        default_factory=list
    )


# ============================================================
# LangGraph State
# ============================================================

class State(TypedDict):
    topic: str

    # Routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # Workers
    sections: Annotated[
        List[tuple[int, str]],
        operator.add
    ]

    # Reducer / image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    # Final result
    final: str


# ============================================================
# Router
# ============================================================

ROUTER_SYSTEM = """
You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:

- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on
  recent facts such as concepts and fundamentals.

- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples,
  tools, models, or releases.

- open_book (needs_research=true):
  Mostly volatile topics such as weekly roundups,
  latest information, rankings, pricing, policy,
  or regulation.

If needs_research=true:

- Output 3–10 high-signal queries.
- Queries should be scoped and specific.
- Avoid generic queries.
- If the user asks for latest/this week/last week,
  include that constraint in the queries.
"""


def router_node(state: State) -> dict:

    topic = state["topic"]

    decider = llm.with_structured_output(
        RouterDecision
    )

    decision = decider.invoke(
        [
            SystemMessage(
                content=ROUTER_SYSTEM
            ),
            HumanMessage(
                content=f"Topic: {topic}"
            ),
        ]
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }


def route_next(state: State) -> str:

    return (
        "research"
        if state["needs_research"]
        else "orchestrator"
    )


# ============================================================
# Research - DuckDuckGo
# ============================================================

def _duckduckgo_search(
    query: str,
    max_results: int = 5,
) -> List[dict]:

    """
    Search the web using DuckDuckGo.

    DuckDuckGoSearchRun returns a text result rather than
    Tavily's structured list, so we normalize it into the
    same evidence format used by the rest of the workflow.
    """

    tool = DuckDuckGoSearchRun()

    result = tool.invoke(query)

    if not result:
        return []

    return [
        {
            "title": f"DuckDuckGo result for: {query}",
            "url": "",
            "snippet": str(result),
            "published_at": None,
            "source": "DuckDuckGo",
        }
    ]


RESEARCH_SYSTEM = """
You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated
list of EvidenceItem objects.

Rules:

- Only include items with a non-empty URL.
- Prefer relevant and authoritative sources.
- If a published date is explicitly present,
  keep it as YYYY-MM-DD.
- If the date is missing or unclear,
  set published_at=null.
- Do NOT guess dates.
- Keep snippets short.
- Deduplicate by URL.
"""


def research_node(state: State) -> dict:

    queries = (
        state.get("queries", [])
        or []
    )

    max_results = 6

    raw_results = []

    for q in queries:

        raw_results.extend(
            _duckduckgo_search(
                q,
                max_results=max_results,
            )
        )

    if not raw_results:
        return {
            "evidence": []
        }

    extractor = llm.with_structured_output(
        EvidencePack
    )

    pack = extractor.invoke(
        [
            SystemMessage(
                content=RESEARCH_SYSTEM
            ),
            HumanMessage(
                content=(
                    "Raw results:\n"
                    f"{raw_results}"
                )
            ),
        ]
    )

    dedup = {}

    for e in pack.evidence:

        if e.url:
            dedup[e.url] = e

    return {
        "evidence": list(
            dedup.values()
        )
    }


# ============================================================
# Orchestrator
# ============================================================

ORCH_SYSTEM = """
You are a senior technical writer and developer advocate.

Your job is to produce a highly actionable outline
for a technical blog post.

Hard requirements:

- Create 5–9 sections.
- Each task must include:
  1. goal
  2. 3–6 concrete bullets
  3. target word count between 120–550

Quality requirements:

- Assume the reader is a developer.
- Use correct terminology.
- Bullets must be actionable.
- Include at least two of:
  - minimal code sketch
  - edge cases
  - performance/cost
  - security/privacy
  - debugging/observability

Grounding rules:

- closed_book:
  Keep the content evergreen.

- hybrid:
  Use evidence for current tools, models,
  releases, and examples.

- open_book:
  Set blog_kind to "news_roundup".
  Every section should summarize events and implications.

Output must strictly match the Plan schema.
"""


def orchestrator_node(state: State) -> dict:

    planner = llm.with_structured_output(
        Plan
    )

    evidence = state.get(
        "evidence",
        []
    )

    mode = state.get(
        "mode",
        "closed_book"
    )

    plan = planner.invoke(
        [
            SystemMessage(
                content=ORCH_SYSTEM
            ),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    "Evidence:\n"
                    f"{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )

    return {
        "plan": plan
    }


# ============================================================
# Fanout
# ============================================================

def fanout(state: State):

    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [
                    e.model_dump()
                    for e in state.get(
                        "evidence",
                        []
                    )
                ],
            },
        )
        for task in state["plan"].tasks
    ]


# ============================================================
# Worker
# ============================================================

WORKER_SYSTEM = """
You are a senior technical writer and developer advocate.

Write ONE section of a technical blog post in Markdown.

Requirements:

- Follow the provided Goal.
- Cover ALL bullets.
- Stay close to the target word count.
- Output ONLY the section content.
- Start with:

## <Section Title>

Grounding:

- If mode == open_book:
  Only use claims supported by Evidence URLs.
- If requires_citations == true:
  Cite outside-world claims using Evidence URLs.
- Evergreen reasoning does not require citations unless
  explicitly requested.

Code:

- If requires_code == true,
  include at least one minimal correct code snippet.

Style:

- Short paragraphs.
- Use bullets where helpful.
- Use code fences for code.
- Avoid fluff.
- Be precise and implementation-oriented.
"""


def worker_node(payload: dict) -> dict:

    task = Task(
        **payload["task"]
    )

    plan = Plan(
        **payload["plan"]
    )

    evidence = [
        EvidenceItem(**e)
        for e in payload.get(
            "evidence",
            []
        )
    ]

    topic = payload["topic"]

    mode = payload.get(
        "mode",
        "closed_book"
    )

    bullets_text = (
        "\n- "
        + "\n- ".join(
            task.bullets
        )
    )

    evidence_text = ""

    if evidence:

        evidence_text = "\n".join(
            (
                f"- {e.title} | "
                f"{e.url} | "
                f"{e.published_at or 'date:unknown'}"
            )
            for e in evidence[:20]
        )

    section_md = llm.invoke(
        [
            SystemMessage(
                content=WORKER_SYSTEM
            ),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"

                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: "
                    f"{task.requires_research}\n"
                    f"requires_citations: "
                    f"{task.requires_citations}\n"
                    f"requires_code: "
                    f"{task.requires_code}\n"

                    f"Bullets:{bullets_text}\n\n"

                    "Evidence:\n"
                    f"{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {
        "sections": [
            (
                task.id,
                section_md
            )
        ]
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    response = llm.invoke(
        "Explain self-attention in transformers "
        "in simple terms."
    )

    print(response.content)