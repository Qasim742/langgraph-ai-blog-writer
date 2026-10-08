from __future__ import annotations

from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from groq import RateLimitError

import operator
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# -----------------------------
# 2) LLM
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Make sure it is present in your .env file."
    )


# ============================================================
# LLM - Groq
# ============================================================

llm = ChatGroq(
    # model="meta-llama/llama-prompt-guard-2-22m",
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY,
)

# ============================================================
# LLM - Hugging face
# ============================================================

# model = HuggingFaceEndpoint(
#     repo_id="openai/gpt-oss-120b",
#     huggingfacehub_api_token=os.getenv("HF_TOKEN"),
#     max_new_tokens=2048,
#     temperature=0.7,
# )

# llm = ChatHuggingFace(llm=model)


# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State) -> dict:
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))

    if not raw:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw}"
                )
            ),
        ]
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""


# ----------------------
#  waiting time for model rate limit errors (Groq) before retrying
# ----------------------

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_random_exponential(min=10, max=30),
    stop=stop_after_attempt(5),
    before_sleep=lambda retry_state: print(
        f"\n⏳ Groq rate limit reached. Agent is waiting "
        f"{retry_state.next_action.sleep:.1f} seconds before retrying "
        f"(attempt {retry_state.attempt_number}/5)...\n"
    ),
    reraise=True,
)



def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""

def decide_images(state: State) -> dict:
    planner = llm.with_structured_output(
            GlobalImagePlan,
            method="json_schema",
            strict=True,
        )
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        # generate only if needed
        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app











# from __future__ import annotations

# import json
# import operator
# import os
# import re
# from datetime import date, timedelta
# from pathlib import Path
# from typing import TypedDict, List, Optional, Literal, Annotated

# from pydantic import BaseModel, Field

# from dotenv import load_dotenv

# from huggingface_hub import InferenceClient

# from langgraph.graph import StateGraph, START, END
# from langgraph.types import Send


# # ============================================================
# # ENVIRONMENT
# # ============================================================

# load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")

# if not HF_TOKEN:
#     raise ValueError(
#         "HF_TOKEN not found. Add HF_TOKEN=hf_... to your .env file."
#     )


# # ============================================================
# # HUGGING FACE MODEL
# # ============================================================

# HF_MODEL = "Qwen/Qwen3-32B"

# # Hugging Face recommends selecting a provider for structured
# # output because model/provider compatibility can vary.
# HF_PROVIDER = "cerebras"

# hf_client = InferenceClient(
#     provider=HF_PROVIDER,
#     api_key=HF_TOKEN,
# )


# # ============================================================
# # 1) PYDANTIC SCHEMAS
# # ============================================================

# class Task(BaseModel):
#     id: int
#     title: str

#     goal: str = Field(
#         ...,
#         description="One sentence describing what the reader should do or understand."
#     )

#     bullets: List[str] = Field(
#         ...,
#         min_length=3,
#         max_length=6
#     )

#     target_words: int = Field(
#         ...,
#         description="Target words between 120 and 550."
#     )

#     tags: List[str] = Field(default_factory=list)

#     requires_research: bool = False
#     requires_citations: bool = False
#     requires_code: bool = False


# class Plan(BaseModel):
#     blog_title: str
#     audience: str
#     tone: str

#     blog_kind: Literal[
#         "explainer",
#         "tutorial",
#         "news_roundup",
#         "comparison",
#         "system_design"
#     ] = "explainer"

#     constraints: List[str] = Field(default_factory=list)

#     tasks: List[Task]


# class EvidenceItem(BaseModel):
#     title: str
#     url: str

#     published_at: Optional[str] = None
#     snippet: Optional[str] = None
#     source: Optional[str] = None


# class RouterDecision(BaseModel):
#     needs_research: bool

#     mode: Literal[
#         "closed_book",
#         "hybrid",
#         "open_book"
#     ]

#     reason: str

#     queries: List[str] = Field(
#         default_factory=list
#     )

#     max_results_per_query: int = Field(
#         5
#     )


# class EvidencePack(BaseModel):
#     evidence: List[EvidenceItem] = Field(
#         default_factory=list
#     )


# # ============================================================
# # IMAGE SCHEMA
# #
# # IMPORTANT:
# # We do NOT ask the model to return the entire Markdown blog.
# # The model only returns image specifications.
# # ============================================================

# class ImageSpec(BaseModel):
#     placeholder: str = Field(
#         ...,
#         description="Placeholder such as [[IMAGE_1]]"
#     )

#     filename: str = Field(
#         ...,
#         description="Filename under images/, e.g. architecture.png"
#     )

#     alt: str

#     caption: str

#     prompt: str = Field(
#         ...,
#         description="Prompt to send to the image generation model."
#     )

#     # Heading after which the image should be inserted.
#     insert_after_heading: str = Field(
#         ...,
#         description="Markdown heading after which the image should be inserted."
#     )

#     size: Literal[
#         "1024x1024",
#         "1024x1536",
#         "1536x1024"
#     ] = "1024x1024"

#     quality: Literal[
#         "low",
#         "medium",
#         "high"
#     ] = "medium"


# class GlobalImagePlan(BaseModel):
#     images: List[ImageSpec] = Field(
#         default_factory=list
#     )


# # ============================================================
# # LANGGRAPH STATE
# # ============================================================

# class State(TypedDict):
#     topic: str

#     # routing / research
#     mode: str
#     needs_research: bool
#     queries: List[str]

#     evidence: List[EvidenceItem]

#     plan: Optional[Plan]

#     # recency
#     as_of: str
#     recency_days: int

#     # workers
#     sections: Annotated[
#         List[tuple[int, str]],
#         operator.add
#     ]

#     # reducer / images
#     merged_md: str
#     md_with_placeholders: str
#     image_specs: List[dict]

#     final: str


# # ============================================================
# # HUGGING FACE HELPERS
# # ============================================================

# def _build_response_format(schema_model: type[BaseModel]) -> dict:
#     """
#     Convert a Pydantic model into Hugging Face JSON Schema
#     structured-output format.
#     """

#     return {
#         "type": "json_schema",
#         "json_schema": {
#             "name": schema_model.__name__,
#             "schema": schema_model.model_json_schema(),
#             "strict": True,
#         },
#     }


# def hf_structured(
#     schema_model: type[BaseModel],
#     messages: list[dict],
#     max_tokens: int = 3000,
# ) -> BaseModel:
#     """
#     Ask Hugging Face to return JSON matching a Pydantic schema.
#     """

#     response_format = _build_response_format(schema_model)

#     response = hf_client.chat_completion(
#         model=HF_MODEL,
#         messages=messages,
#         response_format=response_format,
#         max_tokens=max_tokens,
#         temperature=0.2,
#     )

#     content = response.choices[0].message.content

#     if not content:
#         raise RuntimeError(
#             f"Hugging Face returned empty structured output for "
#             f"{schema_model.__name__}."
#         )

#     try:
#         data = json.loads(content)
#     except json.JSONDecodeError as e:
#         raise RuntimeError(
#             f"Model returned invalid JSON for {schema_model.__name__}:\n"
#             f"{content}"
#         ) from e

#     try:
#         return schema_model.model_validate(data)
#     except Exception as e:
#         raise RuntimeError(
#             f"Model JSON did not match {schema_model.__name__}:\n"
#             f"{data}"
#         ) from e


# def hf_text(
#     messages: list[dict],
#     max_tokens: int = 2500,
#     temperature: float = 0.7,
# ) -> str:
#     """
#     Normal text generation through Hugging Face.
#     """

#     response = hf_client.chat_completion(
#         model=HF_MODEL,
#         messages=messages,
#         max_tokens=max_tokens,
#         temperature=temperature,
#     )

#     content = response.choices[0].message.content

#     if not content:
#         raise RuntimeError(
#             "Hugging Face returned empty text output."
#         )

#     return content.strip()


# # ============================================================
# # 2) ROUTER
# # ============================================================

# ROUTER_SYSTEM = """
# You are a routing module for a technical blog planner.

# Decide whether web research is needed BEFORE planning.

# Modes:

# closed_book:
# - needs_research=false
# - Use for stable, evergreen concepts.

# hybrid:
# - needs_research=true
# - Use for evergreen topics that also need current
#   tools, models, examples, libraries, or technologies.

# open_book:
# - needs_research=true
# - Use for volatile topics such as:
#   latest news, weekly news, current pricing,
#   current policies, recent releases, current events.

# Rules:

# If needs_research=true:
# - Produce 3 to 10 high-signal search queries.
# - Queries should be specific and useful.
# - For open_book weekly/news topics, focus on recent information.

# Return ONLY data matching the required JSON schema.
# """


# def router_node(state: State) -> dict:

#     decision = hf_structured(
#         RouterDecision,
#         [
#             {
#                 "role": "system",
#                 "content": ROUTER_SYSTEM,
#             },
#             {
#                 "role": "user",
#                 "content": (
#                     f"Topic: {state['topic']}\n"
#                     f"As-of date: {state['as_of']}"
#                 ),
#             },
#         ],
#         max_tokens=1000,
#     )

#     if decision.mode == "open_book":
#         recency_days = 7

#     elif decision.mode == "hybrid":
#         recency_days = 45

#     else:
#         recency_days = 3650

#     return {
#         "needs_research": decision.needs_research,
#         "mode": decision.mode,
#         "queries": decision.queries,
#         "recency_days": recency_days,
#     }


# def route_next(state: State) -> str:

#     if state["needs_research"]:
#         return "research"

#     return "orchestrator"


# # ============================================================
# # 3) TAVILY RESEARCH
# # ============================================================

# def _tavily_search(
#     query: str,
#     max_results: int = 5
# ) -> List[dict]:

#     if not os.getenv("TAVILY_API_KEY"):
#         return []

#     try:

#         from langchain_community.tools.tavily_search import (
#             TavilySearchResults
#         )

#         tool = TavilySearchResults(
#             max_results=max_results
#         )

#         results = tool.invoke(
#             {
#                 "query": query
#             }
#         )

#         out: List[dict] = []

#         for r in results or []:

#             out.append(
#                 {
#                     "title": r.get("title") or "",
#                     "url": r.get("url") or "",
#                     "snippet": (
#                         r.get("content")
#                         or r.get("snippet")
#                         or ""
#                     ),
#                     "published_at": (
#                         r.get("published_date")
#                         or r.get("published_at")
#                     ),
#                     "source": r.get("source"),
#                 }
#             )

#         return out

#     except Exception as e:

#         print(
#             f"Tavily search failed: {e}"
#         )

#         return []


# def _iso_to_date(
#     s: Optional[str]
# ) -> Optional[date]:

#     if not s:
#         return None

#     try:
#         return date.fromisoformat(
#             s[:10]
#         )

#     except Exception:
#         return None


# # ============================================================
# # 4) RESEARCH NODE
# # ============================================================

# RESEARCH_SYSTEM = """
# You are a research synthesizer.

# Given raw web search results, produce EvidenceItem objects.

# Rules:

# - Only include items with a non-empty URL.
# - Prefer relevant and authoritative sources.
# - Normalize published_at to ISO YYYY-MM-DD when reliably known.
# - If publication date is uncertain, use null.
# - Do not guess dates.
# - Keep snippets short.
# - Deduplicate sources conceptually.

# Return ONLY data matching the required JSON schema.
# """


# def research_node(state: State) -> dict:

#     queries = (
#         state.get("queries") or []
#     )[:10]

#     raw: List[dict] = []

#     for q in queries:

#         raw.extend(
#             _tavily_search(
#                 q,
#                 max_results=6
#             )
#         )

#     if not raw:

#         return {
#             "evidence": []
#         }

#     extractor = hf_structured(
#         EvidencePack,
#         [
#             {
#                 "role": "system",
#                 "content": RESEARCH_SYSTEM,
#             },
#             {
#                 "role": "user",
#                 "content": (
#                     f"As-of date: {state['as_of']}\n"
#                     f"Recency days: {state['recency_days']}\n\n"
#                     f"Raw search results:\n"
#                     f"{json.dumps(raw, ensure_ascii=False)}"
#                 ),
#             },
#         ],
#         max_tokens=4000,
#     )

#     dedup = {}

#     for e in extractor.evidence:

#         if e.url:
#             dedup[e.url] = e

#     evidence = list(
#         dedup.values()
#     )

#     # Filter recent evidence for open_book.
#     if state.get("mode") == "open_book":

#         as_of = date.fromisoformat(
#             state["as_of"]
#         )

#         cutoff = (
#             as_of
#             - timedelta(
#                 days=int(
#                     state["recency_days"]
#                 )
#             )
#         )

#         evidence = [
#             e
#             for e in evidence
#             if (
#                 d := _iso_to_date(
#                     e.published_at
#                 )
#             )
#             and d >= cutoff
#         ]

#     return {
#         "evidence": evidence
#     }


# # ============================================================
# # 5) ORCHESTRATOR
# # ============================================================

# ORCH_SYSTEM = """
# You are a senior technical writer and developer advocate.

# Create an actionable outline for a technical blog.

# Requirements:

# - Create 5 to 9 tasks.
# - Every task must have:
#   - id
#   - title
#   - goal
#   - 3 to 6 bullets
#   - target_words
# - Target words should normally be between 120 and 550.
# - Tags are optional.

# Grounding:

# closed_book:
# - Evergreen topic.
# - Do not require external evidence.

# hybrid:
# - Use evidence for current tools, models, libraries,
#   examples, or other changing information.
# - Mark relevant tasks requires_research=true.
# - Mark relevant tasks requires_citations=true.

# open_book:
# - Set blog_kind="news_roundup".
# - Focus on current events and implications.
# - Do not invent events.
# - If evidence is weak, explicitly reflect that.

# Return ONLY data matching the required JSON schema.
# """


# def orchestrator_node(state: State) -> dict:

#     mode = state.get(
#         "mode",
#         "closed_book"
#     )

#     evidence = state.get(
#         "evidence",
#         []
#     )

#     forced_kind = (
#         "news_roundup"
#         if mode == "open_book"
#         else None
#     )

#     evidence_data = [
#         e.model_dump()
#         for e in evidence
#     ][:16]

#     plan = hf_structured(
#         Plan,
#         [
#             {
#                 "role": "system",
#                 "content": ORCH_SYSTEM,
#             },
#             {
#                 "role": "user",
#                 "content": (
#                     f"Topic: {state['topic']}\n"
#                     f"Mode: {mode}\n"
#                     f"As-of: {state['as_of']}\n"
#                     f"Recency days: {state['recency_days']}\n"
#                     f"{'Set blog_kind=news_roundup.' if forced_kind else ''}\n\n"
#                     f"Evidence:\n"
#                     f"{json.dumps(evidence_data, ensure_ascii=False)}"
#                 ),
#             },
#         ],
#         max_tokens=5000,
#     )

#     if forced_kind:

#         plan.blog_kind = "news_roundup"

#     return {
#         "plan": plan
#     }


# # ============================================================
# # 6) FANOUT
# # ============================================================

# def fanout(state: State):

#     assert state["plan"] is not None

#     return [
#         Send(
#             "worker",
#             {
#                 "task": task.model_dump(),
#                 "topic": state["topic"],
#                 "mode": state["mode"],
#                 "as_of": state["as_of"],
#                 "recency_days": state["recency_days"],
#                 "plan": state["plan"].model_dump(),
#                 "evidence": [
#                     e.model_dump()
#                     for e in state.get(
#                         "evidence",
#                         []
#                     )
#                 ],
#             },
#         )

#         for task in state["plan"].tasks
#     ]


# # ============================================================
# # 7) WORKER
# # ============================================================

# WORKER_SYSTEM = """
# You are a senior technical writer and developer advocate.

# Write ONE section of a technical blog post in Markdown.

# Rules:

# - Cover ALL provided bullets in order.
# - Stay close to the target word count.
# - Output only section Markdown.
# - Start with:
#   ## <Section Title>

# Scope guard:

# If blog_kind="news_roundup":
# - Do NOT drift into tutorials.
# - Focus on events, developments, and implications.

# Grounding:

# If mode="open_book":
# - Do not introduce specific event/company/model/funding/
#   policy claims unless supported by provided evidence URLs.
# - For supported external claims, use Markdown links.
# - If something cannot be supported, write:
#   "Not found in provided sources."

# If requires_citations=true:
# - Cite relevant evidence URLs.

# Code:

# If requires_code=true:
# - Include at least one minimal useful code snippet.
# """


# def worker_node(payload: dict) -> dict:

#     task = Task(
#         **payload["task"]
#     )

#     plan = Plan(
#         **payload["plan"]
#     )

#     evidence = [
#         EvidenceItem(**e)
#         for e in payload.get(
#             "evidence",
#             []
#         )
#     ]

#     bullets_text = (
#         "\n- "
#         + "\n- ".join(
#             task.bullets
#         )
#     )

#     evidence_text = "\n".join(
#         f"- {e.title} | "
#         f"{e.url} | "
#         f"{e.published_at or 'date:unknown'}"
#         for e in evidence[:20]
#     )

#     section_md = hf_text(
#         [
#             {
#                 "role": "system",
#                 "content": WORKER_SYSTEM,
#             },
#             {
#                 "role": "user",
#                 "content": (
#                     f"Blog title: {plan.blog_title}\n"
#                     f"Audience: {plan.audience}\n"
#                     f"Tone: {plan.tone}\n"
#                     f"Blog kind: {plan.blog_kind}\n"
#                     f"Constraints: {plan.constraints}\n"
#                     f"Topic: {payload['topic']}\n"
#                     f"Mode: {payload.get('mode')}\n"
#                     f"As-of: {payload.get('as_of')}\n"
#                     f"Recency days: {payload.get('recency_days')}\n\n"

#                     f"Section title: {task.title}\n"
#                     f"Goal: {task.goal}\n"
#                     f"Target words: {task.target_words}\n"
#                     f"Tags: {task.tags}\n"
#                     f"requires_research: {task.requires_research}\n"
#                     f"requires_citations: {task.requires_citations}\n"
#                     f"requires_code: {task.requires_code}\n"

#                     f"\nBullets:{bullets_text}\n\n"

#                     f"Evidence "
#                     f"(ONLY cite these URLs):\n"
#                     f"{evidence_text}\n"
#                 ),
#             },
#         ],
#         max_tokens=max(
#             1200,
#             int(
#                 task.target_words * 2
#             )
#         ),
#         temperature=0.7,
#     )

#     return {
#         "sections": [
#             (
#                 task.id,
#                 section_md
#             )
#         ]
#     }


# # ============================================================
# # 8) MERGE CONTENT
# # ============================================================

# def merge_content(state: State) -> dict:

#     plan = state["plan"]

#     if plan is None:
#         raise ValueError(
#             "merge_content called without plan."
#         )

#     ordered_sections = [
#         md
#         for _, md in sorted(
#             state["sections"],
#             key=lambda x: x[0]
#         )
#     ]

#     body = "\n\n".join(
#         ordered_sections
#     ).strip()

#     merged_md = (
#         f"# {plan.blog_title}\n\n"
#         f"{body}\n"
#     )

#     return {
#         "merged_md": merged_md
#     }


# # ============================================================
# # 9) IMAGE PLANNER
# #
# # IMPORTANT CHANGE:
# # The model only returns image specifications.
# # It does NOT return the entire blog Markdown.
# # ============================================================

# DECIDE_IMAGES_SYSTEM = """
# You are an expert technical editor.

# Analyze the provided technical blog and decide whether
# technical diagrams or visuals would materially improve it.

# Rules:

# - Maximum 3 images.
# - Prefer technical diagrams, architecture diagrams,
#   workflows, system diagrams, or conceptual visuals.
# - Avoid decorative images.
# - Each image must provide real explanatory value.
# - If no image is needed, return images=[].

# For every image:

# placeholder:
#   [[IMAGE_1]]
#   [[IMAGE_2]]
#   [[IMAGE_3]]

# filename:
#   A safe filename ending in .png.

# alt:
#   Useful accessibility description.

# caption:
#   Short explanatory caption.

# prompt:
#   Detailed prompt for an image generation model.

# insert_after_heading:
#   EXACT Markdown heading from the blog after which
#   the image should be inserted.

# Return ONLY data matching the required JSON schema.
# """


# def decide_images(state: State) -> dict:

#     plan = state["plan"]

#     if plan is None:
#         raise ValueError(
#             "decide_images called without plan."
#         )

#     merged_md = state["merged_md"]

#     image_plan = hf_structured(
#         GlobalImagePlan,
#         [
#             {
#                 "role": "system",
#                 "content": DECIDE_IMAGES_SYSTEM,
#             },
#             {
#                 "role": "user",
#                 "content": (
#                     f"Blog kind: {plan.blog_kind}\n"
#                     f"Topic: {state['topic']}\n\n"
#                     f"Blog:\n{merged_md}"
#                 ),
#             },
#         ],
#         max_tokens=2500,
#     )

#     md = merged_md

#     image_specs = []

#     for index, image in enumerate(
#         image_plan.images[:3],
#         start=1
#     ):

#         placeholder = (
#             f"[[IMAGE_{index}]]"
#         )

#         # Force the placeholder format
#         # instead of trusting arbitrary model output.
#         image.placeholder = placeholder

#         heading = (
#             image.insert_after_heading.strip()
#         )

#         if heading and heading in md:

#             md = md.replace(
#                 heading,
#                 (
#                     f"{heading}\n\n"
#                     f"{placeholder}"
#                 ),
#                 1,
#             )

#         else:

#             # Graceful fallback:
#             # place image at end if heading wasn't found.
#             md = (
#                 md.rstrip()
#                 + f"\n\n{placeholder}\n"
#             )

#         image_specs.append(
#             image.model_dump()
#         )

#     return {
#         "md_with_placeholders": md,
#         "image_specs": image_specs,
#     }


# # ============================================================
# # 10) GEMINI IMAGE GENERATION
# # ============================================================

# def _gemini_generate_image_bytes(
#     prompt: str
# ) -> bytes:

#     """
#     Generate an image using Gemini.

#     Requires:
#         pip install google-genai

#     Environment:
#         GOOGLE_API_KEY
#     """

#     from google import genai
#     from google.genai import types

#     api_key = os.environ.get(
#         "GOOGLE_API_KEY"
#     )

#     if not api_key:
#         raise RuntimeError(
#             "GOOGLE_API_KEY is not set."
#         )

#     client = genai.Client(
#         api_key=api_key
#     )

#     resp = client.models.generate_content(
#         model="gemini-2.5-flash-image",
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             response_modalities=["IMAGE"],
#             safety_settings=[
#                 types.SafetySetting(
#                     category="HARM_CATEGORY_DANGEROUS_CONTENT",
#                     threshold="BLOCK_ONLY_HIGH",
#                 )
#             ],
#         ),
#     )

#     parts = getattr(
#         resp,
#         "parts",
#         None
#     )

#     if (
#         not parts
#         and getattr(
#             resp,
#             "candidates",
#             None
#         )
#     ):

#         try:

#             parts = (
#                 resp
#                 .candidates[0]
#                 .content
#                 .parts
#             )

#         except Exception:

#             parts = None

#     if not parts:

#         raise RuntimeError(
#             "No image content returned."
#         )

#     for part in parts:

#         inline = getattr(
#             part,
#             "inline_data",
#             None
#         )

#         if (
#             inline
#             and getattr(
#                 inline,
#                 "data",
#                 None
#             )
#         ):

#             return inline.data

#     raise RuntimeError(
#         "No inline image bytes found."
#     )


# # ============================================================
# # 11) SAFE SLUG
# # ============================================================

# def _safe_slug(
#     title: str
# ) -> str:

#     s = title.strip().lower()

#     s = re.sub(
#         r"[^a-z0-9 _-]+",
#         "",
#         s
#     )

#     s = re.sub(
#         r"\s+",
#         "_",
#         s
#     )

#     s = s.strip("_")

#     return s or "blog"


# # ============================================================
# # 12) GENERATE + PLACE IMAGES
# # ============================================================

# def generate_and_place_images(
#     state: State
# ) -> dict:

#     plan = state["plan"]

#     if plan is None:
#         raise ValueError(
#             "generate_and_place_images called without plan."
#         )

#     md = (
#         state.get(
#             "md_with_placeholders"
#         )
#         or state["merged_md"]
#     )

#     image_specs = (
#         state.get(
#             "image_specs",
#             []
#         )
#         or []
#     )

#     # -----------------------------------------
#     # No images
#     # -----------------------------------------

#     if not image_specs:

#         filename = (
#             f"{_safe_slug(plan.blog_title)}.md"
#         )

#         Path(filename).write_text(
#             md,
#             encoding="utf-8"
#         )

#         return {
#             "final": md
#         }

#     # -----------------------------------------
#     # Images directory
#     # -----------------------------------------

#     images_dir = Path(
#         "images"
#     )

#     images_dir.mkdir(
#         exist_ok=True
#     )

#     # -----------------------------------------
#     # Generate images
#     # -----------------------------------------

#     for spec in image_specs:

#         placeholder = spec[
#             "placeholder"
#         ]

#         filename = spec[
#             "filename"
#         ]

#         out_path = (
#             images_dir / filename
#         )

#         # -------------------------------------
#         # Generate only if image doesn't exist
#         # -------------------------------------

#         if not out_path.exists():

#             try:

#                 img_bytes = (
#                     _gemini_generate_image_bytes(
#                         spec["prompt"]
#                     )
#                 )

#                 out_path.write_bytes(
#                     img_bytes
#                 )

#             except Exception as e:

#                 prompt_block = (
#                     f"> **[IMAGE GENERATION FAILED]** "
#                     f"{spec.get('caption', '')}\n"
#                     f">\n"
#                     f"> **Alt:** "
#                     f"{spec.get('alt', '')}\n"
#                     f">\n"
#                     f"> **Prompt:** "
#                     f"{spec.get('prompt', '')}\n"
#                     f">\n"
#                     f"> **Error:** {e}\n"
#                 )

#                 md = md.replace(
#                     placeholder,
#                     prompt_block
#                 )

#                 continue

#         # -------------------------------------
#         # Markdown image
#         # -------------------------------------

#         img_md = (
#             f"![{spec['alt']}]"
#             f"(images/{filename})\n"
#             f"*{spec['caption']}*"
#         )

#         md = md.replace(
#             placeholder,
#             img_md
#         )

#     # -----------------------------------------
#     # Save final Markdown
#     # -----------------------------------------

#     filename = (
#         f"{_safe_slug(plan.blog_title)}.md"
#     )

#     Path(filename).write_text(
#         md,
#         encoding="utf-8"
#     )

#     return {
#         "final": md
#     }


# # ============================================================
# # 13) REDUCER SUBGRAPH
# # ============================================================

# reducer_graph = StateGraph(
#     State
# )

# reducer_graph.add_node(
#     "merge_content",
#     merge_content
# )

# reducer_graph.add_node(
#     "decide_images",
#     decide_images
# )

# reducer_graph.add_node(
#     "generate_and_place_images",
#     generate_and_place_images
# )

# reducer_graph.add_edge(
#     START,
#     "merge_content"
# )

# reducer_graph.add_edge(
#     "merge_content",
#     "decide_images"
# )

# reducer_graph.add_edge(
#     "decide_images",
#     "generate_and_place_images"
# )

# reducer_graph.add_edge(
#     "generate_and_place_images",
#     END
# )

# reducer_subgraph = (
#     reducer_graph.compile()
# )


# # ============================================================
# # 14) MAIN GRAPH
# # ============================================================

# g = StateGraph(
#     State
# )

# g.add_node(
#     "router",
#     router_node
# )

# g.add_node(
#     "research",
#     research_node
# )

# g.add_node(
#     "orchestrator",
#     orchestrator_node
# )

# g.add_node(
#     "worker",
#     worker_node
# )

# g.add_node(
#     "reducer",
#     reducer_subgraph
# )


# # START
# g.add_edge(
#     START,
#     "router"
# )


# # Router -> Research / Orchestrator
# g.add_conditional_edges(
#     "router",
#     route_next,
#     {
#         "research": "research",
#         "orchestrator": "orchestrator",
#     },
# )


# # Research -> Orchestrator
# g.add_edge(
#     "research",
#     "orchestrator"
# )


# # Orchestrator -> Workers
# g.add_conditional_edges(
#     "orchestrator",
#     fanout,
#     ["worker"]
# )


# # Worker -> Reducer
# g.add_edge(
#     "worker",
#     "reducer"
# )


# # Reducer -> END
# g.add_edge(
#     "reducer",
#     END
# )


# # ============================================================
# # 15) COMPILE
# # ============================================================

# app = g.compile()


# # ============================================================
# # OPTIONAL TEST
# # ============================================================

# if __name__ == "__main__":

#     result = app.invoke(
#         {
#             "topic": "How RAG works with LangChain",
#             "mode": "",
#             "needs_research": False,
#             "queries": [],
#             "evidence": [],
#             "plan": None,
#             "as_of": date.today().isoformat(),
#             "recency_days": 3650,
#             "sections": [],
#             "merged_md": "",
#             "md_with_placeholders": "",
#             "image_specs": [],
#             "final": "",
#         }
#     )

#     print("\n")
#     print("=" * 70)
#     print("BLOG GENERATED")
#     print("=" * 70)
#     print(result["final"])