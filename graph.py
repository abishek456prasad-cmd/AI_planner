"""
graph.py — Stage 2: Multi-Agent LangGraph Pipeline

Three-node loop:
  PlannerNode       → drafts day-by-day place sequence
  LogisticsCritic   → validates distance / time / stay-days; may REJECT
  ExperienceRefiner → injects meals, accommodation, tips into final state
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ──────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not _GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

llm = ChatGroq(
    groq_api_key=_GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
)


def _call_llm(system: str, user: str) -> str:
    """Invoke Groq LLM. Returns raw text content."""
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.content


def _extract_json(text: str) -> Any:
    """Pull the first valid JSON block from LLM output."""
    # Try ```json ... ``` fence first
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Fallback: first top-level { } or [ ] (non-greedy to avoid over-capture)
    m = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    raise ValueError(f"No JSON found in LLM output:\n{text[:500]}")


# ──────────────────────────────────────────────
# Shared Graph State
# ──────────────────────────────────────────────

class ItineraryState(TypedDict):
    # RAG context — set once at entry
    arrival_hub: dict
    reachable_hubs: list[str]
    season: str
    duration_days: int
    persona: str
    budget_preference: str
    interests: list[str]
    retrieved_places: list[dict]
    retrieved_dining: list[dict]
    retrieved_accommodations: list[dict]

    # Mutable across iterations
    draft_days: list[dict]
    critic_feedback: str
    critic_approved: bool
    iteration: int        # number of completed planner→critic cycles
    final_itinerary: dict


# ──────────────────────────────────────────────
# Node 1 — Planner
# ──────────────────────────────────────────────

_PLANNER_SYSTEM = """You are a Sikkim travel itinerary planner.
Given candidate places and user preferences, produce a DRAFT day-by-day itinerary as JSON.

RULES (violation = rejection by critic):
- "days" array MUST contain EXACTLY the requested number of days. Never fewer.
- Each day MUST have at least 1 activity and a primary hub.
- Total visit_duration_mins per day must not exceed 480 (8 hours).
- Respect best_time_of_day: morning places first.
- If places run out, add a leisure/exploration day for that hub — never drop days.
- Return ONLY valid JSON, no prose.

Schema:
{
  "days": [
    {
      "day": 1,
      "hub": "Gangtok",
      "activities": [
        { "place_id": "place_001", "name": "...", "duration_mins": 60,
          "distance_from_hub_km": 20, "tip": "..." }
      ]
    }
  ]
}"""


def planner_node(state: ItineraryState) -> dict:
    places_summary = [
        {
            "id": p["id"],
            "name": p["name"],
            "hub": p["nearest_hub"],
            "distance_km": p["distance_from_hub_km"],
            "duration_mins": p["visit_duration_mins"],
            "best_time": p["best_time_of_day"],
            "tip": p.get("tip", ""),
            "interests": p.get("interests", []),
        }
        for p in state["retrieved_places"]
    ]

    prompt = f"""
User profile:
- Persona: {state['persona']}
- Budget: {state['budget_preference']}
- Interests: {state['interests']}
- Arrival hub: {state['arrival_hub']['name']}
- Duration: {state['duration_days']} days
- Season: {state['season']}

Previous critic feedback (empty on first run): "{state.get('critic_feedback', '')}"

Candidate places ({len(places_summary)} total):
{json.dumps(places_summary, indent=2)}

Draft a {state['duration_days']}-day itinerary using the most relevant places.
"""
    raw = _call_llm(_PLANNER_SYSTEM, prompt)
    draft = _extract_json(raw)
    days: list[dict] = draft.get("days", [])

    # Pad missing days with a distinct leisure fallback (never copy Day 1)
    hub_name = state["arrival_hub"]["name"]
    while len(days) < state["duration_days"]:
        next_day = len(days) + 1
        days.append({
            "day": next_day,
            "hub": hub_name,
            "activities": [{
                "place_id": f"leisure_day_{next_day}",
                "name": f"Day {next_day} — Leisure & local exploration in {hub_name}",
                "duration_mins": 240,
                "distance_from_hub_km": 0,
                "tip": "Explore the local market, try street food, or rest at your accommodation.",
            }],
        })

    return {
        "draft_days": days,
        "critic_approved": False,
        "iteration": state.get("iteration", 0) + 1,
    }


# ──────────────────────────────────────────────
# Node 2 — Logistics Critic
# ──────────────────────────────────────────────

_CRITIC_SYSTEM = """You are a logistics critic for Sikkim travel itineraries.
Evaluate the draft against ALL rules:
1. "days" length MUST exactly equal duration_days. Too few = REJECT.
2. Total daily visit_duration_mins ≤ 480.
3. No single day exceeds 120 km total road distance from the hub.
4. High-altitude places (tip mentions "warm layers" or "permit") must not share
   a day with another high-altitude place.

Return ONLY valid JSON:
{
  "approved": true | false,
  "issues": ["issue 1"],
  "feedback": "One concise paragraph for the planner to fix, or empty string if approved."
}"""


def logistics_critic_node(state: ItineraryState) -> dict:
    prompt = f"""
Required duration_days: {state['duration_days']}
Draft itinerary ({len(state['draft_days'])} days submitted):
{json.dumps(state['draft_days'], indent=2)}
"""
    raw = _call_llm(_CRITIC_SYSTEM, prompt)
    result = _extract_json(raw)

    approved = bool(result.get("approved", False))
    # Allow at most 2 planner→critic cycles; force approval on the second pass
    # to prevent an infinite loop. iteration is incremented in planner_node,
    # so after the second planning pass iteration == 2 when critic runs here.
    if state.get("iteration", 1) >= 2:
        approved = True

    return {
        "critic_feedback": result.get("feedback", ""),
        "critic_approved": approved,
    }


# ──────────────────────────────────────────────
# Node 3 — Experience Refiner
# ──────────────────────────────────────────────

_REFINER_SYSTEM = """You are an experience refiner for Sikkim travel itineraries.
Enhance every day of the approved draft with:
- 2 meal stops (breakfast + lunch or lunch + dinner) from the dining list.
- 1 accommodation for overnight stay from the accommodations list.
- A short narrative (2 sentences) for each day.
- Collect unique activity tips into a global pro_tips list.

Return ONLY valid JSON matching this schema exactly:
{
  "itinerary_id": "sikkim_XXXXX",
  "days": [
    {
      "day": 1,
      "hub": "Gangtok",
      "narrative": "...",
      "activities": [...],
      "meals": [
        { "meal": "breakfast|lunch|dinner", "name": "...", "id": "...", "distance_from_hub_km": 2 }
      ],
      "stay": { "name": "...", "id": "...", "type": "...", "budget": "..." }
    }
  ],
  "pro_tips": ["tip 1", "tip 2"]
}"""


def experience_refiner_node(state: ItineraryState) -> dict:
    dining_summary = [
        {
            "id": d["id"], "name": d["name"], "hub": d["nearest_hub"],
            "distance_km": d["distance_from_hub_km"], "budget": d["budget"],
        }
        for d in state["retrieved_dining"]
    ]
    acc_summary = [
        {
            "id": a["id"], "name": a["name"], "hub": a["nearest_hub"],
            "distance_km": a["distance_from_hub_km"], "budget": a["budget"],
            "type": a["type"],
        }
        for a in state["retrieved_accommodations"]
    ]

    prompt = f"""
Enrich ALL {state['duration_days']} days below — do not drop any.

Approved draft:
{json.dumps(state['draft_days'], indent=2)}

Available dining:
{json.dumps(dining_summary, indent=2)}

Available accommodations:
{json.dumps(acc_summary, indent=2)}

User budget: {state['budget_preference']}
Persona: {state['persona']}

Produce the final enriched itinerary with exactly {state['duration_days']} days.
"""
    raw = _call_llm(_REFINER_SYSTEM, prompt)
    final = _extract_json(raw)

    if not final.get("itinerary_id"):
        final["itinerary_id"] = f"sikkim_{uuid.uuid4().hex[:6]}"

    return {"final_itinerary": final}


# ──────────────────────────────────────────────
# Routing
# ──────────────────────────────────────────────

def should_replan(state: ItineraryState) -> str:
    return "refine" if state["critic_approved"] else "replan"


# ──────────────────────────────────────────────
# Graph
# ──────────────────────────────────────────────

def build_itinerary_graph() -> StateGraph:
    g = StateGraph(ItineraryState)

    g.add_node("planner", planner_node)
    g.add_node("critic", logistics_critic_node)
    g.add_node("refiner", experience_refiner_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "critic")
    g.add_conditional_edges(
        "critic",
        should_replan,
        {"replan": "planner", "refine": "refiner"},
    )
    g.add_edge("refiner", END)

    return g.compile()


itinerary_graph = build_itinerary_graph()
