"""
main.py — Stage 3: FastAPI Packaging

Endpoints:
  POST /api/itinerary/generate   — main generation endpoint
  GET  /api/itinerary/health     — liveness probe
  GET  /api/itinerary/hubs       — list available arrival hubs
  GET  /api/itinerary/interests  — list supported interest tags
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from retriever import SikkimRAGRetriever
from graph import itinerary_graph

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()

DATA_PATH = Path(os.environ.get("SIKKIM_DATA_PATH", "hackathon_data.json"))

app = FastAPI(
    title="Sikkim AI Travel Itinerary API",
    description="RAG + LangGraph multi-agent itinerary generator for Sikkim",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

retriever = SikkimRAGRetriever(DATA_PATH)


# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────

class UserProfile(BaseModel):
    persona: str = Field(..., example="couple")
    budget_preference: str = Field(..., example="medium")
    interests: list[str] = Field(default_factory=list, example=["photography", "spiritual"])

    @field_validator("persona", mode="before")
    @classmethod
    def validate_persona(cls, v: str) -> str:
        valid = {"solo", "couple", "friends", "family"}
        v = v.strip().lower()
        if v not in valid:
            raise ValueError(f"persona must be one of {valid}")
        return v

    @field_validator("budget_preference", mode="before")
    @classmethod
    def validate_budget(cls, v: str) -> str:
        valid = {"very low", "low", "medium", "high", "very high", "luxury"}
        v = v.strip().lower()
        if v not in valid:
            raise ValueError(f"budget_preference must be one of {valid}")
        return v


class TravelDetails(BaseModel):
    arrival_hub: str = Field(..., example="Gangtok")
    duration_days: int = Field(..., ge=1, le=14, example=5)
    travel_month: str = Field(..., example="Spring")


class ItineraryRequest(BaseModel):
    user_profile: UserProfile
    travel_details: TravelDetails


class ItineraryResponse(BaseModel):
    itinerary_id: str
    days: list[dict]
    pro_tips: list[str]


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/api/itinerary/health")
async def health():
    return {"status": "ok", "data_loaded": DATA_PATH.exists()}


@app.get("/api/itinerary/hubs")
async def list_hubs():
    """Return all available arrival hub names."""
    return {"hubs": retriever.hub_names}


@app.get("/api/itinerary/interests")
async def list_interests():
    """Return all unique interest tags present in the places data."""
    return {"interests": retriever.interest_tags}


@app.post("/api/itinerary/generate", response_model=ItineraryResponse)
async def generate_itinerary(req: ItineraryRequest):
    """
    Runs the full RAG → LangGraph pipeline and returns a structured itinerary.
    Typical latency: 15–45 s (3 LLM calls + retrieval).
    """
    profile = req.user_profile
    details = req.travel_details

    # Stage 1: RAG retrieval
    context = retriever.retrieve(
        persona=profile.persona,
        budget_preference=profile.budget_preference,
        interests=profile.interests,
        arrival_hub=details.arrival_hub,
        duration_days=details.duration_days,
        travel_month=details.travel_month,
    )

    if not context["retrieved_places"]:
        raise HTTPException(
            status_code=422,
            detail=(
                "No places matched your filters. "
                "Try a different season, persona, or budget level."
            ),
        )

    # Stage 2: LangGraph multi-agent run
    initial_state: dict = {
        **context,
        "draft_days": [],
        "critic_feedback": "",
        "critic_approved": False,
        "iteration": 0,
        "final_itinerary": {},
    }

    try:
        final_state = itinerary_graph.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {exc}")

    # Stage 3: Package response
    result = final_state.get("final_itinerary", {})
    if not result:
        raise HTTPException(status_code=500, detail="Refiner returned an empty itinerary.")

    days = result.get("days", [])
    hub_name = details.arrival_hub

    # Final safety net: pad any days the LLM silently dropped
    while len(days) < details.duration_days:
        next_day = len(days) + 1
        days.append({
            "day": next_day,
            "hub": hub_name,
            "narrative": f"Day {next_day} — free time to explore {hub_name} at your own pace.",
            "activities": [{
                "place_id": f"leisure_day_{next_day}",
                "name": "Leisure & local market walk",
                "duration_mins": 240,
                "distance_from_hub_km": 0,
                "tip": "Great opportunity to pick up local handicrafts and try street food.",
            }],
            "meals": [],
            "stay": days[-1].get("stay", {}) if days else {},
        })

    return ItineraryResponse(
        itinerary_id=result.get("itinerary_id", "sikkim_000"),
        days=days,
        pro_tips=result.get("pro_tips", []),
    )
