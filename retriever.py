"""
retriever.py — Stage 1: Contextual RAG Retriever with Metadata Filtering

No vector DB required for seed data — uses structured in-memory filtering.
Drop-in replacement: swap _filter_* methods with ChromaDB/Pinecone calls
when scaling to a real embedding store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


# ──────────────────────────────────────────────
# Season normalisation
# ──────────────────────────────────────────────

_MONTH_TO_SEASON: dict[str, str] = {
    "january": "Winter",   "february": "Winter",
    "march": "Spring",     "april": "Spring",    "may": "Spring",
    "june": "Summer",      "july": "Monsoon",    "august": "Monsoon",
    "september": "Autumn", "october": "Autumn",  "november": "Autumn",
    "december": "Winter",
}

_CANONICAL_SEASONS = {"winter", "summer", "autumn", "spring", "monsoon"}


def month_to_season(travel_month: str) -> str:
    """Convert a free-text month/season string → consistently title-cased season."""
    key = travel_month.strip().lower()
    for s in _CANONICAL_SEASONS:
        if s in key:
            return s.title()           # "spring" → "Spring", "winter" → "Winter"
    return _MONTH_TO_SEASON.get(key, "Spring")


class SikkimRAGRetriever:
    """
    Loads hackathon_data.json once and exposes filtered views.
    retrieve() is the only public method consumed by the graph.
    """

    def __init__(
        self,
        data_path: str | Path,
        use_chroma: bool = False,
        chroma_collection: str = "sikkim_places",
        chroma_persist_dir: str = "./chroma_db",
    ) -> None:
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self._hubs: list[dict] = raw["hubs"]
        self._places: list[dict] = raw["places"]
        self._dining: list[dict] = raw["dining"]
        self._accommodations: list[dict] = raw["accommodations"]

        self._hub_by_name: dict[str, dict] = {h["name"]: h for h in self._hubs}
        self._place_by_id: dict[str, dict] = {p["id"]: p for p in self._places}

        # Expose lists for endpoint helpers (avoids re-reading the file)
        self.hub_names: list[str] = [h["name"] for h in self._hubs]
        self.interest_tags: list[str] = sorted(
            {tag for p in self._places for tag in p.get("interests", [])}
        )

        self._chroma_collection: Optional[Any] = None
        if use_chroma:
            self._init_chroma_collection(chroma_collection, chroma_persist_dir)

    # ──────────────────────────────────────────────
    # Optional ChromaDB path
    # ──────────────────────────────────────────────

    def _init_chroma_collection(self, collection_name: str, persist_dir: str) -> None:
        try:
            import chromadb
        except Exception as exc:
            raise RuntimeError(
                "ChromaDB requested but not installed. "
                "Install chromadb and an embedding function first."
            ) from exc

        client = chromadb.PersistentClient(path=persist_dir)
        self._chroma_collection = client.get_or_create_collection(name=collection_name)

        ids, documents, metadatas = [], [], []
        for p in self._places:
            ids.append(p["id"])
            documents.append(self._build_place_document(p))
            metadatas.append({
                "id": p["id"],
                "name": p.get("name", ""),
                "nearest_hub": p.get("nearest_hub", ""),
                "best_season_csv": ",".join(p.get("best_season", [])),
                "ideal_for_csv": ",".join(p.get("ideal_for", [])),
                "interests_csv": ",".join(p.get("interests", [])),
                "budget": p.get("budget", "medium"),
                "distance_from_hub_km": p.get("distance_from_hub_km", 999),
            })
        self._chroma_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def _build_place_document(self, p: dict) -> str:
        parts = [
            p.get("name", ""),
            p.get("description", ""),
            " ".join(p.get("interests", [])),
            " ".join(p.get("ideal_for", [])),
            " ".join(p.get("best_season", [])),
        ]
        return " | ".join(x for x in parts if x)

    def _filter_places_chroma(
        self,
        persona: str,
        season: str,
        interests: list[str],
        budget: str,
        reachable_hubs: set[str],
        limit: int,
    ) -> list[dict]:
        if self._chroma_collection is None:
            return self._filter_places(
                persona=persona, season=season, interests=interests,
                budget=budget, reachable_hubs=reachable_hubs, limit=limit,
            )

        query_text = ", ".join(interests) if interests else f"{persona} travel experiences"
        raw = self._chroma_collection.query(query_texts=[query_text], n_results=max(limit * 3, 20))

        ranked: list[tuple[int, dict]] = []
        for pid in raw.get("ids", [[]])[0]:
            p = self._place_by_id.get(pid)
            if not p:
                continue
            if p.get("nearest_hub") not in reachable_hubs:
                continue
            if season not in p.get("best_season", []):
                continue
            if persona not in p.get("ideal_for", []):
                continue
            if not self._budget_matches(p.get("budget", "medium"), budget):
                continue
            overlap = len(set(p.get("interests", [])) & set(interests))
            ranked.append((overlap, p))

        ranked.sort(key=lambda x: (-x[0], x[1].get("distance_from_hub_km", 999)))
        return [p for _, p in ranked[:limit]]

    # ──────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────

    def retrieve(
        self,
        persona: str,
        budget_preference: str,
        interests: list[str],
        arrival_hub: str,
        duration_days: int,
        travel_month: str,
        max_places: int = 30,
        max_dining: int = 15,
        max_acc: int = 10,
    ) -> dict[str, Any]:
        season = month_to_season(travel_month)

        hub = self._hub_by_name.get(arrival_hub) or self._hub_by_name.get("Gangtok", self._hubs[0])
        reachable_hubs = {hub["name"]} | set(hub.get("nearby_hubs", []))

        if self._chroma_collection is not None:
            places = self._filter_places_chroma(
                persona=persona, season=season, interests=interests,
                budget=budget_preference, reachable_hubs=reachable_hubs, limit=max_places,
            )
        else:
            places = self._filter_places(
                persona=persona, season=season, interests=interests,
                budget=budget_preference, reachable_hubs=reachable_hubs, limit=max_places,
            )

        return {
            "arrival_hub": hub,
            "reachable_hubs": list(reachable_hubs),
            "season": season,
            "duration_days": duration_days,
            "persona": persona,
            "budget_preference": budget_preference,
            "interests": interests,
            "retrieved_places": places,
            "retrieved_dining": self._filter_dining(
                persona=persona, budget=budget_preference,
                reachable_hubs=reachable_hubs, limit=max_dining,
            ),
            "retrieved_accommodations": self._filter_accommodations(
                persona=persona, budget=budget_preference,
                reachable_hubs=reachable_hubs, limit=max_acc,
            ),
        }

    # ──────────────────────────────────────────
    # Internal filters
    # ──────────────────────────────────────────

    def _budget_matches(self, item_budget: str, pref: str) -> bool:
        order = ["very low", "low", "medium", "high", "very high", "luxury"]
        try:
            return order.index(item_budget.lower()) <= order.index(pref.lower()) + 1
        except ValueError:
            return True

    def _filter_places(
        self,
        persona: str,
        season: str,
        interests: list[str],
        budget: str,
        reachable_hubs: set[str],
        limit: int,
    ) -> list[dict]:
        scored: list[tuple[int, dict]] = []
        for p in self._places:
            if p.get("nearest_hub") not in reachable_hubs:
                continue
            if season not in p.get("best_season", []):
                continue
            if persona not in p.get("ideal_for", []):
                continue
            if not self._budget_matches(p.get("budget", "medium"), budget):
                continue
            overlap = len(set(p.get("interests", [])) & set(interests))
            scored.append((overlap, p))

        scored.sort(key=lambda x: (-x[0], x[1].get("distance_from_hub_km", 999)))
        return [p for _, p in scored[:limit]]

    def _filter_dining(
        self,
        persona: str,
        budget: str,
        reachable_hubs: set[str],
        limit: int,
    ) -> list[dict]:
        results = [
            d for d in self._dining
            if d.get("nearest_hub") in reachable_hubs
            and persona in d.get("ideal_for", [])
            and self._budget_matches(d.get("budget", "medium"), budget)
        ]
        results.sort(key=lambda x: x.get("distance_from_hub_km", 999))
        return results[:limit]

    def _filter_accommodations(
        self,
        persona: str,
        budget: str,
        reachable_hubs: set[str],
        limit: int,
    ) -> list[dict]:
        results = [
            a for a in self._accommodations
            if a.get("nearest_hub") in reachable_hubs
            and persona in a.get("ideal_for", [])
            and self._budget_matches(a.get("budget", "medium"), budget)
        ]
        results.sort(key=lambda x: x.get("distance_from_hub_km", 999))
        return results[:limit]
