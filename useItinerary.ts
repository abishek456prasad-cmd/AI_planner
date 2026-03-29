/**
 * useItinerary.ts
 * React hook that calls the Sikkim itinerary FastAPI endpoint.
 *
 * Usage:
 *   const { generate, data, loading, error } = useItinerary();
 *   await generate({ user_profile, travel_details });
 */

import { useState, useCallback, useRef, useEffect } from "react";

// ── Types matching the FastAPI schemas ──────────────────────────────────────

export interface UserProfile {
  persona: "solo" | "couple" | "friends" | "family";
  budget_preference: "very low" | "low" | "medium" | "high" | "very high" | "luxury";
  interests: string[];
}

export interface TravelDetails {
  arrival_hub: string;
  duration_days: number;
  travel_month: string;
}

export interface ItineraryRequest {
  user_profile: UserProfile;
  travel_details: TravelDetails;
}

export interface Activity {
  place_id: string;
  name: string;
  duration_mins: number;
  distance_from_hub_km: number;
  tip?: string;
}

export interface Meal {
  meal: "breakfast" | "lunch" | "dinner";
  name: string;
  id: string;
  distance_from_hub_km: number;
}

export interface Stay {
  name: string;
  id: string;
  type: string;
  budget: string;
}

export interface DayPlan {
  day: number;
  hub: string;
  narrative: string;
  activities: Activity[];
  meals: Meal[];
  stay: Stay;
}

export interface ItineraryResponse {
  itinerary_id: string;
  days: DayPlan[];
  pro_tips: string[];
}

// ── Hook ────────────────────────────────────────────────────────────────────

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export function useItinerary() {
  const [data, setData] = useState<ItineraryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Track the active request so we can abort it on unmount or new call
  const abortRef = useRef<AbortController | null>(null);

  // Cancel any in-flight request when the component unmounts
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const generate = useCallback(async (req: ItineraryRequest): Promise<ItineraryResponse | null> => {
    // Cancel previous request if still pending
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);
    setData(null);

    try {
      const res = await fetch(`${API_BASE}/api/itinerary/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
        signal: controller.signal,
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail ?? `HTTP ${res.status}`);
      }

      const json: ItineraryResponse = await res.json();
      setData(json);
      return json;
    } catch (err) {
      if ((err as Error).name === "AbortError") return null;
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  /** Fetch all available hubs for the arrival hub selector. */
  const fetchHubs = useCallback(async (): Promise<string[]> => {
    try {
      const res = await fetch(`${API_BASE}/api/itinerary/hubs`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      return json.hubs ?? [];
    } catch {
      return [];
    }
  }, []);

  /** Fetch all available interest tags for the multi-select. */
  const fetchInterests = useCallback(async (): Promise<string[]> => {
    try {
      const res = await fetch(`${API_BASE}/api/itinerary/interests`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      return json.interests ?? [];
    } catch {
      return [];
    }
  }, []);

  return { generate, data, loading, error, fetchHubs, fetchInterests };
}
