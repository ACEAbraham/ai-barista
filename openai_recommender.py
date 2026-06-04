"""Constrained OpenAI ranking for Smart Match candidate drinks."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import time

import pandas as pd

from openai_client import build_user_memory_summary, get_openai
from supabase_client import get_supabase, insert_row


MODEL = "gpt-5.5"
AI_RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "drink_id": {"type": "string"},
                    "drink_name": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "tradeoffs": {"type": "string"},
                    "preference_matches": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "goal_alignment": {"type": "string"},
                    "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "required": [
                    "drink_id",
                    "drink_name",
                    "reasoning",
                    "tradeoffs",
                    "preference_matches",
                    "goal_alignment",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["recommendations"],
    "additionalProperties": False,
}


def _snapshot_hash(
    user_profile: dict[str, object],
    context: dict[str, object],
    candidate_drinks: pd.DataFrame,
) -> tuple[str, dict[str, object], dict[str, object]]:
    """Create a stable request hash and serializable snapshots."""
    profile_snapshot = {
        key: user_profile.get(key, "")
        for key in (
            "user_id",
            "favorite_milk",
            "favorite_temperature",
            "caffeine_tolerance",
            "preferred_sweetness",
        )
    }
    context_snapshot = dict(sorted(context.items()))
    payload = {
        "profile": profile_snapshot,
        "context": context_snapshot,
        "candidate_ids": candidate_drinks["drink_id"].astype(str).head(10).tolist(),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return digest, profile_snapshot, context_snapshot


def _cached_result(user_id: str, request_hash: str) -> dict[str, object] | None:
    """Load a matching AI ranking generated within the last 24 hours."""
    client = get_supabase()
    if client is None:
        return None
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    try:
        response = (
            client.table("ai_recommendation_cache")
            .select("*")
            .eq("user_id", user_id)
            .eq("request_hash", request_hash)
            .gte("created_at", cutoff)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0].get("recommendation_result") if rows else None
    except Exception:
        return None


def _save_cache(
    user_id: str,
    request_hash: str,
    profile_snapshot: dict[str, object],
    context_snapshot: dict[str, object],
    result: dict[str, object],
) -> None:
    """Best-effort save of a reusable AI ranking."""
    try:
        insert_row(
            "ai_recommendation_cache",
            {
                "user_id": user_id,
                "request_hash": request_hash,
                "profile_snapshot": profile_snapshot,
                "context_snapshot": context_snapshot,
                "recommendation_result": result,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception:
        pass


def _candidate_prompt(candidate_drinks: pd.DataFrame) -> str:
    """Serialize only candidate drinks that GPT is allowed to select."""
    columns = [
        "drink_id",
        "drink_name",
        "base",
        "temperature",
        "milk",
        "sweetness_level",
        "caffeine_level",
        "calories",
        "flavor_profile",
        "recommendation_score",
        "recommendation_explanation",
    ]
    available = [column for column in columns if column in candidate_drinks.columns]
    return candidate_drinks.head(10)[available].to_json(orient="records")


def _validate_result(
    result: dict[str, object],
    candidate_drinks: pd.DataFrame,
) -> dict[str, object]:
    """Reject hallucinated, duplicate, or malformed drink selections."""
    candidates = candidate_drinks.head(10).copy()
    valid_names = candidates.set_index(candidates["drink_id"].astype(str))[
        "drink_name"
    ].astype(str).to_dict()
    recommendations = result.get("recommendations", [])
    if not isinstance(recommendations, list) or len(recommendations) != 3:
        raise ValueError("AI response did not contain exactly three recommendations.")

    seen: set[str] = set()
    for recommendation in recommendations:
        drink_id = str(recommendation.get("drink_id", ""))
        if drink_id not in valid_names:
            raise ValueError(f"AI selected an unknown drink ID: {drink_id}")
        if drink_id in seen:
            raise ValueError(f"AI selected duplicate drink ID: {drink_id}")
        recommendation["drink_name"] = valid_names[drink_id]
        seen.add(drink_id)
    return result


def get_ai_recommendations(
    user_profile: dict[str, object],
    candidate_drinks: pd.DataFrame,
    user_history: pd.DataFrame,
    current_context: dict[str, object] | None = None,
) -> tuple[dict[str, object], bool]:
    """Rank Smart Match candidates with GPT-5.5, returning result and cache status."""
    if candidate_drinks.empty:
        raise ValueError("Smart Match did not produce candidate drinks.")
    context = current_context or {}
    user_id = str(user_profile.get("user_id", ""))
    request_hash, profile_snapshot, context_snapshot = _snapshot_hash(
        user_profile,
        context,
        candidate_drinks,
    )
    cached = _cached_result(user_id, request_hash)
    if cached:
        return _validate_result(cached, candidate_drinks), True

    memory = build_user_memory_summary(user_id)
    history = (
        user_history.tail(5).to_json(orient="records")
        if user_history is not None and not user_history.empty
        else "[]"
    )
    prompt = (
        "You are AI Barista, an experimental explanation layer over Smart Match. "
        "Rank only the candidate drinks listed below. Select exactly three unique "
        "candidate drink IDs. Never invent, customize, rename, or select any drink "
        "outside this list. Reference actual user preferences and today's context. "
        "Explain reasoning, tradeoffs, preference matches, and goal alignment. Do not "
        "claim scientific certainty. Return structured JSON only.\n\n"
        f"PROFILE\n{json.dumps(profile_snapshot, ensure_ascii=True)}\n\n"
        f"MEMORY\n{memory}\n\nRECENT RATINGS\n{history}\n\n"
        f"CURRENT CONTEXT\n{json.dumps(context_snapshot, ensure_ascii=True)}\n\n"
        f"SMART MATCH CANDIDATES\n{_candidate_prompt(candidate_drinks)}"
    )

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = get_openai().with_options(timeout=20.0).responses.create(
                model=MODEL,
                input=prompt,
                reasoning={"effort": "low"},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "ai_barista_candidate_ranking",
                        "strict": True,
                        "schema": AI_RANKING_SCHEMA,
                    }
                },
            )
            result = _validate_result(json.loads(response.output_text), candidate_drinks)
            _save_cache(user_id, request_hash, profile_snapshot, context_snapshot, result)
            return result, False
        except Exception as error:
            last_error = error
            if attempt == 0:
                time.sleep(0.4)
    raise RuntimeError("AI Barista could not rank the Smart Match candidates.") from last_error
