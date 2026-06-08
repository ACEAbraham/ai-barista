"""Private OpenAI selector for the single AI Barista recommendation flow."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import time

import pandas as pd

from openai_client import build_user_memory, build_user_memory_summary, get_openai
from supabase_client import get_supabase, insert_row


MODEL = "gpt-5.5"
RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "drink_id": {"type": "string"},
        "drink_name": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "reasoning": {"type": "string"},
        "matched_preferences": {"type": "array", "items": {"type": "string"}},
        "matched_context": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "drink_id",
        "drink_name",
        "confidence",
        "reasoning",
        "matched_preferences",
        "matched_context",
    ],
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
    """Load a matching recommendation generated within the last 24 hours."""
    client = get_supabase()
    if client is None:
        return None
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    try:
        response = (
            client.table("recommendation_cache")
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
    """Best-effort save of a reusable recommendation."""
    try:
        insert_row(
            "recommendation_cache",
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
    """Serialize only the top candidate drinks that may be selected."""
    columns = [
        "drink_id",
        "drink_name",
        "base",
        "temperature",
        "milk",
        "sweetness_level",
        "caffeine_level",
        "calories",
        "dietary_tags",
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
    """Reject hallucinated or malformed selections."""
    candidates = candidate_drinks.head(10).copy()
    valid_names = candidates.set_index(candidates["drink_id"].astype(str))[
        "drink_name"
    ].astype(str).to_dict()
    drink_id = str(result.get("drink_id", ""))
    if drink_id not in valid_names:
        raise ValueError(f"Selected drink ID is not in candidate set: {drink_id}")
    result["drink_name"] = valid_names[drink_id]
    result["confidence"] = int(result.get("confidence", 0) or 0)
    result["matched_preferences"] = list(result.get("matched_preferences", []))
    result["matched_context"] = list(result.get("matched_context", []))
    return result


def select_best_candidate(
    user_profile: dict[str, object],
    candidate_drinks: pd.DataFrame,
    current_context: dict[str, object],
) -> tuple[dict[str, object], bool]:
    """Select the best candidate drink, returning recommendation and cache status."""
    if candidate_drinks.empty:
        raise ValueError("Candidate generator did not return drinks.")

    user_id = str(user_profile.get("user_id", ""))
    request_hash, profile_snapshot, context_snapshot = _snapshot_hash(
        user_profile,
        current_context,
        candidate_drinks,
    )
    cached = _cached_result(user_id, request_hash)
    if cached:
        return _validate_result(cached, candidate_drinks), True

    memory = build_user_memory(user_id)
    memory_summary = build_user_memory_summary(user_id)
    prompt = (
        "Choose the best drink recommendation from the candidate drinks only. "
        "Do not invent drinks. Do not rename drinks. The selected drink_id must be "
        "one of the candidate drink_id values. Reference actual user preferences "
        "and current context. Return JSON only.\n\n"
        f"USER PROFILE\n{json.dumps(memory['profile'], ensure_ascii=True)}\n\n"
        f"USER MEMORY SUMMARY\n{memory_summary}\n\n"
        f"CURRENT CONTEXT\n{json.dumps(context_snapshot, ensure_ascii=True)}\n\n"
        f"CANDIDATE DRINKS\n{_candidate_prompt(candidate_drinks)}"
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
                        "name": "single_drink_recommendation",
                        "strict": True,
                        "schema": RECOMMENDATION_SCHEMA,
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
    raise RuntimeError("Could not select a candidate drink.") from last_error


def fallback_recommendation(candidate_drinks: pd.DataFrame) -> dict[str, object]:
    """Use the highest-ranked candidate when the private selector is unavailable."""
    top = candidate_drinks.iloc[0]
    return {
        "drink_id": str(top["drink_id"]),
        "drink_name": str(top["drink_name"]),
        "confidence": int(min(100, max(0, round(float(top.get("recommendation_score", 0)) + 60)))),
        "reasoning": "Recommendation generated using profile matching.",
        "matched_preferences": [
            str(top.get("recommendation_summary", "Matches your saved profile and drink history."))
        ],
        "matched_context": [
            str(top.get("recommendation_explanation", "Fits today's recommendation context."))
        ],
    }
