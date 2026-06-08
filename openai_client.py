"""OpenAI reasoning over AI Barista's Supabase-backed user memory."""

import json
from os import getenv

import pandas as pd
import streamlit as st
from openai import OpenAI

from favorites import get_user_favorites
from ingredient_engine import get_ingredient_preferences, load_ingredients
from profile import get_user_history, load_user
from supabase_client import get_supabase


DEFAULT_MODEL = "gpt-5.4-mini"
RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "drink_name": {"type": "string"},
        "base": {"type": "string"},
        "temperature": {"type": "string"},
        "size": {"type": "string"},
        "milk": {"type": "string"},
        "syrup": {"type": "string"},
        "sweetness_level": {"type": "string"},
        "espresso_shots": {"type": "integer", "minimum": 0},
        "caffeine_level": {"type": "string"},
        "ingredients": {"type": "array", "items": {"type": "string"}},
        "why_recommended": {"type": "string"},
        "memory_used": {"type": "array", "items": {"type": "string"}},
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "drink_name",
        "base",
        "temperature",
        "size",
        "milk",
        "syrup",
        "sweetness_level",
        "espresso_shots",
        "caffeine_level",
        "ingredients",
        "why_recommended",
        "memory_used",
        "confidence_score",
    ],
    "additionalProperties": False,
}


def _secret_value(name: str) -> str | None:
    """Read a secret from Streamlit or local environment variables."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return getenv(name)


def openai_is_configured() -> bool:
    """Return True when an OpenAI API key is available."""
    return bool(_secret_value("OPENAI_API_KEY"))


def get_openai() -> OpenAI:
    """Create an authenticated OpenAI client."""
    api_key = _secret_value("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("AI recommendations are not configured yet.")
    return OpenAI(api_key=api_key)


def _safe_recent_rows(
    table_name: str,
    user_id: str,
    limit: int = 5,
    order_column: str = "created_at",
) -> list[dict[str, object]]:
    """Load recent user rows without making optional memory tables fatal."""
    client = get_supabase()
    if client is None:
        return []
    try:
        response = (
            client.table(table_name)
            .select("*")
            .eq("user_id", user_id)
            .order(order_column, desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception:
        try:
            response = client.table(table_name).select("*").eq("user_id", user_id).limit(limit).execute()
            return response.data or []
        except Exception:
            return []


def _drink_names_by_id() -> dict[str, str]:
    """Return drink names for rating-history summaries."""
    from drink_database import load_drinks

    drinks = load_drinks()
    return drinks.set_index(drinks["drink_id"].astype(str))["drink_name"].astype(str).to_dict()


def build_user_memory_summary(user_id: str) -> str:
    """Build a concise Supabase memory summary for OpenAI reasoning."""
    memory = build_user_memory(user_id)

    def line(label: str, values: list[object]) -> str:
        return f"{label}: {', '.join(map(str, values)) if values else 'none yet'}"

    profile = memory["profile"]
    return "\n".join(
        [
            "Profile:",
            f"- favorite milk: {profile.get('favorite_milk', 'not specified')}",
            f"- caffeine tolerance: {profile.get('caffeine_tolerance', 'not specified')}",
            f"- sweetness preference: {profile.get('preferred_sweetness', 'not specified')}",
            line("Liked ingredients", memory["favorite_ingredients"]),
            line("Disliked ingredients", memory["disliked_ingredients"]),
            line("Favorite drinks", memory["favorite_drinks"]),
            line("Recent ratings", memory["recent_ratings"]),
            line("Previous recommendations", memory["previous_recommendations"]),
        ]
    )


def build_user_memory(user_id: str) -> dict[str, object]:
    """Load structured user memory from Supabase for recommendation context."""
    user = load_user(user_id) or {}
    preferences = get_ingredient_preferences(user_id)
    ratings = get_user_history(user_id)
    try:
        favorites = get_user_favorites(user_id)
    except Exception:
        favorites = pd.DataFrame()
    recent_sessions = _safe_recent_rows("sessions", user_id, 5, order_column="timestamp")
    previous_recommendations = _safe_recent_rows("ai_recommendations", user_id, 5)
    drink_names = _drink_names_by_id()
    ingredient_names = load_ingredients().set_index("ingredient_id")["ingredient_name"].to_dict()

    liked: list[str] = []
    disliked: list[str] = []
    if not preferences.empty:
        scored = preferences.copy()
        scored["preference_score"] = pd.to_numeric(scored["preference_score"], errors="coerce").fillna(0)
        scored["ingredient_name"] = scored.apply(
            lambda row: row.get("ingredient_name")
            or ingredient_names.get(row.get("ingredient_id"), row.get("ingredient_id")),
            axis=1,
        )
        liked = [
            f"{row.ingredient_name} {row.preference_score:+g}"
            for row in scored[scored["preference_score"] > 0]
            .sort_values("preference_score", ascending=False)
            .head(5)
            .itertuples()
        ]
        disliked = [
            f"{row.ingredient_name} {row.preference_score:+g}"
            for row in scored[scored["preference_score"] < 0]
            .sort_values("preference_score")
            .head(5)
            .itertuples()
        ]

    recent_rating_items: list[str] = []
    if not ratings.empty:
        recent = ratings.tail(5).copy()
        recent["rating"] = pd.to_numeric(recent["rating"], errors="coerce").fillna(0)
        for row in recent.itertuples():
            recent_rating_items.append(
                f"{drink_names.get(str(row.drink_id), row.drink_id)}: {int(row.rating)}/5"
            )

    favorite_names = (
        favorites["drink_name"].dropna().astype(str).head(5).tolist()
        if not favorites.empty and "drink_name" in favorites
        else []
    )
    session_notes = [
        ", ".join(
            f"{key}={row.get(key)}"
            for key in ("goal", "weather", "stress_level", "drink_id")
            if row.get(key) not in {None, ""}
        )
        for row in recent_sessions
    ]
    recommendation_items = []
    for row in previous_recommendations:
        recommendation = row.get("recommendation_json") or {}
        if isinstance(recommendation, dict):
            recommendation_items.append(str(recommendation.get("drink_name", "")))
    recommendation_items = [item for item in recommendation_items if item][:5]

    return {
        "profile": user,
        "favorite_ingredients": liked,
        "disliked_ingredients": disliked,
        "favorite_drinks": favorite_names,
        "recent_ratings": recent_rating_items,
        "recent_sessions": session_notes,
        "previous_recommendations": recommendation_items,
    }


def _available_options(available_drinks: pd.DataFrame) -> str:
    """Create a bounded list of current drinks and ingredients for the prompt."""
    drink_lines = []
    for drink in available_drinks.head(15).itertuples():
        drink_lines.append(
            f"{drink.drink_name} | base={drink.base} | temp={drink.temperature} | "
            f"milk={drink.milk} | caffeine={drink.caffeine_level} | "
            f"sweetness={drink.sweetness_level}"
        )
    ingredient_names = load_ingredients()["ingredient_name"].dropna().astype(str).head(5).tolist()
    return "Available drinks:\n- " + "\n- ".join(drink_lines) + (
        "\nAvailable ingredients:\n- " + ", ".join(ingredient_names)
    )


def generate_ai_recommendation(
    user_id: str,
    current_context: dict[str, object],
    available_drinks: pd.DataFrame,
) -> dict[str, object]:
    """Use OpenAI to reason over Supabase memory and current context."""
    memory_summary = build_user_memory_summary(user_id)
    prompt = (
        "You are AI Barista. Recommend exactly one drink using the user's durable "
        "Supabase memory and today's context. Prefer a drink in the available dataset. "
        "If you create a custom drink, use only listed ingredients. Avoid disliked "
        "ingredients and respect dietary restrictions. Be clear and friendly, and do "
        "not claim scientific certainty. Return structured JSON only.\n\n"
        f"USER MEMORY\n{memory_summary}\n\n"
        f"CURRENT CONTEXT\n{json.dumps(current_context, ensure_ascii=True)}\n\n"
        f"{_available_options(available_drinks)}"
    )
    response = get_openai().responses.create(
        model=_secret_value("OPENAI_MODEL") or DEFAULT_MODEL,
        input=prompt,
        reasoning={"effort": "low"},
        text={
            "format": {
                "type": "json_schema",
                "name": "ai_barista_recommendation",
                "strict": True,
                "schema": RECOMMENDATION_SCHEMA,
            }
        },
    )
    return json.loads(response.output_text)


def generate_drink_recommendation(context: dict[str, object]) -> dict[str, object]:
    """Backward-compatible wrapper for older callers."""
    from drink_database import load_drinks

    return generate_ai_recommendation(
        str(context.get("user_id", "guest")),
        context,
        load_drinks(),
    )
