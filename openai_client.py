"""OpenAI-powered beverage recommendation helpers."""

import json
from os import getenv

import streamlit as st
from openai import OpenAI


DEFAULT_MODEL = "gpt-5.4-mini"
RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "drink_name": {"type": "string"},
        "size": {"type": "string"},
        "temperature": {"type": "string"},
        "milk": {"type": "string"},
        "syrup": {"type": "string"},
        "sweetness_level": {"type": "string"},
        "espresso_shots": {"type": "integer", "minimum": 0},
        "caffeine_level": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
        },
        "ingredients": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "explanation": {"type": "string"},
        "confidence_score": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": [
        "drink_name",
        "size",
        "temperature",
        "milk",
        "syrup",
        "sweetness_level",
        "espresso_shots",
        "caffeine_level",
        "ingredients",
        "explanation",
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
        raise RuntimeError(
            "OpenAI is not configured. Add OPENAI_API_KEY to Streamlit secrets "
            "or local environment variables."
        )
    return OpenAI(api_key=api_key)


def generate_drink_recommendation(context: dict[str, object]) -> dict[str, object]:
    """Generate exactly one structured beverage recommendation."""
    prompt = (
        "You are AI Barista. Recommend exactly one practical coffee-shop beverage "
        "that fits the user's context and preferences. Respect dietary restrictions "
        "and explicit dislikes. Use common ingredients and make the explanation brief, "
        "specific, and friendly. User context:\n"
        f"{json.dumps(context, ensure_ascii=True)}"
    )
    response = get_openai().responses.create(
        model=_secret_value("OPENAI_MODEL") or DEFAULT_MODEL,
        input=prompt,
        reasoning={"effort": "low"},
        text={
            "format": {
                "type": "json_schema",
                "name": "drink_recommendation",
                "strict": True,
                "schema": RECOMMENDATION_SCHEMA,
            }
        },
    )
    return json.loads(response.output_text)
