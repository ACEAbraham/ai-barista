"""Drink customization and session logging for AI Barista."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from supabase_client import insert_row

COMPONENTS_FILE = Path(__file__).with_name("drink_components.csv")

SESSION_COLUMNS = [
    "user_id",
    "timestamp",
    "sleep_hours",
    "stress_level",
    "goal",
    "weather",
    "drink_id",
    "rating",
]


def load_components() -> pd.DataFrame:
    """Load the available customization components."""
    return pd.read_csv(COMPONENTS_FILE)


def list_component_options(components: pd.DataFrame, column: str) -> list[str]:
    """Return sorted options for one component column."""
    values = components[column].dropna().astype(str).unique()
    return sorted(values)


def _next_custom_drink_id(drinks: pd.DataFrame) -> str:
    """Create the next custom drink ID in the format CUS-0001."""
    custom_ids = drinks[drinks["drink_id"].str.startswith("CUS-")]["drink_id"]
    if custom_ids.empty:
        return "CUS-0001"

    numbers = custom_ids.str.replace("CUS-", "", regex=False).astype(int)
    return f"CUS-{numbers.max() + 1:04d}"


def build_custom_drink(
    drinks: pd.DataFrame,
    base: str,
    milk: str,
    syrup: str,
    toppings: str,
    size: str,
    shots: int,
    temperature: str,
    ice_level: str,
) -> dict[str, object]:
    """Build a custom drink row that matches the main drinks.csv structure."""
    caffeine_level = "none"
    if shots >= 3:
        caffeine_level = "high"
    elif shots >= 1 or base.lower() in {"matcha", "chai", "cold brew"}:
        caffeine_level = "medium"
    elif base.lower() in {"refresher"}:
        caffeine_level = "low"

    sweetness_level = "unsweetened" if syrup.lower() == "none" else "classic"
    calories = 40 + (shots * 5)
    if milk.lower() not in {"none", "nonfat"}:
        calories += 80
    if syrup.lower() != "none":
        calories += 70
    if toppings.lower() != "none":
        calories += 45

    price = 3.50 + (shots * 0.60)
    if milk.lower() not in {"none", "whole", "2%", "nonfat"}:
        price += 0.70
    if syrup.lower() != "none":
        price += 0.60
    if toppings.lower() != "none":
        price += 0.75
    if size.lower() == "grande":
        price += 0.55
    elif size.lower() == "venti":
        price += 1.00

    dietary_tags = []
    if milk.lower() in {"none", "oat", "almond", "soy", "coconut"}:
        dietary_tags.extend(["dairy-free", "vegan"])
    else:
        dietary_tags.append("vegetarian")
    if calories <= 120:
        dietary_tags.append("low-calorie")
    if syrup.lower() == "none":
        dietary_tags.append("no-added-sugar")

    return {
        "drink_id": _next_custom_drink_id(drinks),
        "drink_name": f"Custom {size.title()} {temperature.title()} {base} with {milk} milk",
        "base": base,
        "temperature": temperature,
        "size": size,
        "milk": milk,
        "syrup": syrup,
        "sweetness_level": sweetness_level,
        "espresso_shots": shots,
        "caffeine_level": caffeine_level,
        "calories": calories,
        "price": round(price, 2),
        "dietary_tags": ",".join(dietary_tags),
        "flavor_profile": f"{base}, {milk}, {syrup}, {toppings}",
        "toppings": toppings,
        "ice_level": ice_level,
    }


def save_custom_drink(custom_drink: dict[str, object]) -> None:
    """Save a custom drink to Supabase."""
    insert_row("custom_drinks", custom_drink)


def log_session(
    user_id: str,
    sleep_hours: str,
    stress_level: str,
    goal: str,
    weather: str,
    drink_id: str = "",
    rating: int | str = "",
) -> dict[str, object]:
    """Save a user interaction to Supabase as future training data."""
    session = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "goal": goal,
        "weather": weather,
        "drink_id": drink_id,
        "rating": rating,
    }
    return insert_row("sessions", session)


def log_recommendation_session(
    user_id: str,
    context: dict[str, object],
    drink_id: str,
    score: int | float,
    explanation: str,
    rating: int | str = "",
) -> dict[str, object]:
    """Save a scored recommendation context for future model training."""
    session = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "goal": context.get("goal", ""),
        "sleep_hours": context.get("sleep_hours", ""),
        "stress_level": context.get("stress_level", ""),
        "weather": context.get("weather", ""),
        "temperature_preference": context.get("temperature_preference", ""),
        "caffeine_preference": context.get("caffeine_preference", ""),
        "sweetness_preference": context.get("sweetness_preference", ""),
        "likes": context.get("likes", ""),
        "dislikes": context.get("dislikes", ""),
        "drink_id": drink_id,
        "recommendation_score": float(score),
        "recommendation_explanation": explanation,
        "rating": rating,
    }
    return insert_row("recommendation_sessions", session)
