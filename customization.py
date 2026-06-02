"""Drink customization and session logging for AI Barista."""

from datetime import datetime
from pathlib import Path

import pandas as pd


CUSTOM_DRINKS_FILE = Path(__file__).with_name("custom_drinks.csv")
COMPONENTS_FILE = Path(__file__).with_name("drink_components.csv")
SESSIONS_FILE = Path(__file__).with_name("sessions.csv")

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
    """Append a custom drink to custom_drinks.csv."""
    if CUSTOM_DRINKS_FILE.exists():
        custom_drinks = pd.read_csv(CUSTOM_DRINKS_FILE)
    else:
        custom_drinks = pd.DataFrame(columns=list(custom_drink.keys()))

    updated_drinks = pd.concat(
        [custom_drinks, pd.DataFrame([custom_drink])],
        ignore_index=True,
    )
    updated_drinks.to_csv(CUSTOM_DRINKS_FILE, index=False)


def log_session(
    user_id: str,
    sleep_hours: str,
    stress_level: str,
    goal: str,
    weather: str,
    drink_id: str = "",
    rating: int | str = "",
) -> dict[str, object]:
    """Save a user interaction as future training data."""
    if SESSIONS_FILE.exists():
        sessions = pd.read_csv(SESSIONS_FILE)
    else:
        sessions = pd.DataFrame(columns=SESSION_COLUMNS)

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
    updated_sessions = pd.concat([sessions, pd.DataFrame([session])], ignore_index=True)
    updated_sessions.to_csv(SESSIONS_FILE, index=False)
    return session
