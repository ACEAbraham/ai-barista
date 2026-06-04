"""User profile and rating storage for AI Barista."""

import logging

import pandas as pd

from supabase_client import insert_row, table_to_dataframe

USER_COLUMNS = [
    "user_id",
    "name",
    "favorite_milk",
    "favorite_temperature",
    "caffeine_tolerance",
    "preferred_sweetness",
]
RATING_COLUMNS = ["user_id", "drink_id", "rating", "would_order_again"]
LOGGER = logging.getLogger(__name__)


def load_users() -> pd.DataFrame:
    """Load all user profiles from Supabase."""
    return table_to_dataframe("users", USER_COLUMNS)


def load_ratings() -> pd.DataFrame:
    """Load all ratings from Supabase."""
    return table_to_dataframe("ratings", RATING_COLUMNS)


def _next_user_id(users: pd.DataFrame) -> str:
    """Create the next user ID in the format USR-0001."""
    if users.empty:
        return "USR-0001"

    numbers = users["user_id"].str.replace("USR-", "", regex=False).astype(int)
    return f"USR-{numbers.max() + 1:04d}"


def create_user(
    name: str,
    favorite_milk: str,
    favorite_temperature: str,
    caffeine_tolerance: str,
    preferred_sweetness: str,
) -> dict[str, str]:
    """Create a user profile and save it to Supabase."""
    users = load_users()
    user = {
        "user_id": _next_user_id(users),
        "name": name,
        "favorite_milk": favorite_milk,
        "favorite_temperature": favorite_temperature,
        "caffeine_tolerance": caffeine_tolerance,
        "preferred_sweetness": preferred_sweetness,
    }

    return insert_row("users", user)


def load_user(user_id: str) -> dict[str, str] | None:
    """Load a single user profile by ID."""
    users = load_users()
    matches = users[users["user_id"].str.lower() == user_id.lower()]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def load_user_by_id_or_name(value: str) -> dict[str, str] | None:
    """Load a profile using either its user ID or exact name."""
    users = load_users()
    if users.empty:
        return None

    query = str(value).strip().lower()
    matches = users[
        (users["user_id"].astype(str).str.lower() == query)
        | (users["name"].astype(str).str.lower() == query)
    ]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def save_rating(
    user_id: str,
    drink_id: str,
    rating: int,
    would_order_again: bool,
) -> dict[str, object]:
    """Save a drink rating and immediately learn its ingredient preferences."""
    new_rating = {
        "user_id": user_id,
        "drink_id": drink_id,
        "rating": rating,
        "would_order_again": would_order_again,
    }

    saved_rating = insert_row("ratings", new_rating)

    # Import here to keep profile storage independent during module loading.
    from ingredient_engine import update_ingredient_preferences

    try:
        update_ingredient_preferences(user_id, drink_id, rating)
    except Exception as error:
        LOGGER.warning(
            "Rating saved, but ingredient preference learning failed for user %s: %s",
            user_id,
            error,
        )
    return saved_rating


def get_user_history(user_id: str) -> pd.DataFrame:
    """Return all saved ratings for one user."""
    ratings = load_ratings()
    if ratings.empty:
        return ratings
    return ratings[ratings["user_id"].str.lower() == user_id.lower()]
