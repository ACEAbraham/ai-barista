"""User profile and rating storage for AI Barista."""

from pathlib import Path

import pandas as pd


USERS_FILE = Path(__file__).with_name("users.csv")
RATINGS_FILE = Path(__file__).with_name("ratings.csv")

USER_COLUMNS = [
    "user_id",
    "name",
    "favorite_milk",
    "favorite_temperature",
    "caffeine_tolerance",
    "preferred_sweetness",
]
RATING_COLUMNS = ["user_id", "drink_id", "rating", "would_order_again"]


def _load_csv(csv_path: Path, columns: list[str]) -> pd.DataFrame:
    """Load a CSV file, or return an empty DataFrame if it is missing."""
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=columns)


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
    """Create a user profile and save it to users.csv."""
    users = _load_csv(USERS_FILE, USER_COLUMNS)
    user = {
        "user_id": _next_user_id(users),
        "name": name,
        "favorite_milk": favorite_milk,
        "favorite_temperature": favorite_temperature,
        "caffeine_tolerance": caffeine_tolerance,
        "preferred_sweetness": preferred_sweetness,
    }

    updated_users = pd.concat([users, pd.DataFrame([user])], ignore_index=True)
    updated_users.to_csv(USERS_FILE, index=False)
    return user


def load_user(user_id: str) -> dict[str, str] | None:
    """Load a single user profile by ID."""
    users = _load_csv(USERS_FILE, USER_COLUMNS)
    matches = users[users["user_id"].str.lower() == user_id.lower()]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def save_rating(
    user_id: str,
    drink_id: str,
    rating: int,
    would_order_again: bool,
) -> dict[str, object]:
    """Save a drink rating to ratings.csv."""
    ratings = _load_csv(RATINGS_FILE, RATING_COLUMNS)
    new_rating = {
        "user_id": user_id,
        "drink_id": drink_id,
        "rating": rating,
        "would_order_again": would_order_again,
    }

    updated_ratings = pd.concat([ratings, pd.DataFrame([new_rating])], ignore_index=True)
    updated_ratings.to_csv(RATINGS_FILE, index=False)
    return new_rating


def get_user_history(user_id: str) -> pd.DataFrame:
    """Return all saved ratings for one user."""
    ratings = _load_csv(RATINGS_FILE, RATING_COLUMNS)
    return ratings[ratings["user_id"].str.lower() == user_id.lower()]
