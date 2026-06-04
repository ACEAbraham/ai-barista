"""Supabase-backed favorites storage for AI Barista."""

from datetime import datetime, timezone
import logging

import pandas as pd

from supabase_client import get_supabase, insert_row


FAVORITE_COLUMNS = ["id", "created_at", "user_id", "drink_id", "drink_name"]
LOGGER = logging.getLogger(__name__)


def get_user_favorites(user_id: str) -> pd.DataFrame:
    """Load a user's saved favorite drinks."""
    client = get_supabase()
    if client is None:
        raise RuntimeError(
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY "
            "to Streamlit secrets or local environment variables."
        )

    response = (
        client.table("favorites")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    rows = response.data or []
    if not rows:
        return pd.DataFrame(columns=FAVORITE_COLUMNS)
    favorites = pd.DataFrame(rows)
    for column in FAVORITE_COLUMNS:
        if column not in favorites.columns:
            favorites[column] = None
    return favorites[FAVORITE_COLUMNS]


def is_favorite(user_id: str, drink_id: str) -> bool:
    """Return True when a user has already saved a drink."""
    favorites = get_user_favorites(user_id)
    if favorites.empty:
        return False
    return (
        favorites["drink_id"].astype(str).str.lower() == str(drink_id).lower()
    ).any()


def save_favorite(user_id: str, drink_id: str, drink_name: str) -> dict[str, object] | None:
    """Save a favorite unless the user has already saved that drink."""
    if is_favorite(user_id, drink_id):
        return None
    try:
        return insert_row(
            "favorites",
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "drink_id": drink_id,
                "drink_name": drink_name,
            },
        )
    except Exception:
        # A database-level unique constraint closes the race between check and insert.
        if is_favorite(user_id, drink_id):
            return None
        raise


def remove_favorite(user_id: str, drink_id: str) -> bool:
    """Remove a favorite using its user and drink IDs."""
    client = get_supabase()
    if client is None:
        raise RuntimeError(
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY "
            "to Streamlit secrets or local environment variables."
        )

    response = (
        client.table("favorites")
        .delete()
        .eq("user_id", user_id)
        .eq("drink_id", drink_id)
        .execute()
    )
    return bool(response.data)
