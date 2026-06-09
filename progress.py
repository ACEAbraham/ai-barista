"""Barista progress tracking for AI Barista."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import logging

import pandas as pd

from favorites import get_user_favorites
from profile import get_user_history, load_users
from supabase_client import get_supabase


LOGGER = logging.getLogger(__name__)
PROGRESS_FILE = Path(__file__).with_name("progress.csv")
PROGRESS_COLUMNS = ["user_id", "xp", "last_seen_date", "updated_at"]

XP_AWARDS = {
    "create_profile": 10,
    "rate_drink": 5,
    "save_favorite": 5,
    "generate_recommendation": 1,
    "use_ai_recommendation": 2,
    "create_custom_drink": 10,
    "daily_return": 5,
}

LEVELS = [
    (1, "New Customer", 0),
    (2, "Coffee Explorer", 25),
    (3, "Coffee Enthusiast", 50),
    (4, "Regular", 100),
    (5, "Barista Apprentice", 200),
    (6, "Coffee Expert", 350),
    (7, "Master Barista", 500),
]


def _empty_progress() -> pd.DataFrame:
    return pd.DataFrame(columns=PROGRESS_COLUMNS)


def _load_local_progress() -> pd.DataFrame:
    if not PROGRESS_FILE.exists():
        return _empty_progress()
    progress = pd.read_csv(PROGRESS_FILE)
    for column in PROGRESS_COLUMNS:
        if column not in progress.columns:
            progress[column] = ""
    return progress[PROGRESS_COLUMNS]


def _save_local_progress(progress: pd.DataFrame) -> None:
    progress.to_csv(PROGRESS_FILE, index=False)


def _load_supabase_progress(user_id: str) -> dict[str, object] | None:
    client = get_supabase()
    if client is None:
        return None
    try:
        response = (
            client.table("user_progress")
            .select("*")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None
    except Exception as error:
        LOGGER.warning("Supabase progress load unavailable: %s", error)
        return None


def _save_supabase_progress(row: dict[str, object]) -> bool:
    client = get_supabase()
    if client is None:
        return False
    try:
        client.table("user_progress").upsert(row, on_conflict="user_id").execute()
        return True
    except Exception as error:
        LOGGER.warning("Supabase progress save unavailable: %s", error)
        return False


def _load_progress_row(user_id: str) -> dict[str, object]:
    row = _load_supabase_progress(user_id)
    if row:
        return row

    progress = _load_local_progress()
    if progress.empty:
        return {"user_id": user_id, "xp": 0, "last_seen_date": "", "updated_at": ""}
    matches = progress[progress["user_id"].astype(str) == str(user_id)]
    if matches.empty:
        return {"user_id": user_id, "xp": 0, "last_seen_date": "", "updated_at": ""}
    return matches.iloc[0].to_dict()


def _save_progress_row(row: dict[str, object]) -> None:
    if _save_supabase_progress(row):
        return

    progress = _load_local_progress()
    if progress.empty:
        progress = _empty_progress()
    matches = progress["user_id"].astype(str) == str(row["user_id"])
    if matches.any():
        for key, value in row.items():
            progress.loc[matches, key] = value
    else:
        progress = pd.concat([progress, pd.DataFrame([row])], ignore_index=True)
    _save_local_progress(progress[PROGRESS_COLUMNS])


def _session_rows(user_id: str) -> pd.DataFrame:
    client = get_supabase()
    if client is not None:
        try:
            response = client.table("sessions").select("*").eq("user_id", user_id).execute()
            return pd.DataFrame(response.data or [])
        except Exception:
            pass
    path = Path(__file__).with_name("sessions.csv")
    if not path.exists():
        return pd.DataFrame()
    sessions = pd.read_csv(path)
    if sessions.empty or "user_id" not in sessions.columns:
        return pd.DataFrame()
    return sessions[sessions["user_id"].astype(str) == str(user_id)]


def _recommendation_count(user_id: str) -> int:
    count = 0
    client = get_supabase()
    if client is not None:
        for table in ("recommendation_sessions", "ai_recommendations"):
            try:
                response = client.table(table).select("*").eq("user_id", user_id).execute()
                count += len(response.data or [])
            except Exception:
                continue
    return count


def progress_stats(user_id: str) -> dict[str, int]:
    """Return engagement stats used by the sidebar."""
    ratings = get_user_history(user_id)
    drinks_rated = 0 if ratings.empty else len(ratings)

    try:
        favorites = get_user_favorites(user_id)
        favorites_saved = 0 if favorites.empty else len(favorites)
    except Exception:
        favorites_saved = 0

    sessions = _session_rows(user_id)
    custom_created = 0
    if not sessions.empty and "goal" in sessions.columns:
        custom_created = int(
            sessions["goal"].astype(str).str.lower().str.contains("custom").sum()
        )

    return {
        "drinks_rated": int(drinks_rated),
        "favorites_saved": int(favorites_saved),
        "custom_drinks_created": int(custom_created),
        "recommendations_generated": int(_recommendation_count(user_id)),
    }


def historical_xp(user_id: str) -> int:
    """Calculate baseline XP from already stored user data."""
    stats = progress_stats(user_id)
    xp = (
        stats["drinks_rated"] * XP_AWARDS["rate_drink"]
        + stats["favorites_saved"] * XP_AWARDS["save_favorite"]
        + stats["custom_drinks_created"] * XP_AWARDS["create_custom_drink"]
        + stats["recommendations_generated"] * XP_AWARDS["generate_recommendation"]
    )
    users = load_users()
    if not users.empty and (
        users["user_id"].astype(str).str.lower() == str(user_id).lower()
    ).any():
        xp += XP_AWARDS["create_profile"]

    sessions = _session_rows(user_id)
    if not sessions.empty and "timestamp" in sessions.columns:
        dates = pd.to_datetime(sessions["timestamp"], errors="coerce").dt.date.dropna()
        if not dates.empty:
            xp += max(0, len(set(dates)) - 1) * XP_AWARDS["daily_return"]
    return int(xp)


def award_xp(user_id: str | None, action: str) -> int:
    """Award XP for one action and persist the user's best-known total."""
    if not user_id or action not in XP_AWARDS:
        return 0
    row = _load_progress_row(user_id)
    current = max(int(float(row.get("xp") or 0)), historical_xp(user_id))
    updated = current + XP_AWARDS[action]
    saved = {
        "user_id": user_id,
        "xp": int(updated),
        "last_seen_date": str(row.get("last_seen_date", "") or ""),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_progress_row(saved)
    return int(updated)


def award_daily_return_if_needed(user_id: str | None) -> None:
    """Award return XP once when a user comes back on a different day."""
    if not user_id:
        return
    today = date.today().isoformat()
    row = _load_progress_row(user_id)
    last_seen = str(row.get("last_seen_date", "") or "")
    current = max(int(float(row.get("xp") or 0)), historical_xp(user_id))
    updated = current + (XP_AWARDS["daily_return"] if last_seen and last_seen != today else 0)
    _save_progress_row(
        {
            "user_id": user_id,
            "xp": int(updated),
            "last_seen_date": today,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def get_user_progress(user_id: str | None) -> dict[str, object]:
    """Return XP, level, percent-to-next-level, and engagement stats."""
    if not user_id:
        return {
            "xp": 0,
            "level": 1,
            "title": "New Customer",
            "progress_percent": 0,
            **progress_stats(""),
        }

    row = _load_progress_row(user_id)
    xp = max(int(float(row.get("xp") or 0)), historical_xp(user_id))
    current_level = LEVELS[0]
    next_level = None
    for index, level in enumerate(LEVELS):
        if xp >= level[2]:
            current_level = level
            next_level = LEVELS[index + 1] if index + 1 < len(LEVELS) else None

    if next_level is None:
        progress_percent = 100
    else:
        span = max(1, next_level[2] - current_level[2])
        progress_percent = int(max(0, min(99, ((xp - current_level[2]) / span) * 100)))

    return {
        "xp": int(xp),
        "level": current_level[0],
        "title": current_level[1],
        "progress_percent": progress_percent,
        **progress_stats(user_id),
    }
