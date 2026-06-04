"""Best-effort recommendation analytics."""

from datetime import datetime, timezone

from supabase_client import insert_row


def track_recommendation_event(
    system: str,
    event_type: str,
    user_id: str = "guest",
    drink_id: str = "",
    metadata: dict[str, object] | None = None,
) -> None:
    """Track a recommendation event without interrupting the user experience."""
    try:
        insert_row(
            "recommendation_analytics",
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "system": system,
                "event_type": event_type,
                "user_id": user_id,
                "drink_id": drink_id,
                "metadata": metadata or {},
            },
        )
    except Exception:
        pass
