"""Supabase connection helpers for AI Barista."""

from os import getenv

import pandas as pd
import streamlit as st
from supabase import Client, create_client


def _secret_value(name: str) -> str | None:
    """Read a Supabase secret from Streamlit or local environment variables."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return getenv(name)


def supabase_is_configured() -> bool:
    """Return True when Supabase credentials are available."""
    return bool(_secret_value("SUPABASE_URL") and _secret_value("SUPABASE_KEY"))


def get_supabase() -> Client | None:
    """Create a Supabase client, or return None when credentials are missing."""
    url = _secret_value("SUPABASE_URL")
    key = _secret_value("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def table_to_dataframe(table_name: str, columns: list[str]) -> pd.DataFrame:
    """Load a Supabase table into a DataFrame."""
    client = get_supabase()
    if client is None:
        return pd.DataFrame(columns=columns)

    response = client.table(table_name).select("*").execute()
    return pd.DataFrame(response.data or [], columns=columns)


def insert_row(table_name: str, row: dict[str, object]) -> dict[str, object]:
    """Insert one row into a Supabase table and return the saved row."""
    client = get_supabase()
    if client is None:
        raise RuntimeError(
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY "
            "to Streamlit secrets or local environment variables."
        )

    response = client.table(table_name).insert(row).execute()
    if response.data:
        return response.data[0]
    return row


def upsert_row(
    table_name: str,
    row: dict[str, object],
    on_conflict: str,
) -> dict[str, object]:
    """Insert or update one row using the table's unique conflict columns."""
    client = get_supabase()
    if client is None:
        raise RuntimeError(
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY "
            "to Streamlit secrets or local environment variables."
        )

    response = client.table(table_name).upsert(row, on_conflict=on_conflict).execute()
    if response.data:
        return response.data[0]
    return row


def update_rows(
    table_name: str,
    values: dict[str, object],
    match: dict[str, object],
) -> list[dict[str, object]]:
    """Update rows matching the supplied column values."""
    client = get_supabase()
    if client is None:
        raise RuntimeError(
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY "
            "to Streamlit secrets or local environment variables."
        )

    response = client.table(table_name).update(values).match(match).execute()
    return response.data or []
