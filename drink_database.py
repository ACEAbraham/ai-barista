"""Utilities for loading and inspecting the AI Barista drink database."""

from pathlib import Path

import pandas as pd


DATA_FILE = Path(__file__).with_name("drinks.csv")
CUSTOM_DATA_FILE = Path(__file__).with_name("custom_drinks.csv")


def load_drinks(csv_path: str | Path = DATA_FILE) -> pd.DataFrame:
    """Load base and custom drinks into one pandas DataFrame."""
    drinks = pd.read_csv(csv_path)
    if CUSTOM_DATA_FILE.exists():
        custom_drinks = pd.read_csv(CUSTOM_DATA_FILE)
        if not custom_drinks.empty:
            drinks = pd.concat([drinks, custom_drinks], ignore_index=True)
    return drinks


def list_options(drinks: pd.DataFrame, column: str) -> list[str]:
    """Return sorted unique values for a database column."""
    values = drinks[column].dropna().astype(str).unique()
    return sorted(values)
