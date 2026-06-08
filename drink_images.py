"""Category-safe real drink photos for AI Barista.

Images are curated category photos, not keyword-search results. Flavor words
like peppermint, coconut, raspberry, and pineapple never select images.
"""

from __future__ import annotations

import math


HERO_IMAGE_URL = (
    "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085"
    "?auto=format&fit=crop&w=1600&q=85"
)

# Fixed real beverage photo URLs. Do not replace these with random search/query
# URLs; each category intentionally maps to one vetted drink photo.
CURATED_DRINK_IMAGES = {
    "iced_latte": "https://images.unsplash.com/photo-1517701604599-bb29b565090c?auto=format&fit=crop&w=900&q=85",
    "hot_latte": "https://images.unsplash.com/photo-1570968915860-54d5c301fa9f?auto=format&fit=crop&w=900&q=85",
    "cappuccino": "https://images.unsplash.com/photo-1534778101976-62847782c213?auto=format&fit=crop&w=900&q=85",
    "espresso": "https://images.unsplash.com/photo-1510707577719-ae7c14805e3a?auto=format&fit=crop&w=900&q=85",
    "americano": "https://images.unsplash.com/photo-1509042239860-f550ce710b93?auto=format&fit=crop&w=900&q=85",
    "cold_brew": "https://images.unsplash.com/photo-1461023058943-07fcbe16d735?auto=format&fit=crop&w=900&q=85",
    "mocha": "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?auto=format&fit=crop&w=900&q=85",
    "matcha": "https://images.unsplash.com/photo-1515823662972-da6a2e4d3002?auto=format&fit=crop&w=900&q=85",
    "tea": "https://images.unsplash.com/photo-1544787219-7f47ccb76574?auto=format&fit=crop&w=900&q=85",
    "chai": "https://images.unsplash.com/photo-1571934811356-5cc061b6821f?auto=format&fit=crop&w=900&q=85",
    "frappuccino": "https://images.unsplash.com/photo-1572490122747-3968b75cc699?auto=format&fit=crop&w=900&q=85",
    "refresher": "https://images.unsplash.com/photo-1546173159-315724a31696?auto=format&fit=crop&w=900&q=85",
    "generic_coffee": "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?auto=format&fit=crop&w=900&q=85",
}

SEARCH_FIELDS = (
    "drink_name",
    "base",
    "temperature",
    "ingredients",
    "flavor_profile",
)


def _clean_value(value: object) -> str:
    """Convert possibly missing drink data into searchable lowercase text."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_clean_value(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_clean_value(item) for item in value.values())
    return str(value).lower()


def _identity_text(drink: dict[str, object]) -> str:
    """Search drink name/base first so flavor words cannot override type."""
    return " ".join(_clean_value(drink.get(field, "")) for field in ("drink_name", "base"))


def _fallback_text(drink: dict[str, object]) -> str:
    """Search broader fields only when the drink identity has no category."""
    return " ".join(_clean_value(drink.get(field, "")) for field in SEARCH_FIELDS)


def _is_iced(drink: dict[str, object], searchable: str) -> bool:
    """Infer whether a latte should use the iced or hot category image."""
    temperature = _clean_value(drink.get("temperature", ""))
    return any(word in f"{temperature} {searchable}" for word in ("iced", "cold", "blended"))


def _category_from_text(text: str, drink: dict[str, object], searchable: str) -> str:
    """Resolve the category with strict beverage priority."""
    if "mocha" in text:
        return "mocha"
    if "matcha" in text:
        return "matcha"
    if "chai" in text:
        return "chai"
    if "cold brew" in text:
        return "cold_brew"
    if "cappuccino" in text:
        return "cappuccino"
    if "americano" in text:
        return "americano"
    if "shaken espresso" in text or "espresso" in text:
        return "espresso"
    if "latte" in text:
        return "iced_latte" if _is_iced(drink, searchable) else "hot_latte"
    if any(word in text for word in ("frappuccino", "blended", "frappe", "frap")):
        return "frappuccino"
    if "refresher" in text:
        return "refresher"
    if "tea" in text:
        return "tea"
    return "generic_coffee"


def _drink_image_category(drink: dict[str, object]) -> str:
    """Resolve a safe category; never use syrup/flavor words as categories."""
    searchable = _fallback_text(drink)
    category = _category_from_text(_identity_text(drink), drink, searchable)
    if category != "generic_coffee":
        return category
    return _category_from_text(searchable, drink, searchable)


def get_drink_image(drink: dict[str, object]) -> str:
    """Return the curated real beverage photo URL for a drink."""
    return CURATED_DRINK_IMAGES[_drink_image_category(drink)]


def get_drink_image_alt(drink: dict[str, object]) -> str:
    """Return accessible alt text for the resolved drink image."""
    name = _clean_value(drink.get("drink_name", "")).strip()
    if name:
        return f"{name.title()} beverage"
    base = _clean_value(drink.get("base", "")).strip()
    if base:
        return f"{base.title()} beverage"
    return "Coffee beverage"
