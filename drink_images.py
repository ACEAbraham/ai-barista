"""Single source of truth for AI Barista drink imagery.

Images are category-based, not exact drink images. This keeps every card
beverage-only and prevents flavor words from pulling unrelated photos.
"""

from __future__ import annotations

import math


CURATED_DRINK_IMAGES = {
    "iced_latte": "https://images.unsplash.com/photo-1517701604599-bb29b565090c?auto=format&fit=crop&w=900&q=80",
    "hot_latte": "https://images.unsplash.com/photo-1570968915860-54d5c301fa9f?auto=format&fit=crop&w=900&q=80",
    "cappuccino": "https://images.unsplash.com/photo-1534778101976-62847782c213?auto=format&fit=crop&w=900&q=80",
    "espresso": "https://images.unsplash.com/photo-1510707577719-ae7c14805e3a?auto=format&fit=crop&w=900&q=80",
    "americano": "https://images.unsplash.com/photo-1497636577773-f1231844b336?auto=format&fit=crop&w=900&q=80",
    "cold_brew": "https://images.unsplash.com/photo-1461023058943-07fcbe16d735?auto=format&fit=crop&w=900&q=80",
    "mocha": "https://images.unsplash.com/photo-1579888944880-d98341245702?auto=format&fit=crop&w=900&q=80",
    "matcha": "https://images.unsplash.com/photo-1515823662972-da6a2e4d3002?auto=format&fit=crop&w=900&q=80",
    "tea": "https://images.unsplash.com/photo-1544787219-7f47ccb76574?auto=format&fit=crop&w=900&q=80",
    "chai": "https://images.unsplash.com/photo-1571934811356-5cc061b6821f?auto=format&fit=crop&w=900&q=80",
    "frappuccino": "https://images.unsplash.com/photo-1572490122747-3968b75cc699?auto=format&fit=crop&w=900&q=80",
    "refresher": "https://images.unsplash.com/photo-1546173159-315724a31696?auto=format&fit=crop&w=900&q=80",
    "generic_coffee": "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?auto=format&fit=crop&w=900&q=80",
}

IMAGE_FIELDS = (
    "drink_name",
    "base",
    "temperature",
    "ingredients",
    "flavor_profile",
    "syrup",
    "milk",
    "dietary_tags",
    "toppings",
)


def _clean_value(value: object) -> str:
    """Convert possibly missing drink data into searchable text."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_clean_value(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_clean_value(item) for item in value.values())
    return str(value).lower()


def _search_text(drink: dict[str, object]) -> str:
    """Build text used to infer the visual category."""
    return " ".join(_clean_value(drink.get(field, "")) for field in IMAGE_FIELDS)


def _primary_text(drink: dict[str, object]) -> str:
    """Use drink identity fields before flavor/ingredient details."""
    return " ".join(_clean_value(drink.get(field, "")) for field in ("drink_name", "base"))


def _is_iced(drink: dict[str, object], searchable: str) -> bool:
    """Infer whether a latte should use the iced or hot category image."""
    temperature = _clean_value(drink.get("temperature", ""))
    return any(word in f"{temperature} {searchable}" for word in ("iced", "cold", "blended"))


def _category_from_text(category_text: str, drink: dict[str, object], searchable: str) -> str:
    """Resolve a category from already-selected beverage category text."""
    # Priority is intentionally main drink category first. Syrups and flavor
    # words never override mocha, refresher, latte, or other beverage types.
    if "mocha" in category_text:
        return "mocha"
    if "matcha" in category_text:
        return "matcha"
    if "chai" in category_text:
        return "chai"
    if "cold brew" in category_text:
        return "cold_brew"
    if "cappuccino" in category_text:
        return "cappuccino"
    if "americano" in category_text:
        return "americano"
    if "shaken espresso" in category_text or "espresso" in category_text:
        return "espresso"
    if "latte" in category_text:
        return "iced_latte" if _is_iced(drink, searchable) else "hot_latte"
    if any(word in category_text for word in ("frappuccino", "blended", "frappe", "frap")):
        return "frappuccino"
    if "refresher" in category_text:
        return "refresher"
    if "tea" in category_text:
        return "tea"
    return "generic_coffee"


def _drink_image_category(drink: dict[str, object]) -> str:
    """Resolve the safe beverage category by fixed priority."""
    searchable = _search_text(drink)
    primary_category = _category_from_text(_primary_text(drink), drink, searchable)
    if primary_category != "generic_coffee":
        return primary_category
    return _category_from_text(searchable, drink, searchable)


def get_drink_image(drink: dict[str, object]) -> str:
    """Return the curated beverage-only image URL for a drink."""
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
