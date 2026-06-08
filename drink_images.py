"""Single source of truth for AI Barista drink imagery."""

from __future__ import annotations

import math


DRINK_IMAGES = {
    "cold brew": "https://images.unsplash.com/photo-1461023058943-07fcbe16d735?auto=format&fit=crop&w=900&q=80",
    "frappuccino": "https://images.unsplash.com/photo-1572490122747-3968b75cc699?auto=format&fit=crop&w=900&q=80",
    "refresher": "https://images.unsplash.com/photo-1546173159-315724a31696?auto=format&fit=crop&w=900&q=80",
    "smoothie": "https://images.unsplash.com/photo-1505252585461-04db1eb84625?auto=format&fit=crop&w=900&q=80",
    "matcha": "https://images.unsplash.com/photo-1515823662972-da6a2e4d3002?auto=format&fit=crop&w=900&q=80",
    "mocha": "https://images.unsplash.com/photo-1579888944880-d98341245702?auto=format&fit=crop&w=900&q=80",
    "cappuccino": "https://images.unsplash.com/photo-1534778101976-62847782c213?auto=format&fit=crop&w=900&q=80",
    "americano": "https://images.unsplash.com/photo-1497636577773-f1231844b336?auto=format&fit=crop&w=900&q=80",
    "latte": "https://images.unsplash.com/photo-1570968915860-54d5c301fa9f?auto=format&fit=crop&w=900&q=80",
    "espresso": "https://images.unsplash.com/photo-1510707577719-ae7c14805e3a?auto=format&fit=crop&w=900&q=80",
    "chai": "https://images.unsplash.com/photo-1571934811356-5cc061b6821f?auto=format&fit=crop&w=900&q=80",
    "tea": "https://images.unsplash.com/photo-1544787219-7f47ccb76574?auto=format&fit=crop&w=900&q=80",
    "iced": "https://images.unsplash.com/photo-1517701604599-bb29b565090c?auto=format&fit=crop&w=900&q=80",
    "hot": "https://images.unsplash.com/photo-1509042239860-f550ce710b93?auto=format&fit=crop&w=900&q=80",
    "default": "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?auto=format&fit=crop&w=900&q=80",
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

ALIASES = {
    "blended": "frappuccino",
    "frappe": "frappuccino",
    "frap": "frappuccino",
    "iced coffee": "iced",
    "cold foam": "cold brew",
    "green tea": "matcha",
    "black tea": "tea",
    "herbal": "tea",
    "lemonade": "refresher",
    "acai": "refresher",
    "dragonfruit": "refresher",
}


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


def get_drink_image(drink: dict[str, object]) -> str:
    """Return the best category image URL for a drink."""
    searchable = _search_text(drink)
    for phrase, category in ALIASES.items():
        if phrase in searchable:
            return DRINK_IMAGES[category]
    for category, url in DRINK_IMAGES.items():
        if category != "default" and category in searchable:
            return url
    return DRINK_IMAGES["default"]


def get_drink_image_alt(drink: dict[str, object]) -> str:
    """Return accessible alt text for the resolved drink image."""
    name = _clean_value(drink.get("drink_name", "")).strip()
    if name:
        return f"{name.title()} beverage"
    base = _clean_value(drink.get("base", "")).strip()
    if base:
        return f"{base.title()} beverage"
    return "Coffee beverage"
