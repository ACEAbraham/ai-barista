"""Category-safe drink images for AI Barista.

These images are intentionally category-based, not exact drink photos. The
curated data URLs below avoid external keyword search, food photos, logos,
bottles, restaurants, and flavor-word mismatches.
"""

from __future__ import annotations

import math
from urllib.parse import quote


def _svg_image(label: str, bg: str, cup: str, accent: str, steam: bool = True) -> str:
    """Build a safe beverage-only SVG data URL for one drink category."""
    steam_lines = """
        <path d="M256 118 C236 92 278 78 254 48" />
        <path d="M320 118 C300 92 342 78 318 48" />
        <path d="M384 118 C364 92 406 78 382 48" />
    """ if steam else ""
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 480" role="img" aria-label="{label}">
      <defs>
        <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stop-color="{bg}" />
          <stop offset="1" stop-color="#fff7ea" />
        </linearGradient>
        <linearGradient id="drink" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stop-color="{accent}" />
          <stop offset="1" stop-color="{cup}" />
        </linearGradient>
      </defs>
      <rect width="640" height="480" fill="url(#bg)" />
      <circle cx="492" cy="110" r="78" fill="#ffffff" opacity="0.28" />
      <circle cx="122" cy="374" r="96" fill="#ffffff" opacity="0.20" />
      <g fill="none" stroke="#4A2608" stroke-width="12" stroke-linecap="round" opacity="0.42">
        {steam_lines}
      </g>
      <g transform="translate(176 142)">
        <path d="M56 42 H328 L292 322 H92 Z" fill="#fffdf8" stroke="#4A2608" stroke-width="12" />
        <path d="M78 92 H306 L280 292 H104 Z" fill="url(#drink)" opacity="0.96" />
        <ellipse cx="192" cy="92" rx="114" ry="28" fill="#fffdf8" stroke="#4A2608" stroke-width="10" />
        <ellipse cx="192" cy="92" rx="82" ry="15" fill="{accent}" opacity="0.72" />
        <path d="M328 118 H364 C408 118 408 194 356 198 H318" fill="none" stroke="#4A2608" stroke-width="16" stroke-linecap="round" />
      </g>
      <rect x="104" y="382" width="432" height="54" rx="27" fill="#4A2608" opacity="0.88" />
      <text x="320" y="417" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="800" fill="#FFFFFF">{label}</text>
    </svg>
    """
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


# Hardcoded, curated beverage-only image URLs. Flavor words like peppermint,
# coconut, raspberry, and pineapple are never mapped to separate images.
CURATED_DRINK_IMAGES = {
    "iced_latte": _svg_image("ICED LATTE", "#D0BCA8", "#C79058", "#F6E6CE", steam=False),
    "hot_latte": _svg_image("HOT LATTE", "#D0BCA8", "#B47A44", "#F1D9B7"),
    "cappuccino": _svg_image("CAPPUCCINO", "#D8C8B4", "#9A633C", "#FFF1D8"),
    "espresso": _svg_image("ESPRESSO", "#C7A98F", "#4A2608", "#7A4A27"),
    "americano": _svg_image("AMERICANO", "#CDB9A4", "#5C341C", "#8A5A32"),
    "cold_brew": _svg_image("COLD BREW", "#BFD0D2", "#432410", "#7B4A25", steam=False),
    "mocha": _svg_image("MOCHA", "#C9B09B", "#5B2E16", "#A06A3D"),
    "matcha": _svg_image("MATCHA", "#D8D8B8", "#6E8C48", "#B7D58D"),
    "tea": _svg_image("TEA", "#D8CDB5", "#8B5E2D", "#C99A48"),
    "chai": _svg_image("CHAI", "#D4B99D", "#8A4E2A", "#D19A5A"),
    "frappuccino": _svg_image("FRAPPUCCINO", "#D9C7B1", "#8F623F", "#F2E6D8", steam=False),
    "refresher": _svg_image("REFRESHER", "#D7DCC0", "#D56F4A", "#F5B36C", steam=False),
    "generic_coffee": _svg_image("COFFEE", "#D0BCA8", "#6F4E37", "#A07D61"),
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
    """Search the drink name/base first so flavor words cannot override type."""
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
