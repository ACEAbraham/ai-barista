"""Fast category-based drink image lookup for AI Barista."""


DRINK_IMAGES = {
    "latte": "https://images.unsplash.com/photo-1570968915860-54d5c301fa9f?auto=format&fit=crop&w=900&q=80",
    "cappuccino": "https://images.unsplash.com/photo-1534778101976-62847782c213?auto=format&fit=crop&w=900&q=80",
    "espresso": "https://images.unsplash.com/photo-1510707577719-ae7c14805e3a?auto=format&fit=crop&w=900&q=80",
    "americano": "https://images.unsplash.com/photo-1497636577773-f1231844b336?auto=format&fit=crop&w=900&q=80",
    "cold brew": "https://images.unsplash.com/photo-1461023058943-07fcbe16d735?auto=format&fit=crop&w=900&q=80",
    "mocha": "https://images.unsplash.com/photo-1579888944880-d98341245702?auto=format&fit=crop&w=900&q=80",
    "matcha": "https://images.unsplash.com/photo-1515823662972-da6a2e4d3002?auto=format&fit=crop&w=900&q=80",
    "tea": "https://images.unsplash.com/photo-1544787219-7f47ccb76574?auto=format&fit=crop&w=900&q=80",
    "frappuccino": "https://images.unsplash.com/photo-1572490122747-3968b75cc699?auto=format&fit=crop&w=900&q=80",
    "smoothie": "https://images.unsplash.com/photo-1505252585461-04db1eb84625?auto=format&fit=crop&w=900&q=80",
    "refresher": "https://images.unsplash.com/photo-1546173159-315724a31696?auto=format&fit=crop&w=900&q=80",
    "default": "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?auto=format&fit=crop&w=900&q=80",
}


def get_drink_image(drink: dict[str, object]) -> str:
    """Return the best category image URL for a drink."""
    searchable = " ".join(
        str(drink.get(field, "")).lower()
        for field in ("drink_name", "base", "flavor_profile")
    )
    for category, url in DRINK_IMAGES.items():
        if category != "default" and category in searchable:
            return url
    return DRINK_IMAGES["default"]
