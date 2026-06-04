"""Ingredient-based drink building for AI Barista."""

import logging
from pathlib import Path
import re

import pandas as pd

from supabase_client import insert_row, table_to_dataframe, upsert_row


INGREDIENTS_FILE = Path(__file__).with_name("ingredients.csv")
RECIPES_FILE = Path(__file__).with_name("drink_recipes.csv")
RECIPE_COLUMNS = ["drink_id", "ingredient_id", "quantity", "unit"]
PREFERENCE_COLUMNS = ["user_id", "ingredient_id", "preference_score"]
CUSTOM_DRINK_COLUMNS = [
    "drink_id",
    "drink_name",
    "base",
    "temperature",
    "size",
    "milk",
    "syrup",
    "sweetness_level",
    "espresso_shots",
    "caffeine_level",
    "calories",
    "price",
    "dietary_tags",
    "flavor_profile",
    "toppings",
    "ice_level",
    "flavor_score",
]
RATING_COLUMNS = ["user_id", "drink_id", "rating", "would_order_again"]
INGREDIENT_COLUMNS = [
    "ingredient_id",
    "ingredient_name",
    "category",
    "calories",
    "caffeine",
    "price",
    "default_unit",
]
SUPPORTED_CATEGORIES = [
    "base",
    "milk",
    "syrup",
    "topping",
    "sweetener",
    "flavor",
    "powder",
    "ice",
    "temperature",
    "size",
    "add-in",
]
SUPPORTED_UNITS = ["oz", "ml", "pump", "tsp", "tbsp", "shot", "scoop", "serving", "cup", "g"]

FLAVOR_WEIGHTS = {
    "base": 3,
    "milk": 2,
    "syrup": 2,
    "topping": 1,
    "temperature": 1,
}
LOGGER = logging.getLogger(__name__)


def load_ingredients() -> pd.DataFrame:
    """Load all available ingredients."""
    ingredients = pd.read_csv(INGREDIENTS_FILE)
    if "default_unit" not in ingredients.columns:
        ingredients["default_unit"] = "serving"
    return ingredients


def load_recipes() -> pd.DataFrame:
    """Load saved drink recipes."""
    if RECIPES_FILE.exists():
        recipes = pd.read_csv(RECIPES_FILE)
        if "unit" not in recipes.columns:
            recipes["unit"] = "serving"
        return recipes
    return pd.DataFrame(columns=RECIPE_COLUMNS)


def load_custom_drinks() -> pd.DataFrame:
    """Load custom drinks from Supabase."""
    return table_to_dataframe("custom_drinks", CUSTOM_DRINK_COLUMNS)


def normalize_name(name: str) -> str:
    """Normalize a drink or ingredient name for duplicate checks."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def recipe_signature(recipe_items: list[dict[str, object]]) -> str:
    """Build a stable ingredient + quantity + unit signature."""
    normalized_items = []
    for item in recipe_items:
        ingredient_id = str(item["ingredient_id"]).strip()
        quantity = float(item["quantity"])
        unit = str(item.get("unit", "serving")).strip().lower()
        normalized_items.append(f"{ingredient_id}:{quantity:g}:{unit}")
    return "|".join(sorted(normalized_items))


def next_ingredient_id() -> str:
    """Create the next ingredient ID in the format ING-001."""
    ingredients = load_ingredients()
    if ingredients.empty:
        return "ING-001"

    numbers = ingredients["ingredient_id"].str.replace("ING-", "", regex=False).astype(int)
    return f"ING-{numbers.max() + 1:03d}"


def add_ingredient(
    ingredient_name: str,
    category: str,
    calories: float,
    caffeine: float,
    price: float,
    default_unit: str,
) -> dict[str, object]:
    """Add a new ingredient locally and collect it in Supabase when available."""
    ingredients = load_ingredients()
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(f"Unsupported category: {category}")
    if default_unit not in SUPPORTED_UNITS:
        raise ValueError(f"Unsupported unit: {default_unit}")

    normalized_new_name = normalize_name(ingredient_name)
    existing_names = ingredients["ingredient_name"].apply(normalize_name)
    if normalized_new_name in set(existing_names):
        raise ValueError("This ingredient already exists.")

    ingredient = {
        "ingredient_id": next_ingredient_id(),
        "ingredient_name": ingredient_name.strip(),
        "category": category,
        "calories": float(calories),
        "caffeine": float(caffeine),
        "price": float(price),
        "default_unit": default_unit,
    }
    updated_ingredients = pd.concat(
        [ingredients, pd.DataFrame([ingredient])],
        ignore_index=True,
    )
    updated_ingredients.to_csv(INGREDIENTS_FILE, index=False)
    try:
        insert_row("ingredients", ingredient)
    except Exception as error:
        LOGGER.warning("Ingredient saved locally, but Supabase collection failed: %s", error)
    return ingredient


def custom_drink_exists(drink_name: str, recipe_items: list[dict[str, object]]) -> tuple[bool, str]:
    """Check whether a custom drink duplicates a name or recipe signature."""
    custom_drinks = load_custom_drinks()
    normalized_new_name = normalize_name(drink_name)
    if not custom_drinks.empty and "drink_name" in custom_drinks.columns:
        normalized_names = custom_drinks["drink_name"].astype(str).apply(normalize_name)
        if normalized_new_name in set(normalized_names):
            return True, "A custom drink with that name already exists."

    recipes = load_recipes()
    custom_ids = set(custom_drinks["drink_id"]) if not custom_drinks.empty else set()
    if recipes.empty or not custom_ids:
        return False, ""

    new_signature = recipe_signature(recipe_items)
    for drink_id, recipe_rows in recipes[recipes["drink_id"].isin(custom_ids)].groupby("drink_id"):
        existing_items = recipe_rows[["ingredient_id", "quantity", "unit"]].to_dict("records")
        if recipe_signature(existing_items) == new_signature:
            return True, f"That recipe already exists as {drink_id}."

    return False, ""


def load_ingredient_preferences() -> pd.DataFrame:
    """Load all ingredient preference scores from Supabase."""
    return table_to_dataframe("ingredient_preferences", PREFERENCE_COLUMNS)


def get_ingredient_preferences(user_id: str) -> pd.DataFrame:
    """Load the live ingredient preference scores for one user."""
    preferences = load_ingredient_preferences()
    if preferences.empty:
        return preferences
    return preferences[
        preferences["user_id"].astype(str).str.lower() == str(user_id).lower()
    ].copy()


def _rating_adjustment(rating: int) -> int:
    """Convert a drink rating into an ingredient preference adjustment."""
    if rating >= 5:
        return 3
    if rating == 4:
        return 2
    if rating == 3:
        return 0
    if rating == 2:
        return -2
    return -3


def update_ingredient_preferences(user_id: str, drink_id: str, rating: int) -> pd.DataFrame:
    """Best-effort update of a user's Supabase ingredient scores from a rating."""
    adjustment = _rating_adjustment(rating)
    if adjustment == 0:
        return get_ingredient_preferences(user_id)

    recipes = load_recipes()
    drink_recipe = recipes[recipes["drink_id"].astype(str).str.lower() == drink_id.lower()]
    if drink_recipe.empty:
        return get_ingredient_preferences(user_id)

    preferences = get_ingredient_preferences(user_id)
    current_scores = (
        preferences.set_index("ingredient_id")["preference_score"].astype(float).to_dict()
        if not preferences.empty
        else {}
    )
    ingredients = load_ingredients()
    ingredient_names = (
        ingredients.set_index("ingredient_id")["ingredient_name"].astype(str).to_dict()
    )
    ingredient_quantities = drink_recipe.groupby("ingredient_id")["quantity"].sum()

    for ingredient_id, quantity in ingredient_quantities.items():
        score = current_scores.get(ingredient_id, 0.0) + adjustment * float(quantity)
        try:
            upsert_row(
                "ingredient_preferences",
                {
                    "user_id": user_id,
                    "ingredient_id": ingredient_id,
                    "ingredient_name": ingredient_names.get(ingredient_id, str(ingredient_id)),
                    "preference_score": score,
                },
                on_conflict="user_id,ingredient_id",
            )
        except Exception as error:
            LOGGER.warning(
                "Ingredient preference learning failed for user %s, ingredient %s: %s",
                user_id,
                ingredient_id,
                error,
            )

    try:
        return get_ingredient_preferences(user_id)
    except Exception as error:
        LOGGER.warning(
            "Could not reload ingredient preferences for user %s: %s",
            user_id,
            error,
        )
        return preferences


def get_taste_profile(user_id: str) -> dict[str, pd.DataFrame]:
    """Return favorite, least favorite, and most common ingredients for a user."""
    recipes = load_recipes()
    ingredients = load_ingredients()
    ratings = table_to_dataframe("ratings", RATING_COLUMNS)
    user_preferences = get_ingredient_preferences(user_id)

    if user_preferences.empty:
        scored = pd.DataFrame(columns=["ingredient_id", "ingredient_name", "preference_score"])
    else:
        user_preferences["preference_score"] = user_preferences["preference_score"].astype(float)
        scored = user_preferences.merge(ingredients, on="ingredient_id", how="left")

    if ratings.empty:
        common = pd.DataFrame(columns=["ingredient_id", "ingredient_name", "times_seen"])
    else:
        user_ratings = ratings[
            ratings["user_id"].astype(str).str.lower() == user_id.lower()
        ]
        common = recipes[recipes["drink_id"].isin(user_ratings["drink_id"])]
        common = (
            common.groupby("ingredient_id")["quantity"]
            .sum()
            .reset_index(name="times_seen")
            .merge(ingredients, on="ingredient_id", how="left")
            .sort_values("times_seen", ascending=False)
        )

    return {
        "favorite": scored[scored["preference_score"] > 0].sort_values(
            "preference_score",
            ascending=False,
        ),
        "least_favorite": scored[scored["preference_score"] < 0].sort_values(
            "preference_score",
            ascending=True,
        ),
        "most_common": common,
    }


def _next_custom_drink_id(drinks: pd.DataFrame) -> str:
    """Create the next custom drink ID in the format CUS-0001."""
    recipe_ids = load_recipes()["drink_id"] if RECIPES_FILE.exists() else pd.Series(dtype=str)
    drink_ids = drinks["drink_id"] if "drink_id" in drinks.columns else pd.Series(dtype=str)
    all_ids = pd.concat([drink_ids, recipe_ids], ignore_index=True).astype(str)
    custom_ids = all_ids[all_ids.str.startswith("CUS-")]
    if custom_ids.empty:
        return "CUS-0001"

    numbers = custom_ids.str.replace("CUS-", "", regex=False).astype(int)
    return f"CUS-{numbers.max() + 1:04d}"


def show_ingredients_by_category(ingredients: pd.DataFrame) -> None:
    """Print ingredients grouped by category."""
    print("\nIngredient options")
    for category, rows in ingredients.groupby("category"):
        print(f"\n{category.title()}")
        for _, ingredient in rows.iterrows():
            print(
                f"  {ingredient['ingredient_id']}: {ingredient['ingredient_name']} "
                f"({ingredient['calories']} cal, {ingredient['caffeine']} mg caffeine, "
                f"${ingredient['price']:.2f})"
            )


def parse_recipe_input(recipe_text: str) -> list[dict[str, object]]:
    """Parse recipe input like ING-001:1:shot, ING-010:2:oz."""
    recipe_items = []
    for item in recipe_text.split(","):
        if not item.strip():
            continue

        parts = item.strip().split(":")
        ingredient_id = parts[0].strip()
        quantity = 1.0
        if len(parts) > 1:
            quantity = float(parts[1].strip())
        unit = parts[2].strip() if len(parts) > 2 else "serving"

        recipe_items.append(
            {
                "ingredient_id": ingredient_id,
                "quantity": quantity,
                "unit": unit,
            }
        )
    return recipe_items


def calculate_nutrition(recipe_items: list[dict[str, object]], ingredients: pd.DataFrame) -> dict[str, float]:
    """Calculate calories and caffeine from a recipe."""
    recipe = pd.DataFrame(recipe_items)
    merged = recipe.merge(ingredients, on="ingredient_id", how="left")
    return {
        "calories": float(round((merged["calories"] * merged["quantity"]).sum(), 1)),
        "caffeine": float(round((merged["caffeine"] * merged["quantity"]).sum(), 1)),
    }


def calculate_cost(recipe_items: list[dict[str, object]], ingredients: pd.DataFrame) -> float:
    """Calculate total ingredient cost from a recipe."""
    recipe = pd.DataFrame(recipe_items)
    merged = recipe.merge(ingredients, on="ingredient_id", how="left")
    return round((merged["price"] * merged["quantity"]).sum(), 2)


def calculate_flavor_score(recipe_items: list[dict[str, object]], ingredients: pd.DataFrame) -> int:
    """Score recipe flavor balance with simple category rules."""
    recipe = pd.DataFrame(recipe_items)
    merged = recipe.merge(ingredients, on="ingredient_id", how="left")
    categories = set(merged["category"].dropna().astype(str).str.lower())
    score = sum(points for category, points in FLAVOR_WEIGHTS.items() if category in categories)

    syrup_count = (merged["category"].str.lower() == "syrup").sum()
    topping_count = (merged["category"].str.lower() == "topping").sum()
    if syrup_count > 2:
        score -= 2
    if topping_count > 2:
        score -= 1
    if "base" in categories and ("milk" in categories or "water" in categories):
        score += 2

    return max(0, min(10, int(score)))


def _caffeine_level(caffeine: float) -> str:
    """Convert caffeine milligrams into a friendly caffeine level."""
    if caffeine >= 180:
        return "high"
    if caffeine >= 50:
        return "medium"
    if caffeine > 0:
        return "low"
    return "none"


def _clean_label(value: str, suffixes: tuple[str, ...]) -> str:
    """Remove ingredient suffix words used for readable drink fields."""
    cleaned = value
    for suffix in suffixes:
        if cleaned.lower().endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    return cleaned.strip()


def _first_category_value(merged: pd.DataFrame, category: str, fallback: str) -> str:
    """Return the first ingredient name in a category."""
    matches = merged[merged["category"].str.lower() == category]
    if matches.empty:
        return fallback
    return str(matches.iloc[0]["ingredient_name"])


def build_custom_drink_from_ingredients(
    drinks: pd.DataFrame,
    drink_name: str,
    recipe_items: list[dict[str, object]],
) -> dict[str, object]:
    """Build a custom drink from ingredient recipe rows."""
    if not recipe_items:
        raise ValueError("Recipe must include at least one ingredient.")

    ingredients = load_ingredients()
    recipe = pd.DataFrame(recipe_items)
    merged = recipe.merge(ingredients, on="ingredient_id", how="left")
    if merged["ingredient_name"].isna().any():
        missing = merged[merged["ingredient_name"].isna()]["ingredient_id"].tolist()
        raise ValueError(f"Unknown ingredient ID(s): {', '.join(missing)}")

    nutrition = calculate_nutrition(recipe_items, ingredients)
    price = calculate_cost(recipe_items, ingredients)
    flavor_score = calculate_flavor_score(recipe_items, ingredients)

    base = _first_category_value(merged, "base", "Custom Base")
    milk = _clean_label(_first_category_value(merged, "milk", "none"), (" milk",))
    syrup = _clean_label(
        _first_category_value(merged, "syrup", "none"),
        (" syrup", " sauce"),
    )
    temperature = _clean_label(
        _first_category_value(merged, "temperature", "custom"),
        (" preparation",),
    ).lower()
    size = _clean_label(_first_category_value(merged, "size", "custom"), (" cup",)).lower()
    topping_names = merged[merged["category"].str.lower() == "topping"]["ingredient_name"].tolist()
    toppings = ", ".join(topping_names) if topping_names else "none"
    shots = recipe[recipe["ingredient_id"] == "ING-001"]["quantity"].sum()

    dietary_tags = []
    if milk.lower() in {"oat", "almond", "soy", "coconut", "none"}:
        dietary_tags.extend(["dairy-free", "vegan"])
    else:
        dietary_tags.append("vegetarian")
    if nutrition["calories"] <= 120:
        dietary_tags.append("low-calorie")
    if syrup == "none":
        dietary_tags.append("no-added-sugar")

    return {
        "drink_id": _next_custom_drink_id(drinks),
        "drink_name": drink_name or f"Custom {base}",
        "base": base,
        "temperature": temperature,
        "size": size,
        "milk": milk,
        "syrup": syrup,
        "sweetness_level": "unsweetened" if syrup == "none" else "classic",
        "espresso_shots": int(shots),
        "caffeine_level": _caffeine_level(nutrition["caffeine"]),
        "calories": nutrition["calories"],
        "price": float(price),
        "dietary_tags": ",".join(dietary_tags),
        "flavor_profile": ", ".join(merged["ingredient_name"].astype(str).tolist()),
        "toppings": toppings,
        "ice_level": _first_category_value(merged, "ice", "none"),
        "flavor_score": flavor_score,
    }


def save_custom_drink_recipe(
    custom_drink: dict[str, object],
    recipe_items: list[dict[str, object]],
) -> None:
    """Save a custom drink to Supabase and its ingredient recipe locally."""
    insert_row("custom_drinks", custom_drink)

    recipes = load_recipes()
    recipe_rows = [
        {
            "drink_id": custom_drink["drink_id"],
            "ingredient_id": item["ingredient_id"],
            "quantity": item["quantity"],
            "unit": item.get("unit", "serving"),
        }
        for item in recipe_items
    ]
    updated_recipes = pd.concat([recipes, pd.DataFrame(recipe_rows)], ignore_index=True)
    updated_recipes.to_csv(RECIPES_FILE, index=False)
