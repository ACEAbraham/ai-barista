"""Filtering and scoring helpers for AI Barista recommendations."""

import pandas as pd


PROFILE_SCORE_WEIGHTS = {
    "milk": 3,
    "temperature": 3,
    "caffeine": 2,
    "sweetness": 2,
}
RATING_MULTIPLIER = 2
ORDER_AGAIN_BONUS = 3
INGREDIENT_PREFERENCE_MULTIPLIER = 1


def _is_true(value: object) -> bool:
    """Convert common CSV truthy values into a boolean."""
    return str(value).strip().lower() in {"true", "yes", "y", "1"}


def _add_reason(
    drinks: pd.DataFrame,
    mask: pd.Series,
    points: int,
    reason: str,
) -> None:
    """Add score points and explanation text to matching rows."""
    drinks.loc[mask, "recommendation_score"] += points
    drinks.loc[mask, "recommendation_explanation"] = drinks.loc[
        mask,
        "recommendation_explanation",
    ].apply(lambda text: f"{text}; {reason}" if text else reason)


def filter_by_caffeine(drinks: pd.DataFrame, caffeine_level: str | None) -> pd.DataFrame:
    """Filter drinks by caffeine level."""
    if not caffeine_level:
        return drinks
    return drinks[drinks["caffeine_level"].str.lower() == caffeine_level.lower()]


def filter_by_temperature(drinks: pd.DataFrame, temperature: str | None) -> pd.DataFrame:
    """Filter drinks by hot, iced, or blended temperature."""
    if not temperature:
        return drinks
    return drinks[drinks["temperature"].str.lower() == temperature.lower()]


def filter_by_milk(drinks: pd.DataFrame, milk: str | None) -> pd.DataFrame:
    """Filter drinks by milk type."""
    if not milk:
        return drinks
    return drinks[drinks["milk"].str.lower() == milk.lower()]


def filter_by_budget(drinks: pd.DataFrame, max_price: float | None) -> pd.DataFrame:
    """Filter drinks at or below a user's budget."""
    if max_price is None:
        return drinks
    return drinks[drinks["price"] <= max_price]


def filter_by_sweetness(drinks: pd.DataFrame, sweetness_level: str | None) -> pd.DataFrame:
    """Filter drinks by sweetness level."""
    if not sweetness_level:
        return drinks
    return drinks[drinks["sweetness_level"].str.lower() == sweetness_level.lower()]


def filter_by_dietary_tag(drinks: pd.DataFrame, dietary_tag: str | None) -> pd.DataFrame:
    """Filter drinks that contain a dietary tag, such as vegan or dairy-free."""
    if not dietary_tag:
        return drinks
    tag = dietary_tag.lower()
    return drinks[drinks["dietary_tags"].str.lower().str.contains(tag, na=False)]


def add_recommendation_scores(
    drinks: pd.DataFrame,
    user: dict[str, str] | None = None,
    user_history: pd.DataFrame | None = None,
    ingredient_preferences: pd.DataFrame | None = None,
    drink_recipes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add a rule-based recommendation score to each drink."""
    scored_drinks = drinks.copy()
    scored_drinks["recommendation_score"] = 0
    scored_drinks["recommendation_explanation"] = ""

    if user:
        favorite_milk = str(user.get("favorite_milk", "")).lower()
        favorite_temperature = str(user.get("favorite_temperature", "")).lower()
        caffeine_tolerance = str(user.get("caffeine_tolerance", "")).lower()
        preferred_sweetness = str(user.get("preferred_sweetness", "")).lower()

        _add_reason(
            scored_drinks,
            scored_drinks["milk"].str.lower() == favorite_milk,
            PROFILE_SCORE_WEIGHTS["milk"],
            f"+{PROFILE_SCORE_WEIGHTS['milk']} favorite milk",
        )
        _add_reason(
            scored_drinks,
            scored_drinks["temperature"].str.lower() == favorite_temperature,
            PROFILE_SCORE_WEIGHTS["temperature"],
            f"+{PROFILE_SCORE_WEIGHTS['temperature']} favorite temperature",
        )
        _add_reason(
            scored_drinks,
            scored_drinks["caffeine_level"].str.lower() == caffeine_tolerance,
            PROFILE_SCORE_WEIGHTS["caffeine"],
            f"+{PROFILE_SCORE_WEIGHTS['caffeine']} caffeine tolerance",
        )
        _add_reason(
            scored_drinks,
            scored_drinks["sweetness_level"].str.lower() == preferred_sweetness,
            PROFILE_SCORE_WEIGHTS["sweetness"],
            f"+{PROFILE_SCORE_WEIGHTS['sweetness']} preferred sweetness",
        )

    if user_history is not None and not user_history.empty:
        history = user_history.copy()
        history["rating_points"] = history["rating"].astype(int) * RATING_MULTIPLIER
        history["order_again_points"] = history["would_order_again"].apply(_is_true).astype(int)
        history["order_again_points"] *= ORDER_AGAIN_BONUS
        history["history_score"] = history["rating_points"] + history["order_again_points"]
        history_scores = history.groupby("drink_id")["history_score"].sum()
        history_reasons = {}
        for drink_id, drink_history in history.groupby("drink_id"):
            rating_points = int(drink_history["rating_points"].sum())
            order_again_points = int(drink_history["order_again_points"].sum())
            reasons = [f"+{rating_points} past rating points"]
            if order_again_points:
                reasons.append(f"+{order_again_points} would order again")
            history_reasons[drink_id] = "; ".join(reasons)

        scored_drinks["recommendation_score"] += (
            scored_drinks["drink_id"].map(history_scores).fillna(0).astype(int)
        )
        for drink_id, reason in history_reasons.items():
            mask = scored_drinks["drink_id"] == drink_id
            scored_drinks.loc[mask, "recommendation_explanation"] = scored_drinks.loc[
                mask,
                "recommendation_explanation",
            ].apply(lambda text: f"{text}; {reason}" if text else reason)

    if (
        user
        and ingredient_preferences is not None
        and drink_recipes is not None
        and not ingredient_preferences.empty
        and not drink_recipes.empty
    ):
        user_preferences = ingredient_preferences[
            ingredient_preferences["user_id"].astype(str).str.lower()
            == str(user["user_id"]).lower()
        ].copy()
        if not user_preferences.empty:
            user_preferences["preference_score"] = user_preferences[
                "preference_score"
            ].astype(float)
            recipe_scores = drink_recipes.merge(
                user_preferences[["ingredient_id", "preference_score"]],
                on="ingredient_id",
                how="inner",
            )
            recipe_scores["ingredient_score"] = (
                recipe_scores["preference_score"]
                * recipe_scores["quantity"].astype(float)
                * INGREDIENT_PREFERENCE_MULTIPLIER
            )
            drink_scores = recipe_scores.groupby("drink_id")["ingredient_score"].sum()
            scored_drinks["recommendation_score"] += (
                scored_drinks["drink_id"].map(drink_scores).fillna(0).round().astype(int)
            )

            for drink_id, ingredient_score in drink_scores.items():
                score = int(round(ingredient_score))
                if score == 0:
                    continue
                reason = (
                    f"+{score} liked ingredients"
                    if score > 0
                    else f"{score} disliked ingredients"
                )
                mask = scored_drinks["drink_id"] == drink_id
                scored_drinks.loc[mask, "recommendation_explanation"] = scored_drinks.loc[
                    mask,
                    "recommendation_explanation",
                ].apply(lambda text: f"{text}; {reason}" if text else reason)

    scored_drinks.loc[
        scored_drinks["recommendation_explanation"] == "",
        "recommendation_explanation",
    ] = "No profile or history match yet"

    return scored_drinks.sort_values(
        by=["recommendation_score", "price"],
        ascending=[False, True],
    )


def recommend_drinks(
    drinks: pd.DataFrame,
    caffeine_level: str | None = None,
    temperature: str | None = None,
    milk: str | None = None,
    max_price: float | None = None,
    sweetness_level: str | None = None,
    dietary_tag: str | None = None,
    user: dict[str, str] | None = None,
    user_history: pd.DataFrame | None = None,
    ingredient_preferences: pd.DataFrame | None = None,
    drink_recipes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply filters, score drinks, and return ranked matches."""
    matches = filter_by_caffeine(drinks, caffeine_level)
    matches = filter_by_temperature(matches, temperature)
    matches = filter_by_milk(matches, milk)
    matches = filter_by_budget(matches, max_price)
    matches = filter_by_sweetness(matches, sweetness_level)
    matches = filter_by_dietary_tag(matches, dietary_tag)
    return add_recommendation_scores(
        matches,
        user=user,
        user_history=user_history,
        ingredient_preferences=ingredient_preferences,
        drink_recipes=drink_recipes,
    )
