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
CLOSE_MATCH_WEIGHTS = {
    "milk": 3,
    "temperature": 3,
    "caffeine": 2,
    "sweetness": 2,
    "dietary_tag": 2,
    "budget": 1,
}


def _clean_value(value: object) -> str:
    """Return a safe lowercase string for comparisons."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


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
    return drinks[drinks["caffeine_level"].apply(_clean_value) == caffeine_level.lower()]


def filter_by_temperature(drinks: pd.DataFrame, temperature: str | None) -> pd.DataFrame:
    """Filter drinks by hot, iced, or blended temperature."""
    if not temperature:
        return drinks
    return drinks[drinks["temperature"].apply(_clean_value) == temperature.lower()]


def filter_by_milk(drinks: pd.DataFrame, milk: str | None) -> pd.DataFrame:
    """Filter drinks by milk type."""
    if not milk:
        return drinks
    return drinks[drinks["milk"].apply(_clean_value) == milk.lower()]


def filter_by_budget(drinks: pd.DataFrame, max_price: float | None) -> pd.DataFrame:
    """Filter drinks at or below a user's budget."""
    if max_price is None:
        return drinks
    return drinks[drinks["price"] <= max_price]


def filter_by_sweetness(drinks: pd.DataFrame, sweetness_level: str | None) -> pd.DataFrame:
    """Filter drinks by sweetness level."""
    if not sweetness_level:
        return drinks
    return drinks[drinks["sweetness_level"].apply(_clean_value) == sweetness_level.lower()]


def filter_by_dietary_tag(drinks: pd.DataFrame, dietary_tag: str | None) -> pd.DataFrame:
    """Filter drinks that contain a dietary tag, such as vegan or dairy-free."""
    if not dietary_tag:
        return drinks
    tag = dietary_tag.lower()
    return drinks[drinks["dietary_tags"].apply(_clean_value).str.contains(tag, na=False)]


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
            scored_drinks["milk"].apply(_clean_value) == favorite_milk,
            PROFILE_SCORE_WEIGHTS["milk"],
            f"+{PROFILE_SCORE_WEIGHTS['milk']} favorite milk",
        )
        _add_reason(
            scored_drinks,
            scored_drinks["temperature"].apply(_clean_value) == favorite_temperature,
            PROFILE_SCORE_WEIGHTS["temperature"],
            f"+{PROFILE_SCORE_WEIGHTS['temperature']} favorite temperature",
        )
        _add_reason(
            scored_drinks,
            scored_drinks["caffeine_level"].apply(_clean_value) == caffeine_tolerance,
            PROFILE_SCORE_WEIGHTS["caffeine"],
            f"+{PROFILE_SCORE_WEIGHTS['caffeine']} caffeine tolerance",
        )
        _add_reason(
            scored_drinks,
            scored_drinks["sweetness_level"].apply(_clean_value) == preferred_sweetness,
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


def _selected_filters(
    caffeine_level: str | None,
    temperature: str | None,
    milk: str | None,
    max_price: float | None,
    sweetness_level: str | None,
    dietary_tag: str | None,
) -> dict[str, object]:
    """Collect active filters for fallback scoring."""
    return {
        "milk": milk,
        "temperature": temperature,
        "caffeine": caffeine_level,
        "sweetness": sweetness_level,
        "dietary_tag": dietary_tag,
        "budget": max_price,
    }


def _relaxed_filters(matches: pd.DataFrame, filters: dict[str, object]) -> list[str]:
    """Return filters that were not matched by every fallback result."""
    relaxed = []
    if matches.empty:
        return relaxed

    if filters["milk"] and not (matches["milk"].apply(_clean_value) == filters["milk"].lower()).all():
        relaxed.append("milk")
    if filters["temperature"] and not (
        matches["temperature"].apply(_clean_value) == filters["temperature"].lower()
    ).all():
        relaxed.append("temperature")
    if filters["caffeine"] and not (
        matches["caffeine_level"].apply(_clean_value) == filters["caffeine"].lower()
    ).all():
        relaxed.append("caffeine")
    if filters["sweetness"] and not (
        matches["sweetness_level"].apply(_clean_value) == filters["sweetness"].lower()
    ).all():
        relaxed.append("sweetness")
    if filters["dietary_tag"] and not (
        matches["dietary_tags"].apply(_clean_value).str.contains(filters["dietary_tag"].lower(), na=False)
    ).all():
        relaxed.append("dietary tag")
    if filters["budget"] is not None and not (matches["price"] <= filters["budget"]).all():
        relaxed.append("budget")
    return relaxed


def _close_match_explanation(row: pd.Series, filters: dict[str, object]) -> str:
    """Explain which selected filters matched and which changed."""
    matched = []
    different = []

    checks = [
        ("milk", "milk", filters["milk"]),
        ("temperature", "temperature", filters["temperature"]),
        ("caffeine", "caffeine_level", filters["caffeine"]),
        ("sweetness", "sweetness_level", filters["sweetness"]),
    ]
    for label, column, selected in checks:
        if not selected:
            continue
        if _clean_value(row.get(column, "")) == selected.lower():
            matched.append(label)
        else:
            different.append(label)

    dietary_tag = filters["dietary_tag"]
    if dietary_tag:
        if dietary_tag.lower() in _clean_value(row.get("dietary_tags", "")):
            matched.append("dietary tag")
        else:
            different.append("dietary tag")

    budget = filters["budget"]
    if budget is not None:
        if row.get("price", float("inf")) <= budget:
            matched.append("budget")
        else:
            different.append("budget")

    def joined(values: list[str]) -> str:
        if len(values) <= 1:
            return "".join(values)
        return f"{', '.join(values[:-1])} and {values[-1]}"

    if matched and different:
        verb = "was" if len(different) == 1 else "were"
        return (
            f"Close match: matched {joined(matched)}, "
            f"but {joined(different)} {verb} different."
        )
    if matched:
        return f"Close match: matched {joined(matched)}."
    return "Close match: ranked by your profile and closest available drink attributes."


def recommend_with_fallback(
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
) -> tuple[pd.DataFrame, bool, list[str]]:
    """Recommend exact matches first, then closest matches if exact matches are empty."""
    exact_matches = recommend_drinks(
        drinks=drinks,
        caffeine_level=caffeine_level,
        temperature=temperature,
        milk=milk,
        max_price=max_price,
        sweetness_level=sweetness_level,
        dietary_tag=dietary_tag,
        user=user,
        user_history=user_history,
        ingredient_preferences=ingredient_preferences,
        drink_recipes=drink_recipes,
    )
    if not exact_matches.empty:
        return exact_matches, True, []

    filters = _selected_filters(
        caffeine_level=caffeine_level,
        temperature=temperature,
        milk=milk,
        max_price=max_price,
        sweetness_level=sweetness_level,
        dietary_tag=dietary_tag,
    )
    fallback = add_recommendation_scores(
        drinks,
        user=user,
        user_history=user_history,
        ingredient_preferences=ingredient_preferences,
        drink_recipes=drink_recipes,
    )
    fallback["close_match_score"] = 0

    if milk:
        fallback.loc[fallback["milk"].apply(_clean_value) == milk.lower(), "close_match_score"] += (
            CLOSE_MATCH_WEIGHTS["milk"]
        )
    if temperature:
        fallback.loc[
            fallback["temperature"].apply(_clean_value) == temperature.lower(),
            "close_match_score",
        ] += CLOSE_MATCH_WEIGHTS["temperature"]
    if caffeine_level:
        fallback.loc[
            fallback["caffeine_level"].apply(_clean_value) == caffeine_level.lower(),
            "close_match_score",
        ] += CLOSE_MATCH_WEIGHTS["caffeine"]
    if sweetness_level:
        fallback.loc[
            fallback["sweetness_level"].apply(_clean_value) == sweetness_level.lower(),
            "close_match_score",
        ] += CLOSE_MATCH_WEIGHTS["sweetness"]
    if dietary_tag:
        fallback.loc[
            fallback["dietary_tags"].apply(_clean_value).str.contains(dietary_tag.lower(), na=False),
            "close_match_score",
        ] += CLOSE_MATCH_WEIGHTS["dietary_tag"]
    if max_price is not None:
        fallback.loc[fallback["price"] <= max_price, "close_match_score"] += CLOSE_MATCH_WEIGHTS[
            "budget"
        ]

    fallback["recommendation_score"] += fallback["close_match_score"]
    fallback["recommendation_explanation"] = fallback.apply(
        lambda row: _close_match_explanation(row, filters),
        axis=1,
    )
    fallback = fallback.sort_values(
        by=["recommendation_score", "close_match_score", "price"],
        ascending=[False, False, True],
    )
    relaxed = _relaxed_filters(fallback.head(20), filters)
    return fallback, False, relaxed
