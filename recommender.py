"""Filtering and scoring helpers for AI Barista recommendations."""

import re

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
    component: str | None = None,
) -> None:
    """Add score points and explanation text to matching rows."""
    drinks.loc[mask, "recommendation_score"] += points
    if component:
        drinks.loc[mask, component] += points
    drinks.loc[mask, "recommendation_explanation"] = drinks.loc[
        mask,
        "recommendation_explanation",
    ].apply(lambda text: f"{text}; {reason}" if text else reason)


def _drink_search_text(drink: dict[str, object] | pd.Series) -> str:
    """Combine searchable drink attributes into one normalized string."""
    columns = [
        "drink_name",
        "base",
        "milk",
        "syrup",
        "flavor_profile",
        "ingredients",
        "dietary_tags",
        "toppings",
    ]
    return " ".join(_clean_value(drink.get(column, "")) for column in columns)


def _phrases(value: object) -> list[str]:
    """Split comma-separated preference notes into useful searchable phrases."""
    if value is None or pd.isna(value):
        return []
    return [
        phrase.strip().lower()
        for phrase in re.split(r"[,;|]", str(value))
        if phrase.strip()
    ]


def _calculate_context_score_details(
    drink: dict[str, object] | pd.Series,
    context: dict[str, object] | None,
) -> tuple[int, list[str]]:
    """Return a context score and transparent reasons for one drink."""
    if not context:
        return 0, []

    score = 0
    reasons: list[str] = []
    text = _drink_search_text(drink)
    base = _clean_value(drink.get("base", ""))
    temperature = _clean_value(drink.get("temperature", ""))
    caffeine = _clean_value(drink.get("caffeine_level", ""))
    sweetness = _clean_value(drink.get("sweetness_level", ""))
    milk = _clean_value(drink.get("milk", ""))
    calories = pd.to_numeric(drink.get("calories", 0), errors="coerce")
    calories = 0 if pd.isna(calories) else float(calories)

    def add(points: int, reason: str) -> None:
        nonlocal score
        score += points
        reasons.append(f"{points:+d} {reason}")

    energy_bases = ("espresso", "cold brew", "americano", "shaken espresso")
    focus_bases = ("americano", "cold brew", "matcha", "flat white")
    comfort_bases = ("latte", "mocha", "cappuccino", "chai", "tea")
    workout_bases = ("cold brew", "espresso", "americano", "tea", "refresher")
    treat_bases = ("mocha", "frappuccino", "flavored latte")

    goal = _clean_value(context.get("goal", ""))
    if goal == "energy":
        if caffeine in {"medium", "high"}:
            add(12 if caffeine == "high" else 10, f"goal: energy favors {caffeine} caffeine")
        elif caffeine in {"none", "low"}:
            add(-12, f"goal: energy is less suited to {caffeine} caffeine")
        if any(term in text for term in energy_bases):
            add(10, "goal: energy favors espresso, cold brew, or americano")
    elif goal == "focus":
        if caffeine == "medium":
            add(12, "goal: focus favors medium caffeine")
        if any(term in text for term in focus_bases):
            add(10, "goal: focus favors americano, cold brew, matcha, or flat white")
        if sweetness == "extra":
            add(-8, "goal: focus avoids extra-sweet drinks")
    elif goal == "comfort":
        if temperature == "hot":
            add(10, "goal: comfort favors hot drinks")
        if any(term in text for term in comfort_bases) or milk not in {"", "none"}:
            add(10, "goal: comfort favors creamy latte, mocha, chai, or tea styles")
        if sweetness in {"classic", "extra"}:
            add(4, "goal: comfort slightly favors familiar sweetness")
    elif goal == "workout":
        if calories <= 120:
            add(8, "goal: workout favors lighter drinks")
        elif calories >= 250:
            add(-10, "goal: workout avoids very high-calorie drinks")
        if any(term in text for term in workout_bases):
            add(10, "goal: workout favors cold brew, espresso, americano, tea, or refreshers")
        if sweetness == "extra":
            add(-8, "goal: workout avoids extra sweetness")
        if any(term in text for term in ("heavy cream", "whipped cream", "sweet cream")):
            add(-8, "goal: workout avoids heavy cream")
    elif goal == "treat":
        if any(term in text for term in treat_bases):
            add(12, "goal: treat favors mocha, frappuccino, or flavored latte styles")
        if sweetness in {"classic", "extra"}:
            add(8, "goal: treat favors classic or extra sweetness")
        if _clean_value(drink.get("syrup", "")) not in {"", "none"} or _clean_value(
            drink.get("toppings", "")
        ) not in {"", "none"}:
            add(5, "goal: treat favors syrups and toppings")

    sleep = pd.to_numeric(context.get("sleep_hours"), errors="coerce")
    if not pd.isna(sleep):
        if sleep < 5:
            if caffeine in {"medium", "high"}:
                add(10 if caffeine == "high" else 8, "short sleep favors a caffeine lift")
            elif caffeine == "none":
                add(-10, "short sleep is less suited to no caffeine")
        elif sleep < 7 and caffeine == "medium":
            add(7, "moderate sleep favors medium caffeine")
        elif sleep >= 7 and caffeine == "high":
            add(-3, "rested context avoids over-boosting high caffeine")

    stress = _clean_value(context.get("stress_level", ""))
    if stress == "high":
        if any(term in text for term in comfort_bases):
            add(8, "high stress favors comforting latte, tea, matcha, or chai styles")
        if caffeine == "high":
            add(-12, "high stress avoids very high caffeine")

    weather = _clean_value(context.get("weather", ""))
    if weather in {"sunny", "hot", "warm"}:
        if temperature in {"iced", "blended"}:
            add(10, f"{weather} weather favors iced or blended drinks")
        if any(term in text for term in ("cold brew", "refresher")):
            add(8, f"{weather} weather favors cold brew or refreshers")
    elif weather in {"cold", "rainy", "snowy"}:
        if temperature == "hot":
            add(10, f"{weather} weather favors hot drinks")
        if any(term in text for term in comfort_bases):
            add(8, f"{weather} weather favors latte, mocha, chai, cappuccino, or tea")

    temperature_preference = _clean_value(context.get("temperature_preference", ""))
    if temperature_preference in {"iced", "hot"}:
        if temperature == temperature_preference or (
            temperature_preference == "iced" and temperature == "blended"
        ):
            add(14, f"matches {temperature_preference} temperature preference")
        else:
            add(-6, f"does not match {temperature_preference} temperature preference")

    caffeine_preference = _clean_value(context.get("caffeine_preference", ""))
    if caffeine_preference == "none":
        if caffeine == "none":
            add(24, "matches no-caffeine preference")
        elif caffeine == "low":
            add(5, "low caffeine is close to no-caffeine preference")
        elif caffeine in {"medium", "high"}:
            add(-25, "conflicts with no-caffeine preference")
    elif caffeine_preference in {"low", "medium", "high"}:
        if caffeine == caffeine_preference:
            add(14, f"matches {caffeine_preference} caffeine preference")
        elif caffeine_preference == "low" and caffeine == "high":
            add(-8, "high caffeine conflicts with low-caffeine preference")

    sweetness_preference = _clean_value(context.get("sweetness_preference", ""))
    if sweetness_preference == "unsweetened":
        if sweetness == "unsweetened":
            add(14, "matches unsweetened preference")
        elif sweetness == "light":
            add(5, "light sweetness is close to unsweetened preference")
        elif sweetness == "extra":
            add(-10, "extra sweetness conflicts with unsweetened preference")
    elif sweetness_preference in {"light", "classic", "extra"} and sweetness == sweetness_preference:
        add(14, f"matches {sweetness_preference} sweetness preference")

    for phrase in _phrases(context.get("likes", "")):
        if phrase in text:
            add(6, f"contains liked {phrase}")
    for phrase in _phrases(context.get("dislikes", "")):
        if phrase in text:
            add(-20, f"contains disliked {phrase}")

    return score, reasons


def calculate_context_score(
    drink: dict[str, object] | pd.Series,
    context: dict[str, object] | None,
) -> int:
    """Calculate the context-only score for one drink."""
    return _calculate_context_score_details(drink, context)[0]


def _short_reason(explanation: str) -> str:
    """Create a compact, consumer-friendly reason from detailed score reasons."""
    positive = [
        reason.strip()
        for reason in str(explanation).split(";")
        if reason.strip().startswith("+")
    ]
    if positive:
        selected = [re.sub(r"^\+\d+\s*", "", reason) for reason in positive[:3]]
        return "Based on your preferences and context: " + ", ".join(selected) + "."
    return "Based on your preferences and context, this is one of the closest available drinks."


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


def find_similar_drinks(
    drinks: pd.DataFrame,
    drink: dict[str, object] | pd.Series,
    limit: int = 3,
) -> pd.DataFrame:
    """Find similar drinks using the existing catalog attributes."""
    candidates = drinks.copy()
    drink_id = str(drink.get("drink_id", ""))
    if drink_id and "drink_id" in candidates.columns:
        candidates = candidates[candidates["drink_id"].astype(str) != drink_id]

    candidates["similarity_score"] = 0
    weights = {
        "base": 3,
        "milk": 2,
        "temperature": 2,
        "caffeine_level": 2,
        "sweetness_level": 1,
    }
    for column, weight in weights.items():
        if column not in candidates.columns:
            continue
        selected = _clean_value(drink.get(column, ""))
        if selected:
            candidates.loc[
                candidates[column].apply(_clean_value) == selected,
                "similarity_score",
            ] += weight

    return candidates.sort_values(
        by=["similarity_score", "price"],
        ascending=[False, True],
    ).head(limit)


def add_recommendation_scores(
    drinks: pd.DataFrame,
    user: dict[str, str] | None = None,
    user_history: pd.DataFrame | None = None,
    ingredient_preferences: pd.DataFrame | None = None,
    drink_recipes: pd.DataFrame | None = None,
    context: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Add a rule-based recommendation score to each drink."""
    scored_drinks = drinks.copy()
    details = scored_drinks.apply(
        lambda drink: _calculate_context_score_details(drink, context),
        axis=1,
    )
    scored_drinks["context_score"] = details.apply(lambda detail: detail[0]).astype(int)
    scored_drinks["recommendation_explanation"] = details.apply(
        lambda detail: "; ".join(detail[1])
    )
    scored_drinks["profile_score"] = 0
    scored_drinks["ingredient_preference_score"] = 0
    scored_drinks["past_rating_score"] = 0
    scored_drinks["fallback_similarity_score"] = 0
    scored_drinks["recommendation_score"] = scored_drinks["context_score"]

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
            "profile_score",
        )
        _add_reason(
            scored_drinks,
            scored_drinks["temperature"].apply(_clean_value) == favorite_temperature,
            PROFILE_SCORE_WEIGHTS["temperature"],
            f"+{PROFILE_SCORE_WEIGHTS['temperature']} favorite temperature",
            "profile_score",
        )
        _add_reason(
            scored_drinks,
            scored_drinks["caffeine_level"].apply(_clean_value) == caffeine_tolerance,
            PROFILE_SCORE_WEIGHTS["caffeine"],
            f"+{PROFILE_SCORE_WEIGHTS['caffeine']} caffeine tolerance",
            "profile_score",
        )
        _add_reason(
            scored_drinks,
            scored_drinks["sweetness_level"].apply(_clean_value) == preferred_sweetness,
            PROFILE_SCORE_WEIGHTS["sweetness"],
            f"+{PROFILE_SCORE_WEIGHTS['sweetness']} preferred sweetness",
            "profile_score",
        )

    if user_history is not None and not user_history.empty:
        history = user_history.copy()
        ratings = history["rating"].astype(int)
        history["rating_points"] = ratings.apply(
            lambda rating: (
                rating * RATING_MULTIPLIER
                if rating >= 4
                else -(3 - rating) * RATING_MULTIPLIER
                if rating <= 2
                else 0
            )
        )
        history["order_again_points"] = history["would_order_again"].apply(_is_true).astype(int)
        history["order_again_points"] *= ORDER_AGAIN_BONUS
        history["history_score"] = history["rating_points"] + history["order_again_points"]
        history_scores = history.groupby("drink_id")["history_score"].sum()
        history_reasons = {}
        for drink_id, drink_history in history.groupby("drink_id"):
            rating_points = int(drink_history["rating_points"].sum())
            order_again_points = int(drink_history["order_again_points"].sum())
            reasons = [f"{rating_points:+d} past rating points"] if rating_points else []
            if order_again_points:
                reasons.append(f"+{order_again_points} would order again")
            history_reasons[drink_id] = "; ".join(reasons)

        scored_drinks["recommendation_score"] += (
            scored_drinks["drink_id"].map(history_scores).fillna(0).astype(int)
        )
        scored_drinks["past_rating_score"] += (
            scored_drinks["drink_id"].map(history_scores).fillna(0).astype(int)
        )
        for drink_id, reason in history_reasons.items():
            if not reason:
                continue
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
            scored_drinks["ingredient_preference_score"] += (
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
    scored_drinks["recommendation_summary"] = scored_drinks[
        "recommendation_explanation"
    ].apply(_short_reason)

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
    context: dict[str, object] | None = None,
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
        context=context,
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
    context: dict[str, object] | None = None,
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
        context=context,
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
        context=context,
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

    fallback["fallback_similarity_score"] = fallback["close_match_score"]
    fallback["recommendation_score"] += fallback["fallback_similarity_score"]
    close_reasons = fallback.apply(lambda row: _close_match_explanation(row, filters), axis=1)
    fallback["recommendation_explanation"] = fallback[
        "recommendation_explanation"
    ].astype(str) + "; " + close_reasons
    fallback["recommendation_summary"] = fallback["recommendation_explanation"].apply(
        _short_reason
    )
    fallback = fallback.sort_values(
        by=["recommendation_score", "close_match_score", "price"],
        ascending=[False, False, True],
    )
    relaxed = _relaxed_filters(fallback.head(20), filters)
    return fallback, False, relaxed
