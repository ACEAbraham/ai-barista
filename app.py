"""Streamlit web app for AI Barista data collection."""

from datetime import datetime, timezone
from html import escape

import pandas as pd
import streamlit as st

from customization import log_session
from drink_database import list_options, load_drinks
from ingredient_engine import (
    SUPPORTED_CATEGORIES,
    SUPPORTED_UNITS,
    add_ingredient,
    build_custom_drink_from_ingredients,
    custom_drink_exists,
    get_ingredient_preferences,
    get_taste_profile,
    load_ingredients,
    load_recipes,
    save_custom_drink_recipe,
)
from openai_client import generate_drink_recommendation, openai_is_configured
from profile import create_user, get_user_history, load_user, save_rating
from profile import load_users as load_supabase_users
from recommender import recommend_with_fallback
from supabase_client import insert_row, supabase_is_configured, update_rows


def apply_theme() -> None:
    """Apply the AI Barista premium Streamlit theme."""
    st.markdown(
        """
        <style>
        :root {
            --ai-bg: #DDD4C6;
            --ai-section: #D0BCA8;
            --ai-accent: #A07D61;
            --ai-button: #76563A;
            --ai-text: #4A2608;
            --ai-secondary: #5C4033;
            --ai-input: #FFFDF8;
        }

        .stApp {
            background: var(--ai-bg);
            color: var(--ai-text);
        }

        .stApp p,
        .stApp span,
        .stApp label,
        .stApp input,
        .stApp textarea {
            color: var(--ai-text);
        }

        .stApp small,
        .stApp [data-testid="stCaptionContainer"] {
            color: var(--ai-secondary);
        }

        [data-testid="stSidebar"] {
            background: var(--ai-section);
            border-right: 1px solid var(--ai-accent);
        }

        [data-testid="stSidebar"] * {
            color: var(--ai-text);
        }

        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }

        .ai-title-section {
            background: var(--ai-section);
            border: 1px solid var(--ai-accent);
            border-radius: 22px;
            padding: 2rem 2.25rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 14px 36px rgba(74, 38, 8, 0.08);
        }

        .ai-title-section h1 {
            color: var(--ai-text);
            font-size: 3rem;
            line-height: 1;
            margin: 0 0 0.6rem 0;
            letter-spacing: 0;
        }

        .ai-title-section p {
            color: var(--ai-secondary);
            font-size: 1.06rem;
            margin: 0;
        }

        h2, h3 {
            color: var(--ai-text) !important;
            letter-spacing: 0;
        }

        input,
        textarea,
        [role="combobox"] {
            background-color: var(--ai-input) !important;
            color: var(--ai-text) !important;
            border-color: var(--ai-accent) !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: var(--ai-secondary) !important;
        }

        button[role="tab"],
        button[role="tab"] * {
            color: var(--ai-text) !important;
        }

        button[role="tab"][aria-selected="true"],
        button[role="tab"][aria-selected="true"] * {
            color: var(--ai-text) !important;
            font-weight: 700;
        }

        [role="listbox"],
        [role="listbox"] *,
        [role="option"],
        [role="option"] * {
            color: var(--ai-text) !important;
            background-color: var(--ai-input) !important;
        }

        [role="option"]:hover,
        [role="option"]:hover * {
            color: var(--ai-text) !important;
            background-color: var(--ai-section) !important;
        }

        [role="radiogroup"] label,
        [role="radiogroup"] label * {
            color: var(--ai-text) !important;
        }

        button:not([role="tab"]) {
            background-color: var(--ai-button) !important;
            color: #FFFFFF !important;
            border-color: var(--ai-button) !important;
        }

        button:not([role="tab"]) * {
            color: #FFFFFF !important;
        }

        button:not([role="tab"]):hover {
            background-color: var(--ai-text) !important;
            border-color: var(--ai-text) !important;
            color: #FFFFFF !important;
        }

        .ai-card {
            background: var(--ai-section);
            border: 1px solid var(--ai-accent);
            border-radius: 20px;
            padding: 1.15rem 1.25rem;
            margin: 0.85rem 0;
            box-shadow: 0 12px 30px rgba(74, 38, 8, 0.08);
        }

        .ai-card-title {
            color: var(--ai-text);
            font-size: 1.12rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .ai-card-meta {
            color: var(--ai-secondary);
            font-size: 0.92rem;
            margin-bottom: 0.75rem;
        }

        .ai-score {
            display: inline-block;
            background: var(--ai-input);
            color: var(--ai-button);
            border: 1px solid var(--ai-accent);
            border-radius: 999px;
            padding: 0.22rem 0.65rem;
            font-weight: 800;
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
        }

        .ai-explanation {
            background: var(--ai-input);
            border-left: 4px solid var(--ai-accent);
            color: var(--ai-text);
            border-radius: 14px;
            padding: 0.75rem 0.85rem;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def refresh_data() -> None:
    """Reload CSV-backed app data after a save."""
    st.session_state.drinks = load_drinks()
    st.session_state.users = load_supabase_users()
    st.session_state.ingredients = load_ingredients()


def initialize_state() -> None:
    """Initialize Streamlit session state."""
    if "drinks" not in st.session_state:
        st.session_state.drinks = load_drinks()
    if "users" not in st.session_state:
        st.session_state.users = load_supabase_users()
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "ingredients" not in st.session_state:
        st.session_state.ingredients = load_ingredients()
    if "ai_recommendation" not in st.session_state:
        st.session_state.ai_recommendation = None
    if "ai_recommendation_context" not in st.session_state:
        st.session_state.ai_recommendation_context = None
    if "ai_recommendation_id" not in st.session_state:
        st.session_state.ai_recommendation_id = None


def save_warning() -> None:
    """Show a friendly warning when Supabase is not configured locally."""
    st.warning(
        "Supabase is not configured in this environment. Add SUPABASE_URL "
        "and SUPABASE_KEY to Streamlit secrets or local environment variables "
        "to save shared data."
    )


def current_user_label() -> str:
    """Return a friendly label for the current user."""
    user = st.session_state.current_user
    if not user:
        return "No profile loaded"
    return f"{user['name']} ({user['user_id']})"


def session_context(prefix: str) -> dict[str, object]:
    """Collect contextual fields for training-data sessions."""
    col1, col2 = st.columns(2)
    with col1:
        sleep_hours = st.number_input(
            "Sleep hours",
            min_value=0.0,
            max_value=16.0,
            value=7.0,
            step=0.5,
            key=f"{prefix}_sleep",
        )
        goal = st.selectbox(
            "Goal",
            ["energy", "focus", "comfort", "treat", "refresh", "low caffeine"],
            key=f"{prefix}_goal",
        )
    with col2:
        stress_level = st.selectbox(
            "Stress level",
            ["low", "medium", "high"],
            key=f"{prefix}_stress",
        )
        weather = st.selectbox(
            "Weather",
            ["sunny", "cloudy", "rainy", "snowy", "hot", "cold"],
            key=f"{prefix}_weather",
        )

    return {
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "goal": goal,
        "weather": weather,
    }


def safe_text(value: object, fallback: str = "") -> str:
    """Return escaped display text for HTML snippets."""
    if pd.isna(value):
        return fallback
    return escape(str(value))


def render_recommendation_cards(matches: pd.DataFrame, limit: int = 10) -> None:
    """Render recommendation results as premium cards."""
    for _, drink in matches.head(limit).iterrows():
        name = safe_text(drink.get("drink_name", "Unnamed drink"))
        drink_id = safe_text(drink.get("drink_id", ""))
        price = drink.get("price", 0)
        calories = drink.get("calories", 0)
        caffeine = safe_text(drink.get("caffeine_level", "unknown"))
        score = safe_text(drink.get("recommendation_score", 0))
        explanation = safe_text(
            drink.get("recommendation_explanation", "No explanation available.")
        )

        st.markdown(
            f"""
            <div class="ai-card">
                <div class="ai-card-title">{name}</div>
                <div class="ai-card-meta">
                    {drink_id} &nbsp;|&nbsp; ${float(price):.2f} &nbsp;|&nbsp;
                    {caffeine} caffeine &nbsp;|&nbsp; {calories} calories
                </div>
                <div class="ai-score">Recommendation score: {score}</div>
                <div class="ai-explanation">{explanation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_ai_recommendation(recommendation: dict[str, object]) -> None:
    """Render one OpenAI drink recommendation."""
    ingredients = ", ".join(
        safe_text(item) for item in recommendation.get("ingredients", [])
    )
    st.markdown(
        f"""
        <div class="ai-card">
            <div class="ai-card-title">{safe_text(recommendation.get("drink_name"))}</div>
            <div class="ai-card-meta">
                {safe_text(recommendation.get("caffeine_level"))} caffeine
            </div>
            <p><strong>Ingredients:</strong> {ingredients}</p>
            <div class="ai-explanation">
                {safe_text(recommendation.get("explanation"))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _save_ai_recommendation(
    context: dict[str, object],
    recommendation: dict[str, object],
) -> object | None:
    """Save generated recommendation context to Supabase."""
    if not supabase_is_configured():
        save_warning()
        return None

    row = {
        **context,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recommendation_json": recommendation,
        "rating": None,
        "would_order_again": None,
        "too_sweet": None,
        "too_bitter": None,
        "too_much_caffeine": None,
        "feedback_text": None,
    }
    saved = insert_row("ai_recommendations", row)
    return saved.get("id")


def ai_recommendation_section() -> None:
    """Render the simplified OpenAI recommendation and feedback experience."""
    st.subheader("Quick Recommendation")
    st.caption("One drink recommendation in under 30 seconds.")

    with st.form("ai_recommendation_form"):
        col1, col2 = st.columns(2)
        with col1:
            sleep_hours = st.number_input(
                "Sleep Hours",
                min_value=0.0,
                max_value=16.0,
                value=7.0,
                step=0.5,
                key="ai_sleep_hours",
            )
            stress_level = st.selectbox("Stress Level", ["low", "medium", "high"])
            goal = st.selectbox(
                "Goal",
                ["energy", "focus", "comfort", "study", "workout", "treat"],
            )
            weather = st.selectbox(
                "Weather",
                ["sunny", "cloudy", "rainy", "snowy", "hot", "cold"],
            )
        with col2:
            preferred_temperature = st.selectbox(
                "Temperature Preference",
                ["no preference", "hot", "iced"],
            )
            things_you_love = st.text_area(
                "Things You Love",
                placeholder="Vanilla, oat milk, cinnamon...",
            )
            things_you_hate = st.text_area(
                "Things You Hate",
                placeholder="Bitter flavors, whipped cream...",
            )
        submitted = st.form_submit_button(
            "Get Recommendation",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not openai_is_configured():
            st.error(
                "OpenAI is not configured. Add OPENAI_API_KEY to Streamlit secrets "
                "or local environment variables."
            )
        else:
            context = {
                "user_name": "quick-beta-user",
                "sleep_hours": sleep_hours,
                "stress_level": stress_level,
                "goal": goal,
                "weather": weather,
                "preferred_temperature": preferred_temperature,
                "caffeine_preference": "any",
                "milk_preference": "any",
                "sweetness_preference": "any",
                "dietary_restrictions": "",
                "likes_dislikes": (
                    f"Loves: {things_you_love.strip() or 'not specified'}; "
                    f"Hates: {things_you_hate.strip() or 'not specified'}"
                ),
            }
            try:
                with st.spinner("Your barista is thinking..."):
                    recommendation = generate_drink_recommendation(context)
                    recommendation_id = _save_ai_recommendation(context, recommendation)
            except Exception as error:
                st.error(f"Could not create a recommendation: {error}")
            else:
                st.session_state.ai_recommendation = recommendation
                st.session_state.ai_recommendation_context = context
                st.session_state.ai_recommendation_id = recommendation_id

    recommendation = st.session_state.ai_recommendation
    if not recommendation:
        return

    render_ai_recommendation(recommendation)
    st.markdown("### Rate This Recommendation")
    with st.form("ai_feedback_form"):
        rating = st.slider("Rating", 1, 5, 4)
        would_order_again = st.radio(
            "Would order again?",
            ["Yes", "No"],
            horizontal=True,
        )
        feedback_submitted = st.form_submit_button(
            "Save Rating",
            use_container_width=True,
        )

    if feedback_submitted:
        values = {
            "rating": rating,
            "would_order_again": would_order_again == "Yes",
            "too_sweet": None,
            "too_bitter": None,
            "too_much_caffeine": None,
            "feedback_text": None,
        }
        try:
            recommendation_id = st.session_state.ai_recommendation_id
            if recommendation_id is not None:
                update_rows("ai_recommendations", values, {"id": recommendation_id})
            elif supabase_is_configured():
                row = {
                    **st.session_state.ai_recommendation_context,
                    **values,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "recommendation_json": recommendation,
                }
                saved = insert_row("ai_recommendations", row)
                st.session_state.ai_recommendation_id = saved.get("id")
            else:
                raise RuntimeError("Supabase is not configured.")
        except RuntimeError as error:
            st.error(str(error))
        else:
            st.success("Thanks. Your feedback was saved.")


def ingredient_list_section() -> None:
    """Render the current ingredient catalog."""
    st.subheader("Ingredient List")
    st.dataframe(
        st.session_state.ingredients.sort_values(["category", "ingredient_name"]),
        width="stretch",
    )


def create_profile_section() -> None:
    """Render profile creation UI."""
    st.subheader("Create Profile")
    drinks = st.session_state.drinks

    with st.form("create_profile_form"):
        name = st.text_input("Name")
        favorite_milk = st.selectbox("Favorite milk", list_options(drinks, "milk"))
        favorite_temperature = st.selectbox(
            "Favorite temperature",
            list_options(drinks, "temperature"),
        )
        caffeine_tolerance = st.selectbox(
            "Caffeine tolerance",
            list_options(drinks, "caffeine_level"),
        )
        preferred_sweetness = st.selectbox(
            "Preferred sweetness",
            list_options(drinks, "sweetness_level"),
        )
        submitted = st.form_submit_button("Create profile")

    if submitted:
        if not name.strip():
            st.error("Please enter a name.")
            return
        try:
            user = create_user(
                name=name.strip(),
                favorite_milk=favorite_milk,
                favorite_temperature=favorite_temperature,
                caffeine_tolerance=caffeine_tolerance,
                preferred_sweetness=preferred_sweetness,
            )
        except RuntimeError as error:
            st.error(str(error))
            return
        st.session_state.current_user = user
        refresh_data()
        st.success(f"Created and loaded profile: {user['name']} ({user['user_id']})")


def load_profile_section() -> None:
    """Render profile loading UI."""
    st.subheader("Load Profile")
    users = st.session_state.users

    if users.empty:
        st.info("No profiles exist yet. Create one first.")
        return

    labels = [f"{row.name} ({row.user_id})" for row in users.itertuples()]
    selected = st.selectbox("Choose a profile", labels)
    user_id = selected.split("(")[-1].rstrip(")")

    if st.button("Load selected profile"):
        user = load_user(user_id)
        st.session_state.current_user = user
        st.success(f"Loaded profile: {current_user_label()}")


def recommendation_section() -> None:
    """Render recommendation UI."""
    st.subheader("Find Recommended Drinks")
    drinks = st.session_state.drinks
    user = st.session_state.current_user
    st.caption(f"Current profile: {current_user_label()}")

    with st.form("recommendation_form"):
        context = session_context("recommend")
        col1, col2, col3 = st.columns(3)
        with col1:
            caffeine = st.selectbox(
                "Caffeine filter",
                ["Any"] + list_options(drinks, "caffeine_level"),
            )
            milk = st.selectbox("Milk filter", ["Any"] + list_options(drinks, "milk"))
        with col2:
            temperature = st.selectbox(
                "Temperature filter",
                ["Any"] + list_options(drinks, "temperature"),
            )
            sweetness = st.selectbox(
                "Sweetness filter",
                ["Any"] + list_options(drinks, "sweetness_level"),
            )
        with col3:
            max_price = st.number_input(
                "Maximum price",
                min_value=0.0,
                max_value=20.0,
                value=10.0,
                step=0.25,
            )
            dietary_tag = st.selectbox(
                "Dietary tag",
                ["Any", "dairy-free", "vegan", "vegetarian", "nut-free", "low-calorie"],
            )
        submitted = st.form_submit_button("Find drinks")

    if submitted:
        matches, exact_match, relaxed_filters = recommend_with_fallback(
            drinks=drinks,
            caffeine_level=None if caffeine == "Any" else caffeine,
            temperature=None if temperature == "Any" else temperature,
            milk=None if milk == "Any" else milk,
            max_price=max_price,
            sweetness_level=None if sweetness == "Any" else sweetness,
            dietary_tag=None if dietary_tag == "Any" else dietary_tag,
            user=user,
            user_history=get_user_history(user["user_id"]) if user else None,
            ingredient_preferences=(
                get_ingredient_preferences(user["user_id"]) if user else None
            ),
            drink_recipes=load_recipes(),
        )
        st.session_state.last_matches = matches
        st.session_state.last_exact_match = exact_match
        st.session_state.last_relaxed_filters = relaxed_filters

        if not matches.empty:
            top_match = matches.iloc[0]
            if supabase_is_configured():
                log_session(
                    user_id=user["user_id"] if user else "guest",
                    drink_id=top_match["drink_id"],
                    rating="",
                    **context,
                )
                st.success("Saved this recommendation interaction as training data.")
            else:
                save_warning()

    matches = st.session_state.get("last_matches")
    if matches is not None:
        if matches.empty:
            st.warning("No drinks matched those filters.")
        else:
            exact_match = st.session_state.get("last_exact_match", True)
            relaxed_filters = st.session_state.get("last_relaxed_filters", [])
            if not exact_match:
                st.info("No exact match found, but try this instead:")
                if relaxed_filters:
                    st.caption(f"Relaxed filters: {', '.join(relaxed_filters)}")
            render_recommendation_cards(matches)


def ingredient_recipe_builder(ingredients: pd.DataFrame) -> list[dict[str, object]]:
    """Render ingredient dropdown rows and return selected recipe items."""
    ingredient_options = [
        f"{row.ingredient_name} ({row.ingredient_id})"
        for row in ingredients.itertuples()
    ]
    ingredient_lookup = {
        f"{row.ingredient_name} ({row.ingredient_id})": row.ingredient_id
        for row in ingredients.itertuples()
    }
    unit_lookup = {
        f"{row.ingredient_name} ({row.ingredient_id})": row.default_unit
        for row in ingredients.itertuples()
    }

    recipe_items = []
    for index in range(1, 9):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            selected = st.selectbox(
                f"Ingredient {index}",
                ["None"] + ingredient_options,
                key=f"custom_ingredient_{index}",
            )
        with col2:
            quantity = st.number_input(
                "Quantity",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
                key=f"custom_quantity_{index}",
            )
        with col3:
            default_unit = unit_lookup.get(selected, "serving")
            default_index = SUPPORTED_UNITS.index(default_unit) if default_unit in SUPPORTED_UNITS else 0
            unit = st.selectbox(
                "Unit",
                SUPPORTED_UNITS,
                index=default_index,
                key=f"custom_unit_{index}",
            )
        if selected != "None" and quantity > 0:
            recipe_items.append(
                {
                    "ingredient_id": ingredient_lookup[selected],
                    "quantity": quantity,
                    "unit": unit,
                }
            )

    return recipe_items


def custom_drink_section() -> None:
    """Render ingredient-based custom drink UI."""
    st.subheader("Create Custom Ingredient-Based Drink")
    user = st.session_state.current_user
    ingredients = st.session_state.ingredients

    with st.expander("Ingredient catalog", expanded=False):
        st.dataframe(ingredients, width="stretch")

    with st.form("custom_drink_form"):
        drink_name = st.text_input("Custom drink name")
        recipe_items = ingredient_recipe_builder(ingredients)
        context = session_context("custom")
        rate_now = st.checkbox("Rate this custom drink now", value=False)
        rating = st.slider("Rating", 1, 5, 5, disabled=not rate_now)
        would_order_again = st.checkbox(
            "Would order again",
            value=True,
            disabled=not rate_now,
        )
        submitted = st.form_submit_button("Save custom drink")

    if submitted:
        if not recipe_items:
            st.error("Choose at least one ingredient.")
            return
        try:
            custom_drink = build_custom_drink_from_ingredients(
                drinks=st.session_state.drinks,
                drink_name=drink_name.strip(),
                recipe_items=recipe_items,
            )
        except ValueError as error:
            st.error(str(error))
            return
        exists, duplicate_reason = custom_drink_exists(custom_drink["drink_name"], recipe_items)
        if exists:
            st.warning(duplicate_reason)
            return

        try:
            save_custom_drink_recipe(custom_drink, recipe_items)
        except RuntimeError as error:
            st.error(str(error))
            return
        saved_rating = ""
        if user and rate_now:
            try:
                save_rating(user["user_id"], custom_drink["drink_id"], rating, would_order_again)
            except RuntimeError as error:
                st.error(str(error))
                return
            saved_rating = rating

        try:
            log_session(
                user_id=user["user_id"] if user else "guest",
                drink_id=custom_drink["drink_id"],
                rating=saved_rating,
                **context,
            )
        except RuntimeError as error:
            st.error(str(error))
            return
        refresh_data()
        st.success(f"Saved custom drink: {custom_drink['drink_name']}")
        st.metric("Calories", custom_drink["calories"])
        st.metric("Price", f"${custom_drink['price']:.2f}")
        st.metric("Flavor score", f"{custom_drink['flavor_score']}/10")


def add_ingredient_section() -> None:
    """Render UI for adding a custom ingredient."""
    st.subheader("Add Ingredient")
    st.caption("New ingredients become available immediately in the custom drink builder.")

    with st.form("add_ingredient_form"):
        ingredient_name = st.text_input("Ingredient name")
        category = st.selectbox("Category", SUPPORTED_CATEGORIES)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            calories = st.number_input("Calories", min_value=0.0, max_value=1000.0, value=0.0)
        with col2:
            caffeine = st.number_input("Caffeine", min_value=0.0, max_value=500.0, value=0.0)
        with col3:
            price = st.number_input("Price", min_value=0.0, max_value=20.0, value=0.0, step=0.05)
        with col4:
            default_unit = st.selectbox("Default unit", SUPPORTED_UNITS)
        submitted = st.form_submit_button("Save ingredient")

    if submitted:
        if not ingredient_name.strip():
            st.error("Please enter an ingredient name.")
            return
        try:
            ingredient = add_ingredient(
                ingredient_name=ingredient_name,
                category=category,
                calories=calories,
                caffeine=caffeine,
                price=price,
                default_unit=default_unit,
            )
        except ValueError as error:
            if str(error) == "This ingredient already exists.":
                st.warning("This ingredient already exists.")
            else:
                st.error(str(error))
            return

        refresh_data()
        st.success("Ingredient added successfully.")
        st.caption(f"Saved {ingredient['ingredient_name']} as {ingredient['ingredient_id']}.")

    st.markdown("### Ingredient list")
    st.dataframe(
        st.session_state.ingredients.sort_values(["category", "ingredient_name"]),
        width="stretch",
    )


def rate_drink_section() -> None:
    """Render drink rating UI."""
    st.subheader("Rate a Drink")
    user = st.session_state.current_user
    if not user:
        st.info("Load or create a profile before rating drinks.")
        return

    drinks = st.session_state.drinks
    drink_options = [
        f"{row.drink_name} ({row.drink_id})"
        for row in drinks[["drink_id", "drink_name"]].itertuples()
    ]

    with st.form("rate_drink_form"):
        selected = st.selectbox("Drink", drink_options)
        drink_id = selected.split("(")[-1].rstrip(")")
        rating = st.slider("Rating", 1, 5, 4)
        would_order_again = st.checkbox("Would order again", value=True)
        context = session_context("rating")
        submitted = st.form_submit_button("Save rating")

    if submitted:
        try:
            save_rating(user["user_id"], drink_id, rating, would_order_again)
            log_session(
                user_id=user["user_id"],
                drink_id=drink_id,
                rating=rating,
                **context,
            )
        except RuntimeError as error:
            st.error(str(error))
            return
        st.success("Saved rating, session data, and ingredient preferences.")


def rating_history_section() -> None:
    """Render rating history UI."""
    st.subheader("View Rating History")
    user = st.session_state.current_user
    if not user:
        st.info("Load or create a profile first.")
        return

    history = get_user_history(user["user_id"])
    if history.empty:
        st.info("No ratings saved yet.")
        return

    history = history.merge(st.session_state.drinks, on="drink_id", how="left")
    st.dataframe(
        history[["drink_id", "drink_name", "rating", "would_order_again"]],
        width="stretch",
    )


def taste_profile_section() -> None:
    """Render taste profile UI."""
    st.subheader("View Taste Profile")
    user = st.session_state.current_user
    if not user:
        st.info("Load or create a profile first.")
        return

    profile = get_taste_profile(user["user_id"])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Favorite ingredients")
        st.dataframe(
            profile["favorite"][["ingredient_name", "preference_score"]].head(10),
            width="stretch",
        )
    with col2:
        st.write("Least favorite ingredients")
        st.dataframe(
            profile["least_favorite"][["ingredient_name", "preference_score"]].head(10),
            width="stretch",
        )
    with col3:
        st.write("Most common ingredients")
        st.dataframe(
            profile["most_common"][["ingredient_name", "times_seen"]].head(10),
            width="stretch",
        )


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="AI Barista", layout="wide")
    apply_theme()
    initialize_state()

    st.markdown(
        """
        <section class="ai-title-section">
            <h1>AI Barista</h1>
            <p>A quick drink recommendation for how today feels.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if not supabase_is_configured():
        st.warning(
            "Supabase credentials are not configured locally. Static catalog browsing works, "
            "but recommendations and feedback require SUPABASE_URL and SUPABASE_KEY."
        )

    ai_recommendation_section()

    with st.expander("Advanced Tools", expanded=False):
        advanced = st.tabs(
            [
                "Create Profile",
                "Create Custom Drink",
                "Add Ingredient",
                "Ingredient List",
                "More",
            ]
        )
        with advanced[0]:
            create_profile_section()
        with advanced[1]:
            custom_drink_section()
        with advanced[2]:
            add_ingredient_section()
        with advanced[3]:
            ingredient_list_section()
        with advanced[4]:
            more_tabs = st.tabs(
                [
                    "Rule-based recommendations",
                    "Load profile",
                    "Rate drink",
                    "Rating history",
                    "Taste profile",
                ]
            )
            with more_tabs[0]:
                recommendation_section()
            with more_tabs[1]:
                load_profile_section()
            with more_tabs[2]:
                rate_drink_section()
            with more_tabs[3]:
                rating_history_section()
            with more_tabs[4]:
                taste_profile_section()


if __name__ == "__main__":
    main()
