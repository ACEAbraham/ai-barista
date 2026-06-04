"""Streamlit web app for AI Barista data collection."""

from datetime import datetime, timezone
from html import escape

import pandas as pd
import streamlit as st

from customization import log_session
from drink_database import list_options, load_drinks
from drink_images import get_drink_image
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
from profile import (
    create_user,
    get_user_history,
    load_ratings,
    load_user,
    load_user_by_id_or_name,
    save_rating,
)
from profile import load_users as load_supabase_users
from recommender import find_similar_drinks, recommend_with_fallback
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

        [data-testid="stImage"] img {
            border-radius: 12px;
            border: 1px solid var(--ai-accent);
            object-fit: cover;
        }

        .profile-status {
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 12px;
            padding: 0.7rem 0.9rem;
            margin-bottom: 1rem;
            color: var(--ai-text);
        }

        .flow-progress {
            color: var(--ai-secondary);
            font-size: 0.9rem;
            font-weight: 700;
            margin: 0.25rem 0 1rem 0;
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
    if "home_view" not in st.session_state:
        st.session_state.home_view = "recommend"
    if "selected_drink_id" not in st.session_state:
        st.session_state.selected_drink_id = None
    if "consumer_matches" not in st.session_state:
        st.session_state.consumer_matches = None
    if "flow_step" not in st.session_state:
        st.session_state.flow_step = 1
    if "profile_mode" not in st.session_state:
        st.session_state.profile_mode = None
    if "today_goal" not in st.session_state:
        st.session_state.today_goal = None
    if "today_temperature" not in st.session_state:
        st.session_state.today_temperature = None
    if "today_context" not in st.session_state:
        st.session_state.today_context = {}
    if "flow_message" not in st.session_state:
        st.session_state.flow_message = None


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


def _match_percentage(score: object) -> int:
    """Convert an open-ended recommendation score into a friendly percentage."""
    try:
        return max(55, min(99, 70 + int(float(score)) * 2))
    except (TypeError, ValueError):
        return 70


def _drink_description(drink: dict[str, object] | pd.Series) -> str:
    """Return a short consumer-friendly drink description."""
    profile = str(drink.get("flavor_profile", "")).strip()
    if profile and profile.lower() != "nan":
        return profile.capitalize()
    return (
        f"A {str(drink.get('temperature', '')).lower()} "
        f"{str(drink.get('base', 'beverage')).lower()} made with "
        f"{str(drink.get('milk', 'your preferred')).lower()} milk."
    )


def render_consumer_cards(matches: pd.DataFrame, limit: int = 3) -> None:
    """Render photo-backed recommendation cards with detail actions."""
    rows = list(matches.head(limit).iterrows())
    columns = st.columns(len(rows)) if rows else []
    for column, (_, drink) in zip(columns, rows):
        with column:
            with st.container(border=True):
                st.image(
                    get_drink_image(drink.to_dict()),
                    width="stretch",
                )
                st.markdown(f"### {drink.get('drink_name', 'Recommended drink')}")
                st.markdown(
                    f"**{_match_percentage(drink.get('recommendation_score'))}% match**"
                )
                st.caption(_drink_description(drink))
                st.write(
                    f"{drink.get('caffeine_level', 'unknown')} caffeine · "
                    f"{drink.get('sweetness_level', 'unknown')} sweetness · "
                    f"{drink.get('calories', 'N/A')} cal"
                )
                if st.button(
                    "View Details",
                    key=f"details_{drink.get('drink_id')}",
                    use_container_width=True,
                ):
                    st.session_state.selected_drink_id = drink.get("drink_id")
                    st.session_state.home_view = "details"
                    st.rerun()


def _save_favorite(user_id: str, drink_id: str) -> tuple[bool, str]:
    """Save a favorite, returning a friendly fallback if unavailable."""
    try:
        insert_row(
            "favorites",
            {
                "user_id": user_id,
                "drink_id": drink_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception:
        return False, "Favorites are not available yet, but this drink will stay in your recent recommendations."
    return True, "Saved to favorites."


def drink_detail_section() -> None:
    """Render a detailed drink view with favorites, ratings, and similar drinks."""
    drink_id = st.session_state.selected_drink_id
    drinks = st.session_state.drinks
    matches = drinks[drinks["drink_id"].astype(str) == str(drink_id)]
    if matches.empty:
        st.warning("That drink is no longer available.")
        return

    drink = matches.iloc[0]
    scored = st.session_state.consumer_matches
    if scored is not None and "drink_id" in scored.columns:
        scored_match = scored[scored["drink_id"].astype(str) == str(drink_id)]
        if not scored_match.empty:
            drink = scored_match.iloc[0]

    if st.button("Back to recommendations"):
        st.session_state.home_view = "recommend"
        st.rerun()

    col1, col2 = st.columns([1, 1.25])
    with col1:
        st.image(get_drink_image(drink.to_dict()), width="stretch")
    with col2:
        st.header(str(drink.get("drink_name", "Drink details")))
        st.markdown(f"**{_match_percentage(drink.get('recommendation_score'))}% match**")
        st.write(_drink_description(drink))
        st.info(
            str(
                drink.get(
                    "recommendation_explanation",
                    "Recommended from your taste profile and selected preferences.",
                )
            )
        )
        st.write(f"**Ingredients:** {drink.get('flavor_profile', 'Not available')}")
        st.write(
            f"**Caffeine:** {drink.get('caffeine_level', 'unknown')}  \n"
            f"**Calories:** {drink.get('calories', 'N/A')}  \n"
            f"**Sweetness:** {drink.get('sweetness_level', 'unknown')}  \n"
            f"**Dietary tags:** {drink.get('dietary_tags', 'none listed')}"
        )

        user = st.session_state.current_user
        if st.button("Save to Favorites", use_container_width=True):
            if not user:
                st.info("Load or create a profile to save favorites.")
            else:
                saved, message = _save_favorite(user["user_id"], str(drink_id))
                (st.success if saved else st.info)(message)

    ratings = load_ratings()
    drink_ratings = (
        ratings[ratings["drink_id"].astype(str) == str(drink_id)]
        if not ratings.empty
        else ratings
    )
    if not drink_ratings.empty:
        st.caption(
            f"User rating: {drink_ratings['rating'].astype(float).mean():.1f}/5 "
            f"from {len(drink_ratings)} rating(s)"
        )

    st.subheader("Rate this drink")
    user = st.session_state.current_user
    if not user:
        st.info("Load or create a profile to rate this drink.")
    else:
        with st.form(f"detail_rating_{drink_id}"):
            rating = st.slider("Rating", 1, 5, 4, key=f"detail_rating_value_{drink_id}")
            would_order_again = st.radio(
                "Would order again?",
                ["Yes", "No"],
                horizontal=True,
                key=f"detail_order_again_{drink_id}",
            )
            feedback_text = st.text_area(
                "Optional feedback",
                key=f"detail_feedback_{drink_id}",
            )
            submitted = st.form_submit_button("Save rating")
        if submitted:
            try:
                save_rating(user["user_id"], str(drink_id), rating, would_order_again == "Yes")
                log_session(
                    user_id=user["user_id"],
                    sleep_hours="",
                    stress_level="",
                    goal="drink detail rating",
                    weather="",
                    drink_id=str(drink_id),
                    rating=rating,
                )
                if feedback_text.strip():
                    try:
                        insert_row(
                            "drink_feedback",
                            {
                                "user_id": user["user_id"],
                                "drink_id": str(drink_id),
                                "feedback_text": feedback_text.strip(),
                                "created_at": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    except Exception:
                        st.info("Your rating was saved. Text feedback storage is not available yet.")
                st.success("Rating saved.")
            except RuntimeError as error:
                st.error(str(error))

    st.subheader("Similar drinks")
    similar = find_similar_drinks(drinks, drink, limit=3)
    render_consumer_cards(similar, limit=3)


def consumer_recommendation_section() -> None:
    """Render the profile-aware consumer recommendation flow."""
    user = st.session_state.current_user
    st.subheader("Get Recommendation")
    st.caption(
        "Your taste profile shapes every result."
        if user
        else "Create or load a profile for personalized results."
    )

    with st.form("consumer_recommendation_form"):
        col1, col2 = st.columns(2)
        with col1:
            goal = st.selectbox("Goal", ["Energy", "Focus", "Comfort", "Workout", "Treat"])
        with col2:
            temperature = st.selectbox(
                "Temperature preference",
                ["No Preference", "Iced", "Hot"],
            )
        with st.expander("Optional refinement", expanded=False):
            milk = st.selectbox(
                "Milk",
                ["Any", "whole", "2%", "oat", "almond", "soy", "coconut", "none"],
            )
            caffeine = st.selectbox("Caffeine", ["Any", "none", "low", "medium", "high"])
            sweetness = st.selectbox(
                "Sweetness",
                ["Any", "unsweetened", "light", "classic", "extra"],
            )
            likes = st.text_input("Likes")
            dislikes = st.text_input("Dislikes")
        submitted = st.form_submit_button(
            "Get Recommendation",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        matches, _, _ = recommend_with_fallback(
            drinks=st.session_state.drinks,
            temperature=None if temperature == "No Preference" else temperature.lower(),
            milk=None if milk == "Any" else milk,
            caffeine_level=None if caffeine == "Any" else caffeine,
            sweetness_level=None if sweetness == "Any" else sweetness,
            user=user,
            user_history=get_user_history(user["user_id"]) if user else None,
            ingredient_preferences=get_ingredient_preferences(user["user_id"]) if user else None,
            drink_recipes=load_recipes(),
        )
        query = f"{likes} {dislikes}".strip().lower()
        if query:
            matches["recommendation_explanation"] = matches[
                "recommendation_explanation"
            ].astype(str) + f"; considered your notes: {query}"
        st.session_state.consumer_matches = matches
        if not matches.empty and supabase_is_configured():
            try:
                log_session(
                    user_id=user["user_id"] if user else "guest",
                    sleep_hours="",
                    stress_level="",
                    goal=goal.lower(),
                    weather="",
                    drink_id=str(matches.iloc[0]["drink_id"]),
                    rating="",
                )
            except RuntimeError as error:
                st.error(str(error))

    matches = st.session_state.consumer_matches
    if matches is not None and not matches.empty:
        render_consumer_cards(matches, limit=3)


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


def _set_flow_step(step: int) -> None:
    """Move the guided experience to a new step."""
    st.session_state.flow_step = step
    st.rerun()


def guided_welcome_step() -> None:
    """Render the welcome step."""
    st.markdown("## Welcome")
    st.write("Create a taste profile, tell us what today needs, and meet your drink.")
    if st.button("Start", type="primary", use_container_width=True):
        _set_flow_step(2)


def guided_profile_step() -> None:
    """Render profile creation or loading as one guided step."""
    st.markdown("## Create or Load Profile")
    st.write("Do you already have a profile?")
    create_col, load_col = st.columns(2)
    with create_col:
        if st.button("Create New Profile", use_container_width=True):
            st.session_state.profile_mode = "create"
            st.rerun()
    with load_col:
        if st.button("Load Existing Profile", use_container_width=True):
            st.session_state.profile_mode = "load"
            st.rerun()

    if st.session_state.profile_mode == "create":
        with st.form("guided_create_profile"):
            name = st.text_input("Name")
            favorite_milk = st.selectbox(
                "Favorite milk",
                list_options(st.session_state.drinks, "milk"),
            )
            caffeine_tolerance = st.selectbox(
                "Caffeine tolerance",
                list_options(st.session_state.drinks, "caffeine_level"),
            )
            preferred_sweetness = st.selectbox(
                "Sweetness preference",
                list_options(st.session_state.drinks, "sweetness_level"),
            )
            favorite_flavors = st.text_input(
                "Favorite flavors",
                placeholder="Vanilla, caramel, cinnamon...",
            )
            disliked_flavors = st.text_input(
                "Disliked flavors",
                placeholder="Bitter chocolate, coconut...",
            )
            submitted = st.form_submit_button(
                "Create Profile",
                type="primary",
                use_container_width=True,
            )
        if submitted:
            if not name.strip():
                st.error("Please enter your name.")
                return
            try:
                user = create_user(
                    name=name.strip(),
                    favorite_milk=favorite_milk,
                    favorite_temperature="no preference",
                    caffeine_tolerance=caffeine_tolerance,
                    preferred_sweetness=preferred_sweetness,
                )
            except RuntimeError as error:
                st.error(str(error))
                return
            st.session_state.current_user = user
            st.session_state.profile_flavors = {
                "favorite": favorite_flavors.strip(),
                "disliked": disliked_flavors.strip(),
            }
            refresh_data()
            st.session_state.flow_message = "Profile created successfully."
            st.session_state.flow_step = 3
            st.rerun()

    elif st.session_state.profile_mode == "load":
        with st.form("guided_load_profile"):
            lookup = st.text_input("Profile ID or name")
            submitted = st.form_submit_button(
                "Load Profile",
                type="primary",
                use_container_width=True,
            )
        if submitted:
            user = load_user_by_id_or_name(lookup)
            if user is None:
                st.error("No profile matched that ID or name.")
                return
            st.session_state.current_user = user
            st.session_state.flow_message = f"Welcome back, {user['name']}."
            st.session_state.flow_step = 3
            st.rerun()


def guided_context_step() -> None:
    """Render today's visual context choices."""
    st.markdown("## Today's Context")
    st.write("What are you looking for today?")
    goals = [
        ("⚡ Energy", "energy"),
        ("🎯 Focus", "focus"),
        ("😌 Comfort", "comfort"),
        ("🏋️ Workout", "workout"),
        ("🍰 Treat", "treat"),
    ]
    goal_columns = st.columns(len(goals))
    for column, (label, value) in zip(goal_columns, goals):
        with column:
            button_label = f"✓ {label}" if st.session_state.today_goal == value else label
            if st.button(button_label, key=f"guided_goal_{value}", use_container_width=True):
                st.session_state.today_goal = value
                st.rerun()

    st.write("Hot or iced?")
    temperatures = [
        ("🧊 Iced", "iced"),
        ("🔥 Hot", "hot"),
        ("🤷 No Preference", None),
    ]
    temperature_columns = st.columns(3)
    for column, (label, value) in zip(temperature_columns, temperatures):
        with column:
            selected = st.session_state.today_temperature == value and (
                value is not None or "temperature_selected" in st.session_state
            )
            button_label = f"✓ {label}" if selected else label
            if st.button(
                button_label,
                key=f"guided_temperature_{value}",
                use_container_width=True,
            ):
                st.session_state.today_temperature = value
                st.session_state.temperature_selected = True
                st.rerun()

    with st.expander("Fine-tune recommendation", expanded=False):
        weather = st.selectbox(
            "Weather",
            ["Not specified", "Sunny", "Cloudy", "Rainy", "Snowy", "Hot", "Cold"],
            key="guided_weather",
        )
        sleep_hours = st.number_input(
            "Sleep hours",
            min_value=0.0,
            max_value=16.0,
            value=7.0,
            step=0.5,
            key="guided_sleep",
        )
        stress_level = st.selectbox(
            "Stress level",
            ["Not specified", "Low", "Medium", "High"],
            key="guided_stress",
        )
        love_today = st.text_input("Anything you love today", key="guided_love")
        avoid_today = st.text_input("Anything you want to avoid", key="guided_avoid")

    ready = (
        st.session_state.today_goal is not None
        and "temperature_selected" in st.session_state
    )
    if st.button(
        "Get My Drink",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    ):
        user = st.session_state.current_user
        matches, _, _ = recommend_with_fallback(
            drinks=st.session_state.drinks,
            temperature=st.session_state.today_temperature,
            user=user,
            user_history=get_user_history(user["user_id"]) if user else None,
            ingredient_preferences=get_ingredient_preferences(user["user_id"]) if user else None,
            drink_recipes=load_recipes(),
        )
        profile_flavors = st.session_state.get("profile_flavors", {})
        notes = " ".join(
            value
            for value in [
                str(profile_flavors.get("favorite", "")).strip(),
                str(profile_flavors.get("disliked", "")).strip(),
                love_today.strip(),
                avoid_today.strip(),
            ]
            if value
        )
        if notes:
            matches["recommendation_explanation"] = (
                matches["recommendation_explanation"].astype(str)
                + f"; considered today's notes: {notes}"
            )
        st.session_state.consumer_matches = matches
        st.session_state.today_context = {
            "sleep_hours": sleep_hours,
            "stress_level": stress_level.lower(),
            "goal": st.session_state.today_goal,
            "weather": weather.lower(),
        }
        if not matches.empty and supabase_is_configured():
            try:
                log_session(
                    user_id=user["user_id"] if user else "guest",
                    drink_id=str(matches.iloc[0]["drink_id"]),
                    rating="",
                    **st.session_state.today_context,
                )
            except RuntimeError as error:
                st.error(str(error))
                return
        _set_flow_step(4)


def render_guided_recommendation_cards(matches: pd.DataFrame, limit: int = 3) -> None:
    """Render recommendation cards that advance into the guided detail step."""
    rows = list(matches.head(limit).iterrows())
    for column, (_, drink) in zip(st.columns(len(rows)), rows):
        with column:
            with st.container(border=True):
                st.image(get_drink_image(drink.to_dict()), width="stretch")
                st.markdown(f"### {drink.get('drink_name', 'Recommended drink')}")
                st.markdown(f"**{_match_percentage(drink.get('recommendation_score'))}% match**")
                st.caption(str(drink.get("recommendation_explanation", _drink_description(drink))))
                if st.button(
                    "View Details",
                    key=f"guided_details_{drink.get('drink_id')}",
                    use_container_width=True,
                ):
                    st.session_state.selected_drink_id = drink.get("drink_id")
                    _set_flow_step(5)


def guided_recommendation_step() -> None:
    """Render up to three guided recommendations."""
    st.markdown("## Your Recommendations")
    matches = st.session_state.consumer_matches
    if matches is None or matches.empty:
        st.warning("No recommendations are ready yet.")
        if st.button("Back to Today's Context"):
            _set_flow_step(3)
        return
    render_guided_recommendation_cards(matches)


def _selected_guided_drink() -> pd.Series | None:
    """Return the selected drink, including its recommendation score."""
    drink_id = st.session_state.selected_drink_id
    scored = st.session_state.consumer_matches
    if scored is not None:
        matches = scored[scored["drink_id"].astype(str) == str(drink_id)]
        if not matches.empty:
            return matches.iloc[0]
    matches = st.session_state.drinks[
        st.session_state.drinks["drink_id"].astype(str) == str(drink_id)
    ]
    return None if matches.empty else matches.iloc[0]


def guided_detail_step() -> None:
    """Render guided drink details without showing the rating form yet."""
    drink = _selected_guided_drink()
    if drink is None:
        st.warning("That drink is no longer available.")
        return

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.image(get_drink_image(drink.to_dict()), width="stretch")
    with col2:
        st.header(str(drink.get("drink_name", "Drink details")))
        st.info(str(drink.get("recommendation_explanation", "Recommended for your taste.")))
        st.write(f"**Ingredients:** {drink.get('flavor_profile', 'Not available')}")
        st.write(
            f"**Caffeine:** {drink.get('caffeine_level', 'unknown')}  \n"
            f"**Calories:** {drink.get('calories', 'N/A')}  \n"
            f"**Sweetness:** {drink.get('sweetness_level', 'unknown')}  \n"
            f"**Dietary tags:** {drink.get('dietary_tags', 'none listed')}"
        )
        rate_col, another_col = st.columns(2)
        with rate_col:
            if st.button("Rate this drink", type="primary", use_container_width=True):
                _set_flow_step(6)
        with another_col:
            if st.button("Recommend another", use_container_width=True):
                st.session_state.consumer_matches = None
                st.session_state.selected_drink_id = None
                _set_flow_step(3)

    st.subheader("Similar drinks")
    similar = find_similar_drinks(st.session_state.drinks, drink, limit=3)
    render_guided_recommendation_cards(similar, limit=3)


def guided_rating_step() -> None:
    """Collect feedback and complete the guided journey."""
    drink = _selected_guided_drink()
    user = st.session_state.current_user
    if drink is None or not user:
        st.warning("Load a profile and select a drink before rating.")
        return

    st.markdown("## How good was this recommendation?")
    with st.form("guided_rating_form"):
        rating = st.slider("Rating", 1, 5, 4)
        would_order_again = st.radio("Would order again?", ["Yes", "No"], horizontal=True)
        feedback_text = st.text_area("Optional feedback text")
        submitted = st.form_submit_button(
            "Save Feedback",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        try:
            save_rating(
                user["user_id"],
                str(drink["drink_id"]),
                rating,
                would_order_again == "Yes",
            )
            log_session(
                user_id=user["user_id"],
                drink_id=str(drink["drink_id"]),
                rating=rating,
                **st.session_state.today_context,
            )
            if feedback_text.strip():
                try:
                    insert_row(
                        "drink_feedback",
                        {
                            "user_id": user["user_id"],
                            "drink_id": str(drink["drink_id"]),
                            "feedback_text": feedback_text.strip(),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                except Exception:
                    pass
        except RuntimeError as error:
            st.error(str(error))
            return
        st.session_state.feedback_saved = True

    if st.session_state.get("feedback_saved"):
        st.success("Thanks — your feedback helps AI Barista learn your taste.")
        another_col, profile_col = st.columns(2)
        with another_col:
            if st.button("Get another recommendation", use_container_width=True):
                st.session_state.feedback_saved = False
                st.session_state.consumer_matches = None
                st.session_state.selected_drink_id = None
                _set_flow_step(3)
        with profile_col:
            if st.button("View profile", use_container_width=True):
                st.session_state.feedback_saved = False
                _set_flow_step(7)


def guided_profile_summary_step() -> None:
    """Show the loaded profile after the journey."""
    user = st.session_state.current_user
    st.markdown("## Your Profile")
    if not user:
        st.info("No profile is loaded.")
        return
    st.write(f"**Name:** {user.get('name')}")
    st.write(f"**Favorite milk:** {user.get('favorite_milk')}")
    st.write(f"**Caffeine tolerance:** {user.get('caffeine_tolerance')}")
    st.write(f"**Preferred sweetness:** {user.get('preferred_sweetness')}")
    with st.expander("Taste profile", expanded=False):
        taste_profile_section()
    if st.button("Get another recommendation", type="primary", use_container_width=True):
        _set_flow_step(3)


def guided_flow() -> None:
    """Render exactly one step of the main onboarding and recommendation flow."""
    step = int(st.session_state.flow_step)
    st.markdown(f'<div class="flow-progress">Step {min(step, 6)} of 6</div>', unsafe_allow_html=True)
    if st.session_state.flow_message:
        st.success(st.session_state.flow_message)
        st.session_state.flow_message = None
    if step == 1:
        guided_welcome_step()
    elif step == 2:
        guided_profile_step()
    elif step == 3:
        guided_context_step()
    elif step == 4:
        guided_recommendation_step()
    elif step == 5:
        guided_detail_step()
    elif step == 6:
        guided_rating_step()
    else:
        guided_profile_summary_step()


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="AI Barista", layout="wide")
    apply_theme()
    initialize_state()

    st.markdown(
        """
        <section class="ai-title-section">
            <h1>☕ AI Barista</h1>
            <p>Your personal drink recommendation engine.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if not supabase_is_configured():
        st.warning(
            "Supabase credentials are not configured locally. Static catalog browsing works, "
            "but recommendations and feedback require SUPABASE_URL and SUPABASE_KEY."
        )

    guided_flow()

    with st.expander("Advanced Tools", expanded=False):
        advanced = st.tabs(
            [
                "Create Custom Drink",
                "Add Ingredient",
                "Ingredient List",
                "Admin / Debug",
            ]
        )
        with advanced[0]:
            custom_drink_section()
        with advanced[1]:
            add_ingredient_section()
        with advanced[2]:
            ingredient_list_section()
        with advanced[3]:
            admin_tabs = st.tabs(
                [
                    "OpenAI beta",
                    "Rule-based recommendations",
                    "Rate drink",
                    "Rating history",
                    "Taste profile",
                ]
            )
            with admin_tabs[0]:
                ai_recommendation_section()
            with admin_tabs[1]:
                recommendation_section()
            with admin_tabs[2]:
                rate_drink_section()
            with admin_tabs[3]:
                rating_history_section()
            with admin_tabs[4]:
                taste_profile_section()


if __name__ == "__main__":
    main()
