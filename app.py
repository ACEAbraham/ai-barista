"""Streamlit web app for AI Barista data collection."""

from datetime import datetime, timezone
from html import escape
import re

import pandas as pd
import streamlit as st

from analytics import track_recommendation_event
from customization import log_recommendation_session, log_session
from drink_database import list_options, load_drinks
from drink_images import get_drink_image
from favorites import get_user_favorites, remove_favorite, save_favorite
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
from openai_client import build_user_memory_summary, openai_is_configured
from openai_recommender import fallback_recommendation, select_best_candidate
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
            background:
                linear-gradient(90deg, rgba(0, 0, 0, 0.62), rgba(0, 0, 0, 0.45)),
                url("https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?auto=format&fit=crop&w=1600&q=80");
            background-size: cover;
            background-position: center;
            border: 1px solid var(--ai-accent);
            border-radius: 10px;
            padding: 4.2rem 2.8rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 18px 42px rgba(74, 38, 8, 0.18);
        }

        .ai-title-section h1 {
            color: #FFFFFF !important;
            font-size: 4.6rem;
            line-height: 1;
            margin: 0 0 0.6rem 0;
            letter-spacing: 0;
            font-weight: 800;
            text-shadow: 0 2px 12px rgba(0, 0, 0, 0.5);
        }

        .ai-title-section p {
            color: rgba(255, 255, 255, 0.9) !important;
            font-size: 1.15rem;
            margin: 0;
            max-width: 560px;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.38);
        }

        .home-section-title {
            color: var(--ai-text);
            font-size: 1.35rem;
            font-weight: 850;
            margin: 1.4rem 0 0.4rem 0;
        }

        .rail-card {
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 8px;
            overflow: hidden;
            min-height: 292px;
            box-shadow: 0 10px 24px rgba(74, 38, 8, 0.10);
            transition: transform 140ms ease, box-shadow 140ms ease;
        }

        .rail-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 30px rgba(74, 38, 8, 0.16);
        }

        .rail-card-title {
            color: var(--ai-text);
            font-weight: 800;
            font-size: 0.96rem;
            height: 3rem;
            line-height: 1.2;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        .rail-card-meta {
            color: var(--ai-secondary);
            font-size: 0.84rem;
            height: 2.35rem;
            line-height: 1.25;
            overflow: hidden;
            margin-top: 0.35rem;
        }

        .rail-card-image {
            width: 100%;
            height: 150px;
            object-fit: cover;
            display: block;
        }

        .rail-card-body {
            padding: 0.75rem;
            min-height: 7.2rem;
        }

        .recommendation-feature {
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 18px 42px rgba(74, 38, 8, 0.14);
        }

        .recommendation-feature-image {
            width: 100%;
            height: 320px;
            object-fit: cover;
            border-radius: 8px;
            border: 1px solid var(--ai-accent);
        }

        .secondary-card {
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 8px;
            padding: 0.75rem;
            min-height: 250px;
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

        .experimental-card {
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 8px;
            box-shadow: 0 0 16px rgba(118, 86, 58, 0.20);
            padding: 1rem;
            margin-bottom: 0.75rem;
        }

        .experimental-badge {
            display: inline-block;
            background: var(--ai-button);
            color: #FFFFFF !important;
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 700;
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

        .profile-card {
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 18px;
            padding: 1.4rem;
            margin: 0.75rem 0 1.25rem 0;
            box-shadow: 0 10px 26px rgba(74, 38, 8, 0.07);
        }

        .profile-card-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .profile-avatar {
            width: 64px;
            height: 64px;
            border-radius: 50%;
            background: var(--ai-button);
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            font-weight: 800;
            flex: 0 0 64px;
        }

        .profile-name {
            color: var(--ai-text);
            font-size: 1.45rem;
            font-weight: 800;
        }

        .profile-badges,
        .taste-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }

        .profile-badge,
        .taste-chip {
            background: var(--ai-section);
            border: 1px solid var(--ai-accent);
            border-radius: 999px;
            color: var(--ai-text);
            padding: 0.38rem 0.7rem;
            font-size: 0.88rem;
            font-weight: 700;
        }

        .taste-section {
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 14px;
            padding: 1rem;
            min-height: 130px;
        }

        .taste-section-title {
            color: var(--ai-text);
            font-weight: 800;
            margin-bottom: 0.7rem;
        }

        .taste-empty {
            color: var(--ai-secondary);
            font-size: 0.92rem;
        }

        div[data-testid="stExpander"] {
            background: var(--ai-input);
            border: 1px solid var(--ai-button);
            border-radius: 14px;
            margin: 1rem 0;
            overflow: hidden;
        }

        div[data-testid="stExpander"] summary {
            background: var(--ai-section);
            padding: 0.85rem 1rem;
            color: var(--ai-text);
            font-weight: 800;
        }

        div[data-testid="stExpander"] summary:hover {
            background: var(--ai-accent);
        }

        div[data-testid="stExpander"] details > div {
            background: var(--ai-input);
            padding: 0.4rem 0.9rem 0.9rem 0.9rem;
        }

        [data-testid="stDataFrame"] {
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 12px;
            padding: 0.35rem;
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
    if "ai_memory_summary" not in st.session_state:
        st.session_state.ai_memory_summary = None
    if "ai_fallback_matches" not in st.session_state:
        st.session_state.ai_fallback_matches = None
    if "ai_candidate_drinks" not in st.session_state:
        st.session_state.ai_candidate_drinks = None
    if "ai_selected_drink_id" not in st.session_state:
        st.session_state.ai_selected_drink_id = None
    if "ai_cache_used" not in st.session_state:
        st.session_state.ai_cache_used = False
    if "home_view" not in st.session_state:
        st.session_state.home_view = "recommend"
    if "selected_drink_id" not in st.session_state:
        st.session_state.selected_drink_id = None
    if "selected_drink" not in st.session_state:
        st.session_state.selected_drink = None
    if "consumer_matches" not in st.session_state:
        st.session_state.consumer_matches = None
    if "recommendation_results" not in st.session_state:
        st.session_state.recommendation_results = None
    if st.session_state.recommendation_results is None and st.session_state.consumer_matches is not None:
        st.session_state.recommendation_results = st.session_state.consumer_matches
    if st.session_state.selected_drink is None and st.session_state.selected_drink_id is not None:
        st.session_state.selected_drink = {"drink_id": st.session_state.selected_drink_id}
    if "flow_step" not in st.session_state:
        st.session_state.flow_step = 1
    if "current_step" not in st.session_state:
        st.session_state.current_step = st.session_state.flow_step
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "home"
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


def _initials(name: object) -> str:
    """Return up to two initials for a profile avatar."""
    parts = [part for part in str(name).strip().split() if part]
    return "".join(part[0].upper() for part in parts[:2]) or "AB"


def _chip_html(label: object) -> str:
    """Return one escaped taste-profile chip."""
    return f'<span class="taste-chip">{safe_text(label)}</span>'


def key_slug(value: object) -> str:
    """Return a stable key-safe slug for repeated Streamlit widgets."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_").lower() or "item"


def _ingredient_chips(
    rows: pd.DataFrame,
    value_column: str | None = None,
    limit: int = 8,
) -> str:
    """Render ingredient rows as compact HTML chips."""
    if rows.empty:
        return '<div class="taste-empty">Rate a few drinks to build your taste profile.</div>'

    chips = []
    for _, row in rows.head(limit).iterrows():
        label = str(row.get("ingredient_name", "Ingredient"))
        if value_column and value_column in row and not pd.isna(row[value_column]):
            label = f"{label} · {float(row[value_column]):g}"
        chips.append(_chip_html(label))
    return f'<div class="taste-chips">{"".join(chips)}</div>'


def render_profile_card(user: dict[str, object]) -> None:
    """Render profile identity and preference badges."""
    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-card-header">
                <div class="profile-avatar">{_initials(user.get("name", ""))}</div>
                <div>
                    <div class="profile-name">{safe_text(user.get("name", "Taste Explorer"))}</div>
                    <div class="ai-card-meta">{safe_text(user.get("user_id", ""))}</div>
                </div>
            </div>
            <div class="profile-badges">
                <span class="profile-badge">Favorite Milk: {safe_text(user.get("favorite_milk", "Not set"))}</span>
                <span class="profile-badge">Caffeine: {safe_text(user.get("caffeine_tolerance", "Not set"))}</span>
                <span class="profile-badge">Sweetness: {safe_text(user.get("preferred_sweetness", "Not set"))}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_taste_profile_cards(user_id: str) -> None:
    """Render learned ingredient preferences without raw dataframes."""
    profile = get_taste_profile(user_id)
    col1, col2, col3 = st.columns(3)
    sections = [
        ("Favorite ingredients", profile["favorite"], "preference_score"),
        ("Least favorite ingredients", profile["least_favorite"], "preference_score"),
        ("Most common ingredients", profile["most_common"], "times_seen"),
    ]
    for column, (title, rows, score_column) in zip((col1, col2, col3), sections):
        with column:
            st.markdown(
                f"""
                <div class="taste-section">
                    <div class="taste-section-title">{title}</div>
                    {_ingredient_chips(rows, score_column)}
                </div>
                """,
                unsafe_allow_html=True,
            )


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


def save_favorite_action(
    user_id: str,
    drink_id: str,
    drink_name: str,
) -> tuple[bool, str]:
    """Save a favorite and return a friendly duplicate-aware message."""
    try:
        saved = save_favorite(user_id, drink_id, drink_name)
    except Exception as error:
        return False, f"Could not save favorite: {error}"
    if saved is None:
        return False, "Already in favorites."
    return True, "Saved to favorites."


def render_favorites_section(user_id: str) -> None:
    """Render a user's favorite drinks as compact cards."""
    st.markdown("### My Favorites")
    try:
        favorites = get_user_favorites(user_id)
    except Exception as error:
        st.info(f"Favorites are unavailable right now: {error}")
        return

    if favorites.empty:
        st.info("Save a drink you love and it will appear here.")
        return

    rows = list(favorites.head(6).iterrows())
    for start in range(0, len(rows), 3):
        row_items = rows[start : start + 3]
        for offset, (column, (_, favorite)) in enumerate(zip(st.columns(len(row_items)), row_items)):
            card_index = start + offset
            drink_id = favorite.get("drink_id")
            drink_matches = st.session_state.drinks[
                st.session_state.drinks["drink_id"].astype(str)
                == str(drink_id)
            ]
            drink = (
                drink_matches.iloc[0].to_dict()
                if not drink_matches.empty
                else {
                    "drink_id": favorite.get("drink_id"),
                    "drink_name": favorite.get("drink_name"),
                }
            )
            with column:
                with st.container(border=True):
                    st.image(get_drink_image(drink), width="stretch")
                    st.markdown(f"**{safe_text(favorite.get('drink_name', 'Favorite drink'))}**")
                    if st.button(
                        "View Details",
                        key=f"favorites_profile_{key_slug(drink_id)}_{card_index}_view_details",
                        use_container_width=True,
                    ):
                        st.session_state.selected_drink_id = drink_id
                        st.session_state.selected_drink = drink
                        _set_flow_step(5)
                    if st.button(
                        "Remove",
                        key=f"favorites_profile_{key_slug(drink_id)}_{card_index}_remove",
                        use_container_width=True,
                    ):
                        try:
                            remove_favorite(user_id, str(drink_id))
                        except Exception as error:
                            st.error(f"Could not remove favorite: {error}")
                        else:
                            st.success("Removed from favorites.")
                            st.rerun()


def _open_home_drink(drink_id: object) -> None:
    """Open a drink from a homepage rail in the shared detail view."""
    st.session_state.selected_drink_id = drink_id
    st.session_state.selected_drink = {"drink_id": drink_id}
    st.session_state.home_view = "details"
    st.rerun()


def _rail_meta(drink: pd.Series, score_label: str | None = None) -> str:
    """Build compact rail-card metadata."""
    if score_label:
        return score_label
    if "recommendation_score" in drink and not pd.isna(drink.get("recommendation_score")):
        return f"{_match_percentage(drink.get('recommendation_score'))}% match"
    return (
        f"{drink.get('temperature', 'drink')} · "
        f"{drink.get('caffeine_level', 'unknown')} caffeine · "
        f"{drink.get('calories', 'N/A')} cal"
    )


def render_drink_rail(
    title: str,
    drinks: pd.DataFrame,
    key_prefix: str,
    limit: int = 5,
    empty_text: str = "Nothing here yet.",
    score_column: str | None = None,
) -> None:
    """Render a horizontal row of drink cards."""
    section_slug = key_slug(title)
    st.markdown(f'<div class="home-section-title">{safe_text(title)}</div>', unsafe_allow_html=True)
    if drinks is None or drinks.empty:
        st.caption(empty_text)
        return

    rows = list(drinks.head(limit).iterrows())
    columns = st.columns(len(rows))
    for index, (column, (_, drink)) in enumerate(zip(columns, rows)):
        drink_dict = drink.to_dict()
        drink_id = drink.get("drink_id")
        score_label = None
        if score_column and score_column in drink and not pd.isna(drink.get(score_column)):
            score_label = f"{float(drink.get(score_column)):g}/5 rating"
        with column:
            st.markdown(
                f"""
                <div class="rail-card">
                    <img class="rail-card-image" src="{safe_text(get_drink_image(drink_dict))}" alt="{safe_text(drink.get("drink_name", "Drink"))}">
                    <div class="rail-card-body">
                        <div class="rail-card-title">{safe_text(drink.get("drink_name", "Drink"))}</div>
                        <div class="rail-card-meta">{safe_text(_rail_meta(drink, score_label))}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "View Details",
                key=f"{key_prefix}_{section_slug}_{key_slug(drink_id)}_{index}_view_details",
                use_container_width=True,
            ):
                _open_home_drink(drink_id)


def homepage_rail_data() -> dict[str, pd.DataFrame]:
    """Build homepage rail datasets from local catalog and loaded user memory."""
    drinks = st.session_state.drinks
    user = st.session_state.current_user
    if user:
        recommended, _, _ = recommend_with_fallback(
            drinks=drinks,
            temperature=(
                None
                if str(user.get("favorite_temperature", "")).lower()
                in {"", "any", "no preference"}
                else str(user.get("favorite_temperature", "")).lower()
            ),
            user=user,
            user_history=get_user_history(user["user_id"]),
            ingredient_preferences=get_ingredient_preferences(user["user_id"]),
            drink_recipes=load_recipes(),
            context={
                "goal": "comfort",
                "temperature_preference": user.get("favorite_temperature", "no preference"),
                "caffeine_preference": user.get("caffeine_tolerance", "any"),
                "sweetness_preference": user.get("preferred_sweetness", "any"),
            },
        )
        recommended = recommended.head(8)
    else:
        recommended = drinks.head(5).copy()

    ratings = load_ratings()
    if ratings.empty:
        popular = drinks.sort_values(["calories", "price"], ascending=[True, True]).head(8)
        recently_rated = pd.DataFrame(columns=drinks.columns)
    else:
        rating_values = ratings.copy()
        rating_values["rating"] = pd.to_numeric(rating_values["rating"], errors="coerce")
        popular_scores = (
            rating_values.groupby("drink_id")["rating"]
            .mean()
            .reset_index(name="avg_rating")
            .sort_values("avg_rating", ascending=False)
        )
        popular = popular_scores.merge(drinks, on="drink_id", how="left")
        if user:
            user_ratings = rating_values[
                rating_values["user_id"].astype(str).str.lower()
                == str(user["user_id"]).lower()
            ].tail(8)
        else:
            user_ratings = rating_values.tail(8)
        recently_rated = user_ratings.merge(drinks, on="drink_id", how="left")

    if user:
        try:
            favorites = get_user_favorites(user["user_id"])
            favorite_drinks = favorites.merge(drinks, on="drink_id", how="left")
            if "drink_name_x" in favorite_drinks.columns:
                favorite_drinks["drink_name"] = favorite_drinks["drink_name_y"].fillna(
                    favorite_drinks["drink_name_x"]
                )
        except Exception:
            favorite_drinks = pd.DataFrame(columns=drinks.columns)
    else:
        favorite_drinks = pd.DataFrame(columns=drinks.columns)

    custom_creations = drinks[drinks["drink_id"].astype(str).str.startswith("CUS-")]
    if custom_creations.empty:
        custom_creations = drinks[drinks["base"].astype(str).str.lower().isin(["mocha", "frappuccino", "matcha latte"])].head(8)

    return {
        "recommended": recommended,
        "popular": popular,
        "recent": recently_rated,
        "favorites": favorite_drinks,
        "custom": custom_creations,
    }


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
                st.info("Load or create a profile first.")
            else:
                saved, message = save_favorite_action(
                    user["user_id"],
                    str(drink_id),
                    str(drink.get("drink_name", "Favorite drink")),
                )
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
        st.session_state.recommendation_results = matches
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
    """Render one memory-informed drink recommendation."""
    ingredients = ", ".join(
        safe_text(item) for item in recommendation.get("ingredients", [])
    )
    memory_used = ", ".join(
        safe_text(item) for item in recommendation.get("memory_used", [])
    )
    confidence = float(recommendation.get("confidence_score", 0) or 0)
    st.markdown(
        f"""
        <div class="ai-card">
            <div class="ai-card-title">{safe_text(recommendation.get("drink_name"))}</div>
            <div class="ai-card-meta">
                {safe_text(recommendation.get("size"))} ·
                {safe_text(recommendation.get("temperature"))} ·
                {safe_text(recommendation.get("caffeine_level"))} caffeine ·
                {confidence:.0%} confidence
            </div>
            <p><strong>Ingredients:</strong> {ingredients}</p>
            <div class="ai-explanation">
                {safe_text(recommendation.get("why_recommended"))}
            </div>
            <p><strong>Memory used:</strong> {memory_used or "Current profile and context"}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _save_ai_recommendation(
    user_id: str,
    context: dict[str, object],
    memory_summary: str,
    recommendation: dict[str, object],
) -> object | None:
    """Save generated recommendation context to Supabase."""
    if not supabase_is_configured():
        save_warning()
        return None

    row = {
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "context": context,
        "memory_summary": memory_summary,
        "recommendation_json": recommendation,
        "rating": None,
        "would_order_again": None,
        "feedback_text": None,
    }
    saved = insert_row("ai_recommendations", row)
    return saved.get("id")


def _save_ai_feedback(
    recommendation_id: object | None,
    user_id: str,
    context: dict[str, object],
    memory_summary: str,
    recommendation: dict[str, object],
    values: dict[str, object],
) -> object | None:
    """Update a recommendation or insert it with feedback when needed."""
    if recommendation_id is not None:
        update_rows("ai_recommendations", values, {"id": recommendation_id})
        return recommendation_id
    row = {
        "user_id": user_id,
        "context": context,
        "memory_summary": memory_summary,
        **values,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recommendation_json": recommendation,
    }
    return insert_row("ai_recommendations", row).get("id")


def ai_recommendation_section() -> None:
    """Deprecated compatibility wrapper for the single recommendation flow."""
    st.markdown("## Recommendation")
    st.caption("Uses your saved taste profile, ratings, favorites, and recent context.")
    user = st.session_state.current_user
    if not user:
        st.info("Load or create a profile first.")
        create_col, load_col = st.columns(2)
        with create_col:
            if st.button("Create Profile", key="ai_create_profile", use_container_width=True):
                st.session_state.profile_mode = "create"
                _set_flow_step(2)
        with load_col:
            if st.button("Load Profile", key="ai_load_profile", use_container_width=True):
                st.session_state.profile_mode = "load"
                _set_flow_step(2)
        return

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
            caffeine_preference = st.selectbox(
                "Caffeine Preference",
                ["any", "none", "low", "medium", "high"],
            )
            sweetness_preference = st.selectbox(
                "Sweetness Preference",
                ["any", "unsweetened", "light", "classic", "extra"],
            )
            dietary_restrictions = st.text_input("Dietary Restrictions")
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
        context = {
            "goal": goal,
            "sleep_hours": sleep_hours,
            "stress_level": stress_level,
            "weather": weather,
            "temperature_preference": preferred_temperature,
            "caffeine_preference": caffeine_preference,
            "sweetness_preference": sweetness_preference,
            "likes": things_you_love.strip(),
            "dislikes": things_you_hate.strip(),
            "dietary_restrictions": dietary_restrictions.strip(),
        }
        if not openai_is_configured():
            st.info("Recommendation generated using profile matching.")
            matches, _, _ = recommend_with_fallback(
                drinks=st.session_state.drinks,
                temperature=None if preferred_temperature == "no preference" else preferred_temperature,
                caffeine_level=None if caffeine_preference == "any" else caffeine_preference,
                sweetness_level=None if sweetness_preference == "any" else sweetness_preference,
                dietary_tag=dietary_restrictions.strip() or None,
                user=user,
                user_history=get_user_history(user["user_id"]),
                ingredient_preferences=get_ingredient_preferences(user["user_id"]),
                drink_recipes=load_recipes(),
                context=context,
            )
            st.session_state.ai_fallback_matches = matches
            st.session_state.ai_recommendation = None
        else:
            try:
                with st.spinner("Reasoning over your taste memory..."):
                    memory_summary = build_user_memory_summary(user["user_id"])
                    recommendation = generate_ai_recommendation(
                        user["user_id"],
                        context,
                        st.session_state.drinks,
                    )
                try:
                    recommendation_id = _save_ai_recommendation(
                        user["user_id"],
                        context,
                        memory_summary,
                        recommendation,
                    )
                except Exception:
                    recommendation_id = None
                    st.warning("Your recommendation is ready, but it could not be saved yet.")
            except Exception:
                st.info("Recommendation generated using profile matching.")
                matches, _, _ = recommend_with_fallback(
                    drinks=st.session_state.drinks,
                    temperature=None if preferred_temperature == "no preference" else preferred_temperature,
                    caffeine_level=None if caffeine_preference == "any" else caffeine_preference,
                    sweetness_level=None if sweetness_preference == "any" else sweetness_preference,
                    dietary_tag=dietary_restrictions.strip() or None,
                    user=user,
                    user_history=get_user_history(user["user_id"]),
                    ingredient_preferences=get_ingredient_preferences(user["user_id"]),
                    drink_recipes=load_recipes(),
                    context=context,
                )
                st.session_state.ai_fallback_matches = matches
                st.session_state.ai_recommendation = None
            else:
                st.session_state.ai_recommendation = recommendation
                st.session_state.ai_recommendation_context = context
                st.session_state.ai_recommendation_id = recommendation_id
                st.session_state.ai_memory_summary = memory_summary
                st.session_state.ai_fallback_matches = None

    fallback_matches = st.session_state.ai_fallback_matches
    if fallback_matches is not None and not fallback_matches.empty:
        st.markdown("### Recommendation")
        render_guided_recommendation_cards(fallback_matches, limit=3)

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
        feedback_text = st.text_area("Optional feedback")
        feedback_submitted = st.form_submit_button(
            "Save Feedback",
            use_container_width=True,
        )

    if feedback_submitted:
        values = {
            "rating": rating,
            "would_order_again": would_order_again == "Yes",
            "feedback_text": feedback_text.strip() or None,
        }
        try:
            recommendation_id = st.session_state.ai_recommendation_id
            if not supabase_is_configured():
                raise RuntimeError("Supabase is not configured.")
            st.session_state.ai_recommendation_id = _save_ai_feedback(
                recommendation_id,
                user["user_id"],
                st.session_state.ai_recommendation_context,
                st.session_state.ai_memory_summary,
                recommendation,
                values,
            )
            catalog_match = st.session_state.drinks[
                st.session_state.drinks["drink_name"].astype(str).str.lower()
                == str(recommendation.get("drink_name", "")).lower()
            ]
            if not catalog_match.empty:
                try:
                    save_rating(
                        user["user_id"],
                        str(catalog_match.iloc[0]["drink_id"]),
                        rating,
                        would_order_again == "Yes",
                    )
                except Exception:
                    pass
        except Exception:
            st.warning("Your feedback could not be saved yet. The recommendation is still available.")
        else:
            st.success("Thanks. Your feedback was saved and will inform future recommendations.")


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


def generate_personalized_recommendation(
    user: dict[str, object] | None,
    context: dict[str, object],
    temperature: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, bool, list[str]]:
    """Generate top candidates internally, then select one displayed recommendation."""
    candidates, exact_match, relaxed_filters = recommend_with_fallback(
        drinks=st.session_state.drinks,
        temperature=temperature,
        user=user,
        user_history=get_user_history(user["user_id"]) if user else None,
        ingredient_preferences=get_ingredient_preferences(user["user_id"]) if user else None,
        drink_recipes=load_recipes(),
        context=context,
    )
    if candidates.empty:
        return candidates, candidates, exact_match, relaxed_filters

    top_candidates = candidates.head(10).copy()
    selector_result = None
    cache_used = False
    if user and openai_is_configured():
        try:
            selector_result, cache_used = select_best_candidate(
                user,
                top_candidates,
                context,
            )
        except Exception:
            selector_result = None
    if selector_result is None:
        selector_result = fallback_recommendation(top_candidates)

    selected_id = str(selector_result["drink_id"])
    selected = top_candidates[top_candidates["drink_id"].astype(str) == selected_id].copy()
    if selected.empty:
        selected = top_candidates.head(1).copy()
        selector_result = fallback_recommendation(top_candidates)

    selected.loc[:, "selector_confidence"] = int(selector_result.get("confidence", 0) or 0)
    selected.loc[:, "selector_reasoning"] = str(selector_result.get("reasoning", ""))
    selected.loc[:, "matched_preferences"] = [selector_result.get("matched_preferences", [])]
    selected.loc[:, "matched_context"] = [selector_result.get("matched_context", [])]
    selected.loc[:, "cache_used"] = cache_used
    selected.loc[:, "recommendation_summary"] = selected["selector_reasoning"]
    selected.loc[:, "recommendation_explanation"] = selected["selector_reasoning"]
    return selected, top_candidates, exact_match, relaxed_filters


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
    st.subheader("Your Taste Profile")
    user = st.session_state.current_user
    if not user:
        st.info("Load or create a profile first.")
        return

    render_profile_card(user)
    render_taste_profile_cards(user["user_id"])


def _set_flow_step(step: int) -> None:
    """Move the guided experience to a new step."""
    st.session_state.flow_step = step
    st.session_state.current_step = step
    st.session_state.selected_page = "guided"
    st.rerun()


def _go_to_page(page: str) -> None:
    """Navigate to a manual page."""
    st.session_state.selected_page = page
    if page == "home":
        st.session_state.home_view = "recommend"
    st.rerun()


def _start_recommendation() -> None:
    """Start recommendations, guiding new users through profiles first."""
    _set_flow_step(3 if st.session_state.current_user else 2)


def profile_gate_actions() -> None:
    """Render the Netflix-style entry gate before a profile is loaded."""
    st.markdown(
        """
        <section class="ai-title-section">
            <h1>AI Barista</h1>
            <p>Personalized drink recommendations powered by your taste profile.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        if st.button("Create Profile", type="primary", use_container_width=True):
            st.session_state.profile_mode = "create"
            _go_to_page("profile")
    with col2:
        if st.button("Load Profile", use_container_width=True):
            st.session_state.profile_mode = "load"
            _go_to_page("profile")


def home_actions() -> None:
    """Render the premium coffee-shop homepage."""
    if not st.session_state.current_user:
        profile_gate_actions()
        return

    st.markdown(
        """
        <section class="ai-title-section">
            <h1>AI Barista</h1>
            <p>Your personal drink recommendation engine.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, _ = st.columns([1.1, 1, 2.2])
    with col1:
        if st.button(
            "Get Recommendation",
            type="primary",
            use_container_width=True,
        ):
            _start_recommendation()
    with col2:
        if st.button("My Profile", use_container_width=True):
            _go_to_page("profile")

    rails = homepage_rail_data()
    render_drink_rail(
        "Recommended for You",
        rails["recommended"],
        "rail_recommended",
        empty_text="Create a profile to unlock personalized drink picks.",
    )
    render_drink_rail("Popular Drinks", rails["popular"], "rail_popular", score_column="avg_rating")
    render_drink_rail(
        "Recently Rated",
        rails["recent"],
        "rail_recent",
        empty_text="Rate a few drinks to fill this row.",
        score_column="rating",
    )
    render_drink_rail(
        "Favorites",
        rails["favorites"],
        "rail_favorites",
        empty_text="Save favorite drinks and they will appear here.",
    )
    render_drink_rail("Custom Creations", rails["custom"], "rail_custom")


def guided_welcome_step() -> None:
    """Render the welcome step."""
    home_actions()


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
            st.session_state.current_step = 3
            st.session_state.selected_page = "guided"
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
            st.session_state.current_step = 3
            st.session_state.selected_page = "guided"
            st.rerun()


def render_profile_create_load(after: str = "profile") -> None:
    """Render create/load profile forms and stay on the requested page after success."""
    create_col, load_col = st.columns(2)
    with create_col:
        if st.button("Create Profile", key=f"{after}_choose_create_profile", use_container_width=True):
            st.session_state.profile_mode = "create"
            st.rerun()
    with load_col:
        if st.button("Load Profile", key=f"{after}_choose_load_profile", use_container_width=True):
            st.session_state.profile_mode = "load"
            st.rerun()

    if st.session_state.profile_mode == "create":
        with st.form(f"{after}_create_profile_form"):
            name = st.text_input("Name")
            favorite_milk = st.selectbox("Favorite milk", list_options(st.session_state.drinks, "milk"))
            caffeine_tolerance = st.selectbox(
                "Caffeine tolerance",
                list_options(st.session_state.drinks, "caffeine_level"),
            )
            preferred_sweetness = st.selectbox(
                "Sweetness preference",
                list_options(st.session_state.drinks, "sweetness_level"),
            )
            submitted = st.form_submit_button("Create Profile", type="primary", use_container_width=True)
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
            refresh_data()
            st.session_state.flow_message = "Profile created successfully."
            if after == "home":
                _go_to_page("home")
            else:
                _go_to_page("profile")

    elif st.session_state.profile_mode == "load":
        with st.form(f"{after}_load_profile_form"):
            lookup = st.text_input("Profile ID or name")
            submitted = st.form_submit_button("Load Profile", type="primary", use_container_width=True)
        if submitted:
            user = load_user_by_id_or_name(lookup)
            if user is None:
                st.error("No profile matched that ID or name.")
                return
            st.session_state.current_user = user
            st.session_state.flow_message = f"Welcome back, {user['name']}."
            if after == "home":
                _go_to_page("home")
            else:
                _go_to_page("profile")


def guided_context_step() -> None:
    """Render today's visual context choices."""
    st.markdown("## Today's Context")
    st.write("What are you looking for today?")
    goals = [
        ("Energy", "energy"),
        ("Focus", "focus"),
        ("Comfort", "comfort"),
        ("Workout", "workout"),
        ("Treat", "treat"),
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
        ("Iced", "iced"),
        ("Hot", "hot"),
        ("No Preference", None),
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

    with st.expander(
        "Fine-tune recommendation · Add sleep, stress, weather, or flavor preferences.",
        expanded=False,
    ):
        st.caption("Optional details can make today's recommendation feel more precise.")
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
        caffeine_preference = st.selectbox(
            "Caffeine preference",
            ["Any", "None", "Low", "Medium", "High"],
            key="guided_caffeine_preference",
        )
        sweetness_preference = st.selectbox(
            "Sweetness preference",
            ["Any", "Unsweetened", "Light", "Classic", "Extra"],
            key="guided_sweetness_preference",
        )
        love_today = st.text_input("Things I love today", key="guided_love")
        avoid_today = st.text_input("Things I want to avoid", key="guided_avoid")

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
        profile_flavors = st.session_state.get("profile_flavors", {})
        likes = ", ".join(
            value
            for value in [
                str(profile_flavors.get("favorite", "")).strip(),
                love_today.strip(),
            ]
            if value
        )
        dislikes = ", ".join(
            value
            for value in [
                str(profile_flavors.get("disliked", "")).strip(),
                avoid_today.strip(),
            ]
            if value
        )
        context = {
            "sleep_hours": sleep_hours,
            "stress_level": stress_level.lower(),
            "goal": st.session_state.today_goal,
            "weather": weather.lower(),
            "temperature_preference": st.session_state.today_temperature or "no preference",
            "caffeine_preference": caffeine_preference.lower(),
            "sweetness_preference": sweetness_preference.lower(),
            "likes": likes,
            "dislikes": dislikes,
        }
        matches, candidates, exact_match, relaxed_filters = generate_personalized_recommendation(
            user=user,
            context=context,
            temperature=st.session_state.today_temperature,
        )
        st.session_state.consumer_matches = matches
        st.session_state.recommendation_results = matches
        st.session_state.ai_candidate_drinks = candidates
        st.session_state.guided_exact_match = exact_match
        st.session_state.guided_relaxed_filters = relaxed_filters
        st.session_state.today_context = context
        if not matches.empty and supabase_is_configured():
            try:
                top_match = matches.iloc[0]
                log_session(
                    user_id=user["user_id"] if user else "guest",
                    drink_id=str(top_match["drink_id"]),
                    rating="",
                    sleep_hours=str(context["sleep_hours"]),
                    stress_level=str(context["stress_level"]),
                    goal=str(context["goal"]),
                    weather=str(context["weather"]),
                )
                try:
                    log_recommendation_session(
                        user_id=user["user_id"] if user else "guest",
                        context=context,
                        drink_id=str(top_match["drink_id"]),
                        score=top_match.get("recommendation_score", 0),
                        explanation=str(top_match.get("recommendation_explanation", "")),
                    )
                except Exception:
                    pass
                if user:
                    try:
                        insert_row(
                            "ai_recommendations",
                            {
                                "user_id": user["user_id"],
                                "created_at": datetime.now(timezone.utc).isoformat(),
                                "context": context,
                                "memory_summary": build_user_memory_summary(user["user_id"]),
                                "recommendation_json": {
                                    "drink_id": str(top_match["drink_id"]),
                                    "drink_name": str(top_match["drink_name"]),
                                    "confidence": int(top_match.get("selector_confidence", 0) or 0),
                                    "reasoning": str(top_match.get("selector_reasoning", "")),
                                    "matched_preferences": top_match.get("matched_preferences", []),
                                    "matched_context": top_match.get("matched_context", []),
                                },
                                "rating": None,
                                "would_order_again": None,
                                "feedback_text": None,
                            },
                        )
                    except Exception:
                        pass
            except RuntimeError as error:
                st.error(str(error))
                return
        _set_flow_step(4)


def render_guided_recommendation_cards(
    matches: pd.DataFrame,
    limit: int = 3,
    key_prefix: str = "guided_recommendation",
) -> None:
    """Render recommendation cards that advance into the guided detail step."""
    rows = list(matches.head(limit).iterrows())
    for index, (column, (_, drink)) in enumerate(zip(st.columns(len(rows)), rows)):
        with column:
            with st.container(border=True):
                st.image(get_drink_image(drink.to_dict()), width="stretch")
                st.markdown(f"### {drink.get('drink_name', 'Recommended drink')}")
                confidence = drink.get("selector_confidence", None)
                if confidence is not None and str(confidence) != "nan":
                    st.markdown(f"**{int(float(confidence))}% confidence**")
                else:
                    st.markdown(f"**{_match_percentage(drink.get('recommendation_score'))}% match**")
                st.caption(str(drink.get("recommendation_summary", _drink_description(drink))))
                if st.button(
                    "View Details",
                    key=f"{key_prefix}_{key_slug(drink.get('drink_id'))}_{index}_view_details",
                    use_container_width=True,
                ):
                    st.session_state.selected_drink_id = drink.get("drink_id")
                    st.session_state.selected_drink = drink.to_dict()
                    _set_flow_step(5)


def render_best_recommendation(drink: pd.Series) -> None:
    """Render the primary recommendation as the largest visual element."""
    confidence = drink.get("selector_confidence", None)
    if confidence is not None and str(confidence) != "nan":
        score_text = f"{int(float(confidence))}% confidence"
    else:
        score_text = f"{_match_percentage(drink.get('recommendation_score'))}% match"
    explanation = str(
        drink.get(
            "selector_reasoning",
            drink.get("recommendation_summary", "Selected from your profile and today's context."),
        )
    )
    col1, col2 = st.columns([1.25, 1])
    with col1:
        st.markdown(
            f'<img class="recommendation-feature-image" src="{safe_text(get_drink_image(drink.to_dict()))}" alt="{safe_text(drink.get("drink_name", "Drink"))}">',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="recommendation-feature">
                <div class="ai-card-meta">Best match</div>
                <div class="ai-card-title">{safe_text(drink.get("drink_name", "Recommended drink"))}</div>
                <div class="ai-score">{safe_text(score_text)}</div>
                <div class="ai-explanation">Best match because {safe_text(explanation)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "View Details",
            key=f"best_recommendation_{key_slug(drink.get('drink_id'))}_0_view_details",
            use_container_width=True,
        ):
            st.session_state.selected_drink_id = drink.get("drink_id")
            st.session_state.selected_drink = drink.to_dict()
            _set_flow_step(5)


def render_secondary_recommendations(candidates: pd.DataFrame, selected_id: object) -> None:
    """Render smaller secondary recommendations from the internal candidate list."""
    if candidates is None or candidates.empty:
        return
    secondary = candidates[candidates["drink_id"].astype(str) != str(selected_id)].head(3)
    if secondary.empty:
        return
    st.markdown("### Also Close")
    columns = st.columns(len(secondary))
    for index, (column, (_, drink)) in enumerate(zip(columns, secondary.iterrows())):
        reason = str(drink.get("recommendation_summary", drink.get("recommendation_explanation", "")))
        with column:
            st.markdown(
                f"""
                <div class="secondary-card">
                    <img class="rail-card-image" src="{safe_text(get_drink_image(drink.to_dict()))}" alt="{safe_text(drink.get("drink_name", "Drink"))}">
                    <div class="rail-card-title">{safe_text(drink.get("drink_name", "Drink"))}</div>
                    <div class="rail-card-meta">Also close because {safe_text(reason)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "View Details",
                key=f"secondary_recommendation_{key_slug(drink.get('drink_id'))}_{index}_view_details",
                use_container_width=True,
            ):
                st.session_state.selected_drink_id = drink.get("drink_id")
                st.session_state.selected_drink = drink.to_dict()
                _set_flow_step(5)


def guided_recommendation_step() -> None:
    """Render the selected recommendation."""
    st.markdown("## Your Drink Recommendation")
    matches = st.session_state.consumer_matches
    if matches is None or matches.empty:
        st.warning("No recommendations are ready yet.")
        if st.button("Back to Today's Context"):
            _set_flow_step(3)
        return
    if not st.session_state.get("guided_exact_match", True):
        st.info("No exact match found, but this is close.")
        relaxed = st.session_state.get("guided_relaxed_filters", [])
        if relaxed:
            st.caption(f"Relaxed filters: {', '.join(relaxed)}")
    best = matches.iloc[0]
    render_best_recommendation(best)
    render_secondary_recommendations(
        st.session_state.get("ai_candidate_drinks"),
        best.get("drink_id"),
    )


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
        confidence = drink.get("selector_confidence", None)
        if confidence is not None and str(confidence) != "nan":
            st.markdown(f"**{int(float(confidence))}% confidence**")
        else:
            st.markdown(f"**{_match_percentage(drink.get('recommendation_score'))}% match**")
        st.write(_drink_description(drink))
        st.subheader("Why this was recommended")
        st.info(str(drink.get("selector_reasoning", drink.get("recommendation_explanation", "Recommended for your taste."))))
        matched_preferences = drink.get("matched_preferences", [])
        matched_context = drink.get("matched_context", [])
        if isinstance(matched_preferences, list) and matched_preferences:
            st.markdown("**Matched preferences**")
            for item in matched_preferences:
                st.write(f"- {item}")
        if isinstance(matched_context, list) and matched_context:
            st.markdown("**Matched context**")
            for item in matched_context:
                st.write(f"- {item}")
        score_parts = [
            ("Context", drink.get("context_score", 0)),
            ("Profile", drink.get("profile_score", 0)),
            ("Ingredients", drink.get("ingredient_preference_score", 0)),
            ("Past ratings", drink.get("past_rating_score", 0)),
            ("Close match", drink.get("fallback_similarity_score", 0)),
        ]
        st.caption(
            "Score: "
            + " + ".join(f"{label} {int(float(value or 0))}" for label, value in score_parts)
        )
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
                st.session_state.recommendation_results = None
                st.session_state.selected_drink_id = None
                st.session_state.selected_drink = None
                _set_flow_step(3)

        user = st.session_state.current_user
        if st.button("Save to Favorites", use_container_width=True, key="guided_save_favorite"):
            if not user:
                st.info("Load or create a profile first.")
            else:
                saved, message = save_favorite_action(
                    user["user_id"],
                    str(drink["drink_id"]),
                    str(drink.get("drink_name", "Favorite drink")),
                )
                (st.success if saved else st.info)(message)

    ratings = load_ratings()
    drink_ratings = (
        ratings[ratings["drink_id"].astype(str) == str(drink["drink_id"])]
        if not ratings.empty
        else ratings
    )
    if not drink_ratings.empty:
        st.caption(
            f"User rating: {drink_ratings['rating'].astype(float).mean():.1f}/5 "
            f"from {len(drink_ratings)} rating(s)"
        )

    st.subheader("Similar drinks")
    similar = find_similar_drinks(st.session_state.drinks, drink, limit=3)
    render_guided_recommendation_cards(
        similar,
        limit=3,
        key_prefix=f"similar_{key_slug(drink.get('drink_id'))}",
    )


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
                sleep_hours=str(st.session_state.today_context.get("sleep_hours", "")),
                stress_level=str(st.session_state.today_context.get("stress_level", "")),
                goal=str(st.session_state.today_context.get("goal", "")),
                weather=str(st.session_state.today_context.get("weather", "")),
            )
            try:
                log_recommendation_session(
                    user_id=user["user_id"],
                    context=st.session_state.today_context,
                    drink_id=str(drink["drink_id"]),
                    score=drink.get("recommendation_score", 0),
                    explanation=str(drink.get("recommendation_explanation", "")),
                    rating=rating,
                )
            except Exception:
                pass
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
                st.session_state.recommendation_results = None
                st.session_state.selected_drink_id = None
                st.session_state.selected_drink = None
                _set_flow_step(3)
        with profile_col:
            if st.button("View profile", use_container_width=True):
                st.session_state.feedback_saved = False
                _set_flow_step(7)


def guided_profile_summary_step() -> None:
    """Show the loaded profile after the journey."""
    user = st.session_state.current_user
    st.markdown("## Your Taste Profile")
    if not user:
        st.info("No profile is loaded.")
        return
    render_profile_card(user)
    render_taste_profile_cards(user["user_id"])
    render_favorites_section(user["user_id"])
    if st.button("Get another recommendation", type="primary", use_container_width=True):
        _set_flow_step(3)


def guided_flow() -> None:
    """Render exactly one step of the main onboarding and recommendation flow."""
    step = int(st.session_state.current_step)
    st.session_state.flow_step = step
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


def my_profile_page() -> None:
    """Render direct profile access from navigation."""
    user = st.session_state.current_user
    st.markdown("## My Profile")
    if not user:
        st.markdown(
            """
            <div class="ai-card">
                <div class="ai-card-title">Start your taste profile</div>
                <div class="ai-card-meta">
                    Create a profile or load an existing one to personalize every recommendation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_profile_create_load(after="profile")
        return

    st.markdown("### Profile Dashboard")
    render_profile_card(user)
    action_col, switch_col = st.columns(2)
    with action_col:
        if st.button(
            "Get Recommendation",
            type="primary",
            use_container_width=True,
            key="profile_get_recommendation",
        ):
            _set_flow_step(3)
    with switch_col:
        if st.button("Switch Profile", use_container_width=True):
            st.session_state.profile_mode = "load"
            _set_flow_step(2)

    render_taste_profile_cards(user["user_id"])
    render_favorites_section(user["user_id"])


def sidebar_navigation() -> None:
    """Render persistent manual navigation."""
    st.sidebar.markdown("### AI Barista")
    st.sidebar.caption(current_user_label())
    if st.sidebar.button("Home", key="nav_home", use_container_width=True):
        _go_to_page("home")
    if st.sidebar.button("My Profile", key="nav_profile", use_container_width=True):
        _go_to_page("profile")
    if st.session_state.current_user:
        if st.sidebar.button(
            "Recommendation",
            key="nav_recommendation",
            use_container_width=True,
        ):
            _start_recommendation()
        if st.sidebar.button("Advanced Tools", key="nav_advanced", use_container_width=True):
            _go_to_page("advanced")


def advanced_tools_section(expanded: bool = False) -> None:
    """Render advanced customization and database tools."""
    with st.expander(
        "⚙️ Advanced Tools · Create custom drinks, add ingredients, and explore your data.",
        expanded=expanded,
    ):
        st.caption("Database and customization tools for deeper exploration.")
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
                    "Rule-based recommendations",
                    "Rate drink",
                    "Rating history",
                    "Taste profile",
                ]
            )
            with admin_tabs[0]:
                recommendation_section()
            with admin_tabs[1]:
                rate_drink_section()
            with admin_tabs[2]:
                rating_history_section()
            with admin_tabs[3]:
                taste_profile_section()


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="AI Barista", layout="wide")
    apply_theme()
    initialize_state()
    sidebar_navigation()

    if not supabase_is_configured():
        st.warning(
            "Supabase credentials are not configured locally. Static catalog browsing works, "
            "but recommendations and feedback require SUPABASE_URL and SUPABASE_KEY."
        )

    page = st.session_state.selected_page
    if not st.session_state.current_user and page in {"advanced", "guided"}:
        page = "home"
        st.session_state.selected_page = "home"
    if page == "home":
        if st.session_state.home_view == "details" and st.session_state.selected_drink_id:
            drink_detail_section()
        else:
            home_actions()
    elif page == "profile":
        my_profile_page()
    elif page == "advanced":
        st.markdown("## Advanced Tools")
        advanced_tools_section(expanded=True)
    else:
        guided_flow()

    if st.session_state.current_user and page != "advanced":
        advanced_tools_section(expanded=False)


if __name__ == "__main__":
    main()
