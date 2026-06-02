"""Streamlit web app for AI Barista data collection."""

import pandas as pd
import streamlit as st

from customization import log_session
from drink_database import list_options, load_drinks
from ingredient_engine import (
    build_custom_drink_from_ingredients,
    get_taste_profile,
    load_ingredient_preferences,
    load_ingredients,
    load_recipes,
    save_custom_drink_recipe,
    update_ingredient_preferences,
)
from profile import create_user, get_user_history, load_user, save_rating
from recommender import recommend_drinks


USERS_FILE = "users.csv"


def load_users() -> pd.DataFrame:
    """Load all user profiles."""
    return pd.read_csv(USERS_FILE)


def refresh_data() -> None:
    """Reload CSV-backed app data after a save."""
    st.session_state.drinks = load_drinks()
    st.session_state.users = load_users()


def initialize_state() -> None:
    """Initialize Streamlit session state."""
    if "drinks" not in st.session_state:
        st.session_state.drinks = load_drinks()
    if "users" not in st.session_state:
        st.session_state.users = load_users()
    if "current_user" not in st.session_state:
        st.session_state.current_user = None


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
        user = create_user(
            name=name.strip(),
            favorite_milk=favorite_milk,
            favorite_temperature=favorite_temperature,
            caffeine_tolerance=caffeine_tolerance,
            preferred_sweetness=preferred_sweetness,
        )
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
        matches = recommend_drinks(
            drinks=drinks,
            caffeine_level=None if caffeine == "Any" else caffeine,
            temperature=None if temperature == "Any" else temperature,
            milk=None if milk == "Any" else milk,
            max_price=max_price,
            sweetness_level=None if sweetness == "Any" else sweetness,
            dietary_tag=None if dietary_tag == "Any" else dietary_tag,
            user=user,
            user_history=get_user_history(user["user_id"]) if user else None,
            ingredient_preferences=load_ingredient_preferences(),
            drink_recipes=load_recipes(),
        )
        st.session_state.last_matches = matches

        if not matches.empty:
            top_match = matches.iloc[0]
            log_session(
                user_id=user["user_id"] if user else "guest",
                drink_id=top_match["drink_id"],
                rating="",
                **context,
            )
            st.success("Saved this recommendation interaction as training data.")

    matches = st.session_state.get("last_matches")
    if matches is not None:
        if matches.empty:
            st.warning("No drinks matched those filters.")
        else:
            st.dataframe(
                matches[
                    [
                        "drink_id",
                        "drink_name",
                        "recommendation_score",
                        "recommendation_explanation",
                        "price",
                        "calories",
                        "caffeine_level",
                    ]
                ].head(20),
                width="stretch",
            )


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

    recipe_items = []
    for index in range(1, 9):
        col1, col2 = st.columns([3, 1])
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
        if selected != "None" and quantity > 0:
            recipe_items.append(
                {
                    "ingredient_id": ingredient_lookup[selected],
                    "quantity": quantity,
                }
            )

    return recipe_items


def custom_drink_section() -> None:
    """Render ingredient-based custom drink UI."""
    st.subheader("Create Custom Ingredient-Based Drink")
    user = st.session_state.current_user
    ingredients = load_ingredients()

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

        save_custom_drink_recipe(custom_drink, recipe_items)
        saved_rating = ""
        if user and rate_now:
            save_rating(user["user_id"], custom_drink["drink_id"], rating, would_order_again)
            update_ingredient_preferences(user["user_id"], custom_drink["drink_id"], rating)
            saved_rating = rating

        log_session(
            user_id=user["user_id"] if user else "guest",
            drink_id=custom_drink["drink_id"],
            rating=saved_rating,
            **context,
        )
        refresh_data()
        st.success(f"Saved custom drink: {custom_drink['drink_name']}")
        st.metric("Calories", custom_drink["calories"])
        st.metric("Price", f"${custom_drink['price']:.2f}")
        st.metric("Flavor score", f"{custom_drink['flavor_score']}/10")


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
        save_rating(user["user_id"], drink_id, rating, would_order_again)
        update_ingredient_preferences(user["user_id"], drink_id, rating)
        log_session(
            user_id=user["user_id"],
            drink_id=drink_id,
            rating=rating,
            **context,
        )
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
    initialize_state()

    st.title("AI Barista")
    st.caption("A shareable, no-ML beverage data collection app.")
    st.sidebar.write(f"Current profile: **{current_user_label()}**")

    sections = st.tabs(
        [
            "Create profile",
            "Load profile",
            "Find recommended drinks",
            "Create custom drink",
            "Rate a drink",
            "Rating history",
            "Taste profile",
        ]
    )

    with sections[0]:
        create_profile_section()
    with sections[1]:
        load_profile_section()
    with sections[2]:
        recommendation_section()
    with sections[3]:
        custom_drink_section()
    with sections[4]:
        rate_drink_section()
    with sections[5]:
        rating_history_section()
    with sections[6]:
        taste_profile_section()


if __name__ == "__main__":
    main()
