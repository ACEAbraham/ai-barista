"""Command-line interface for AI Barista."""

from customization import log_session
from drink_database import list_options, load_drinks
from ingredient_engine import (
    build_custom_drink_from_ingredients,
    get_ingredient_preferences,
    get_taste_profile,
    load_ingredients,
    load_recipes,
    parse_recipe_input,
    save_custom_drink_recipe,
    show_ingredients_by_category,
)
from profile import create_user, get_user_history, load_user, save_rating
from recommender import recommend_drinks


def ask_optional(prompt: str) -> str | None:
    """Ask for an optional text preference."""
    value = input(prompt).strip()
    return value or None


def ask_budget() -> float | None:
    """Ask for an optional maximum drink price."""
    value = input("Maximum price, or press Enter to skip: $").strip()
    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        print("That price was not valid, so the budget filter will be skipped.")
        return None


def ask_rating() -> int | None:
    """Ask for a rating from 1 to 5."""
    value = input("Rating from 1 to 5: ").strip()
    try:
        rating = int(value)
    except ValueError:
        print("That rating was not valid.")
        return None

    if rating < 1 or rating > 5:
        print("Ratings must be between 1 and 5.")
        return None
    return rating


def ask_yes_no(prompt: str) -> bool:
    """Ask a yes/no question."""
    value = input(prompt).strip().lower()
    return value in {"y", "yes"}


def ask_session_context() -> dict[str, str]:
    """Collect context that can become future training data."""
    print("\nSession context")
    return {
        "sleep_hours": input("Sleep hours: ").strip(),
        "stress_level": input("Stress level: ").strip(),
        "goal": input("Goal, such as focus, comfort, energy, or treat: ").strip(),
        "weather": input("Weather: ").strip(),
    }


def show_options(drinks) -> None:
    """Print available preference options for the user."""
    print("\nAvailable options")
    print(f"Caffeine levels: {', '.join(list_options(drinks, 'caffeine_level'))}")
    print(f"Temperatures: {', '.join(list_options(drinks, 'temperature'))}")
    print(f"Milk types: {', '.join(list_options(drinks, 'milk'))}")
    print(f"Sweetness levels: {', '.join(list_options(drinks, 'sweetness_level'))}")
    print("Dietary tags examples: dairy-free, vegan, nut-free, low-calorie")


def display_matches(matches, limit: int = 10) -> None:
    """Display matching drinks in a beginner-friendly format."""
    if matches.empty:
        print("\nNo drinks matched those preferences. Try removing one filter.")
        return

    print(f"\nFound {len(matches)} matching drink(s). Showing up to {limit}:\n")
    for _, drink in matches.head(limit).iterrows():
        print(f"{drink['drink_id']}: {drink['drink_name']}")
        print(
            f"  {drink['size']} {drink['temperature']} | {drink['milk']} | "
            f"{drink['syrup']} | {drink['espresso_shots']} shot(s)"
        )
        print(
            f"  Caffeine: {drink['caffeine_level']} | Sweetness: "
            f"{drink['sweetness_level']} | Calories: {drink['calories']} | "
            f"Price: ${drink['price']:.2f}"
        )
        if "recommendation_score" in drink:
            print(f"  Recommendation score: {drink['recommendation_score']}")
        if "recommendation_explanation" in drink:
            print(f"  Why: {drink['recommendation_explanation']}")
        if "flavor_score" in drink and not pd_is_empty(drink["flavor_score"]):
            print(f"  Flavor score: {drink['flavor_score']}/10")
        print(f"  Tags: {drink['dietary_tags']}")
        print(f"  Flavor: {drink['flavor_profile']}\n")


def create_profile() -> dict[str, str]:
    """Ask for profile details and save a new user."""
    print("\nCreate profile")
    user = create_user(
        name=input("Name: ").strip(),
        favorite_milk=input("Favorite milk: ").strip(),
        favorite_temperature=input("Favorite temperature: ").strip(),
        caffeine_tolerance=input("Caffeine tolerance: ").strip(),
        preferred_sweetness=input("Preferred sweetness: ").strip(),
    )
    print(f"\nCreated profile for {user['name']} with ID {user['user_id']}.")
    return user


def load_profile() -> dict[str, str] | None:
    """Ask for a user ID and load the matching profile."""
    user_id = input("\nUser ID: ").strip()
    user = load_user(user_id)
    if user is None:
        print("No profile found with that user ID.")
        return None

    print(f"Loaded profile for {user['name']} ({user['user_id']}).")
    return user


def find_drinks(drinks, current_user: dict[str, str] | None) -> None:
    """Find drinks using typed preferences or a loaded profile."""
    context = ask_session_context()
    show_options(drinks)

    if current_user:
        print("\nYour saved profile will boost matching drinks in the ranking.")
        default_caffeine = current_user["caffeine_tolerance"]
        default_temperature = current_user["favorite_temperature"]
        default_milk = current_user["favorite_milk"]
        default_sweetness = current_user["preferred_sweetness"]
    else:
        default_caffeine = None
        default_temperature = None
        default_milk = None
        default_sweetness = None

    caffeine_level = ask_optional(f"\nFilter caffeine level [{default_caffeine or 'no filter'}]: ")
    temperature = ask_optional(f"Filter temperature [{default_temperature or 'no filter'}]: ")
    milk = ask_optional(f"Filter milk type [{default_milk or 'no filter'}]: ")
    max_price = ask_budget()
    sweetness_level = ask_optional(f"Filter sweetness level [{default_sweetness or 'no filter'}]: ")
    dietary_tag = ask_optional("Dietary tag, or press Enter to skip: ")

    matches = recommend_drinks(
        drinks=drinks,
        caffeine_level=caffeine_level,
        temperature=temperature,
        milk=milk,
        max_price=max_price,
        sweetness_level=sweetness_level,
        dietary_tag=dietary_tag,
        user=current_user,
        user_history=(
            get_user_history(current_user["user_id"])
            if current_user
            else None
        ),
        ingredient_preferences=(
            get_ingredient_preferences(current_user["user_id"])
            if current_user
            else None
        ),
        drink_recipes=load_recipes(),
    )
    display_matches(matches)
    if not matches.empty:
        top_match = matches.iloc[0]
        log_session(
            user_id=current_user["user_id"] if current_user else "guest",
            drink_id=top_match["drink_id"],
            rating="",
            **context,
        )
        print("Saved this recommendation interaction as training data.")


def rate_drink(drinks, current_user: dict[str, str] | None) -> None:
    """Save a user's rating for a drink."""
    if not current_user:
        print("\nLoad or create a profile before rating drinks.")
        return

    drink_id = input("\nDrink ID to rate: ").strip()
    drink = drinks[drinks["drink_id"].str.lower() == drink_id.lower()]
    if drink.empty:
        print("No drink found with that ID.")
        return

    rating = ask_rating()
    if rating is None:
        return

    context = ask_session_context()
    would_order_again = ask_yes_no("Would you order it again? (yes/no): ")
    saved = save_rating(
        user_id=current_user["user_id"],
        drink_id=drink.iloc[0]["drink_id"],
        rating=rating,
        would_order_again=would_order_again,
    )
    print(
        f"Saved rating {saved['rating']} for "
        f"{drink.iloc[0]['drink_name']}."
    )
    print("Updated your ingredient taste profile.")
    log_session(
        user_id=current_user["user_id"],
        drink_id=drink.iloc[0]["drink_id"],
        rating=rating,
        **context,
    )
    print("Saved this rating interaction as training data.")


def pd_is_empty(value) -> bool:
    """Return True for blank or pandas missing values."""
    return str(value).strip().lower() in {"", "nan", "none"}


def create_custom_drink(drinks, current_user: dict[str, str] | None):
    """Create and save a fully custom ingredient-based drink."""
    ingredients = load_ingredients()
    show_ingredients_by_category(ingredients)

    print("\nBuild a custom drink from ingredients")
    print("Enter ingredients as ingredient_id:quantity, separated by commas.")
    print("Example: ING-001:2, ING-010:1, ING-015:1, ING-032:1, ING-035:1, ING-038:1")
    drink_name = input("Drink name: ").strip()
    recipe_text = input("Recipe: ").strip()
    try:
        recipe_items = parse_recipe_input(recipe_text)
        custom_drink = build_custom_drink_from_ingredients(
            drinks=drinks,
            drink_name=drink_name,
            recipe_items=recipe_items,
        )
    except ValueError as error:
        print(f"Could not build custom drink: {error}")
        return drinks

    save_custom_drink_recipe(custom_drink, recipe_items)
    print(f"\nSaved {custom_drink['drink_name']} as {custom_drink['drink_id']}.")
    print(
        f"Nutrition: {custom_drink['calories']} calories, "
        f"{custom_drink['caffeine_level']} caffeine"
    )
    print(f"Cost: ${custom_drink['price']:.2f}")
    print(f"Flavor score: {custom_drink['flavor_score']}/10")

    context = ask_session_context()
    rating = ""
    if current_user and ask_yes_no("Rate this custom drink now? (yes/no): "):
        rating_value = ask_rating()
        if rating_value is not None:
            rating = rating_value
            would_order_again = ask_yes_no("Would you order it again? (yes/no): ")
            save_rating(
                user_id=current_user["user_id"],
                drink_id=custom_drink["drink_id"],
                rating=rating_value,
                would_order_again=would_order_again,
            )
            print("Updated your ingredient taste profile.")

    log_session(
        user_id=current_user["user_id"] if current_user else "guest",
        drink_id=custom_drink["drink_id"],
        rating=rating,
        **context,
    )
    print("Saved this custom drink interaction as training data.")
    return load_drinks()


def display_ingredient_list(title: str, ingredients, score_column: str) -> None:
    """Print a compact ingredient list for a taste profile section."""
    print(f"\n{title}")
    if ingredients.empty:
        print("  No data yet.")
        return

    for _, row in ingredients.head(5).iterrows():
        print(
            f"  {row['ingredient_name']} "
            f"({row['ingredient_id']}): {row[score_column]}"
        )


def view_taste_profile(current_user: dict[str, str] | None) -> None:
    """Show ingredient preferences learned from user ratings."""
    if not current_user:
        print("\nLoad or create a profile before viewing a taste profile.")
        return

    profile = get_taste_profile(current_user["user_id"])
    print(f"\nTaste profile for {current_user['name']}")
    display_ingredient_list(
        "Favorite ingredients",
        profile["favorite"],
        "preference_score",
    )
    display_ingredient_list(
        "Least favorite ingredients",
        profile["least_favorite"],
        "preference_score",
    )
    display_ingredient_list(
        "Most common ingredients",
        profile["most_common"],
        "times_seen",
    )


def show_history(drinks, current_user: dict[str, str] | None) -> None:
    """Display saved ratings for the loaded profile."""
    if not current_user:
        print("\nLoad or create a profile before viewing history.")
        return

    history = get_user_history(current_user["user_id"])
    if history.empty:
        print("\nNo ratings saved yet.")
        return

    history_with_drinks = history.merge(drinks, on="drink_id", how="left")
    print(f"\nRating history for {current_user['name']}:\n")
    for _, row in history_with_drinks.iterrows():
        order_again = "yes" if row["would_order_again"] else "no"
        print(
            f"{row['drink_id']}: {row['drink_name']} | "
            f"Rating: {row['rating']} | Order again: {order_again}"
        )


def show_menu(current_user: dict[str, str] | None) -> None:
    """Display the main menu."""
    profile_name = current_user["name"] if current_user else "none"
    print(f"\nCurrent profile: {profile_name}")
    print("1. Create profile")
    print("2. Load profile")
    print("3. Find drinks")
    print("4. Rate drink")
    print("5. View rating history")
    print("6. Create custom drink")
    print("7. View taste profile")
    print("8. Quit")


def main() -> None:
    """Run the AI Barista command-line app."""
    drinks = load_drinks()
    current_user = None

    print("Welcome to AI Barista")
    print("Collect user preferences and ratings for future recommendations.")

    while True:
        show_menu(current_user)
        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            current_user = create_profile()
        elif choice == "2":
            loaded_user = load_profile()
            if loaded_user:
                current_user = loaded_user
        elif choice == "3":
            find_drinks(drinks, current_user)
        elif choice == "4":
            rate_drink(drinks, current_user)
        elif choice == "5":
            show_history(drinks, current_user)
        elif choice == "6":
            drinks = create_custom_drink(drinks, current_user)
        elif choice == "7":
            view_taste_profile(current_user)
        elif choice == "8":
            print("Goodbye.")
            break
        else:
            print("Please choose a menu option from 1 to 8.")


if __name__ == "__main__":
    main()
