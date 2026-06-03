# AI Barista

AI Barista is the first foundation for a "Spotify for beverages" app. This version does not use machine learning. It focuses on a customizable Starbucks-style drink database, user profiles, custom drink building, and interaction logging for future recommendation models.

## Project Files

- `drinks.csv` stores the drink catalog.
- `custom_drinks.csv` stores drinks built by users.
- `drink_components.csv` stores available custom drink parts.
- `ingredients.csv` stores ingredient nutrition, caffeine, and price data.
- `drink_recipes.csv` maps drinks to ingredient quantities.
- `ingredient_preferences.csv` stores learned per-user ingredient preference scores.
- `users.csv` stores user beverage preferences.
- `ratings.csv` stores drink ratings and repeat-order feedback.
- `sessions.csv` stores interaction context for future training data.
- `drink_database.py` loads the CSV with pandas and lists available options.
- `customization.py` builds custom drinks and logs session data.
- `ingredient_engine.py` calculates nutrition, cost, and flavor scores from recipes.
- `ingredient_engine.py` also updates ingredient preference scores from user ratings.
- `recommender.py` contains reusable filter functions and a rule-based recommendation score.
- `profile.py` creates users, loads users, saves ratings, and reads user history.
- `main.py` provides a simple command-line interface.
- `app.py` provides the Streamlit web app.
- `requirements.txt` lists deployable app dependencies.

## CSV Fields

Each drink includes:

- `drink_id`
- `drink_name`
- `base`
- `temperature`
- `size`
- `milk`
- `syrup`
- `sweetness_level`
- `espresso_shots`
- `caffeine_level`
- `calories`
- `price`
- `dietary_tags`
- `flavor_profile`

Each user includes:

- `user_id`
- `name`
- `favorite_milk`
- `favorite_temperature`
- `caffeine_tolerance`
- `preferred_sweetness`

Each rating includes:

- `user_id`
- `drink_id`
- `rating`
- `would_order_again`

Each component row includes:

- `base`
- `milk`
- `syrup`
- `toppings`
- `size`
- `shots`
- `temperature`
- `ice_level`

Each session includes:

- `user_id`
- `timestamp`
- `sleep_hours`
- `stress_level`
- `goal`
- `weather`
- `drink_id`
- `rating`

Each ingredient includes:

- `ingredient_id`
- `ingredient_name`
- `category`
- `calories`
- `caffeine`
- `price`
- `default_unit`

Each recipe row includes:

- `drink_id`
- `ingredient_id`
- `quantity`
- `unit`

Each ingredient preference includes:

- `user_id`
- `ingredient_id`
- `preference_score`

## Setup

Install dependencies if they are not already installed:

```bash
pip install -r requirements.txt
```

## Run the Streamlit App Locally

From this folder, run:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

## Deploy to Streamlit Cloud

1. Push this project folder to a GitHub repository.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Create a new app from your repository.
4. Set the app entry point to `app.py`.
5. Confirm `requirements.txt` is in the same folder as `app.py`.
6. Deploy the app.

Streamlit Cloud will install `pandas` and `streamlit` from `requirements.txt`.

## CLI Option

The original command-line app is still available:

```bash
python main.py
```

Use the web app to create or load a profile, find drinks, rate drinks, add ingredients, create ingredient-based custom drinks, view rating history, and view a taste profile.

## Custom Ingredients

The Streamlit app includes an `Add Ingredient` section. New ingredients are saved to `ingredients.csv` with a unique `ingredient_id` and become available in the custom drink builder after saving.

Supported categories are:

- `base`
- `milk`
- `syrup`
- `topping`
- `sweetener`
- `flavor`
- `powder`
- `ice`
- `temperature`
- `size`
- `add-in`

## Recipe Units

Each recipe item in `drink_recipes.csv` stores both `quantity` and `unit`. Supported units are:

- `oz`
- `ml`
- `pump`
- `tsp`
- `tbsp`
- `shot`
- `scoop`
- `serving`
- `cup`
- `g`

Nutrition and cost calculations currently use `quantity`. The `unit` is saved for display and future conversion work.

## Duplicate Prevention

Before saving a custom drink, AI Barista checks for duplicates by:

- normalized drink name, using lowercase and removing spaces and punctuation
- ingredient recipe signature, using `ingredient_id + quantity + unit`

If either check matches an existing custom drink, the app warns the user and does not save the duplicate.

Find Drinks ranks results by `recommendation_score`. The score is based on profile matches, high past ratings, whether the user said they would order a drink again, and learned ingredient preference scores. Results also include an explanation showing how the score was calculated. Custom drinks are reloaded after saving, so they appear in future recommendation results.

## Example Preferences

- Caffeine level: `medium`
- Temperature: `iced`
- Milk type: `oat`
- Maximum price: `6`
- Sweetness level: `classic`
- Dietary tag: `vegan`

## Notes

This project intentionally avoids machine learning for now. Custom drink nutrition, caffeine, cost, flavor scores, and taste profiles are calculated with rule-based logic. The collected profiles, ingredient preferences, recipes, ratings, and session context can later become training data, scoring features, or model inputs.
