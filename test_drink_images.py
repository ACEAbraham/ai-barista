"""Regression tests for category-safe drink images."""

from drink_images import CURATED_DRINK_IMAGES, FIXED_IMAGE_CATEGORIES, get_drink_image


def test_fixed_category_image_set() -> None:
    assert set(FIXED_IMAGE_CATEGORIES) == {
        "iced_latte",
        "hot_latte",
        "mocha",
        "americano",
        "espresso",
        "cold_brew",
        "matcha",
        "chai",
        "tea",
        "frappuccino",
        "refresher",
        "generic_coffee",
    }
    assert len(CURATED_DRINK_IMAGES) == 12
    for image_url in CURATED_DRINK_IMAGES.values():
        assert image_url.startswith("https://images.unsplash.com/photo-")
        assert "source.unsplash" not in image_url
        assert not image_url.startswith("data:image")


def test_peppermint_mocha_uses_mocha_image() -> None:
    drink = {"drink_name": "Grande Iced Extra Nonfat Mocha with Peppermint"}

    assert get_drink_image(drink) == CURATED_DRINK_IMAGES["mocha"]
    assert get_drink_image(drink).startswith("https://images.unsplash.com/photo-")
    assert not get_drink_image(drink).startswith("data:image")
    assert "photo-1579888944880" not in get_drink_image(drink)


def test_flavor_words_do_not_override_mocha() -> None:
    for name in (
        "Peppermint Mocha",
        "Coconut Mocha",
        "Raspberry Mocha",
        "Pineapple Mocha",
    ):
        assert get_drink_image({"drink_name": name}) == CURATED_DRINK_IMAGES["mocha"]


def test_mocha_syrup_does_not_override_latte_identity() -> None:
    drink = {
        "drink_name": "Grande Iced Latte",
        "temperature": "iced",
        "syrup": "mocha",
    }

    assert get_drink_image(drink) == CURATED_DRINK_IMAGES["iced_latte"]
