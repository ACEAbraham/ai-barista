"""Regression tests for category-safe drink images."""

from drink_images import CURATED_DRINK_IMAGES, get_drink_image


def test_peppermint_mocha_uses_mocha_image() -> None:
    drink = {"drink_name": "Grande Iced Extra Nonfat Mocha with Peppermint"}

    assert get_drink_image(drink) == CURATED_DRINK_IMAGES["mocha"]


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
