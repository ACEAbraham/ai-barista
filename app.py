"""Streamlit web app for AI Barista data collection."""

from datetime import datetime, timezone
from html import escape
import logging
import re
from urllib.parse import quote

import pandas as pd
import streamlit as st

from analytics import track_recommendation_event
from customization import log_recommendation_session, log_session
from drink_database import list_options, load_drinks
from drink_images import HERO_IMAGE_URL, get_drink_image, get_drink_image_alt
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
from progress import award_daily_return_if_needed, award_xp, get_user_progress
from recommender import find_similar_drinks, recommend_with_fallback
from supabase_client import insert_row, supabase_is_configured, update_rows


LOGGER = logging.getLogger(__name__)


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
                linear-gradient(90deg, rgba(0, 0, 0, 0.68), rgba(0, 0, 0, 0.44)),
                url("__HERO_IMAGE_URL__");
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
            color: rgba(255, 253, 248, 0.94) !important;
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
            height: 388px;
            min-height: 388px;
            max-height: 388px;
            display: flex;
            flex-direction: column;
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
            height: 178px;
            min-height: 178px;
            max-height: 178px;
            aspect-ratio: 4 / 3;
            object-fit: cover;
            display: block;
            flex: 0 0 178px;
        }

        .card-drink-image,
        .favorite-drink-image,
        .detail-drink-image {
            width: 100%;
            object-fit: cover;
            display: block;
            border-radius: 8px;
            border: 1px solid var(--ai-accent);
            background: var(--ai-input);
        }

        .card-drink-image {
            height: 180px;
            aspect-ratio: 4 / 3;
        }

        .favorite-drink-image {
            height: 180px;
            aspect-ratio: 4 / 3;
        }

        .detail-drink-image {
            height: 360px;
            aspect-ratio: 16 / 10;
        }

        .rail-card-body {
            padding: 0.75rem;
            height: 210px;
            min-height: 210px;
            max-height: 210px;
            display: flex;
            flex-direction: column;
        }

        .rail-card-action-space {
            flex: 1 1 auto;
            min-height: 2.75rem;
            border-top: 1px solid rgba(160, 125, 97, 0.22);
            margin-top: 0.65rem;
        }

        .recommendation-feature {
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 12px;
            padding: 1.25rem;
            min-height: 360px;
            box-shadow: 0 20px 46px rgba(74, 38, 8, 0.16);
        }

        .featured-title {
            color: var(--ai-text);
            font-size: 1.55rem;
            font-weight: 900;
            line-height: 1.15;
            max-height: 3.6rem;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            margin-bottom: 0.7rem;
        }

        .featured-score {
            display: inline-block;
            background: var(--ai-button);
            color: #FFFFFF;
            border-radius: 999px;
            padding: 0.35rem 0.85rem;
            font-weight: 900;
            margin-bottom: 0.8rem;
        }

        .featured-reason {
            color: var(--ai-secondary);
            font-size: 0.95rem;
            line-height: 1.45;
            max-height: 4.15rem;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            margin-top: 0.8rem;
        }

        .recommendation-feature-image {
            width: 100%;
            height: 360px;
            aspect-ratio: 16 / 10;
            object-fit: cover;
            border-radius: 8px;
            border: 1px solid var(--ai-accent);
        }

        .secondary-card {
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 8px;
            padding: 0;
            height: 388px;
            min-height: 388px;
            max-height: 388px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 10px 24px rgba(74, 38, 8, 0.10);
        }

        .secondary-card .rail-card-image {
            height: 100%;
            flex: none;
            border-radius: 0;
            border: 0;
        }

        .recommendation-card-title {
            color: var(--ai-text);
            font-weight: 850;
            font-size: 0.98rem;
            height: 2.45rem;
            line-height: 1.22;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            margin-top: 0.7rem;
        }

        .recommendation-card-description {
            color: var(--ai-secondary);
            font-size: 0.84rem;
            height: 2.3rem;
            line-height: 1.35;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            margin-top: 0.45rem;
        }

        .recommendation-card-chips {
            height: 4.1rem;
            min-height: 4.1rem;
            max-height: 4.1rem;
            overflow: hidden;
            margin-top: 0.5rem;
        }

        .card-ingredients {
            height: 4.1rem;
            min-height: 4.1rem;
            max-height: 4.1rem;
            overflow: hidden;
        }

        .card-image-zone {
            height: 178px;
            min-height: 178px;
            max-height: 178px;
            flex: 0 0 178px;
            overflow: hidden;
        }

        .card-image-zone img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

        .card-content-zone {
            padding: 0.75rem;
            height: 210px;
            min-height: 210px;
            max-height: 210px;
            display: grid;
            grid-template-rows: 2.45rem 2.25rem 4.1rem 2.35rem 2.75rem;
            gap: 0.45rem;
        }

        .card-title-zone,
        .card-meta-zone,
        .card-chip-zone,
        .card-description-zone,
        .card-actions-zone {
            overflow: hidden;
        }

        .card-title-zone {
            color: var(--ai-text);
            font-weight: 850;
            font-size: 0.96rem;
            line-height: 1.22;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        .card-meta-zone {
            color: var(--ai-secondary);
            font-size: 0.82rem;
            line-height: 1.32;
        }

        .card-chip-zone {
            display: flex;
            flex-wrap: wrap;
            align-content: flex-start;
            gap: 0.35rem;
        }

        .card-description-zone {
            color: var(--ai-secondary);
            font-size: 0.84rem;
            line-height: 1.35;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        .card-actions-zone {
            border-top: 1px solid rgba(160, 125, 97, 0.22);
        }

        .recommendation-button-spacer {
            flex: 1 1 auto;
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

        input:not(:focus),
        textarea:not(:focus),
        div[data-baseweb="select"] input:not(:focus) {
            caret-color: transparent !important;
        }

        input:focus,
        textarea:focus,
        div[data-baseweb="select"] input:focus {
            caret-color: auto !important;
        }

        div[data-baseweb="select"] input:not(:focus) {
            cursor: default !important;
        }

        div[data-baseweb="select"]:focus-within input {
            cursor: text !important;
        }

        [role="radiogroup"] {
            gap: 0.45rem;
        }

        [role="radiogroup"] label {
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 999px;
            padding: 0.32rem 0.58rem;
            margin-bottom: 0.25rem;
        }

        [role="radiogroup"] label:has(input:checked) {
            background: var(--ai-section);
            border-color: var(--ai-button);
            font-weight: 800;
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

        .card-ingredients,
        .ingredient-chip-row,
        .detail-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-top: 0.55rem;
        }

        .ingredient-chip,
        .detail-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            border: 1px solid var(--ai-accent);
            background: var(--ai-input);
            color: var(--ai-text);
            padding: 0.26rem 0.56rem;
            font-size: 0.78rem;
            font-weight: 750;
            line-height: 1.1;
        }

        .ingredient-chip.muted {
            color: var(--ai-secondary);
            font-weight: 650;
        }

        .detail-badge {
            background: var(--ai-section);
            font-size: 0.84rem;
        }

        .detail-section {
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 12px;
            padding: 0.9rem;
            margin: 0.75rem 0;
        }

        .detail-section-title {
            color: var(--ai-text);
            font-weight: 850;
            margin-bottom: 0.55rem;
        }

        .detail-hero-card {
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 14px;
            padding: 0.75rem;
            box-shadow: 0 14px 32px rgba(74, 38, 8, 0.11);
        }

        .detail-title {
            color: var(--ai-text);
            font-size: 1.8rem;
            font-weight: 900;
            line-height: 1.15;
            margin-bottom: 0.55rem;
        }

        .detail-description {
            color: var(--ai-secondary);
            font-size: 0.98rem;
            line-height: 1.45;
            margin-bottom: 0.8rem;
        }

        .detail-score-pill {
            display: inline-block;
            background: var(--ai-button);
            color: #FFFFFF;
            border-radius: 999px;
            padding: 0.32rem 0.8rem;
            font-weight: 900;
            margin-bottom: 0.75rem;
        }

        .detail-score-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.45rem;
        }

        .detail-score-card {
            background: var(--ai-section);
            border: 1px solid var(--ai-accent);
            border-radius: 8px;
            padding: 0.55rem;
            text-align: center;
        }

        .detail-score-label {
            color: var(--ai-secondary);
            font-size: 0.72rem;
            font-weight: 800;
        }

        .detail-score-value {
            color: var(--ai-text);
            font-size: 1rem;
            font-weight: 900;
        }

        .detail-actions {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.6rem;
            margin-top: 0.9rem;
        }

        .also-try-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin-top: 0.65rem;
        }

        .also-try-card {
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 10px;
            overflow: hidden;
            min-height: 188px;
            box-shadow: 0 8px 18px rgba(74, 38, 8, 0.08);
        }

        .also-try-image {
            width: 100%;
            height: 96px;
            object-fit: cover;
            display: block;
        }

        .also-try-body {
            padding: 0.55rem;
        }

        .also-try-title {
            color: var(--ai-text);
            font-size: 0.82rem;
            font-weight: 850;
            line-height: 1.2;
            height: 2rem;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        .ingredient-card-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.8rem 0;
        }

        .ingredient-card {
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 10px;
            padding: 0.85rem;
            min-height: 110px;
            box-shadow: 0 8px 18px rgba(74, 38, 8, 0.08);
        }

        .ingredient-card-title {
            color: var(--ai-text);
            font-weight: 850;
            margin-bottom: 0.35rem;
        }

        .ingredient-card-meta {
            color: var(--ai-secondary);
            font-size: 0.84rem;
            line-height: 1.35;
        }

        .progress-card {
            background: transparent;
            border-top: 1px solid rgba(118, 86, 58, 0.35);
            border-radius: 0;
            padding: 0.95rem 0.1rem 0.25rem 0.1rem;
            margin-top: 1.35rem;
            box-shadow: none;
        }

        .sidebar-progress-push {
            height: clamp(1.5rem, 10vh, 5.5rem);
        }

        .progress-name {
            color: var(--ai-text);
            font-weight: 900;
            font-size: 0.92rem;
            line-height: 1.2;
        }

        .progress-title {
            color: var(--ai-button);
            font-weight: 850;
            font-size: 0.88rem;
            margin-top: 0.32rem;
        }

        .progress-level {
            color: var(--ai-secondary);
            font-size: 0.78rem;
            font-weight: 700;
            margin-top: 0.28rem;
        }

        .mug-wrap {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.35rem;
            margin: 0.62rem 0 0.35rem 0;
        }

        .coffee-mug {
            position: relative;
            width: 88px;
            height: 54px;
            border: 2px solid var(--ai-button);
            border-top: 0;
            border-radius: 0 0 20px 20px;
            background: linear-gradient(180deg, rgba(255, 253, 248, 0.78), rgba(208, 188, 168, 0.38));
            overflow: visible;
            box-shadow: inset 0 -5px 10px rgba(74, 38, 8, 0.08);
        }

        .coffee-mug::after {
            content: "";
            position: absolute;
            right: -18px;
            top: 12px;
            width: 22px;
            height: 25px;
            border: 2px solid var(--ai-button);
            border-left: 0;
            border-radius: 0 14px 14px 0;
            background: transparent;
        }

        .coffee-mug::before {
            content: "";
            position: absolute;
            left: -9px;
            right: -17px;
            bottom: -9px;
            height: 7px;
            border-radius: 999px;
            background: rgba(160, 125, 97, 0.34);
        }

        .coffee-fill {
            position: absolute;
            left: 5px;
            right: 5px;
            bottom: 0;
            height: var(--fill);
            max-height: calc(100% - 5px);
            background: linear-gradient(180deg, #A07D61 0%, #76563A 48%, #4A2608 100%);
            border-radius: 10px 10px 16px 16px;
            box-shadow: inset 0 2px 0 rgba(255, 253, 248, 0.25);
        }

        .progress-percent {
            color: var(--ai-text);
            font-weight: 850;
            font-size: 0.8rem;
            line-height: 1.1;
            white-space: nowrap;
        }

        .progress-xp {
            color: var(--ai-secondary);
            font-size: 0.82rem;
            font-weight: 850;
            margin-top: 0.32rem;
        }

        .progress-stats {
            display: block;
            margin-top: 0.45rem;
            color: var(--ai-secondary);
            font-size: 0.76rem;
            font-weight: 800;
            line-height: 1.35;
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

        .category-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 0.65rem 0 1.6rem 0;
        }

        .category-card {
            display: block;
            overflow: hidden;
            min-height: 250px;
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 10px;
            text-decoration: none !important;
            box-shadow: 0 10px 24px rgba(74, 38, 8, 0.10);
            transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
        }

        .category-card:hover {
            transform: translateY(-2px);
            border-color: var(--ai-button);
            box-shadow: 0 15px 32px rgba(74, 38, 8, 0.16);
        }

        .category-card-image {
            width: 100%;
            height: 145px;
            object-fit: cover;
            display: block;
        }

        .category-card-body {
            padding: 0.82rem;
        }

        .category-card-title {
            color: var(--ai-text);
            font-size: 1rem;
            font-weight: 850;
            margin-bottom: 0.35rem;
        }

        .category-card-description {
            color: var(--ai-secondary);
            font-size: 0.85rem;
            line-height: 1.35;
        }

        .homepage-feature {
            background: var(--ai-section);
            border: 1px solid var(--ai-accent);
            border-radius: 14px;
            padding: 1rem;
            margin: 1.2rem 0;
            box-shadow: 0 14px 32px rgba(74, 38, 8, 0.10);
        }

        .section-subtitle {
            color: var(--ai-secondary);
            font-size: 0.95rem;
            margin: -0.2rem 0 0.85rem 0;
        }

        .custom-card-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.8rem 0;
        }

        .custom-drink-card {
            background: var(--ai-input);
            border: 1px solid var(--ai-accent);
            border-radius: 10px;
            overflow: hidden;
            height: 390px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 10px 24px rgba(74, 38, 8, 0.10);
        }

        .custom-drink-body {
            padding: 0.75rem;
            display: flex;
            flex-direction: column;
            flex: 1 1 auto;
        }

        .custom-drink-title {
            color: var(--ai-text);
            font-weight: 900;
            font-size: 1rem;
            line-height: 1.2;
            height: 2.45rem;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        .custom-drink-meta {
            color: var(--ai-secondary);
            font-size: 0.82rem;
            line-height: 1.35;
            height: 2.25rem;
            overflow: hidden;
            margin: 0.35rem 0;
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

        @media (min-width: 1025px) {
            .block-container {
                max-width: 1280px;
                padding-left: 2.6rem;
                padding-right: 2.6rem;
            }

            .ai-title-section {
                min-height: 360px;
                padding: 5rem 3.4rem;
            }

            .ai-title-section h1 {
                font-size: 5rem;
            }

            .rail-card-image,
            .card-drink-image,
            .favorite-drink-image {
                height: 190px;
            }

            .recommendation-feature-image {
                height: 390px;
            }

            .detail-drink-image {
                height: 420px;
            }
        }

        @media (min-width: 769px) and (max-width: 1024px) {
            .block-container {
                max-width: 960px;
                padding-left: 1.4rem;
                padding-right: 1.4rem;
            }

            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
            }

            [data-testid="column"] {
                min-width: calc(33.333% - 1rem) !important;
                width: calc(33.333% - 1rem) !important;
                flex: 1 1 calc(33.333% - 1rem) !important;
            }

            .ai-title-section {
                min-height: 300px;
                padding: 3.6rem 2.2rem;
            }

            .ai-title-section h1 {
                font-size: 4rem;
            }

            .ai-title-section p {
                font-size: 1.05rem;
                max-width: 500px;
            }

            .rail-card {
                height: 360px;
                min-height: 360px;
                max-height: 360px;
            }

            .secondary-card {
                height: 360px;
                min-height: 360px;
                max-height: 360px;
            }

            .rail-card-image,
            .card-drink-image,
            .favorite-drink-image {
                height: 165px;
            }

            .card-image-zone {
                height: 165px;
                min-height: 165px;
                max-height: 165px;
                flex-basis: 165px;
            }

            .card-content-zone {
                height: 195px;
                min-height: 195px;
                max-height: 195px;
                grid-template-rows: 2.35rem 2.05rem 3.75rem 2.1rem 2.45rem;
                gap: 0.38rem;
            }

            .recommendation-feature-image {
                height: 300px;
            }

            .detail-drink-image {
                height: 320px;
            }

            .profile-card,
            .taste-section,
            .ai-card {
                border-radius: 14px;
            }

            .category-grid {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }

            .custom-card-grid {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }

            .ingredient-card-grid {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }

            .detail-score-grid {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }
        }

        @media (max-width: 768px) {
            .block-container {
                padding: 1rem 0.9rem 2rem 0.9rem;
                max-width: 100%;
            }

            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
            }

            [data-testid="stSidebar"] {
                background: var(--ai-input);
                border-right: 0;
                border-bottom: 1px solid var(--ai-accent);
            }

            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
                gap: 0.35rem;
            }

            .ai-title-section {
                min-height: 230px;
                padding: 2.4rem 1.15rem;
                border-radius: 8px;
                margin-bottom: 0.9rem;
                background-position: center;
            }

            .ai-title-section h1 {
                font-size: 2.65rem;
                line-height: 1.04;
            }

            .ai-title-section p {
                font-size: 0.98rem;
                line-height: 1.45;
                max-width: 100%;
            }

            .home-section-title {
                font-size: 1.08rem;
                margin-top: 1rem;
            }

            [data-testid="column"] {
                min-width: 100% !important;
                width: 100% !important;
                flex: 1 1 100% !important;
            }

            .rail-card,
            .secondary-card,
            .ai-card,
            .profile-card,
            .taste-section,
            .experimental-card {
                border-radius: 8px;
                margin-bottom: 0.85rem;
            }

            .rail-card {
                height: 356px;
                min-height: 356px;
                max-height: 356px;
            }

            .secondary-card {
                height: 356px;
                min-height: 356px;
                max-height: 356px;
            }

            .secondary-card .rail-card-image {
                height: 100%;
                flex-basis: auto;
            }

            .rail-card-image,
            .card-drink-image,
            .favorite-drink-image {
                height: 160px;
            }

            .card-image-zone {
                height: 160px;
                min-height: 160px;
                max-height: 160px;
                flex-basis: 160px;
            }

            .recommendation-feature-image {
                height: 220px;
            }

            .detail-drink-image {
                height: 230px;
            }

            .rail-card-body {
                height: 196px;
                min-height: 196px;
                max-height: 196px;
                padding: 0.7rem;
            }

            .card-content-zone {
                height: 196px;
                min-height: 196px;
                max-height: 196px;
                padding: 0.7rem;
                grid-template-rows: 2.3rem 2.2rem 3.75rem 2.2rem 2.35rem;
                gap: 0.36rem;
            }

            .rail-card-title {
                height: 2.3rem;
                min-height: 2.3rem;
                font-size: 0.95rem;
            }

            .rail-card-meta {
                height: 2.2rem;
                min-height: 2.2rem;
            }

            .recommendation-feature {
                padding: 0.85rem;
                border-radius: 8px;
            }

            .ai-card-title,
            .profile-name {
                font-size: 1.05rem;
            }

            .profile-card-header {
                align-items: flex-start;
            }

            .profile-avatar {
                width: 52px;
                height: 52px;
                flex-basis: 52px;
                font-size: 1rem;
            }

            .profile-badge,
            .taste-chip {
                font-size: 0.8rem;
                padding: 0.32rem 0.58rem;
            }

            h2 {
                font-size: 1.35rem !important;
            }

            h3 {
                font-size: 1.08rem !important;
            }

            button:not([role="tab"]),
            div[data-testid="stFormSubmitButton"] button {
                width: 100% !important;
                min-height: 44px;
                font-size: 0.95rem;
            }

            input,
            textarea,
            [role="combobox"] {
                min-height: 42px;
                font-size: 0.95rem !important;
            }

            .category-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.7rem;
            }

            .custom-card-grid {
                grid-template-columns: 1fr;
            }

            .category-card {
                min-height: 218px;
            }

            .category-card-image {
                height: 118px;
            }

            .category-card-body {
                padding: 0.68rem;
            }

            .category-card-title {
                font-size: 0.92rem;
            }

            .category-card-description {
                font-size: 0.78rem;
            }

            .ingredient-card-grid {
                grid-template-columns: repeat(1, minmax(0, 1fr));
            }

            .detail-title {
                font-size: 1.35rem;
            }

            .detail-score-grid,
            .detail-actions {
                grid-template-columns: 1fr;
            }

            .also-try-grid {
                grid-template-columns: 1fr;
            }

            .sidebar-progress-push {
                height: 0.75rem;
            }
        }
        </style>
        """.replace("__HERO_IMAGE_URL__", HERO_IMAGE_URL),
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
    if "recommendation_fallback_message" not in st.session_state:
        st.session_state.recommendation_fallback_message = None
    if "duplicate_profile_to_load" not in st.session_state:
        st.session_state.duplicate_profile_to_load = None
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = None


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
        goal = fixed_choice(
            "Goal",
            GOAL_OPTIONS,
            key=f"{prefix}_goal",
        )
    with col2:
        stress_level = fixed_choice(
            "Stress level",
            STRESS_OPTIONS,
            key=f"{prefix}_stress",
            default_index=2,
        )
        weather = fixed_choice(
            "Weather",
            WEATHER_OPTIONS,
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


def _drink_dict(drink: object) -> dict[str, object]:
    """Return drink data as a plain dictionary for shared render helpers."""
    if hasattr(drink, "to_dict"):
        return drink.to_dict()
    if isinstance(drink, dict):
        return drink
    return {}


def drink_image_url(drink: object) -> str:
    """Resolve the shared drink image URL."""
    return get_drink_image(_drink_dict(drink))


def drink_image_alt(drink: object) -> str:
    """Resolve the shared drink image alt text."""
    return get_drink_image_alt(_drink_dict(drink))


def drink_image_html(drink: object, css_class: str) -> str:
    """Render a consistently styled drink image."""
    return (
        f'<img class="{safe_text(css_class)}" '
        f'src="{safe_text(drink_image_url(drink))}" '
        f'alt="{safe_text(drink_image_alt(drink))}">'
    )


GOAL_OPTIONS = ["Energy", "Focus", "Comfort", "Workout", "Treat"]
WEATHER_OPTIONS = ["Not specified", "Sunny", "Cloudy", "Rainy", "Cold", "Hot"]
STRESS_OPTIONS = ["Not specified", "Low", "Medium", "High"]
CAFFEINE_OPTIONS = ["Any", "None", "Low", "Medium", "High"]
SWEETNESS_OPTIONS = ["Any", "Unsweetened", "Light", "Classic", "Extra"]
TEMPERATURE_OPTIONS = ["No preference", "Iced", "Hot"]
BASE_OPTIONS = [
    "Espresso",
    "Latte",
    "Cold Brew",
    "Americano",
    "Mocha",
    "Matcha",
    "Chai",
    "Tea",
    "Refresher",
]
SIZE_OPTIONS = ["Short", "Tall", "Grande", "Venti"]
SYRUP_OPTIONS = [
    "None",
    "Vanilla",
    "Caramel",
    "Mocha",
    "Hazelnut",
    "Brown Sugar",
    "Peppermint",
    "Cinnamon Dolce",
]
MILK_OPTIONS = [
    "No preference",
    "Whole milk",
    "2%",
    "Nonfat",
    "Oat milk",
    "Almond milk",
    "Soy milk",
    "Coconut milk",
]
CAFFEINE_TOLERANCE_OPTIONS = ["No preference", "Low", "Medium", "High"]


def _choice_value(label: str) -> str:
    """Convert display labels to stored preference values."""
    mapping = {
        "No preference": "no preference",
        "Not specified": "not specified",
        "Iced": "iced",
        "Hot": "hot",
        "Whole milk": "whole",
        "Oat milk": "oat",
        "Almond milk": "almond",
        "Soy milk": "soy",
        "Coconut milk": "coconut",
        "None": "none",
    }
    return mapping.get(label, label.lower())


def fixed_choice(
    label: str,
    options: list[str],
    key: str,
    default_index: int = 0,
    horizontal: bool = True,
) -> str:
    """Render fixed options as tap/click radio choices instead of selectboxes."""
    selected = st.radio(
        label,
        options,
        index=default_index,
        horizontal=horizontal,
        key=key,
    )
    return _choice_value(selected)


def _split_tags(value: object, limit: int = 6) -> list[str]:
    """Split comma-ish display values into clean labels."""
    if value is None or pd.isna(value):
        return []
    parts = re.split(r"[,;/|]", str(value))
    return [part.strip() for part in parts if part.strip() and part.strip().lower() != "nan"][:limit]


def _drink_ingredients(drink: object, limit: int = 5) -> list[str]:
    """Return compact ingredient labels for cards and details."""
    data = _drink_dict(drink)
    labels = []
    for field in ("base", "milk", "syrup", "toppings"):
        value = str(data.get(field, "")).strip()
        if value and value.lower() not in {"nan", "none", "no syrup", "no milk"}:
            labels.append(value)
    if not labels:
        labels = _split_tags(data.get("flavor_profile", ""), limit=limit)
    return labels[:limit]


def chips_html(items: list[object], css_class: str = "ingredient-chip") -> str:
    """Render small rounded chips."""
    if not items:
        return '<span class="ingredient-chip muted">Ingredients coming soon</span>'
    return "".join(f'<span class="{css_class}">{safe_text(item)}</span>' for item in items)


def limited_chips_html(
    items: list[object],
    limit: int = 3,
    css_class: str = "ingredient-chip",
) -> str:
    """Render a fixed-size chip preview with a +N overflow chip."""
    clean_items = [item for item in items if str(item).strip()]
    visible = clean_items[:limit]
    html = chips_html(visible, css_class)
    remaining = max(0, len(clean_items) - len(visible))
    if remaining:
        html += f'<span class="{css_class} muted">+{remaining} more</span>'
    return html


def drink_badges_html(drink: object) -> str:
    """Render drink detail badges."""
    data = _drink_dict(drink)
    badges = [
        f"Caffeine: {data.get('caffeine_level', 'unknown')}",
        f"Sweetness: {data.get('sweetness_level', 'unknown')}",
        f"{data.get('calories', 'N/A')} calories",
    ]
    badges.extend(_split_tags(data.get("dietary_tags", ""), limit=4))
    return chips_html(badges, "detail-badge")


def _list_field(value: object) -> list[object]:
    """Return list-like recommendation fields safely."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None or pd.isna(value):
        return []
    return _split_tags(value, limit=6)


def render_score_breakdown(drink: object) -> None:
    """Render scoring details with native Streamlit cards so no raw HTML can leak."""
    data = _drink_dict(drink)
    parts = [
        ("Context", data.get("context_score", 0)),
        ("Profile", data.get("profile_score", 0)),
        ("Ingredients", data.get("ingredient_preference_score", 0)),
        ("Past ratings", data.get("past_rating_score", 0)),
        ("Close match", data.get("fallback_similarity_score", 0)),
    ]
    columns = st.columns(len(parts))
    for column, (label, value) in zip(columns, parts):
        with column:
            with st.container(border=True):
                st.caption(label)
                st.markdown(f"**{int(_safe_float(value))}**")


def detail_summary_html(drink: object) -> str:
    """Render compact drink detail summary cards."""
    data = _drink_dict(drink)
    confidence = data.get("selector_confidence", None)
    if confidence is not None and str(confidence) != "nan":
        score_text = f"{int(_safe_float(confidence))}% confidence"
    else:
        score_text = f"{_match_percentage(data.get('recommendation_score'))}% match"
    reason = str(
        data.get(
            "selector_reasoning",
            data.get("recommendation_explanation", "Recommended for your taste."),
        )
    )
    preference_chips = limited_chips_html(_list_field(data.get("matched_preferences")), limit=5, css_class="detail-badge")
    context_chips = limited_chips_html(_list_field(data.get("matched_context")), limit=5, css_class="detail-badge")
    creator_section = ""
    if str(data.get("drink_id", "")).startswith("CUS-"):
        creator = data.get("creator_name") or "Unknown"
        creator_section = f"""
        <div class="detail-section">
            <div class="detail-section-title">Creator</div>
            <div class="detail-description">Created by {safe_text(creator)}</div>
        </div>
        """
    return f"""
        <div class="detail-title">{safe_text(data.get("drink_name", "Drink details"))}</div>
        <div class="detail-score-pill">{safe_text(score_text)}</div>
        <div class="detail-description">{safe_text(_drink_description(data))}</div>
        {creator_section}
        <div class="detail-section">
            <div class="detail-section-title">Why this was recommended</div>
            <div class="detail-description">{safe_text(reason)}</div>
        </div>
        <div class="detail-section">
            <div class="detail-section-title">Matched Preferences</div>
            <div class="detail-badge-row">{preference_chips}</div>
        </div>
        <div class="detail-section">
            <div class="detail-section-title">Matched Context</div>
            <div class="detail-badge-row">{context_chips}</div>
        </div>
        <div class="detail-section">
            <div class="detail-section-title">Ingredients</div>
            <div class="ingredient-chip-row">{limited_chips_html(_drink_ingredients(data, limit=8), limit=6)}</div>
        </div>
        <div class="detail-section">
            <div class="detail-section-title">Drink Profile</div>
            <div class="detail-badge-row">{drink_badges_html(data)}</div>
        </div>
    """


def render_drink_card_grid(
    drinks: pd.DataFrame,
    key_prefix: str,
    limit: int = 3,
    mode: str = "home",
) -> None:
    """Render reusable compact drink cards with safe HTML rendering and native buttons."""
    if drinks is None or drinks.empty:
        st.info("Similar drinks will appear here.")
        return

    rows = list(drinks.head(limit).iterrows())
    columns = st.columns(len(rows))
    for index, (column, (_, drink)) in enumerate(zip(columns, rows)):
        drink_dict = drink.to_dict()
        drink_id = drink.get("drink_id")
        with column:
            st.markdown(
                f"""
                <div class="rail-card">
                    <div class="card-image-zone">{drink_image_html(drink_dict, "rail-card-image")}</div>
                    <div class="card-content-zone">
                        <div class="card-title-zone">{safe_text(drink.get("drink_name", "Drink"))}</div>
                        <div class="card-meta-zone">{safe_text(_rail_meta(drink))}</div>
                        <div class="card-chip-zone">{limited_chips_html(_drink_ingredients(drink_dict, limit=6), limit=3)}</div>
                        <div class="card-description-zone">{safe_text(_drink_description(drink_dict))}</div>
                        <div class="card-actions-zone"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "View Details",
                key=f"{key_prefix}_{key_slug(drink_id)}_{index}_view_details",
                use_container_width=True,
            ):
                st.session_state.selected_drink_id = drink_id
                st.session_state.selected_drink = drink_dict
                if mode == "guided":
                    _set_flow_step(5)
                else:
                    st.session_state.home_view = "details"
                    st.rerun()


def profile_name_exists(name: str) -> dict[str, object] | None:
    """Return an existing profile with the same name, if present."""
    query = str(name).strip().lower()
    if not query:
        return None
    users = load_supabase_users()
    if users.empty:
        return None
    matches = users[users["name"].astype(str).str.strip().str.lower() == query]
    return None if matches.empty else matches.iloc[0].to_dict()


def render_duplicate_profile_prompt(destination: str = "profile") -> None:
    """Render a persistent duplicate-profile load prompt."""
    existing_user = st.session_state.get("duplicate_profile_to_load")
    if not existing_user:
        return
    st.warning("Profile already exists. Load this profile instead?")
    if st.button(
        f"Load {existing_user.get('name', 'profile')}",
        key=f"{destination}_load_duplicate_{key_slug(existing_user.get('user_id', 'profile'))}",
        use_container_width=True,
    ):
        st.session_state.current_user = existing_user
        st.session_state.duplicate_profile_to_load = None
        st.session_state.flow_message = f"Welcome back, {existing_user['name']}."
        if destination == "guided":
            st.session_state.flow_step = 3
            st.session_state.current_step = 3
            st.session_state.selected_page = "guided"
            st.rerun()
        _go_to_page(destination)


def _safe_float(value: object, fallback: float = 0.0) -> float:
    """Convert display values to floats without crashing on missing data."""
    try:
        if pd.isna(value):
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


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
                {drink_image_html(drink, "card-drink-image")}
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
                st.markdown(
                    drink_image_html(drink, "card-drink-image"),
                    unsafe_allow_html=True,
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
    award_xp(user_id, "save_favorite")
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
                    st.markdown(
                        drink_image_html(drink, "favorite-drink-image"),
                        unsafe_allow_html=True,
                    )
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


def render_recently_rated_section(user_id: str, limit: int = 6) -> None:
    """Render recently rated drinks as compact cards."""
    st.markdown("### Recently Rated Drinks")
    history = get_user_history(user_id)
    if history.empty:
        st.info("Rate a few drinks and they will appear here.")
        return
    drinks = history.tail(limit).merge(st.session_state.drinks, on="drink_id", how="left")
    if "drink_name_x" in drinks.columns:
        drinks["drink_name"] = drinks["drink_name_y"].fillna(drinks["drink_name_x"])
    render_drink_rail("Recently Rated Drinks", drinks, "profile_recent", limit=limit, score_column="rating")


def render_custom_creations_section(limit: int = 6, user_id: str | None = None) -> None:
    """Render custom drinks as compact cards."""
    custom = st.session_state.drinks[
        st.session_state.drinks["drink_id"].astype(str).str.startswith("CUS-")
    ].copy()
    title = "Custom Creations"
    empty_text = "Create a custom drink and it will appear here."
    if user_id:
        title = "My Custom Drinks"
        empty_text = "Create a custom drink and it will appear here."
        if "creator_user_id" in custom.columns:
            custom = custom[
                custom["creator_user_id"].fillna("").astype(str).str.lower()
                == str(user_id).lower()
            ]
        else:
            custom = custom.iloc[0:0]
    st.markdown(f"### {title}")
    if custom.empty:
        st.info(empty_text)
        return
    render_custom_drink_cards(custom.head(limit), "profile_custom", limit=limit)


def render_ingredient_cards(ingredients: pd.DataFrame, limit: int = 24) -> None:
    """Render ingredients as styled cards instead of a raw table."""
    if ingredients.empty:
        st.info("No ingredients found.")
        return
    cards = []
    for _, ingredient in ingredients.head(limit).iterrows():
        calories = _safe_float(ingredient.get("calories", 0))
        caffeine = _safe_float(ingredient.get("caffeine", 0))
        price = _safe_float(ingredient.get("price", 0))
        cards.append(
            f"""
            <div class="ingredient-card">
                <div class="ingredient-card-title">{safe_text(ingredient.get("ingredient_name", "Ingredient"))}</div>
                <div class="ingredient-card-meta">
                    {safe_text(ingredient.get("category", "category"))}<br>
                    {calories:g} cal Â· {caffeine:g} mg caffeine<br>
                    ${price:.2f} per {safe_text(ingredient.get("default_unit", "serving"))}
                </div>
            </div>
            """
        )
    st.markdown(
        f'<div class="ingredient-card-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_ingredient_cards(ingredients: pd.DataFrame, limit: int = 24) -> None:
    """Render ingredients as native cards so raw HTML never appears."""
    if ingredients.empty:
        st.info("No ingredients found.")
        return
    rows = list(ingredients.head(limit).iterrows())
    for start in range(0, len(rows), 4):
        row_items = rows[start : start + 4]
        columns = st.columns(len(row_items))
        for column, (_, ingredient) in zip(columns, row_items):
            calories = _safe_float(ingredient.get("calories", 0))
            caffeine = _safe_float(ingredient.get("caffeine", 0))
            price = _safe_float(ingredient.get("price", 0))
            with column:
                with st.container(border=True):
                    st.markdown(f"**{ingredient.get('ingredient_name', 'Ingredient')}**")
                    st.caption(str(ingredient.get("category", "category")))
                    st.write(f"{calories:g} cal")
                    st.write(f"{caffeine:g} mg caffeine")
                    st.write(f"${price:.2f} per {ingredient.get('default_unit', 'serving')}")


def _custom_drinks() -> pd.DataFrame:
    """Return saved user-created drinks from the loaded drink catalog."""
    drinks = st.session_state.drinks
    if drinks.empty or "drink_id" not in drinks.columns:
        return pd.DataFrame(columns=drinks.columns)
    return drinks[drinks["drink_id"].astype(str).str.startswith("CUS-")].copy()


def render_custom_drink_cards(
    custom_drinks: pd.DataFrame,
    key_prefix: str,
    limit: int = 8,
) -> None:
    """Render custom drinks as polished cards with detail actions."""
    if custom_drinks is None or custom_drinks.empty:
        st.info("No custom drinks yet. Create one and it will appear here.")
        return

    rows = list(custom_drinks.head(limit).iterrows())
    columns = st.columns(min(4, len(rows)))
    for index, (column, (_, drink)) in enumerate(zip(columns * ((len(rows) // len(columns)) + 1), rows)):
        drink_dict = drink.to_dict()
        drink_id = drink.get("drink_id")
        creator = drink.get("creator_name") or drink.get("created_by") or "Unknown"
        rating_label = ""
        if "avg_rating" in drink and not pd.isna(drink.get("avg_rating")):
            rating_label = f"Rating {float(drink.get('avg_rating')):.1f}/5"
        meta = (
            f"Created by {safe_text(creator)}<br>"
            f"{safe_text(drink.get('caffeine_level', 'unknown'))} caffeine · "
            f"{safe_text(drink.get('sweetness_level', 'classic'))} sweetness"
        )
        if rating_label:
            meta += f"<br>{safe_text(rating_label)}"
        with column:
            st.markdown(
                f"""
                <div class="custom-drink-card">
                    <div class="card-image-zone">{drink_image_html(drink_dict, "rail-card-image")}</div>
                    <div class="card-content-zone">
                        <div class="card-title-zone">{safe_text(drink.get("drink_name", "Custom drink"))}</div>
                        <div class="card-meta-zone">{meta}</div>
                        <div class="card-chip-zone">{limited_chips_html(_drink_ingredients(drink_dict, limit=6), limit=3)}</div>
                        <div class="card-description-zone">{safe_text(_drink_description(drink_dict))}</div>
                        <div class="card-actions-zone"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "View Details",
                key=f"{key_prefix}_{key_slug(drink_id)}_{index}_view_details",
                use_container_width=True,
            ):
                _open_home_drink(drink_id)


def render_custom_drinks_home_section() -> None:
    """Render custom drink discovery and creation on the homepage."""
    st.markdown('<div class="homepage-feature">', unsafe_allow_html=True)
    st.markdown('<div class="home-section-title">Custom Drinks</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Create your own drink or explore drinks made by users.</div>',
        unsafe_allow_html=True,
    )
    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        if st.button("Create Custom Drink", type="primary", key="home_create_custom_drink", use_container_width=True):
            st.session_state.show_custom_drink_builder = not st.session_state.get(
                "show_custom_drink_builder",
                False,
            )
    with col2:
        if st.button("Explore Custom Drinks", key="home_explore_custom_drinks", use_container_width=True):
            st.session_state.show_custom_drinks = True

    if st.session_state.get("show_custom_drink_builder", False):
        with st.expander("Create Custom Drink", expanded=True):
            custom_drink_section(show_catalog=False)

    custom_drinks = _custom_drinks()
    ratings = load_ratings()
    if not custom_drinks.empty and not ratings.empty:
        rating_values = ratings.copy()
        rating_values["rating"] = pd.to_numeric(rating_values["rating"], errors="coerce")
        custom_ratings = (
            rating_values.groupby("drink_id")["rating"]
            .mean()
            .reset_index(name="avg_rating")
        )
        custom_drinks = custom_drinks.merge(custom_ratings, on="drink_id", how="left")

    if not custom_drinks.empty:
        st.markdown("#### Recent custom drinks")
        render_custom_drink_cards(custom_drinks.tail(8).iloc[::-1], "home_custom_recent", limit=4)
        popular_custom = custom_drinks.dropna(subset=["avg_rating"]) if "avg_rating" in custom_drinks.columns else pd.DataFrame()
        if not popular_custom.empty:
            st.markdown("#### Popular custom drinks")
            render_custom_drink_cards(
                popular_custom.sort_values("avg_rating", ascending=False),
                "home_custom_popular",
                limit=4,
            )
    else:
        st.info("Create the first custom drink for this profile.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_ingredients_home_section() -> None:
    """Render ingredient discovery and add-ingredient entry point."""
    st.markdown('<div class="homepage-feature">', unsafe_allow_html=True)
    st.markdown('<div class="home-section-title">Ingredients</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Add ingredients and use them in custom drinks.</div>',
        unsafe_allow_html=True,
    )
    if st.button("Add Ingredient", key="home_add_ingredient", use_container_width=True):
        st.session_state.show_add_ingredient = not st.session_state.get("show_add_ingredient", False)

    if st.session_state.get("show_add_ingredient", False):
        with st.expander("Add Ingredient", expanded=True):
            add_ingredient_section(show_list=False)

    ingredients = st.session_state.ingredients.sort_values(["category", "ingredient_name"])
    categories = ["All"] + [
        category.title() if category != "add-in" else "Add-in"
        for category in sorted(ingredients["category"].dropna().astype(str).unique())
    ]
    selected_category = fixed_choice(
        "Ingredient category",
        categories,
        "home_ingredient_category_filter",
    )
    filtered = ingredients
    if selected_category != "all":
        filtered = ingredients[ingredients["category"].astype(str).str.lower() == selected_category]
    render_ingredient_cards(filtered, limit=12)
    st.markdown("</div>", unsafe_allow_html=True)


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
                    <div class="card-image-zone">{drink_image_html(drink_dict, "rail-card-image")}</div>
                    <div class="card-content-zone">
                        <div class="card-title-zone">{safe_text(drink.get("drink_name", "Drink"))}</div>
                        <div class="card-meta-zone">{safe_text(_rail_meta(drink, score_label))}</div>
                        <div class="card-chip-zone">{limited_chips_html(_drink_ingredients(drink_dict, limit=6), limit=3)}</div>
                        <div class="card-description-zone">{safe_text(_drink_description(drink_dict))}</div>
                        <div class="card-actions-zone"></div>
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


CATEGORY_CARDS = [
    {
        "key": "espresso",
        "name": "Espresso",
        "description": "Bold espresso-forward drinks with a clean coffee finish.",
        "sample": {"drink_name": "Espresso", "base": "Espresso"},
    },
    {
        "key": "lattes",
        "name": "Lattes",
        "description": "Creamy milk-based coffee drinks for smooth everyday sipping.",
        "sample": {"drink_name": "Hot Latte", "base": "Latte", "temperature": "hot"},
    },
    {
        "key": "refreshers",
        "name": "Refreshers",
        "description": "Bright, chilled fruit-style drinks for a lighter pick-me-up.",
        "sample": {"drink_name": "Refresher", "base": "Refresher", "temperature": "iced"},
    },
    {
        "key": "matcha",
        "name": "Matcha",
        "description": "Earthy green tea drinks with a smooth cafe feel.",
        "sample": {"drink_name": "Matcha Latte", "base": "Matcha Latte"},
    },
    {
        "key": "cold_brew",
        "name": "Cold Brew",
        "description": "Chilled, slow-steeped coffee drinks with a bold profile.",
        "sample": {"drink_name": "Cold Brew", "base": "Cold Brew", "temperature": "iced"},
    },
]


def _category_matches(drinks: pd.DataFrame, category_key: str) -> pd.DataFrame:
    """Return drinks matching a homepage discovery category."""
    text = (
        drinks.get("drink_name", pd.Series(dtype=str)).astype(str).str.lower()
        + " "
        + drinks.get("base", pd.Series(dtype=str)).astype(str).str.lower()
    )
    if category_key == "espresso":
        shots = pd.to_numeric(drinks.get("espresso_shots", 0), errors="coerce").fillna(0)
        mask = text.str.contains("espresso|americano|cappuccino", na=False) | (shots > 0)
    elif category_key == "lattes":
        mask = text.str.contains("latte", na=False) & ~text.str.contains(
            "matcha|chai",
            na=False,
        )
    elif category_key == "refreshers":
        mask = text.str.contains("refresher", na=False)
    elif category_key == "matcha":
        mask = text.str.contains("matcha", na=False)
    elif category_key == "cold_brew":
        mask = text.str.contains("cold brew", na=False)
    else:
        mask = pd.Series(False, index=drinks.index)
    return drinks[mask].copy()


def render_category_explorer(drinks: pd.DataFrame) -> None:
    """Render clickable category cards and selected category drinks."""
    st.markdown('<div class="home-section-title">Explore Categories</div>', unsafe_allow_html=True)
    cards = []
    for category in CATEGORY_CARDS:
        sample = category["sample"]
        href = f"?category={quote(category['key'])}"
        cards.append(
            f"""
            <a class="category-card" href="{href}">
                {drink_image_html(sample, "category-card-image")}
                <div class="category-card-body">
                    <div class="category-card-title">{safe_text(category["name"])}</div>
                    <div class="category-card-description">{safe_text(category["description"])}</div>
                </div>
            </a>
            """
        )
    st.markdown(
        f'<div class="category-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )

    selected_category = st.query_params.get("category")
    category = next((item for item in CATEGORY_CARDS if item["key"] == selected_category), None)
    if category:
        matches = _category_matches(drinks, category["key"])
        render_drink_rail(
            f"{category['name']} Picks",
            matches,
            f"category_{category['key']}",
            empty_text=f"No {category['name'].lower()} drinks are available yet.",
        )
        st.markdown('<a class="ai-card-meta" href="?">Clear category</a>', unsafe_allow_html=True)


def render_category_explorer(drinks: pd.DataFrame) -> None:
    """Render category discovery with native Streamlit cards and buttons."""
    st.markdown("### Explore Categories")
    columns = st.columns(len(CATEGORY_CARDS))
    for column, category in zip(columns, CATEGORY_CARDS):
        with column:
            with st.container(border=True):
                st.image(
                    drink_image_url(category["sample"]),
                    caption=category["name"],
                    use_container_width=True,
                )
                st.caption(category["description"])
                if st.button(
                    f"View {category['name']}",
                    key=f"home_category_{category['key']}",
                    use_container_width=True,
                ):
                    st.session_state.selected_category = category["key"]
                    st.rerun()

    selected_category = st.session_state.get("selected_category")
    category = next((item for item in CATEGORY_CARDS if item["key"] == selected_category), None)
    if category:
        matches = _category_matches(drinks, category["key"])
        render_drink_rail(
            f"{category['name']} Picks",
            matches,
            f"category_{category['key']}",
            empty_text=f"No {category['name'].lower()} drinks are available yet.",
        )
        if st.button("Clear category", key="clear_home_category"):
            st.session_state.selected_category = None
            st.rerun()


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

    similar = find_similar_drinks(drinks, drink, limit=3)
    col1, col2 = st.columns([1, 1.25])
    with col1:
        st.markdown(
            f"""
            <div class="detail-hero-card">
                {drink_image_html(drink, "detail-drink-image")}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Also try")
        render_drink_card_grid(similar, "detail_also_try", limit=3, mode="home")
    with col2:
        st.markdown(detail_summary_html(drink), unsafe_allow_html=True)
        with st.expander("View scoring details"):
            render_score_breakdown(drink)

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
                award_xp(user["user_id"], "rate_drink")
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
            goal = fixed_choice("Goal", GOAL_OPTIONS, "consumer_goal")
        with col2:
            temperature = fixed_choice(
                "Temperature preference",
                TEMPERATURE_OPTIONS,
                "consumer_temperature",
            )
        with st.expander("Optional refinement", expanded=False):
            milk = fixed_choice(
                "Milk",
                ["Any"] + MILK_OPTIONS[1:],
                "consumer_milk",
            )
            caffeine = fixed_choice("Caffeine", CAFFEINE_OPTIONS, "consumer_caffeine")
            sweetness = fixed_choice(
                "Sweetness",
                SWEETNESS_OPTIONS,
                "consumer_sweetness",
            )
            likes = st.text_input("Things I love today")
            dislikes = st.text_input("Things I want to avoid")
        submitted = st.form_submit_button(
            "Get Recommendation",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        matches, _, _ = recommend_with_fallback(
            drinks=st.session_state.drinks,
            temperature=None if temperature == "no preference" else temperature,
            milk=None if milk == "any" else milk,
            caffeine_level=None if caffeine == "any" else caffeine,
            sweetness_level=None if sweetness == "any" else sweetness,
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
            stress_level = fixed_choice(
                "Stress level",
                STRESS_OPTIONS,
                "ai_stress_level",
                default_index=2,
            )
            goal = fixed_choice(
                "Goal",
                GOAL_OPTIONS,
                "ai_goal",
            )
            weather = fixed_choice(
                "Weather",
                WEATHER_OPTIONS,
                "ai_weather",
            )
        with col2:
            preferred_temperature = fixed_choice(
                "Temperature preference",
                TEMPERATURE_OPTIONS,
                "ai_temperature",
            )
            caffeine_preference = fixed_choice(
                "Caffeine preference",
                CAFFEINE_OPTIONS,
                "ai_caffeine_preference",
            )
            sweetness_preference = fixed_choice(
                "Sweetness preference",
                SWEETNESS_OPTIONS,
                "ai_sweetness_preference",
            )
            dietary_restrictions = fixed_choice(
                "Dietary restrictions",
                ["None", "Dairy-free", "Vegan", "Vegetarian", "Nut-free", "Low calorie"],
                "ai_dietary_restrictions",
            )
            things_you_love = st.text_area(
                "Things I love today",
                placeholder="Vanilla, oat milk, cinnamon...",
            )
            things_you_hate = st.text_area(
                "Things I want to avoid",
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
    render_ingredient_cards(
        st.session_state.ingredients.sort_values(["category", "ingredient_name"]),
        limit=80,
    )


def create_profile_section() -> None:
    """Render profile creation UI."""
    st.subheader("Create Profile")
    drinks = st.session_state.drinks
    render_duplicate_profile_prompt("profile")

    with st.form("create_profile_form"):
        name = st.text_input("Name")
        favorite_milk = fixed_choice("Favorite milk", MILK_OPTIONS, "create_favorite_milk")
        favorite_temperature = fixed_choice(
            "Favorite temperature",
            ["No preference", "Hot", "Iced", "Blended"],
            "create_favorite_temperature",
        )
        caffeine_tolerance = fixed_choice(
            "Caffeine tolerance",
            CAFFEINE_TOLERANCE_OPTIONS,
            "create_caffeine_tolerance",
        )
        preferred_sweetness = fixed_choice(
            "Preferred sweetness",
            SWEETNESS_OPTIONS,
            "create_preferred_sweetness",
        )
        submitted = st.form_submit_button("Create profile")

    if submitted:
        if not name.strip():
            st.error("Please enter a name.")
            return
        existing_user = profile_name_exists(name)
        if existing_user:
            st.session_state.duplicate_profile_to_load = existing_user
            st.rerun()
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
        award_xp(user["user_id"], "create_profile")
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
            caffeine = fixed_choice(
                "Caffeine filter",
                CAFFEINE_OPTIONS,
                "recommend_caffeine_filter",
            )
            milk = fixed_choice(
                "Milk filter",
                ["Any"] + MILK_OPTIONS[1:],
                "recommend_milk_filter",
            )
        with col2:
            temperature = fixed_choice(
                "Temperature filter",
                ["Any", "Iced", "Hot", "Blended"],
                "recommend_temperature_filter",
            )
            sweetness = fixed_choice(
                "Sweetness filter",
                SWEETNESS_OPTIONS,
                "recommend_sweetness_filter",
            )
        with col3:
            max_price = st.number_input(
                "Maximum price",
                min_value=0.0,
                max_value=20.0,
                value=10.0,
                step=0.25,
            )
            dietary_tag = fixed_choice(
                "Dietary tag",
                ["Any", "dairy-free", "vegan", "vegetarian", "nut-free", "low-calorie"],
                "recommend_dietary_filter",
            )
        submitted = st.form_submit_button("Find drinks")

    if submitted:
        matches, exact_match, relaxed_filters = recommend_with_fallback(
            drinks=drinks,
            caffeine_level=None if caffeine == "any" else caffeine,
            temperature=None if temperature == "any" else temperature,
            milk=None if milk == "any" else milk,
            max_price=max_price,
            sweetness_level=None if sweetness == "any" else sweetness,
            dietary_tag=None if dietary_tag == "any" else dietary_tag,
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
            unit = fixed_choice(
                "Unit",
                SUPPORTED_UNITS,
                key=f"custom_unit_{index}",
                default_index=default_index,
                horizontal=False,
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


def custom_drink_section(show_catalog: bool = True) -> None:
    """Render ingredient-based custom drink UI."""
    st.subheader("Create Custom Ingredient-Based Drink")
    user = st.session_state.current_user
    ingredients = st.session_state.ingredients

    if show_catalog:
        with st.expander("Ingredient catalog", expanded=False):
            render_ingredient_cards(ingredients.sort_values(["category", "ingredient_name"]), limit=24)

    with st.form("custom_drink_form"):
        drink_name = st.text_input("Custom drink name")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            base = fixed_choice("Base", BASE_OPTIONS, "custom_base")
            temperature = fixed_choice("Temperature", TEMPERATURE_OPTIONS[1:], "custom_temperature")
        with col_b:
            size = fixed_choice("Size", SIZE_OPTIONS, "custom_size", default_index=2)
            milk = fixed_choice("Milk", MILK_OPTIONS, "custom_milk")
        with col_c:
            syrup = fixed_choice("Syrup / flavor", SYRUP_OPTIONS, "custom_syrup")
            espresso_shots = st.number_input(
                "Espresso shots",
                min_value=0,
                max_value=6,
                value=1,
                step=1,
            )
        caffeine_level = fixed_choice("Caffeine level", CAFFEINE_OPTIONS[1:], "custom_caffeine_level")
        roast_blend = fixed_choice(
            "Roast / blend type",
            ["House", "Blonde", "Medium", "Dark", "Decaf"],
            "custom_roast_blend",
        )
        sweetness_level = fixed_choice("Sweetness", SWEETNESS_OPTIONS[1:], "custom_sweetness_level")
        calories_override = st.number_input("Calories", min_value=0.0, max_value=1200.0, value=0.0)
        dietary_tags = st.text_input("Dietary tags", placeholder="vegan, dairy-free, nut-free")
        uploaded_image = st.file_uploader("Optional drink image", type=["png", "jpg", "jpeg"])
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
        custom_drink["base"] = base
        custom_drink["temperature"] = temperature
        custom_drink["size"] = size
        custom_drink["milk"] = "none" if milk == "no preference" else milk
        custom_drink["syrup"] = syrup
        custom_drink["espresso_shots"] = int(espresso_shots)
        custom_drink["caffeine_level"] = caffeine_level
        custom_drink["sweetness_level"] = sweetness_level
        custom_drink["creator_user_id"] = user["user_id"] if user else "guest"
        custom_drink["creator_name"] = user["name"] if user else "Guest"
        if calories_override > 0:
            custom_drink["calories"] = calories_override
        if dietary_tags.strip():
            custom_drink["dietary_tags"] = dietary_tags.strip()
        if roast_blend:
            custom_drink["flavor_profile"] = (
                f"{custom_drink.get('flavor_profile', '')}, {base}, {milk}, {syrup}, {roast_blend} roast/blend"
            ).strip(", ")
        if uploaded_image is not None:
            st.info("Image upload is preview-only for now. The custom drink was saved with category imagery.")
        exists, duplicate_reason = custom_drink_exists(custom_drink["drink_name"], recipe_items)
        if exists:
            st.warning(duplicate_reason)
            return

        try:
            save_custom_drink_recipe(custom_drink, recipe_items)
        except RuntimeError as error:
            st.error(str(error))
            return
        if user:
            award_xp(user["user_id"], "create_custom_drink")
        saved_rating = ""
        if user and rate_now:
            try:
                save_rating(user["user_id"], custom_drink["drink_id"], rating, would_order_again)
                award_xp(user["user_id"], "rate_drink")
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


def add_ingredient_section(show_list: bool = True) -> None:
    """Render UI for adding a custom ingredient."""
    st.subheader("Add Ingredient")
    st.caption("New ingredients become available immediately in the custom drink builder.")

    with st.form("add_ingredient_form"):
        ingredient_name = st.text_input("Ingredient name")
        category = fixed_choice(
            "Category",
            [category_name.title() if category_name != "add-in" else "Add-in" for category_name in SUPPORTED_CATEGORIES],
            "add_ingredient_category",
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            calories = st.number_input("Calories", min_value=0.0, max_value=1000.0, value=0.0)
        with col2:
            caffeine = st.number_input("Caffeine", min_value=0.0, max_value=500.0, value=0.0)
        with col3:
            price = st.number_input("Price", min_value=0.0, max_value=20.0, value=0.0, step=0.05)
        with col4:
            default_unit = fixed_choice(
                "Default unit",
                SUPPORTED_UNITS,
                "add_ingredient_default_unit",
                default_index=SUPPORTED_UNITS.index("serving") if "serving" in SUPPORTED_UNITS else 0,
            )
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

    if show_list:
        st.markdown("### Ingredient list")
        render_ingredient_cards(
            st.session_state.ingredients.sort_values(["category", "ingredient_name"]),
            limit=48,
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
    used_profile_matching = False
    if user and openai_is_configured():
        try:
            selector_result, cache_used = select_best_candidate(
                user,
                top_candidates,
                context,
            )
        except Exception as error:
            LOGGER.warning("OpenAI selector failed; using profile matching fallback: %s", error)
            used_profile_matching = True
            selector_result = None
    if selector_result is None:
        selector_result = fallback_recommendation(top_candidates)
        used_profile_matching = True

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
    selected.loc[:, "used_profile_matching"] = used_profile_matching
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
            award_xp(user["user_id"], "rate_drink")
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
    render_custom_drinks_home_section()
    render_ingredients_home_section()
    render_drink_rail(
        "Favorites",
        rails["favorites"],
        "rail_favorites",
        empty_text="Save favorite drinks and they will appear here.",
    )
    render_category_explorer(st.session_state.drinks)


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
        render_duplicate_profile_prompt("guided")
        with st.form("guided_create_profile"):
            name = st.text_input("Name")
            favorite_milk = fixed_choice(
                "Favorite milk",
                MILK_OPTIONS,
                "guided_favorite_milk",
            )
            caffeine_tolerance = fixed_choice(
                "Caffeine tolerance",
                CAFFEINE_TOLERANCE_OPTIONS,
                "guided_caffeine_tolerance",
            )
            preferred_sweetness = fixed_choice(
                "Sweetness preference",
                SWEETNESS_OPTIONS,
                "guided_preferred_sweetness",
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
            existing_user = profile_name_exists(name)
            if existing_user:
                st.session_state.duplicate_profile_to_load = existing_user
                st.rerun()
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
            award_xp(user["user_id"], "create_profile")
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
        render_duplicate_profile_prompt("home" if after == "home" else "profile")
        with st.form(f"{after}_create_profile_form"):
            name = st.text_input("Name")
            favorite_milk = fixed_choice(
                "Favorite milk",
                MILK_OPTIONS,
                f"{after}_favorite_milk",
            )
            caffeine_tolerance = fixed_choice(
                "Caffeine tolerance",
                CAFFEINE_TOLERANCE_OPTIONS,
                f"{after}_caffeine_tolerance",
            )
            preferred_sweetness = fixed_choice(
                "Sweetness preference",
                SWEETNESS_OPTIONS,
                f"{after}_preferred_sweetness",
            )
            submitted = st.form_submit_button("Create Profile", type="primary", use_container_width=True)
        if submitted:
            if not name.strip():
                st.error("Please enter your name.")
                return
            existing_user = profile_name_exists(name)
            if existing_user:
                st.session_state.duplicate_profile_to_load = existing_user
                st.rerun()
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
            award_xp(user["user_id"], "create_profile")
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
        weather = fixed_choice(
            "Weather",
            WEATHER_OPTIONS,
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
        stress_level = fixed_choice(
            "Stress level",
            STRESS_OPTIONS,
            key="guided_stress",
        )
        caffeine_preference = fixed_choice(
            "Caffeine preference",
            CAFFEINE_OPTIONS,
            key="guided_caffeine_preference",
        )
        sweetness_preference = fixed_choice(
            "Sweetness preference",
            SWEETNESS_OPTIONS,
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
        if not matches.empty and bool(matches.iloc[0].get("used_profile_matching", False)):
            st.session_state.recommendation_fallback_message = (
                "Recommendation generated using profile matching."
            )
        else:
            st.session_state.recommendation_fallback_message = None
        if user and not matches.empty:
            award_xp(user["user_id"], "generate_recommendation")
            if not bool(matches.iloc[0].get("used_profile_matching", False)):
                award_xp(user["user_id"], "use_ai_recommendation")
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
            confidence = drink.get("selector_confidence", None)
            if confidence is not None and str(confidence) != "nan":
                meta = f"{int(float(confidence))}% confidence"
            else:
                meta = f"{_match_percentage(drink.get('recommendation_score'))}% match"
            description = str(drink.get("recommendation_summary", _drink_description(drink)))
            st.markdown(
                f"""
                <div class="rail-card">
                    <div class="card-image-zone">{drink_image_html(drink, "rail-card-image")}</div>
                    <div class="card-content-zone">
                        <div class="card-title-zone">{safe_text(drink.get("drink_name", "Recommended drink"))}</div>
                        <div class="card-meta-zone">{safe_text(meta)}</div>
                        <div class="card-chip-zone">{limited_chips_html(_drink_ingredients(drink, limit=6), limit=3)}</div>
                        <div class="card-description-zone">{safe_text(description)}</div>
                        <div class="card-actions-zone"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
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
            drink_image_html(drink, "recommendation-feature-image"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="recommendation-feature">
                <div class="ai-card-meta">Best match</div>
                <div class="featured-title">{safe_text(drink.get("drink_name", "Recommended drink"))}</div>
                <div class="featured-score">{safe_text(score_text)}</div>
                <div class="recommendation-card-chips">
                    <div class="card-ingredients">{limited_chips_html(_drink_ingredients(drink, limit=7), limit=5)}</div>
                </div>
                <div class="featured-reason">Best match because {safe_text(explanation)}</div>
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
    secondary = candidates[candidates["drink_id"].astype(str) != str(selected_id)].head(8)
    if secondary.empty:
        return
    st.markdown("### Also Close")
    rows = list(secondary.iterrows())
    for start in range(0, len(rows), 5):
        row_items = rows[start : start + 5]
        columns = st.columns(len(row_items))
        for offset, (column, (_, drink)) in enumerate(zip(columns, row_items)):
            index = start + offset
            reason = str(drink.get("recommendation_summary", drink.get("recommendation_explanation", "")))
            with column:
                st.markdown(
                    f"""
                    <div class="secondary-card">
                        <div class="card-image-zone">{drink_image_html(drink, "rail-card-image")}</div>
                        <div class="card-content-zone">
                            <div class="card-title-zone">{safe_text(drink.get("drink_name", "Drink"))}</div>
                            <div class="card-meta-zone">{_match_percentage(drink.get('recommendation_score'))}% match</div>
                            <div class="card-chip-zone">{limited_chips_html(_drink_ingredients(drink, limit=6), limit=3)}</div>
                            <div class="card-description-zone">Also close because {safe_text(reason)}</div>
                            <div class="card-actions-zone"></div>
                        </div>
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
    if st.session_state.get("recommendation_fallback_message"):
        st.info(st.session_state.recommendation_fallback_message)
    if not st.session_state.get("guided_exact_match", True):
        st.info("No exact match found, but this is close.")
        relaxed = st.session_state.get("guided_relaxed_filters", [])
        if relaxed:
            st.caption(f"Relaxed filters: {', '.join(relaxed)}")
    best = matches.iloc[0]
    st.markdown("### Featured For You")
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

    similar = find_similar_drinks(st.session_state.drinks, drink, limit=3)
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown(
            f"""
            <div class="detail-hero-card">
                {drink_image_html(drink, "detail-drink-image")}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Also try")
        render_drink_card_grid(similar, "guided_also_try", limit=3, mode="guided")
    with col2:
        st.markdown(detail_summary_html(drink), unsafe_allow_html=True)
        with st.expander("View scoring details"):
            render_score_breakdown(drink)

        rate_col, another_col, favorite_col = st.columns(3)
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
        with favorite_col:
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
            award_xp(user["user_id"], "rate_drink")
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
    render_recently_rated_section(user["user_id"])
    render_custom_creations_section(user_id=user["user_id"])
    if st.button("Get another recommendation", type="primary", use_container_width=True):
        _set_flow_step(3)


def render_sidebar_categories() -> None:
    """Render compact category navigation in the sidebar."""
    st.sidebar.markdown("#### Explore Categories")
    for category in CATEGORY_CARDS:
        selected = st.session_state.get("selected_category") == category["key"]
        label = f"{category['name']}"
        if selected:
            label = f"{category['name']} selected"
        if st.sidebar.button(
            label,
            key=f"sidebar_category_{category['key']}",
            use_container_width=True,
        ):
            st.session_state.selected_category = category["key"]
            st.session_state.selected_page = "home"
            st.session_state.home_view = "recommend"
            st.rerun()
    if st.session_state.get("selected_category"):
        if st.sidebar.button("Clear category", key="sidebar_clear_category", use_container_width=True):
            st.session_state.selected_category = None
            st.session_state.selected_page = "home"
            st.rerun()


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
    render_recently_rated_section(user["user_id"])
    render_custom_creations_section(user_id=user["user_id"])


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
    render_sidebar_categories()
    st.sidebar.markdown('<div class="sidebar-progress-push"></div>', unsafe_allow_html=True)
    render_barista_progress()


def render_barista_progress() -> None:
    """Render the sidebar coffee mug progress tracker."""
    user = st.session_state.current_user
    if not user:
        st.sidebar.markdown(
            """
            <div class="progress-card">
                <div class="progress-name">Barista Progress</div>
                <div class="progress-title">Level 1 - New Customer</div>
                <div class="mug-wrap">
                    <div class="coffee-mug"><div class="coffee-fill" style="--fill: 0%;"></div></div>
                    <div class="progress-percent">0% to next level</div>
                </div>
                <div class="progress-xp">0 XP</div>
                <div class="progress-stats">Create or load a profile to start earning XP.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    award_daily_return_if_needed(str(user["user_id"]))
    progress = get_user_progress(str(user["user_id"]))
    fill = int(progress["progress_percent"])
    st.sidebar.markdown(
        f"""
        <div class="progress-card">
            <div class="progress-name">Barista Progress</div>
            <div class="progress-title">Level {safe_text(progress["level"])} - {safe_text(progress["title"])}</div>
            <div class="mug-wrap">
                <div class="coffee-mug"><div class="coffee-fill" style="--fill: {fill}%;"></div></div>
                <div class="progress-percent">{fill}% to next level</div>
            </div>
            <div class="progress-xp">{safe_text(progress["xp"])} XP</div>
            <div class="progress-stats">{safe_text(progress["drinks_rated"])} rated - {safe_text(progress["favorites_saved"])} saved - {safe_text(progress["custom_drinks_created"])} custom</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def advanced_tools_section(expanded: bool = False) -> None:
    """Render advanced customization and database tools."""
    with st.expander(
        "Advanced Tools · Admin and debug utilities.",
        expanded=expanded,
    ):
        st.caption("Maintenance utilities and data inspection for debugging.")
        advanced = st.tabs(
            [
                "Ingredient List",
                "Rule-based recommendations",
                "Rate drink",
                "Rating history",
                "Taste profile",
            ]
        )
        with advanced[0]:
            ingredient_list_section()
        with advanced[1]:
            recommendation_section()
        with advanced[2]:
            rate_drink_section()
        with advanced[3]:
            rating_history_section()
        with advanced[4]:
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
