# config.py — Central event configuration
# Edit this file to switch between events (Black Friday, Easter, Christmas, etc.)

import os

import pandas as pd

# ── Event identity ──────────────────────────────────────────────────────────────
EVENT_NAME = "Paris Fashion Week"      # Display name used in chart titles and reports
EVENT_KEY  = "paris_fashion_week"      # Slug used in filenames (lowercase, underscores)
ARTICLE_LANGUAGE = "fr"       # ISO 639-1 language code used by langdetect, e.g. "en" or "fr"

# ── Historical anchor dates + forecast target ───────────────────────────────────
# Add or remove years to match your available data.
EVENT_ANCHORS = {
    2025: pd.Timestamp("2025-09-29"),
    2026: pd.Timestamp("2026-09-28"),
}
EVENT_ENDS = {
    2025: pd.Timestamp("2025-10-07"),
    2026: pd.Timestamp("2026-10-06"),
}
FORECAST_YEAR = 2026   # Year to produce the forecast for
BASE_YEAR     = 2025   # Earliest year with data (used for YoY growth baseline)
COMPARE_YEAR  = 2025   # Most recent complete year (used as the projection base)

# ── Forecast activation window ─────────────────────────────────────────────────
# Recent pre-season data can inform topic discovery, but the deliverable forecast
# should stay inside the activation months for the event.
FORECAST_WINDOW_START = pd.Timestamp("2026-09-01")
FORECAST_WINDOW_END   = pd.Timestamp("2026-10-31")
MIN_EVERGREEN_IMPRESSION_SHARE = 0.50
MIN_EVERGREEN_ARTICLE_SHARE = 0.50

# ── Single-year baseline forecast settings ─────────────────────────────────────
# Used by 3-forecast-baseline.py when BASE_YEAR data is missing. A multiplier of
# 1.0 means "repeat the COMPARE_YEAR curve"; 1.10 means "+10% vs compare year".
BASELINE_YEAR                  = COMPARE_YEAR
BASELINE_IMPRESSION_MULTIPLIER = 1.0
BASELINE_ARTICLE_MULTIPLIER    = 1.0

# ── Topic modelling stopwords ───────────────────────────────────────────────────
# Words to strip from the bag-of-words so BERTopic clusters separate by
# *what's being covered* rather than the shared event vocabulary.
DOMAIN_STOPWORDS = {
    "fashion", "week", "paris", "mode", "defile", "defile", "defiles", "collection",
    "collections", "show", "shows", "look", "looks", "tendance", "tendances",
    "style", "styles", "createur", "createurs", "creation", "maison", "maisons",
    "couture", "haute", "pret", "porter", "printemps", "ete", "automne", "hiver",
    "nouveau", "nouvelle", "nouvelles", "meilleur", "meilleurs", "top", "guide",
    "article", "articles", "photo", "photos", "video", "videos",
    "plus", "cette", "comme", "aussi", "tout", "bien", "chez", "fait", "faire",
    "ans", "jeune", "jeunes", "fermer", "ouvrir", "panneau", "diaporama",
    "2025", "2026",
}

# ── Derived constants (no need to edit) ────────────────────────────────────────
EVENT_ANCHOR  = EVENT_ANCHORS[FORECAST_YEAR]   # Anchor date for the forecast year
EVENT_END     = EVENT_ENDS[FORECAST_YEAR]      # Final date of the forecast-year event
DEFAULT_REPORT_PATH = f"reports/{EVENT_KEY}_{FORECAST_YEAR}_forecast_report_v4.html"
DEFAULT_RECO_CSV_PATH = f"data/{EVENT_KEY}_{FORECAST_YEAR}_recommendations.csv"

REPORT_PATH   = os.getenv("REPORT_PATH", DEFAULT_REPORT_PATH)
RECO_CSV_PATH = os.getenv("RECO_CSV_PATH", DEFAULT_RECO_CSV_PATH)
