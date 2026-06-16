# input.py -- required event inputs
# Edit this file for every new forecast.

import pandas as pd

# Event identity
EVENT_NAME = "Toronto International Film Festival"
EVENT_KEY = "toronto_international_film_festival"
ARTICLE_LANGUAGE = "en"  # ISO 639-1 language code, e.g. "en" or "fr"

# Event dates
# Anchors are event start dates. Add or remove historical years to match your
# available files, e.g. data/2025.csv.
EVENT_ANCHORS = {
    2025: pd.Timestamp("2025-09-04"),
    2026: pd.Timestamp("2026-09-10"),
}

# End dates are the final dates of each event edition.
EVENT_ENDS = {
    2025: pd.Timestamp("2025-09-14"),
    2026: pd.Timestamp("2026-09-20"),
}

# Forecast target and historical baseline years
FORECAST_YEAR = 2026
BASE_YEAR = 2025
COMPARE_YEAR = 2025

# Forecast report window
# This controls which forecast dates are allowed to appear in the report.
FORECAST_WINDOW_START = pd.Timestamp("2026-09-01")
FORECAST_WINDOW_END = pd.Timestamp("2026-10-31")
