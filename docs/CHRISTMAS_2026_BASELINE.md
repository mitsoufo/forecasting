# Christmas 2026 Baseline Forecast

Use this version when you only have 2025 Christmas data and no 2024 comparison
year.

## Required Input

Place one CSV at:

```text
data/2025.csv
```

It should contain Christmas-relevant articles for a consistent pre-event window,
for example September 1 to December 31, 2025.

Required columns:

```text
url
page_content_title_formalized
page_content_body_formalized
date
page_categories_tier1
ClientCreativeImpression
```

## Config

In `config.py`, set the event to Christmas:

```python
EVENT_NAME = "Christmas"
EVENT_KEY = "christmas"
EVENT_ANCHORS = {
    2025: pd.Timestamp("2025-12-25"),
    2026: pd.Timestamp("2026-12-25"),
}
FORECAST_YEAR = 2026
COMPARE_YEAR = 2025
BASELINE_YEAR = COMPARE_YEAR
```

Optional growth assumptions:

```python
BASELINE_IMPRESSION_MULTIPLIER = 1.0  # 1.10 means +10%
BASELINE_ARTICLE_MULTIPLIER = 1.0     # 0.95 means -5%
```

## Run Order

```bash
python3 1-classify-content.py
python3 2-find-topics.py
python3 3-forecast-baseline.py
```

The baseline forecast writes separate files:

```text
data/2026_baseline_2025_daily_impression_forecast_by_content_type.csv
data/2026_baseline_2025_daily_impression_forecast_by_topic.csv
```

This is a baseline projection, not a YoY model: it repeats the 2025 timing,
content-type mix, and topic pattern onto 2026 unless you set multipliers.
