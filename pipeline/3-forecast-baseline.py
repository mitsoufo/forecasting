"""Single-year baseline forecast.

Use this when the BASE_YEAR data is missing. It projects the baseline year's
daily event curve onto FORECAST_YEAR and applies optional multipliers from
config.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BASELINE_ARTICLE_MULTIPLIER,
    BASELINE_IMPRESSION_MULTIPLIER,
    BASELINE_YEAR,
    EVENT_ANCHORS,
    FORECAST_YEAR,
)


TOPIC_DATA_PATH = Path("data/with_topics.csv")
SOURCE_DATA_PATH = TOPIC_DATA_PATH if TOPIC_DATA_PATH.exists() else Path("data/classified.csv")

OUT_TYPE_PATH = (
    f"data/{FORECAST_YEAR}_baseline_{BASELINE_YEAR}_"
    "daily_impression_forecast_by_content_type.csv"
)
OUT_TOPIC_PATH = (
    f"data/{FORECAST_YEAR}_baseline_{BASELINE_YEAR}_"
    "daily_impression_forecast_by_topic.csv"
)


if BASELINE_YEAR not in EVENT_ANCHORS:
    raise KeyError(f"BASELINE_YEAR {BASELINE_YEAR} is missing from EVENT_ANCHORS")

if FORECAST_YEAR not in EVENT_ANCHORS:
    raise KeyError(f"FORECAST_YEAR {FORECAST_YEAR} is missing from EVENT_ANCHORS")

if not SOURCE_DATA_PATH.exists():
    raise FileNotFoundError(
        "Run 1-classify-content.py first, or provide data/classified.csv."
    )


df = pd.read_csv(SOURCE_DATA_PATH)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year
df["event_date"] = df["year"].map(EVENT_ANCHORS)
df = df.dropna(subset=["date", "event_date"])
df["event_day"] = (df["date"] - df["event_date"]).dt.days

baseline_df = df[df["year"] == BASELINE_YEAR].copy()
if baseline_df.empty:
    raise ValueError(
        f"No {BASELINE_YEAR} rows found in {SOURCE_DATA_PATH}. "
        f"Provide data/{BASELINE_YEAR}.csv and rerun classification."
    )


daily = (
    baseline_df.groupby(["content_type", "event_day"], as_index=False)
    .agg(
        ClientCreativeImpression=("ClientCreativeImpression", "sum"),
        article_count=("url", "count"),
    )
)

forecast_by_type = daily.rename(
    columns={
        "ClientCreativeImpression": f"base_{BASELINE_YEAR}_impressions",
        "article_count": f"base_{BASELINE_YEAR}_articles",
    }
)
forecast_by_type[f"forecast_impressions_{FORECAST_YEAR}"] = (
    forecast_by_type[f"base_{BASELINE_YEAR}_impressions"]
    * BASELINE_IMPRESSION_MULTIPLIER
).round().astype(int)
forecast_by_type[f"forecast_articles_{FORECAST_YEAR}"] = (
    forecast_by_type[f"base_{BASELINE_YEAR}_articles"]
    * BASELINE_ARTICLE_MULTIPLIER
).round().astype(int)
forecast_by_type["projection_method"] = f"{BASELINE_YEAR}_baseline"
forecast_by_type["impression_multiplier"] = BASELINE_IMPRESSION_MULTIPLIER
forecast_by_type["article_multiplier"] = BASELINE_ARTICLE_MULTIPLIER
forecast_by_type["date"] = (
    EVENT_ANCHORS[FORECAST_YEAR]
    + pd.to_timedelta(forecast_by_type["event_day"], unit="D")
)

forecast_by_type = forecast_by_type[
    [
        "date",
        "event_day",
        "content_type",
        f"base_{BASELINE_YEAR}_impressions",
        f"base_{BASELINE_YEAR}_articles",
        "projection_method",
        "impression_multiplier",
        "article_multiplier",
        f"forecast_impressions_{FORECAST_YEAR}",
        f"forecast_articles_{FORECAST_YEAR}",
    ]
]
forecast_by_type.to_csv(OUT_TYPE_PATH, index=False)
print(f"Saved: {OUT_TYPE_PATH}")
print(
    forecast_by_type.groupby("content_type")[
        [f"forecast_impressions_{FORECAST_YEAR}", f"forecast_articles_{FORECAST_YEAR}"]
    ].sum()
)


if {"topic_key", "topic_id", "topic_label"}.issubset(baseline_df.columns):
    df_topics = baseline_df.dropna(subset=["topic_key", "topic_id", "topic_label"]).copy()
    daily_topic = (
        df_topics.groupby(
            ["content_type", "topic_key", "topic_id", "topic_label", "event_day"],
            as_index=False,
        )
        .agg(
            ClientCreativeImpression=("ClientCreativeImpression", "sum"),
            article_count=("url", "count"),
        )
    )

    if daily_topic.empty:
        print("No topic forecasts created. Topic columns exist, but no topic rows were found.")
    else:
        forecast_by_topic = daily_topic.rename(
            columns={
                "ClientCreativeImpression": f"base_{BASELINE_YEAR}_impressions",
                "article_count": f"base_{BASELINE_YEAR}_articles",
            }
        )
        forecast_by_topic["independent_topic_forecast"] = (
            forecast_by_topic[f"base_{BASELINE_YEAR}_impressions"]
            * BASELINE_IMPRESSION_MULTIPLIER
        )

        content_type_daily_total = forecast_by_type[
            ["content_type", "event_day", f"forecast_impressions_{FORECAST_YEAR}"]
        ].rename(columns={f"forecast_impressions_{FORECAST_YEAR}": "content_type_forecast"})

        forecast_by_topic = forecast_by_topic.merge(
            content_type_daily_total,
            on=["content_type", "event_day"],
            how="left",
        )
        forecast_by_topic["topic_day_total_forecast"] = (
            forecast_by_topic.groupby(["content_type", "event_day"])
            ["independent_topic_forecast"]
            .transform("sum")
        )
        forecast_by_topic["raw_reconciled_forecast"] = np.where(
            forecast_by_topic["topic_day_total_forecast"] > 0,
            forecast_by_topic["independent_topic_forecast"]
            / forecast_by_topic["topic_day_total_forecast"]
            * forecast_by_topic["content_type_forecast"],
            0,
        )
        forecast_by_topic["forecast_floor"] = np.floor(
            forecast_by_topic["raw_reconciled_forecast"]
        ).astype(int)
        forecast_by_topic["rounding_remainder"] = (
            forecast_by_topic["content_type_forecast"].fillna(0).astype(int)
            - forecast_by_topic.groupby(["content_type", "event_day"])
            ["forecast_floor"]
            .transform("sum")
        )
        forecast_by_topic["rounding_fraction"] = (
            forecast_by_topic["raw_reconciled_forecast"]
            - forecast_by_topic["forecast_floor"]
        )
        forecast_by_topic["rounding_rank"] = (
            forecast_by_topic.groupby(["content_type", "event_day"])
            ["rounding_fraction"]
            .rank(method="first", ascending=False)
        )
        forecast_by_topic[f"forecast_impressions_{FORECAST_YEAR}"] = (
            forecast_by_topic["forecast_floor"]
            + (
                forecast_by_topic["rounding_rank"]
                <= forecast_by_topic["rounding_remainder"]
            ).astype(int)
        ).astype(int)
        forecast_by_topic[f"forecast_articles_{FORECAST_YEAR}"] = (
            forecast_by_topic[f"base_{BASELINE_YEAR}_articles"]
            * BASELINE_ARTICLE_MULTIPLIER
        ).round().astype(int)
        forecast_by_topic["projection_method"] = f"{BASELINE_YEAR}_baseline"
        forecast_by_topic["impression_multiplier"] = BASELINE_IMPRESSION_MULTIPLIER
        forecast_by_topic["article_multiplier"] = BASELINE_ARTICLE_MULTIPLIER
        forecast_by_topic["date"] = (
            EVENT_ANCHORS[FORECAST_YEAR]
            + pd.to_timedelta(forecast_by_topic["event_day"], unit="D")
        )

        forecast_by_topic = forecast_by_topic[
            [
                "date",
                "event_day",
                "content_type",
                "topic_key",
                "topic_id",
                "topic_label",
                f"base_{BASELINE_YEAR}_impressions",
                f"base_{BASELINE_YEAR}_articles",
                "projection_method",
                "impression_multiplier",
                "article_multiplier",
                "independent_topic_forecast",
                "content_type_forecast",
                f"forecast_impressions_{FORECAST_YEAR}",
                f"forecast_articles_{FORECAST_YEAR}",
            ]
        ]
        forecast_by_topic.to_csv(OUT_TOPIC_PATH, index=False)
        print(f"Saved: {OUT_TOPIC_PATH}")
else:
    print("No topic columns found. Run 2-find-topics.py first to create topic forecasts.")
