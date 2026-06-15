import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    EVENT_ANCHORS,
    FORECAST_YEAR,
    BASE_YEAR,
    COMPARE_YEAR,
    FORECAST_WINDOW_START,
    FORECAST_WINDOW_END,
    MIN_EVERGREEN_IMPRESSION_SHARE,
)

TOPIC_DATA_PATH = Path("data/with_topics.csv")
SOURCE_DATA_PATH = TOPIC_DATA_PATH if TOPIC_DATA_PATH.exists() else Path("data/classified.csv")

df = pd.read_csv(SOURCE_DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year

df["event_day"] = df.apply(
    lambda r: (r["date"] - EVENT_ANCHORS[r["year"]]).days,
    axis=1
)
df["forecast_date"] = EVENT_ANCHORS[FORECAST_YEAR] + pd.to_timedelta(df["event_day"], unit="D")
before_window = len(df)
df = df[
    df["forecast_date"].between(FORECAST_WINDOW_START, FORECAST_WINDOW_END)
].copy()
print(
    f"Forecast window: {FORECAST_WINDOW_START.date()} to {FORECAST_WINDOW_END.date()} "
    f"({len(df):,}/{before_window:,} rows retained)"
)

daily = (
    df.groupby(["content_type", "year", "event_day"], as_index=False)
      .agg(
          ClientCreativeImpression=("ClientCreativeImpression", "sum"),
          article_count=("url", "count"),
      )
)

forecasts = []

for content_type, g in daily.groupby("content_type"):
    pivot = g.pivot(index="event_day", columns="year", values="ClientCreativeImpression")
    pivot = pivot.dropna(subset=[BASE_YEAR, COMPARE_YEAR])

    pivot_arts = g.pivot(index="event_day", columns="year", values="article_count")
    pivot_arts = pivot_arts.reindex(pivot.index)

    growth_factor = pivot[COMPARE_YEAR].sum() / pivot[BASE_YEAR].sum()
    art_growth_factor = pivot_arts[COMPARE_YEAR].sum() / pivot_arts[BASE_YEAR].sum() if pivot_arts[BASE_YEAR].sum() > 0 else growth_factor

    fc = pivot[[COMPARE_YEAR]].rename(columns={COMPARE_YEAR: f"base_{COMPARE_YEAR}_impressions"})
    fc[f"forecast_impressions_{FORECAST_YEAR}"] = (
        fc[f"base_{COMPARE_YEAR}_impressions"] * growth_factor
    ).round().astype(int)

    fc[f"base_{COMPARE_YEAR}_articles"] = pivot_arts[COMPARE_YEAR].reindex(fc.index)
    fc[f"forecast_articles_{FORECAST_YEAR}"] = (
        fc[f"base_{COMPARE_YEAR}_articles"] * art_growth_factor
    ).round().astype(int)

    fc = fc.reset_index()
    fc["content_type"] = content_type
    fc[f"growth_factor_{BASE_YEAR}_to_{COMPARE_YEAR}"] = growth_factor
    fc["date"] = EVENT_ANCHORS[FORECAST_YEAR] + pd.to_timedelta(fc["event_day"], unit="D")

    forecasts.append(fc)

forecast_2026_by_type = pd.concat(forecasts, ignore_index=True)

# Stabilize content-type impression allocation. Use historical impression share
# (not article-count share) so that the efficiency difference between Evergreen
# and Spontaneous content is preserved in the 2026 forecast.
_imps_col = f"forecast_impressions_{FORECAST_YEAR}"
_independent_imps_col = f"independent_forecast_impressions_{FORECAST_YEAR}"
forecast_2026_by_type[_independent_imps_col] = forecast_2026_by_type[_imps_col]

impression_share = (
    df.groupby(["year", "content_type"])["ClientCreativeImpression"]
    .sum()
    .unstack(fill_value=0)
)
target_shares = impression_share.div(impression_share.sum(axis=1), axis=0).mean()
target_shares = target_shares / target_shares.sum()

if (
    "Evergreen" in target_shares.index
    and target_shares["Evergreen"] < MIN_EVERGREEN_IMPRESSION_SHARE
):
    non_evergreen = target_shares.index != "Evergreen"
    remaining_share = 1 - MIN_EVERGREEN_IMPRESSION_SHARE
    non_evergreen_total = target_shares.loc[non_evergreen].sum()
    if non_evergreen_total > 0:
        target_shares.loc[non_evergreen] = (
            target_shares.loc[non_evergreen] / non_evergreen_total * remaining_share
        )
    target_shares["Evergreen"] = MIN_EVERGREEN_IMPRESSION_SHARE

target_shares = target_shares / target_shares.sum()

total_impressions_target = int(forecast_2026_by_type[_independent_imps_col].sum())
target_totals = (target_shares * total_impressions_target).round().astype(int)
rounding_delta = total_impressions_target - int(target_totals.sum())
if rounding_delta:
    target_totals.loc[target_totals.idxmax()] += rounding_delta

for content_type, target_total in target_totals.items():
    mask = forecast_2026_by_type["content_type"] == content_type
    current = forecast_2026_by_type.loc[mask, _independent_imps_col]
    if current.sum() > 0:
        raw = current / current.sum() * target_total
    else:
        raw = pd.Series(target_total / mask.sum(), index=current.index)

    floor = np.floor(raw).astype(int)
    remainder = int(target_total - floor.sum())
    rank = (raw - floor).rank(method="first", ascending=False)
    forecast_2026_by_type.loc[mask, _imps_col] = (
        floor + (rank <= remainder).astype(int)
    ).astype(int)

print("Target impression share from classified article mix:")
print((target_shares * 100).round(1).to_string())

forecast_2026_by_type = forecast_2026_by_type[
    [
        "date",
        "event_day",
        "content_type",
        f"base_{COMPARE_YEAR}_impressions",
        f"base_{COMPARE_YEAR}_articles",
        f"growth_factor_{BASE_YEAR}_to_{COMPARE_YEAR}",
        _independent_imps_col,
        f"forecast_impressions_{FORECAST_YEAR}",
        f"forecast_articles_{FORECAST_YEAR}",
    ]
]

_ct_csv = f"data/{FORECAST_YEAR}_daily_impression_forecast_by_content_type.csv"
forecast_2026_by_type.to_csv(_ct_csv, index=False)
print(f"Saved: {_ct_csv}")
print(forecast_2026_by_type.groupby("content_type")[[f"forecast_impressions_{FORECAST_YEAR}", f"forecast_articles_{FORECAST_YEAR}"]].sum())

if {"topic_key", "topic_id", "topic_label"}.issubset(df.columns):
    df_topics = df.dropna(subset=["topic_key", "topic_id", "topic_label"]).copy()

    daily_topic = (
        df_topics.groupby(
            ["content_type", "topic_key", "topic_id", "topic_label", "year", "event_day"],
            as_index=False,
        )
        .agg(
            ClientCreativeImpression=("ClientCreativeImpression", "sum"),
            article_count=("url", "count"),
        )
    )

    topic_forecasts = []

    for (content_type, topic_key, topic_id, topic_label), g in daily_topic.groupby(
        ["content_type", "topic_key", "topic_id", "topic_label"]
    ):
        pivot = g.pivot(index="event_day", columns="year", values="ClientCreativeImpression")
        if BASE_YEAR not in pivot.columns or COMPARE_YEAR not in pivot.columns:
            continue
        pivot = pivot.dropna(subset=[BASE_YEAR, COMPARE_YEAR])

        if pivot.empty or pivot[BASE_YEAR].sum() == 0:
            continue

        pivot_arts = g.pivot(index="event_day", columns="year", values="article_count")
        pivot_arts = pivot_arts.reindex(pivot.index)

        growth_factor = pivot[COMPARE_YEAR].sum() / pivot[BASE_YEAR].sum()
        art_growth_factor = pivot_arts[COMPARE_YEAR].sum() / pivot_arts[BASE_YEAR].sum() if pivot_arts[BASE_YEAR].sum() > 0 else growth_factor

        fc = pivot[[COMPARE_YEAR]].rename(columns={COMPARE_YEAR: f"base_{COMPARE_YEAR}_impressions"})
        fc[f"forecast_impressions_{FORECAST_YEAR}"] = (
            fc[f"base_{COMPARE_YEAR}_impressions"] * growth_factor
        ).round().astype(int)

        fc[f"base_{COMPARE_YEAR}_articles"] = pivot_arts[COMPARE_YEAR].reindex(fc.index)
        fc[f"forecast_articles_{FORECAST_YEAR}"] = (
            fc[f"base_{COMPARE_YEAR}_articles"] * art_growth_factor
        ).round().astype(int)

        fc = fc.reset_index()
        fc["content_type"] = content_type
        fc["topic_key"] = topic_key
        fc["topic_id"] = topic_id
        fc["topic_label"] = topic_label
        fc[f"growth_factor_{BASE_YEAR}_to_{COMPARE_YEAR}"] = growth_factor
        fc["date"] = EVENT_ANCHORS[FORECAST_YEAR] + pd.to_timedelta(fc["event_day"], unit="D")

        topic_forecasts.append(fc)

    if topic_forecasts:
        _imps_col  = f"forecast_impressions_{FORECAST_YEAR}"
        _arts_col  = f"forecast_articles_{FORECAST_YEAR}"
        _base_imps = f"base_{COMPARE_YEAR}_impressions"
        _base_arts = f"base_{COMPARE_YEAR}_articles"
        _growth    = f"growth_factor_{BASE_YEAR}_to_{COMPARE_YEAR}"

        forecast_2026_by_topic = pd.concat(topic_forecasts, ignore_index=True)
        forecast_2026_by_topic = forecast_2026_by_topic.rename(
            columns={_imps_col: "independent_topic_forecast"}
        )

        content_type_daily_total = forecast_2026_by_type[
            ["content_type", "event_day", _imps_col]
        ].rename(columns={_imps_col: "content_type_forecast"})

        forecast_2026_by_topic = forecast_2026_by_topic.merge(
            content_type_daily_total,
            on=["content_type", "event_day"],
            how="left",
        )
        forecast_2026_by_topic["topic_day_total_forecast"] = (
            forecast_2026_by_topic.groupby(["content_type", "event_day"])
            ["independent_topic_forecast"]
            .transform("sum")
        )
        forecast_2026_by_topic["raw_reconciled_forecast"] = np.where(
            forecast_2026_by_topic["topic_day_total_forecast"] > 0,
            forecast_2026_by_topic["independent_topic_forecast"]
            / forecast_2026_by_topic["topic_day_total_forecast"]
            * forecast_2026_by_topic["content_type_forecast"],
            0,
        )
        forecast_2026_by_topic["forecast_floor"] = np.floor(
            forecast_2026_by_topic["raw_reconciled_forecast"]
        ).astype(int)
        forecast_2026_by_topic["rounding_remainder"] = (
            forecast_2026_by_topic["content_type_forecast"].astype(int)
            - forecast_2026_by_topic.groupby(["content_type", "event_day"])
            ["forecast_floor"]
            .transform("sum")
        )
        forecast_2026_by_topic["rounding_fraction"] = (
            forecast_2026_by_topic["raw_reconciled_forecast"]
            - forecast_2026_by_topic["forecast_floor"]
        )
        forecast_2026_by_topic["rounding_rank"] = (
            forecast_2026_by_topic.groupby(["content_type", "event_day"])
            ["rounding_fraction"]
            .rank(method="first", ascending=False)
        )
        forecast_2026_by_topic[_imps_col] = (
            forecast_2026_by_topic["forecast_floor"]
            + (
                forecast_2026_by_topic["rounding_rank"]
                <= forecast_2026_by_topic["rounding_remainder"]
            ).astype(int)
        )

        forecast_2026_by_topic = forecast_2026_by_topic[
            [
                "date",
                "event_day",
                "content_type",
                "topic_key",
                "topic_id",
                "topic_label",
                _base_imps,
                _base_arts,
                _growth,
                "independent_topic_forecast",
                "content_type_forecast",
                _imps_col,
                _arts_col,
            ]
        ]

        _topic_csv = f"data/{FORECAST_YEAR}_daily_impression_forecast_by_topic.csv"
        forecast_2026_by_topic.to_csv(_topic_csv, index=False)
        print(f"Saved: {_topic_csv}")

        print(
            forecast_2026_by_topic.groupby(
                ["content_type", "topic_key", "topic_label"]
            )[_imps_col].sum()
        )
    else:
        print("No topic forecasts created. Check that each topic has data in both BASE_YEAR and COMPARE_YEAR.")
else:
    print("No topic columns found. Run 2-find-topics.py first to create data/with_topics.csv.")
