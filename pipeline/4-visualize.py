# 4-visualize.py — Visualize topics and impression forecasts

import pandas as pd
import plotly.express as px

from config import EVENT_NAME, FORECAST_YEAR

_imps_col = f"forecast_impressions_{FORECAST_YEAR}"

# ── Load data ────────────────────────────────────────────
topics = pd.read_csv("data/with_topics.csv")
forecast_by_type = pd.read_csv(f"data/{FORECAST_YEAR}_daily_impression_forecast_by_content_type.csv")
forecast_by_topic = pd.read_csv(f"data/{FORECAST_YEAR}_daily_impression_forecast_by_topic.csv")

# ── Topic distribution (one chart per content type) ──────────
topic_counts = (
    topics.dropna(subset=["topic_id", "topic_label"])
    .query("topic_id != -1")
    .groupby(["content_type", "topic_label"], as_index=False)
    .size()
    .rename(columns={"size": "article_count"})
)

for ct, group in topic_counts.groupby("content_type"):
    group = group.sort_values("article_count", ascending=True)
    fig = px.bar(
        group,
        x="article_count",
        y="topic_label",
        orientation="h",
        title=f"Topic Distribution — {ct}",
        labels={"article_count": "Articles", "topic_label": "Topic"},
        height=400,
    )
    fig.show()

# ── Forecast by content type ─────────────────────────────────
fig = px.line(
    forecast_by_type,
    x="event_day",
    y=_imps_col,
    color="content_type",
    markers=True,
    title=f"{EVENT_NAME} {FORECAST_YEAR} — Daily Impression Forecast by Content Type",
    labels={
        "event_day": f"Days Relative to {EVENT_NAME}",
        _imps_col: "Forecast Impressions",
        "content_type": "Content Type",
    },
)
fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Event day")
fig.show()

# ── Forecast by topic ──────────────────────────────────────────
fig = px.line(
    forecast_by_topic,
    x="event_day",
    y=_imps_col,
    color="topic_label",
    facet_col="content_type",
    facet_col_wrap=2,
    markers=True,
    title=f"{EVENT_NAME} {FORECAST_YEAR} — Daily Impression Forecast by Topic & Content Type",
    labels={
        "event_day": f"Days Relative to {EVENT_NAME}",
        _imps_col: "Forecast Impressions",
        "topic_label": "Topic",
    },
)
fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Event day")
fig.update_layout(height=600)
fig.show()
