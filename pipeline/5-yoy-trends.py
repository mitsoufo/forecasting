# 5-yoy-trends.py — Temporal analysis, YoY trends, and forecast topic projections
# Input:  data/with_topics.csv  (from 2-find-topics.py)
#         Falls back to data/classified.csv
# Output: data/timing_all.csv
#         data/timing_by_year.csv
#         data/yoy_breakdown.csv
#         data/{FORECAST_YEAR}_topic_projections.csv

from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────
TEXT_COL  = "page_content_title_formalized"
BODY_COL  = "page_content_body_formalized"
URL_COL   = "url"
LABEL_COL = "content_type"

from config import (
    EVENT_NAME,
    EVENT_ANCHORS,
    FORECAST_YEAR,
    FORECAST_WINDOW_START,
    FORECAST_WINDOW_END,
)

# ── Load data ────────────────────────────────────────────────
TOPIC_DATA_PATH  = Path("data/with_topics.csv")
SOURCE_DATA_PATH = TOPIC_DATA_PATH if TOPIC_DATA_PATH.exists() else Path("data/classified.csv")

df = pd.read_csv(SOURCE_DATA_PATH)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
print(f"Loaded {len(df):,} rows from {SOURCE_DATA_PATH}")

# ── Temporal alignment (days_to_event) ──────────────────────
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df_temporal = df.dropna(subset=["date"]).copy()
df_temporal["year"] = df_temporal["date"].dt.year

anchors = EVENT_ANCHORS
df_temporal["event_date"] = df_temporal["year"].map(anchors)
df_temporal = df_temporal.dropna(subset=["event_date"])
df_temporal["days_to_event"] = (df_temporal["date"] - df_temporal["event_date"]).dt.days
df_temporal["forecast_date"] = (
    EVENT_ANCHORS[FORECAST_YEAR]
    + pd.to_timedelta(df_temporal["days_to_event"], unit="D")
)
before_window = len(df_temporal)
df_temporal = df_temporal[
    df_temporal["forecast_date"].between(FORECAST_WINDOW_START, FORECAST_WINDOW_END)
].copy()

print(f"Rows with valid dates: {len(df_temporal):,}")
print(
    f"Forecast window: {FORECAST_WINDOW_START.date()} to {FORECAST_WINDOW_END.date()} "
    f"({len(df_temporal):,}/{before_window:,} rows retained)"
)
print(f"Years covered: {sorted(df_temporal['year'].unique())}")

# ── Evergreen subset ─────────────────────────────────────────
df_ev = df_temporal[df_temporal[LABEL_COL] == "Evergreen"].copy()
print(f"Evergreen articles: {len(df_ev):,}")

# ── Volume curve: articles & impressions by day ──────────────
timing_all = (
    df_ev
    .groupby("days_to_event")
    .agg(articles=(URL_COL, "count"), impressions=("ClientCreativeImpression", "sum"))
    .reset_index()
    .sort_values("days_to_event")
)

timing_by_year = (
    df_ev
    .groupby(["year", "days_to_event"])
    .agg(articles=(URL_COL, "count"), impressions=("ClientCreativeImpression", "sum"))
    .reset_index()
    .sort_values(["year", "days_to_event"])
)

peak = timing_all.nlargest(5, "impressions")[["days_to_event", "articles", "impressions"]]
print("\nTop 5 days by impressions:")
print(peak.to_string(index=False))

# Save timing data before plotting so report inputs exist even in headless runs.
timing_all.to_csv("data/timing_all.csv", index=False)
timing_by_year.to_csv("data/timing_by_year.csv", index=False)
print("Saved: data/timing_all.csv")
print("Saved: data/timing_by_year.csv")

# ── Plot volume curves per year ──────────────────────────────
colors = {2024: "#4C72B0", 2025: "#DD8452"}
for year, grp in timing_by_year.groupby("year"):
    color = colors.get(year, "gray")
    anchor = EVENT_ANCHORS[year]
    actual_dates = anchor + pd.to_timedelta(grp["days_to_event"], unit="D")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.bar(actual_dates, grp["articles"],    color=color, alpha=0.7, width=0.8)
    ax2.bar(actual_dates, grp["impressions"], color=color, alpha=0.7, width=0.8)

    ax1.set_ylabel("Articles")
    ax1.set_title(f"Evergreen volume — {EVENT_NAME} {year}")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.set_ylabel("Impressions")
    ax2.set_xlabel("Date")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.axvline(anchor, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Event day")
    ax1.legend()
    ax2.axvline(anchor, color="red", linestyle="--", linewidth=1, alpha=0.6)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.close(fig)

# ── Topic impressions summary (requires 2-find-topics.py output) ─
has_topics = (
    {"topic_id", "topic_label"}.issubset(df_ev.columns)
    and df_ev["topic_id"].notna().any()
)

if has_topics:
    df_ev_t = df_ev[df_ev["topic_id"].notna() & (df_ev["topic_id"] != -1)].copy()
    df_ev_t["topic_id"] = df_ev_t["topic_id"].astype(int)

    topic_summary = (
        df_ev_t
        .groupby(["topic_id", "topic_label"])
        .agg(
            articles          = (URL_COL, "count"),
            impressions       = ("ClientCreativeImpression", "sum"),
            avg_days_to_event = ("days_to_event", "mean"),
            earliest_day      = ("days_to_event", "min"),
            latest_day        = ("days_to_event", "max"),
        )
        .reset_index()
    )

    # Bar chart: top 15 topics by impressions
    plot_df = (
        topic_summary.nlargest(15, "impressions")
        .sort_values("impressions", ascending=True)
        .copy()
    )
    plot_df["label_compact"] = (
        "T" + plot_df["topic_id"].astype(str)
        + " | " + plot_df["topic_label"].str.replace(" | ", ", ", regex=False)
    )
    plot_df["label_wrapped"] = plot_df["label_compact"].apply(lambda x: fill(str(x), width=28))
    plot_df["timing_label"]  = plot_df["avg_days_to_event"].apply(lambda x: f"avg D{x:+.0f}")
    plot_df["imps_M"]        = (plot_df["impressions"] / 1e6).round(2)

    fig, ax = plt.subplots(figsize=(14, 8))
    y    = np.arange(len(plot_df))
    bars = ax.barh(y, plot_df["imps_M"], color="#4C72B0", height=0.72)
    ax.set_yticks(y, labels=plot_df["label_wrapped"])
    ax.set_xlabel("Impressions (M)")
    ax.set_title(
        f"Top Evergreen Topics — {EVENT_NAME}",
        loc="left", fontsize=15, weight="bold",
    )
    ax.xaxis.grid(True, linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    xmax = plot_df["imps_M"].max()
    ax.set_xlim(0, xmax * 1.25)
    for bar, imps_m, timing in zip(bars, plot_df["imps_M"], plot_df["timing_label"]):
        ax.text(
            imps_m + xmax * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{imps_m:.2f}M  |  {timing}", va="center", ha="left", fontsize=9, color="#4B5563",
        )
    plt.tight_layout()
    plt.close(fig)

    # ── YoY Breakdown ────────────────────────────────────────
    forecast_table = (
        df_ev_t
        .groupby(["topic_label", "year"])
        .agg(
            articles          = (URL_COL, "count"),
            impressions       = ("ClientCreativeImpression", "sum"),
            avg_days_to_event = ("days_to_event", "mean"),
            earliest_day      = ("days_to_event", "min"),
            latest_day        = ("days_to_event", "max"),
        )
        .reset_index()
    )
    # NOTE: keep raw `impressions` for projection math; only round the *display*
    # column (`impressions_M`) to 3 decimals. Previously the rounded value fed
    # the projection, which zeroed out any topic with <500 imps.
    forecast_table["impressions_M"]     = forecast_table["impressions"] / 1e6
    forecast_table["avg_days_to_event"] = forecast_table["avg_days_to_event"].round(1)

    forecast_table_total = (
        df_ev_t
        .groupby("topic_label")
        .agg(
            articles          = (URL_COL, "count"),
            impressions       = ("ClientCreativeImpression", "sum"),
            avg_days_to_event = ("days_to_event", "mean"),
            earliest_day      = ("days_to_event", "min"),
            latest_day        = ("days_to_event", "max"),
        )
        .sort_values("impressions", ascending=False)
        .reset_index()
    )
    forecast_table_total["impressions_M"]     = (forecast_table_total["impressions"] / 1e6).round(3)
    forecast_table_total["avg_days_to_event"] = forecast_table_total["avg_days_to_event"].round(1)
    forecast_table_total["window"] = forecast_table_total.apply(
        lambda r: f"D{r['earliest_day']:+d} to D{r['latest_day']:+d}", axis=1
    )

    yoy = forecast_table.pivot_table(
        index="topic_label", columns="year",
        values=["articles", "impressions_M"], aggfunc="sum", fill_value=0,
    )
    yoy.columns = [f"{m}_{int(y)}" for m, y in yoy.columns]
    yoy = yoy.reset_index()

    for metric in ["articles", "impressions_M"]:
        c24, c25 = f"{metric}_2024", f"{metric}_2025"
        if c24 in yoy.columns and c25 in yoy.columns:
            yoy[f"{metric}_growth"] = (
                ((yoy[c25] - yoy[c24]) / yoy[c24].replace(0, np.nan) * 100).round(1)
            )

    sort_col = "impressions_M_2025" if "impressions_M_2025" in yoy.columns else yoy.columns[1]
    yoy = yoy.sort_values(sort_col, ascending=False).reset_index(drop=True)

    print("\n=== YEAR-OVER-YEAR BREAKDOWN ===")
    print(yoy.to_string())

    # ── 2026 Projection per Topic ─────────────────────────────
    proj = yoy[["topic_label"]].copy()

    for metric in ["articles", "impressions_M"]:
        c24, c25 = f"{metric}_2024", f"{metric}_2025"
        has_both = c24 in yoy.columns and c25 in yoy.columns

        if has_both:
            proj[f"{metric}_2026_proj"] = yoy.apply(
                lambda r: round(
                    r[c25] * (1 + r[f"{metric}_growth"] / 100)
                    if pd.notna(r.get(f"{metric}_growth")) and r[c24] > 0
                    else (r[c24] + r[c25]) / 2,
                    3,
                ),
                axis=1,
            )
        elif c25 in yoy.columns:
            proj[f"{metric}_2026_proj"] = yoy[c25]
        elif c24 in yoy.columns:
            proj[f"{metric}_2026_proj"] = yoy[c24]

    proj = proj.merge(
        forecast_table_total[["topic_label", "avg_days_to_event", "window"]],
        on="topic_label", how="left",
    ).sort_values("impressions_M_2026_proj", ascending=False).reset_index(drop=True)

    # Save report inputs before optional chart rendering.
    topic_projection_path = f"data/{FORECAST_YEAR}_topic_projections.csv"
    proj.to_csv(topic_projection_path, index=False)
    print(f"\nSaved: {topic_projection_path}")

    yoy.to_csv("data/yoy_breakdown.csv", index=False)
    print("Saved: data/yoy_breakdown.csv")

    # ── Grouped bar: 2024 vs 2025 vs 2026 ────────────────────
    top15 = proj.head(15)
    x = np.arange(len(top15))
    w = 0.25
    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (col, yr, color, hatch) in enumerate([
        ("impressions_M_2024",      "2024",        "#4C72B0", None),
        ("impressions_M_2025",      "2025",        "#DD8452", None),
        ("impressions_M_2026_proj", "2026 (proj)", "#55A868", "//"),
    ]):
        vals = top15[col].fillna(0).tolist() if col in top15.columns else [0] * len(top15)
        ax.bar(x + i * w, vals, width=w, label=yr, color=color, hatch=hatch, edgecolor="white")
    ax.set_xticks(x + w)
    ax.set_xticklabels(top15["topic_label"].tolist(), rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Impressions (M)")
    ax.set_title("Topic impressions: 2024 vs 2025 vs 2026 projection")
    ax.legend()
    plt.tight_layout()
    plt.close(fig)

    print("\n=== 2026 FORECAST PER TOPIC ===")
    print(proj.to_string())

else:
    print("\nNo topic columns found. Run 2-find-topics.py first.")
