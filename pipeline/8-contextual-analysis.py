import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    EVENT_ANCHORS,
    EVENT_NAME,
    FORECAST_WINDOW_END,
    FORECAST_WINDOW_START,
    FORECAST_YEAR,
)


VERTICAL = "Fashion"
COUNTRY = "France"
THEME = "Fashion Week"
SOURCE_PATH = Path("data/with_topics.csv")
OUT_HTML = Path("reports/fashion_france_fashion_week_contextual_analysis_2026.html")
OUT_CSV = Path("data/fashion_france_fashion_week_contextual_signals_2026.csv")

URL_COL = "url"
IMPS_COL = "ClientCreativeImpression"
SIGNAL_COL = "page_categories_tier1"
TOPIC_COL = "topic_label"
TYPE_COL = "content_type"

BG = "#EBE6E4"
TEXT = "#000000"
MUTED = "#8D8A89"
BORDER = "#D4D0CE"
ACCENT = "#FF6B7C"
BLUE = "#5476FF"
PURPLE = "#EB66F4"
GREY = "#948A8A"
GREEN = "#459084"
RED = "#DD4C45"
PALETTE = [BLUE, PURPLE, GREY, "#67C9FE", "#FFA071", "#A36AFF", "#F4D56D"]


def clean_signal(value):
    if pd.isna(value) or str(value).strip() == "":
        return "Unclassified"
    return str(value).strip()


def fmt_int(value):
    return f"{int(round(value)):,}"


def fmt_pct(value):
    return f"{float(value):.1f}%"


def fig_to_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def table_html(df, columns):
    if df.empty:
        return '<p class="empty">No data available.</p>'
    return df[columns].to_html(index=False, classes="data-table", border=0, escape=False)


def apply_theme(fig, height=420):
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Instrument Sans, Arial, sans-serif", color=TEXT, size=15),
        title=dict(font=dict(size=24, color=TEXT)),
        height=height,
        margin=dict(l=70, r=28, t=70, b=60),
        colorway=PALETTE,
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
        hoverlabel=dict(
            bgcolor="rgba(235,230,228,0.95)",
            bordercolor=BORDER,
            font=dict(color="#2F2E2E", size=15),
        ),
    )
    fig.update_xaxes(
        title_font=dict(size=15, color="#2F2E2E"),
        gridcolor="rgba(0,0,0,0.1)",
        linecolor="rgba(0,0,0,0.3)",
        tickfont=dict(color="#2F2E2E", size=10),
    )
    fig.update_yaxes(
        title_font=dict(size=15, color="#2F2E2E"),
        gridcolor="rgba(0,0,0,0.1)",
        linecolor="rgba(0,0,0,0.3)",
        tickfont=dict(color="#2F2E2E", size=10),
    )
    return fig


def assign_universe(topic, signal):
    text = f"{topic} {signal}".lower()
    if any(k in text for k in ["beauty", "makeup", "hair"]):
        return "Beauty, Hair & Makeup"
    if any(k in text for k in ["sneaker", "footwear", "accessory", "color", "shopping"]):
        return "Accessories & Retail Cues"
    if any(k in text for k in ["luxury", "couture", "brand", "runway", "style-and-fashion"]):
        return "Luxury & Runway"
    if any(k in text for k in ["celebrity", "kardashian", "k-pop", "model", "influencer", "pop-culture"]):
        return "Celebrity & Influence"
    if any(k in text for k in ["events", "music", "television", "movies"]):
        return "Culture & Entertainment"
    return "General Lifestyle Context"


df = pd.read_csv(SOURCE_PATH)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year
df["event_date"] = df["year"].map(EVENT_ANCHORS)
df = df.dropna(subset=["date", "event_date"]).copy()
df["event_day"] = (df["date"] - df["event_date"]).dt.days
df["forecast_date"] = EVENT_ANCHORS[FORECAST_YEAR] + pd.to_timedelta(df["event_day"], unit="D")
df = df[df["forecast_date"].between(FORECAST_WINDOW_START, FORECAST_WINDOW_END)].copy()
df[SIGNAL_COL] = df[SIGNAL_COL].apply(clean_signal)
df[TOPIC_COL] = df[TOPIC_COL].fillna("Unclassified Topic")
df["contextual_universe"] = df.apply(
    lambda row: assign_universe(row[TOPIC_COL], row[SIGNAL_COL]),
    axis=1,
)

actual_start = df["date"].min()
actual_end = df["date"].max()
total_articles = len(df)
total_imps = df[IMPS_COL].fillna(0).sum()
classified_signal_share = (df[SIGNAL_COL] != "Unclassified").mean() * 100

signal_perf = (
    df.groupby(SIGNAL_COL, as_index=False)
    .agg(
        articles=(URL_COL, "count"),
        impressions=(IMPS_COL, "sum"),
        topics=(TOPIC_COL, "nunique"),
        evergreen_articles=(TYPE_COL, lambda s: int((s == "Evergreen").sum())),
    )
)
signal_perf["share_impressions_pct"] = signal_perf["impressions"] / total_imps * 100
signal_perf["avg_impressions_per_article"] = signal_perf["impressions"] / signal_perf["articles"]
signal_perf["evergreen_share_pct"] = signal_perf["evergreen_articles"] / signal_perf["articles"] * 100
signal_perf = signal_perf.sort_values("impressions", ascending=False)
signal_perf.to_csv(OUT_CSV, index=False)

signal_display = signal_perf.copy()
signal_display["Articles"] = signal_display["articles"].map(fmt_int)
signal_display["Impressions"] = signal_display["impressions"].map(fmt_int)
signal_display["Share"] = signal_display["share_impressions_pct"].map(fmt_pct)
signal_display["Avg / Article"] = signal_display["avg_impressions_per_article"].map(lambda v: f"{v:,.1f}")
signal_display["Evergreen Mix"] = signal_display["evergreen_share_pct"].map(fmt_pct)
signal_display = signal_display.rename(columns={SIGNAL_COL: "Contextual Signal", "topics": "Topics"})

topic_perf = (
    df.groupby(["contextual_universe", TOPIC_COL, TYPE_COL], as_index=False)
    .agg(articles=(URL_COL, "count"), impressions=(IMPS_COL, "sum"))
    .sort_values("impressions", ascending=False)
)
topic_totals = (
    df.groupby(["contextual_universe", TOPIC_COL], as_index=False)
    .agg(articles=(URL_COL, "count"), impressions=(IMPS_COL, "sum"))
    .sort_values("impressions", ascending=False)
)
topic_totals["share_impressions_pct"] = topic_totals["impressions"] / total_imps * 100

topic_display = topic_totals.head(14).copy()
topic_display["Articles"] = topic_display["articles"].map(fmt_int)
topic_display["Impressions"] = topic_display["impressions"].map(fmt_int)
topic_display["Share"] = topic_display["share_impressions_pct"].map(fmt_pct)
topic_display = topic_display.rename(
    columns={"contextual_universe": "Fashion Context", TOPIC_COL: "Topic"}
)

universe_perf = (
    df.groupby("contextual_universe", as_index=False)
    .agg(articles=(URL_COL, "count"), impressions=(IMPS_COL, "sum"), topics=(TOPIC_COL, "nunique"))
    .sort_values("impressions", ascending=False)
)
universe_perf["share_impressions_pct"] = universe_perf["impressions"] / total_imps * 100

daily = (
    df.groupby(["forecast_date", TYPE_COL], as_index=False)
    .agg(articles=(URL_COL, "count"), impressions=(IMPS_COL, "sum"))
    .sort_values("forecast_date")
)

known_signal_perf = signal_perf[signal_perf[SIGNAL_COL] != "Unclassified"]
top_signal = known_signal_perf.iloc[0] if not known_signal_perf.empty else signal_perf.iloc[0]
top_topic = topic_totals.iloc[0]
top_universe = universe_perf.iloc[0]
universe_cumulative = universe_perf["share_impressions_pct"].cumsum()
main_context_count = int((universe_cumulative < 90).sum() + 1)
main_context_share = float(universe_perf.head(main_context_count)["share_impressions_pct"].sum())

strategy_rows = []
for _, row in universe_perf.head(5).iterrows():
    action = {
        "Celebrity & Influence": "Use talent, front-row, and social proof angles for reach-led fashion adjacency.",
        "Beauty, Hair & Makeup": "Package beauty routines, runway hair, and makeup trend contexts for consideration.",
        "Luxury & Runway": "Prioritize premium maison, runway, and collection adjacency for brand equity.",
        "Accessories & Retail Cues": "Use accessory, color, and footwear contexts for shoppable product discovery.",
        "Culture & Entertainment": "Pair event coverage with celebrity media and entertainment moments.",
    }.get(row["contextual_universe"], "Use as a broad lifestyle context for incremental reach.")
    strategy_rows.append(
        {
            "Fashion Context": row["contextual_universe"],
            "Scale": fmt_int(row["impressions"]),
            "Share": fmt_pct(row["share_impressions_pct"]),
            "Activation Role": action,
        }
    )
strategy_df = pd.DataFrame(strategy_rows)

known_categories = known_signal_perf.head(8).copy()
category_rows = [
    {"name": row[SIGNAL_COL], "imps": int(row["impressions"])}
    for _, row in known_categories.iterrows()
]
context_rows = [
    {"name": row["contextual_universe"], "imps": int(row["impressions"])}
    for _, row in universe_perf.head(8).iterrows()
]
topic_pills = [
    {
        "name": row[TOPIC_COL],
        "imps": int(row["impressions"]),
        "share": round(float(row["share_impressions_pct"]), 2),
        "color": PALETTE[i % len(PALETTE)],
    }
    for i, (_, row) in enumerate(topic_totals.head(7).iterrows())
]
mix_rows = (
    df.groupby(TYPE_COL, as_index=False)
    .agg(articles=(URL_COL, "count"), impressions=(IMPS_COL, "sum"))
    .sort_values("impressions", ascending=False)
)
mix_total = mix_rows["impressions"].sum()
intent_rows = [
    {
        "name": row[TYPE_COL],
        "pct": round(float(row["impressions"] / mix_total * 100), 1),
        "imps": int(row["impressions"]),
        "color": BLUE if row[TYPE_COL] == "Evergreen" else PURPLE,
    }
    for _, row in mix_rows.iterrows()
]
article_source = (
    df[[URL_COL, "page_content_title_formalized", SIGNAL_COL, IMPS_COL]]
    .dropna(subset=[URL_COL])
    .sort_values(IMPS_COL, ascending=False)
    .drop_duplicates(URL_COL)
    .head(8)
)
article_rows = [
    {
        "url": str(row[URL_COL]).split("/")[0],
        "title": str(row["page_content_title_formalized"])[:90],
        "cat": row[SIGNAL_COL],
        "imps": int(row[IMPS_COL]),
        "share": round(float(row[IMPS_COL] / total_imps * 100), 2),
        "color": ACCENT if row[SIGNAL_COL] == "style-and-fashion" else BLUE,
    }
    for _, row in article_source.iterrows()
]
perf_rows = [
    {
        "signal": row[SIGNAL_COL],
        "articles": int(row["articles"]),
        "imps": int(row["impressions"]),
        "share": round(float(row["share_impressions_pct"]), 2),
        "avg": round(float(row["avg_impressions_per_article"]), 1),
        "evergreen": round(float(row["evergreen_share_pct"]), 1),
    }
    for _, row in signal_perf.head(14).iterrows()
]
monthly_rows = (
    df.groupby(df["forecast_date"].dt.strftime("%b"), sort=False)[IMPS_COL]
    .sum()
    .reset_index(name="imps")
)
daily_rows = (
    df.groupby("forecast_date", as_index=False)[IMPS_COL]
    .sum()
    .sort_values("forecast_date")
)
daily_chart_rows = [
    {"date": row["forecast_date"].strftime("%d %b"), "imps": int(row[IMPS_COL])}
    for _, row in daily_rows.iterrows()
]
chart_context_rows = [
    {"name": row["contextual_universe"], "imps": int(row["impressions"])}
    for _, row in universe_perf.head(8).iterrows()
]

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{VERTICAL} {COUNTRY} {THEME} — Seedtag Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Instrument+Serif:ital@1&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  :root {{
    --bg-primary: #EBE6E4;
    --text-primary: #000000;
    --text-secondary: #8D8A89;
    --accent-brand: #FF6B7C;
    --border-primary: #D4D0CE;
    --bg-box: #FFFFFF;
    --axis-color: #2F2E2E;
    --chart-1: #5476FF;
    --chart-2: #EB66F4;
    --chart-3: #948A8A;
    --chart-4: #67C9FE;
    --chart-5: #FFA071;
    --chart-6: #A36AFF;
    --chart-7: #F4D56D;
    --chart-safe: #459084;
    --chart-unsafe: #DD4C45;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #EBE6E4; color: #000; font-family: 'Instrument Sans', sans-serif; min-height: 100vh; padding: 48px 56px; max-width: 1280px; margin: 0 auto; }}
  .marker {{ font-size: 7px; font-weight: 600; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-secondary); }}
  .title-sans {{ font-size: 36px; font-weight: 700; line-height: 0.9; color: var(--text-primary); }}
  .title-serif {{ font-family: 'Instrument Serif', serif; font-style: italic; font-size: 46px; line-height: 0.9; color: var(--accent-brand); }}
  .section-label {{ font-size: 7px; font-weight: 600; letter-spacing: 0.18em; text-transform: uppercase; color: var(--text-secondary); border-bottom: 0.5px solid var(--border-primary); padding-bottom: 10px; margin-bottom: 18px; }}
  .header {{ display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 0.5px solid var(--border-primary); padding-bottom: 28px; margin-bottom: 36px; }}
  .header-titles {{ display: flex; align-items: baseline; gap: 12px; margin: 8px 0 6px; }}
  .header-meta {{ text-align: right; display: flex; flex-direction: column; gap: 6px; align-items: flex-end; }}
  .header-meta .pill {{ font-size: 8px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; background: var(--bg-box); color: var(--text-secondary); padding: 4px 10px; border-radius: 2px; }}
  .kpi-strip {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--border-primary); border: 0.5px solid var(--border-primary); margin-bottom: 32px; }}
  .kpi {{ background: #fff; padding: 20px 18px; }}
  .kpi-value {{ font-size: 36px; font-weight: 400; line-height: 1; color: var(--text-primary); }}
  .kpi-value.text {{ font-size: 24px; line-height: 1; }}
  .kpi-value em {{ color: var(--accent-brand); font-style: normal; }}
  .kpi-label {{ font-size: 7px; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-secondary); margin-top: 10px; }}
  .kpi-sub {{ font-size: 8px; color: var(--text-secondary); margin-top: 3px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: #C8C3C1; border: 1px solid #C8C3C1; margin-bottom: 1px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1px; background: #C8C3C1; border: 1px solid #C8C3C1; margin-bottom: 1px; }}
  .grid-section {{ margin-bottom: 32px; }}
  .panel {{ background: #fff; padding: 22px 20px; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.75); }}
  .bar-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
  .bar-label {{ font-size: 10px; color: var(--text-primary); width: 150px; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .bar-track {{ flex: 1; background: #EBE6E4; height: 3px; border-radius: 1px; }}
  .bar-fill {{ height: 3px; border-radius: 1px; }}
  .bar-val {{ font-size: 10px; color: var(--text-secondary); width: 48px; text-align: right; flex-shrink: 0; }}
  .pills-wrap {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .emotion-pill {{ display: inline-flex; align-items: center; gap: 6px; padding: 7px 11px; border: 0.5px solid var(--border-primary); border-radius: 2px; }}
  .emotion-dot {{ width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }}
  .emotion-name {{ font-size: 10px; color: var(--text-primary); }}
  .emotion-ctr {{ font-size: 8px; color: var(--accent-brand); font-weight: 600; }}
  .intent-row {{ margin-bottom: 14px; }}
  .intent-head {{ display: flex; justify-content: space-between; margin-bottom: 5px; }}
  .intent-name {{ font-size: 10px; color: var(--text-primary); text-transform: capitalize; }}
  .intent-pct {{ font-size: 10px; color: var(--text-secondary); }}
  .intent-track {{ background: #EBE6E4; height: 3px; border-radius: 1px; }}
  .intent-fill {{ height: 3px; border-radius: 1px; }}
  .intent-sub {{ font-size: 8px; color: var(--text-secondary); margin-top: 3px; }}
  .article-row {{ display: flex; align-items: center; gap: 10px; padding: 9px 0; border-bottom: 0.5px solid var(--border-primary); }}
  .article-idx {{ font-size: 8px; color: var(--text-secondary); width: 14px; flex-shrink: 0; }}
  .article-url {{ font-size: 10px; color: var(--text-primary); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .article-cat {{ font-size: 7px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; padding: 2px 7px; border-radius: 2px; flex-shrink: 0; }}
  .article-imp {{ font-size: 10px; color: var(--text-secondary); width: 44px; text-align: right; flex-shrink: 0; }}
  .article-ctr {{ font-size: 10px; color: var(--accent-brand); font-weight: 600; width: 44px; text-align: right; flex-shrink: 0; }}
  .perf-table {{ width: 100%; border-collapse: collapse; }}
  .perf-table th {{ font-size: 7px; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--text-secondary); text-align: right; padding: 0 0 10px; border-bottom: 0.5px solid var(--border-primary); }}
  .perf-table th:first-child {{ text-align: left; }}
  .perf-table td {{ font-size: 10px; color: var(--text-primary); text-align: right; padding: 8px 0; border-bottom: 0.5px solid var(--border-primary); }}
  .perf-table td:first-child {{ text-align: left; }}
  .perf-table tr:nth-child(even) td {{ background: #EBE6E4; }}
  .fmt-tag {{ font-size: 7px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; padding: 2px 6px; border-radius: 2px; margin-left: 5px; vertical-align: middle; }}
  .fmt-d {{ background: rgba(84,118,255,0.15); color: #5476FF; }}
  .fmt-v {{ background: rgba(235,102,244,0.15); color: #EB66F4; }}
  .hi {{ color: var(--chart-safe) !important; }}
  .lo {{ color: var(--chart-unsafe) !important; }}
  .donut-wrap {{ display: flex; align-items: center; gap: 28px; height: 220px; }}
  .donut-legend {{ display: flex; flex-direction: column; gap: 16px; }}
  .donut-legend-item .dl-marker {{ font-size: 7px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 2px; }}
  .donut-legend-item .dl-val {{ font-size: 24px; font-weight: 400; line-height: 1; }}
  .donut-legend-item .dl-sub {{ font-size: 8px; color: var(--text-secondary); margin-top: 2px; }}
  .footer {{ border-top: 0.5px solid var(--border-primary); margin-top: 40px; padding-top: 16px; display: flex; justify-content: space-between; align-items: center; }}
  .footer span {{ font-size: 8px; color: var(--text-secondary); }}
  .footer .brand {{ color: var(--accent-brand); font-weight: 600; letter-spacing: 0.1em; }}
  @media(max-width:900px) {{ body {{ padding: 28px 18px; }} .header {{ display:block; }} .header-meta {{ align-items:flex-start; text-align:left; margin-top:18px; }} .kpi-strip,.grid-2,.grid-3 {{ grid-template-columns:1fr; }} .header-titles {{ flex-wrap:wrap; }} }}
</style>
</head>
<body>
<header class="header">
  <div>
    <span class="marker">Seedtag · Intelligence Report</span>
    <div class="header-titles">
      <span class="title-sans">{VERTICAL}</span>
      <span class="title-serif">{COUNTRY}</span>
    </div>
    <span style="font-size:10px;color:var(--text-secondary);">Vertical Contextual Analysis · {EVENT_NAME} · {FORECAST_YEAR}</span>
  </div>
  <div class="header-meta">
    <span class="marker">Contextual Signal Analysis</span>
    <span class="pill">{df[SIGNAL_COL].nunique()} contextual signals</span>
    <span class="pill">{df[TOPIC_COL].nunique()} topic clusters</span>
    <span class="pill">{actual_start:%d %b %Y} - {actual_end:%d %b %Y}</span>
  </div>
</header>

<div class="kpi-strip">
  <div class="kpi"><div class="kpi-value"><em>{fmt_int(total_articles)}</em></div><div class="kpi-label">Articles</div><div class="kpi-sub">in inferred Fashion Week window</div></div>
  <div class="kpi"><div class="kpi-value"><em>{total_imps/1e6:.2f}</em>M</div><div class="kpi-label">Impressions</div><div class="kpi-sub">historical performance base</div></div>
  <div class="kpi"><div class="kpi-value"><em>{main_context_count}</em></div><div class="kpi-label">Main Categories</div><div class="kpi-sub">{main_context_share:.1f}% of impressions</div></div>
  <div class="kpi"><div class="kpi-value"><em>{df[TOPIC_COL].nunique()}</em></div><div class="kpi-label">Topics</div><div class="kpi-sub">Fashion Week clusters</div></div>
</div>

<div class="grid-section">
  <div class="grid-2">
    <div class="panel"><div class="section-label">Top contextual signals · impressions</div><div id="catBars"></div></div>
    <div class="panel"><div class="section-label">Fashion context universes · impressions</div><div id="contextBars"></div></div>
  </div>
</div>

<div class="grid-section">
  <div class="grid-3">
    <div class="panel"><div class="section-label">Top topics · share of impressions</div><div class="pills-wrap" id="topicPills"></div></div>
    <div class="panel"><div class="section-label">Content type mix · distribution</div><div id="intentBars"></div></div>
    <div class="panel"><div class="section-label">Top articles · impressions + share</div><div id="articleList"></div></div>
  </div>
</div>

<div class="grid-section">
  <div style="border: 0.5px solid var(--border-primary);">
    <div class="panel">
      <div class="section-label">Performance by contextual signal · local data</div>
      <table class="perf-table">
        <thead><tr><th>Signal</th><th>Articles</th><th>Impressions</th><th>Share %</th><th>Avg / Art.</th><th>Evergreen %</th></tr></thead>
        <tbody id="perfBody"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="grid-section">
  <div class="grid-2">
    <div class="panel" style="border: 0.5px solid var(--border-primary);">
      <div class="section-label">Context universe scale · top 8</div>
      <div style="position:relative;height:240px;"><canvas id="contextChart" role="img" aria-label="Fashion France context universe scale"></canvas></div>
    </div>
    <div class="panel" style="border: 0.5px solid var(--border-primary);">
      <div class="section-label">Mix evergreen / spontaneous · impressions</div>
      <div class="donut-wrap">
        <div style="position:relative;height:200px;width:200px;flex-shrink:0;"><canvas id="donutChart" role="img" aria-label="Fashion France content type mix"></canvas></div>
        <div class="donut-legend" id="donutLegend"></div>
      </div>
    </div>
  </div>
</div>

<div class="grid-section">
  <div class="grid-2">
    <div class="panel" style="border:0.5px solid var(--border-primary);">
      <div class="section-label">Daily impressions · activation window</div>
      <div style="position:relative;height:220px;"><canvas id="dailyChart" role="img" aria-label="Daily impressions Fashion France"></canvas></div>
    </div>
    <div class="panel" style="border:0.5px solid var(--border-primary);">
      <div class="section-label">Signal source health · known vs unclassified</div>
      <div style="position:relative;height:220px;"><canvas id="coverageChart" role="img" aria-label="Known versus unclassified contextual signals"></canvas></div>
    </div>
  </div>
</div>

<footer class="footer">
  <span>Source: {SOURCE_PATH} · vertical = {VERTICAL} · country = {COUNTRY} · inferred date range = {actual_start:%Y-%m-%d} to {actual_end:%Y-%m-%d}</span>
  <span class="brand">Seedtag Intelligence</span>
</footer>

<script>
  const cats = {json.dumps(category_rows, ensure_ascii=False)};
  const contexts = {json.dumps(context_rows, ensure_ascii=False)};
  const topics = {json.dumps(topic_pills, ensure_ascii=False)};
  const intents = {json.dumps(intent_rows, ensure_ascii=False)};
  const articles = {json.dumps(article_rows, ensure_ascii=False)};
  const perfData = {json.dumps(perf_rows, ensure_ascii=False)};
  const contextChartRows = {json.dumps(chart_context_rows, ensure_ascii=False)};
  const dailyRows = {json.dumps(daily_chart_rows, ensure_ascii=False)};
  const knownCoverage = {classified_signal_share:.4f};
  const palette = ['#5476FF','#EB66F4','#948A8A','#67C9FE','#FFA071','#A36AFF','#F4D56D'];

  function fmtM(n){{ return n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?Math.round(n/1e3)+'K':n; }}
  function esc(s){{ return String(s).replace(/[&<>"']/g, m => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m])); }}

  const maxCat = Math.max(...cats.map(c=>c.imps), 1);
  document.getElementById('catBars').innerHTML = cats.map(c=>`
    <div class="bar-row"><div class="bar-label" title="${{esc(c.name)}}">${{esc(c.name)}}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${{Math.round(c.imps/maxCat*100)}}%;background:#5476FF"></div></div>
      <div class="bar-val">${{fmtM(c.imps)}}</div></div>`).join('');

  const maxContext = Math.max(...contexts.map(c=>c.imps), 1);
  document.getElementById('contextBars').innerHTML = contexts.map((c,i)=>`
    <div class="bar-row"><div class="bar-label" title="${{esc(c.name)}}">${{esc(c.name)}}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${{Math.round(c.imps/maxContext*100)}}%;background:${{palette[i%palette.length]}}"></div></div>
      <div class="bar-val">${{fmtM(c.imps)}}</div></div>`).join('');

  document.getElementById('topicPills').innerHTML = topics.map(t=>`
    <div class="emotion-pill"><span class="emotion-dot" style="background:${{t.color}}"></span>
      <span class="emotion-name">${{esc(t.name)}}</span><span class="emotion-ctr">${{t.share.toFixed(2)}}%</span></div>`).join('');

  document.getElementById('intentBars').innerHTML = intents.map(it=>`
    <div class="intent-row"><div class="intent-head"><span class="intent-name">${{esc(it.name)}}</span><span class="intent-pct">${{it.pct}}%</span></div>
      <div class="intent-track"><div class="intent-fill" style="width:${{it.pct}}%;background:${{it.color}}"></div></div>
      <div class="intent-sub">${{fmtM(it.imps)}} imps</div></div>`).join('');

  document.getElementById('articleList').innerHTML = articles.map((a,i)=>`
    <div class="article-row" title="${{esc(a.title)}}"><span class="article-idx">${{i+1}}</span><span class="article-url">${{esc(a.url)}}</span>
      <span class="article-cat" style="background:${{a.color}}22;color:${{a.color}}">${{esc(a.cat).slice(0,14)}}</span>
      <span class="article-imp">${{fmtM(a.imps)}}</span><span class="article-ctr">${{a.share.toFixed(2)}}%</span></div>`).join('');

  const tbody = document.getElementById('perfBody');
  perfData.forEach(r=>{{
    const evC = r.evergreen>=70?'hi':r.evergreen<25?'lo':'';
    const shareC = r.share>=1?'hi':r.share<0.05?'lo':'';
    tbody.innerHTML += `<tr><td>${{esc(r.signal)}}<span class="fmt-tag ${{r.signal==='Unclassified'?'fmt-v':'fmt-d'}}">${{r.signal==='Unclassified'?'Gap':'Known'}}</span></td>
      <td>${{r.articles.toLocaleString()}}</td><td>${{fmtM(r.imps)}}</td><td class="${{shareC}}">${{r.share.toFixed(2)}}%</td>
      <td>${{r.avg.toLocaleString()}}</td><td class="${{evC}}">${{r.evergreen.toFixed(1)}}%</td></tr>`;
  }});

  const chartDefaults = {{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ display:false }} }},
    scales:{{
      x:{{ ticks:{{ font:{{size:8,family:'Instrument Sans'}}, color:'#8D8A89', maxRotation:40, autoSkip:false }}, grid:{{ display:false }}, border:{{ color:'#D4D0CE' }} }},
      y:{{ ticks:{{ font:{{size:8,family:'Instrument Sans'}}, color:'#8D8A89', callback:v=>fmtM(v) }}, grid:{{ color:'rgba(0,0,0,0.08)' }}, border:{{ display:false }} }}
    }}
  }};

  new Chart(document.getElementById('contextChart'), {{
    type:'bar',
    data:{{ labels: contextChartRows.map(r=>r.name), datasets:[{{ data: contextChartRows.map(r=>r.imps), backgroundColor:'#5476FF', borderRadius:2 }}] }},
    options: chartDefaults
  }});

  document.getElementById('donutLegend').innerHTML = intents.map((it,i)=>`
    <div class="donut-legend-item"><div class="dl-marker">${{esc(it.name)}}</div>
      <div class="dl-val" style="color:${{it.color}}">${{it.pct}}%</div><div class="dl-sub">${{fmtM(it.imps)}} imps</div></div>`).join('');
  new Chart(document.getElementById('donutChart'), {{
    type:'doughnut',
    data:{{ labels:intents.map(i=>i.name), datasets:[{{ data:intents.map(i=>i.pct), backgroundColor:intents.map(i=>i.color), borderWidth:0, hoverOffset:4 }}] }},
    options:{{ responsive:false, cutout:'70%', plugins:{{ legend:{{ display:false }} }} }}
  }});

  new Chart(document.getElementById('dailyChart'), {{
    type:'line',
    data:{{ labels: dailyRows.map(r=>r.date), datasets:[{{ data: dailyRows.map(r=>r.imps), borderColor:'#FF6B7C', backgroundColor:'rgba(255,107,124,0.15)', fill:true, tension:0.28, pointRadius:0 }}] }},
    options: chartDefaults
  }});

  new Chart(document.getElementById('coverageChart'), {{
    type:'bar',
    data:{{ labels:['Known signals','Unclassified'], datasets:[{{ data:[knownCoverage, 100-knownCoverage], backgroundColor:['#459084','#DD4C45'], borderRadius:2 }}] }},
    options:{{ ...chartDefaults, scales:{{ ...chartDefaults.scales, y:{{ ...chartDefaults.scales.y, max:100, ticks:{{ font:{{size:8,family:'Instrument Sans'}}, color:'#8D8A89', callback:v=>v+'%' }} }} }} }}
  }});
</script>
</body>
</html>
"""

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
OUT_HTML.write_text(html, encoding="utf-8")

print(f"HTML report: {OUT_HTML}")
print(f"Signal CSV : {OUT_CSV}")
print(f"Rows       : {len(df):,}")
print(f"Date range : {actual_start.date()} to {actual_end.date()}")
print(f"Signals    : {df[SIGNAL_COL].nunique()}")
print(f"Topics     : {df[TOPIC_COL].nunique()}")
