# 6-report.py — Spontaneous budget estimation + interactive HTML report with all visuals
# Input:  data/with_topics.csv                                      (from 2-find-topics.py)
#         data/{FORECAST_YEAR}_daily_impression_forecast_by_content_type.csv (from 3-forecast.py)
#         data/{FORECAST_YEAR}_daily_impression_forecast_by_topic.csv        (from 3-forecast.py)
#         data/timing_all.csv                                        (from 5-yoy-trends.py)
#         data/timing_by_year.csv                                    (from 5-yoy-trends.py)
#         data/yoy_breakdown.csv                                     (from 5-yoy-trends.py)
#         data/{FORECAST_YEAR}_topic_projections.csv                 (from 5-yoy-trends.py)
# Output: {EVENT_KEY}_{FORECAST_YEAR}_forecast_report_v2.html
#         data/{EVENT_KEY}_{FORECAST_YEAR}_recommendations.csv

import datetime
import html as _html_mod
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ──────────────────────────────────────────────────
from config import (
    EVENT_NAME,
    EVENT_KEY,
    EVENT_ANCHORS,
    FORECAST_YEAR,
    EVENT_ANCHOR,
    EVENT_END,
    REPORT_PATH,
    RECO_CSV_PATH,
    FORECAST_WINDOW_START,
    FORECAST_WINDOW_END,
)

URL_COL    = "url"
TEXT_COL   = "page_content_title_formalized"
BODY_COL   = "page_content_body_formalized"
LABEL_COL  = "content_type"
EVENT_LABEL = EVENT_NAME
BF_2026     = EVENT_ANCHOR
FW_END_2026 = EVENT_END
TODAY       = datetime.date.today().strftime("%d %B %Y")
# Fashion Week can have one dominant cluster, so a 5% cutoff hides nearly every
# other usable topic. Keep small-but-actionable topics visible in the report.
MIN_TOPIC_IMPRESSION_SHARE = 0.001
# Maximum number of topics to surface in the activation plan and topic browser.
MAX_TOPICS = 5
USE_FASHION_WEEK_OVERRIDES = EVENT_KEY == "paris_fashion_week"

TOPIC_PRESENTATION_OVERRIDES = {
    "Evergreen_3": {
        "old_label": "Social Media & Celebrity Style",
        "label": "Celebrity News & Fashion Buzz",
        "description": (
            "High-volume celebrity, entertainment, and fashion-industry stories that create broad "
            "cultural attention around the event. This cluster is less about specific outfits and "
            "more about people, media narratives, and public-interest moments connected to fashion."
        ),
        "example_terms": (
            "véronique nichanian",
            "veronique nichanian",
            "hermès",
            "hermes",
            "jaden smith",
            "louboutin",
            "milan fashion week",
            "camille",
            "slimane",
            "laeticia",
            "zahia",
            "angelina",
        ),
        "preferred_url_terms": (
            "veronique-nichanian",
            "jaden-smith",
            "jetais-a-la-milan-fashion-week",
        ),
    },
    "Evergreen_9": {
        "old_label": "Iconic Model & Celebrity Style",
        "label": "Model & Red-Carpet Looks",
        "description": (
            "Look-led coverage of models, actresses, and public figures at runways, red carpets, "
            "and fashion events. This cluster is about the outfit, silhouette, styling, and visual "
            "impact of appearances rather than wider celebrity news."
        ),
        "example_terms": (
            "iris mittenaere",
            "charlize theron",
            "laetitia casta",
            "valeria bruni",
            "bella hadid",
            "ester expósito",
            "ester exposito",
            "virginie efira",
        ),
        "preferred_url_terms": (
            "iris-mittenaere",
            "charlize-theron",
            "laetitia-casta",
            "valeria-bruni-tedeschi",
            "bella-hadid",
        ),
    },
}

TOPIC_EXAMPLE_OVERRIDES = {
    "Evergreen_13": {
        "example_terms": (
            "vanessa paradis",
            "jack depp",
            "lily-rose depp",
            "deva cassel",
            "monica bellucci",
            "chanel",
            "dior haute couture",
        ),
        "preferred_url_terms": (
            "ca-minspire-vanessa-paradis",
            "vanessa-paradis-son-fils-jack",
            "deva-cassel-la-fille-de-monica-bellucci",
        ),
    },
    "Evergreen_6": {
        "example_terms": (
            "ballerines",
            "chaussure",
            "pull",
            "jean",
            "baskets",
            "mi-saison",
            "automne",
        ),
        "preferred_url_terms": (
            "adieu-les-ballerines",
            "le-pull-parfait-existe",
            "il-est-parfait-je-lai-pris-en-3-couleurs",
        ),
    },
    "Evergreen_4": {
        "example_terms": (
            "brigitte macron",
            "carla bruni",
            "robe",
            "veste",
            "manteau",
            "dior",
        ),
        "preferred_url_terms": (
            "brigitte-macron-porte-la-tendance-robe",
            "brigitte-macron-porte-la-tendance-veste",
            "carla-bruni-ne-jure-que-par-ce-manteau-long",
        ),
    },
}


def topic_example_config(topic_key):
    cfg = {}
    if USE_FASHION_WEEK_OVERRIDES:
        cfg.update(TOPIC_PRESENTATION_OVERRIDES.get(str(topic_key), {}))
        cfg.update(TOPIC_EXAMPLE_OVERRIDES.get(str(topic_key), {}))
    return cfg


def apply_topic_presentation_overrides(frame):
    if frame is None or frame.empty or "topic_label" not in frame.columns:
        return frame
    if not USE_FASHION_WEEK_OVERRIDES:
        return frame
    frame = frame.copy()
    old_to_new = {
        cfg["old_label"]: cfg["label"]
        for cfg in TOPIC_PRESENTATION_OVERRIDES.values()
    }
    if "topic_key" in frame.columns:
        for topic_key, cfg in TOPIC_PRESENTATION_OVERRIDES.items():
            mask = frame["topic_key"].astype(str).eq(topic_key)
            frame.loc[mask, "topic_label"] = cfg["label"]
            if "topic_description" in frame.columns:
                frame.loc[mask, "topic_description"] = cfg["description"]
    frame["topic_label"] = frame["topic_label"].replace(old_to_new)
    return frame


def report_date(value, with_weekday=False):
    fmt = "%a %d %b %Y" if with_weekday else "%d %b %Y"
    return pd.to_datetime(value).strftime(fmt)


def event_offset_date(offset, with_weekday=False):
    return report_date(BF_2026 + pd.Timedelta(days=int(offset)), with_weekday)


def event_range_label():
    return f"{report_date(BF_2026)} to {report_date(FW_END_2026)}"


def event_range_x1():
    return (FW_END_2026 + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def window_to_dates(window):
    match = re.fullmatch(r"\s*D([+-]\d+)\s+to\s+D([+-]\d+)\s*", str(window))
    if not match:
        return window
    start, end = (int(match.group(1)), int(match.group(2)))
    return f"{event_offset_date(start)} to {event_offset_date(end)}"

# Seedtag chart theme from font.html. Keep Plotly charts aligned with the
# surrounding report instead of falling back to Plotly's blue/orange defaults.
CHART_BG = "#EBE6E4"
CHART_TEXT = "#2F2E2E"
CHART_GRID = "#D4D0CE"
CHART_ACCENT = "#FF6B7C"
CHART_BLUE = "#5476FF"
CHART_PINK = "#FF6B7C"
CHART_GRAY = "#948A8A"
CHART_CYAN = "#67C9FE"
CHART_ORANGE = "#FFA071"
CHART_PURPLE = "#A36AFF"
CHART_YELLOW = "#F4D56D"
CHART_PALETTE = [
    CHART_BLUE,
    CHART_PINK,
    CHART_GRAY,
    CHART_CYAN,
    CHART_ORANGE,
    CHART_PURPLE,
    CHART_YELLOW,
]
CONTENT_COLORS = {"Evergreen": CHART_BLUE, "Spontaneous": CHART_PINK}
CONTENT_LINE_COLORS = {"Evergreen": CHART_TEXT, "Spontaneous": "#CC3347"}

# ── Load source data for spontaneous budget ──────────────────
df_src = pd.read_csv("data/with_topics.csv")
df_src = df_src.loc[:, ~df_src.columns.str.startswith("Unnamed")]
df_src["date"] = pd.to_datetime(df_src["date"], errors="coerce")
df_src["year"] = df_src["date"].dt.year

anchors = EVENT_ANCHORS
df_src["event_date"] = df_src["year"].map(anchors)
df_temporal = df_src.dropna(subset=["date", "event_date"]).copy()
df_temporal["days_to_event"] = (df_temporal["date"] - df_temporal["event_date"]).dt.days
df_temporal["forecast_date"] = (
    EVENT_ANCHORS[FORECAST_YEAR]
    + pd.to_timedelta(df_temporal["days_to_event"], unit="D")
)
df_temporal = df_temporal[
    df_temporal["forecast_date"].between(FORECAST_WINDOW_START, FORECAST_WINDOW_END)
].copy()

# ── Load intermediate data from 5-yoy-trends.py ─────────────
proj           = pd.read_csv(f"data/{FORECAST_YEAR}_topic_projections.csv")
yoy            = pd.read_csv("data/yoy_breakdown.csv")
timing_all     = pd.read_csv("data/timing_all.csv")
timing_by_year = pd.read_csv("data/timing_by_year.csv")

# ── Load topic descriptions (LLM generated) ─────────────────
topic_desc_path = Path("data/topic_descriptions.csv")
if topic_desc_path.exists():
    topic_desc_df = pd.read_csv(topic_desc_path)
    topic_desc_df = apply_topic_presentation_overrides(topic_desc_df)
    desc_by_label = dict(zip(topic_desc_df["topic_label"], topic_desc_df["topic_description"].fillna("")))
    desc_by_key   = dict(zip(topic_desc_df["topic_key"],   topic_desc_df["topic_description"].fillna("")))
else:
    topic_desc_df = pd.DataFrame(columns=["topic_key", "topic_label", "topic_description"])
    desc_by_label, desc_by_key = {}, {}

def _wrap_desc(text: str, width: int = 70) -> str:
    """Wrap long descriptions for plotly hover tooltips (uses <br>)."""
    if not isinstance(text, str) or not text:
        return ""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur: lines.append(cur)
    return "<br>".join(lines)

# ── Load step-3 forecast CSVs ────────────────────────────────
forecast_by_type  = pd.read_csv(f"data/{FORECAST_YEAR}_daily_impression_forecast_by_content_type.csv", parse_dates=["date"])
forecast_by_topic_df = (
    pd.read_csv(f"data/{FORECAST_YEAR}_daily_impression_forecast_by_topic.csv")
    if Path(f"data/{FORECAST_YEAR}_daily_impression_forecast_by_topic.csv").exists()
    else pd.DataFrame()
)
topic_exact_forecasts = pd.DataFrame()
_topic_imps_col = f"forecast_impressions_{FORECAST_YEAR}"
_topic_arts_col = f"forecast_articles_{FORECAST_YEAR}"
if (
    not forecast_by_topic_df.empty
    and {"topic_label", _topic_imps_col, _topic_arts_col}.issubset(forecast_by_topic_df.columns)
):
    topic_forecast_source = forecast_by_topic_df
    if LABEL_COL in topic_forecast_source.columns:
        topic_forecast_source = topic_forecast_source[
            topic_forecast_source[LABEL_COL] == "Evergreen"
        ]
    topic_exact_forecasts = (
        topic_forecast_source
        .groupby("topic_label", as_index=False)
        .agg(
            exact_forecast_impressions=(_topic_imps_col, "sum"),
            exact_forecast_articles=(_topic_arts_col, "sum"),
        )
    )
    proj = proj.merge(topic_exact_forecasts, on="topic_label", how="left")
    proj["impressions_M_2026_proj"] = (
        proj["exact_forecast_impressions"]
        .fillna(proj["impressions_M_2026_proj"] * 1e6)
        / 1e6
    )
    proj["articles_2026_proj"] = proj["exact_forecast_articles"].fillna(proj["articles_2026_proj"])
    proj = proj.drop(columns=["exact_forecast_impressions", "exact_forecast_articles"])

df_src = apply_topic_presentation_overrides(df_src)
df_temporal = apply_topic_presentation_overrides(df_temporal)
proj = apply_topic_presentation_overrides(proj)
yoy = apply_topic_presentation_overrides(yoy)
timing_all = apply_topic_presentation_overrides(timing_all)
timing_by_year = apply_topic_presentation_overrides(timing_by_year)
forecast_by_topic_df = apply_topic_presentation_overrides(forecast_by_topic_df)

# ── Spontaneous budget ───────────────────────────────────────
spont_base = (
    df_temporal
    .groupby(["year", LABEL_COL])
    .agg(articles=(URL_COL, "count"), impressions=("ClientCreativeImpression", "sum"))
    .reset_index()
)
totals = spont_base.groupby("year")[["articles", "impressions"]].transform("sum")
spont_base["pct_articles"]    = (spont_base["articles"]    / totals["articles"]    * 100).round(1)
spont_base["pct_impressions"] = (spont_base["impressions"] / totals["impressions"] * 100).round(1)
spont_base["impressions_M"]   = (spont_base["impressions"] / 1e6).round(3)

spont_only    = spont_base[spont_base[LABEL_COL] == "Spontaneous"].set_index("year")
spont_avg_pct = spont_only["pct_impressions"].mean().round(1)
spont_article_pct = spont_only["pct_articles"].mean().round(1)
spont_avg_M   = spont_only["impressions_M"].mean()

print("=== SPONTANEOUS BASELINE BY YEAR ===")
print(
    spont_base[
        ["year", LABEL_COL, "articles", "pct_articles", "impressions_M", "pct_impressions"]
    ].to_string(index=False)
)
print(f"\nAvg spontaneous impression share: {spont_avg_pct:.1f}%")
print(f"Avg spontaneous article share: {spont_article_pct:.1f}%")

# ── Topic heatmap data ────────────────────────────────────────
df_ev_t = pd.DataFrame()
forecast_table_total = pd.DataFrame()

if "topic_label" in df_temporal.columns and "topic_id" in df_temporal.columns:
    df_ev_t = df_temporal[
        (df_temporal[LABEL_COL] == "Evergreen")
        & df_temporal["topic_label"].notna()
        & (df_temporal["topic_id"] != -1)
    ].copy()

    if not df_ev_t.empty:
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

heat = pd.DataFrame()
if not df_ev_t.empty:
    heat = (
        df_ev_t
        .groupby(["topic_label", "days_to_event"])["ClientCreativeImpression"]
        .sum()
        .unstack(fill_value=0)
    )

# ── Scale daily curve to 2026 totals ─────────────────────────
# Use the daily forecast file (3-forecast.py) as source of truth for the
# Evergreen / total impression projections. This file applies the YoY growth
# factor to actual 2025 daily impressions and INCLUDES outlier topics, so it
# matches the real impression universe (~141K total vs the topic-projection
# subset which only sums non-outlier evergreen topics).
if f"forecast_impressions_{FORECAST_YEAR}" in forecast_by_type.columns:
    total_ev_proj_imps = int(
        forecast_by_type.loc[
            forecast_by_type["content_type"] == "Evergreen",
            f"forecast_impressions_{FORECAST_YEAR}",
        ].sum()
    )
    total_all_proj_imps = int(forecast_by_type[f"forecast_impressions_{FORECAST_YEAR}"].sum())
else:
    total_ev_proj_imps  = int(proj["impressions_M_2026_proj"].sum() * 1e6)
    total_all_proj_imps = int(total_ev_proj_imps / max(1 - spont_avg_pct / 100, 1e-9))

total_ev_proj_M = total_ev_proj_imps / 1e6
total_proj_M    = total_all_proj_imps / 1e6
total_arts_2026 = proj["articles_2026_proj"].sum()

total_ev_arts_2026 = int(forecast_by_type.loc[forecast_by_type["content_type"] == "Evergreen", f"forecast_articles_{FORECAST_YEAR}"].sum()) if f"forecast_articles_{FORECAST_YEAR}" in forecast_by_type.columns else 0
total_all_arts_2026 = int(forecast_by_type[f"forecast_articles_{FORECAST_YEAR}"].sum()) if f"forecast_articles_{FORECAST_YEAR}" in forecast_by_type.columns else 0

projected_budget_rows = pd.DataFrame()
if f"forecast_impressions_{FORECAST_YEAR}" in forecast_by_type.columns:
    projected_content_totals = (
        forecast_by_type
        .groupby(LABEL_COL, as_index=False)
        .agg(
            impressions=(f"forecast_impressions_{FORECAST_YEAR}", "sum"),
            articles=(f"forecast_articles_{FORECAST_YEAR}", "sum"),
        )
    )
    projected_total_imps = projected_content_totals["impressions"].sum()
    projected_total_articles = projected_content_totals["articles"].sum()
    projected_content_totals["pct_impressions"] = (
        projected_content_totals["impressions"] / projected_total_imps * 100
    ).round(1)
    projected_content_totals["pct_articles"] = (
        projected_content_totals["articles"] / projected_total_articles * 100
    ).round(1)
    projected_content_totals["year"] = FORECAST_YEAR
    projected_content_totals["impressions_M"] = (
        projected_content_totals["impressions"] / 1e6
    ).round(3)
    projected_budget_rows = projected_content_totals[
        ["year", LABEL_COL, "articles", "pct_articles", "impressions_M", "pct_impressions", "impressions"]
    ]
    projected_spont = projected_content_totals[
        projected_content_totals[LABEL_COL] == "Spontaneous"
    ]
    if not projected_spont.empty:
        spont_avg_pct = float(projected_spont["pct_impressions"].iloc[0])
        spont_article_pct = float(projected_spont["pct_articles"].iloc[0])

daily = timing_all.copy()
daily["imps_2026"] = (
    daily["impressions"] / daily["impressions"].sum() * total_ev_proj_imps
).round(0)
daily["arts_2026"] = (
    daily["articles"] / daily["articles"].sum() * total_arts_2026
).round(0)
daily["date_2026"] = daily["days_to_event"].apply(
    lambda d: BF_2026 + pd.Timedelta(days=int(d))
)

peak_idx      = daily["imps_2026"].idxmax()
peak_day_imps = daily.loc[peak_idx, "imps_2026"]
peak_d_offset = int(daily.loc[peak_idx, "days_to_event"])
peak_day_date = daily.loc[peak_idx, "date_2026"].strftime("%d %b")

# ── Helper functions ─────────────────────────────────────────
def apply_seedtag_chart_theme(fig):
    fig.update_layout(
        template="none",
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        colorway=CHART_PALETTE,
        font=dict(family="Poppins, sans-serif", color=CHART_TEXT),
        hoverlabel=dict(bgcolor=CHART_BG, font_color=CHART_TEXT),
    )
    fig.update_xaxes(
        color=CHART_TEXT,
        linecolor=CHART_TEXT,
        gridcolor=CHART_GRID,
        zerolinecolor=CHART_GRID,
    )
    fig.update_yaxes(
        color=CHART_TEXT,
        linecolor=CHART_TEXT,
        gridcolor=CHART_GRID,
        zerolinecolor=CHART_GRID,
    )
    for shape in fig.layout.shapes or []:
        if getattr(getattr(shape, "line", None), "color", None) in {"red", "#FF0000"}:
            shape.line.color = CHART_ACCENT
    for annotation in fig.layout.annotations or []:
        if not getattr(annotation, "font", None):
            annotation.font = {}
        annotation.font.color = CHART_TEXT
    return fig


def fig_to_html(fig):
    apply_seedtag_chart_theme(fig)
    return fig.to_html(full_html=False, include_plotlyjs=False)

def df_to_html(df, cols=None, no_toggle=False):
    sub = df[cols] if cols else df
    classes = "data-table no-toggle" if no_toggle else "data-table"
    return sub.to_html(index=False, classes=classes, border=0)

# ── Plotly figures ────────────────────────────────────────────
colors_yr = {2024: CHART_GRAY, 2025: CHART_BLUE, 2026: CHART_ACCENT}

# Fig 1: Evergreen volume curve 2024 vs 2025
fig1 = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    subplot_titles=("Articles per day", "Impressions (M) per day"),
)
for yr, grp in timing_by_year.groupby("year"):
    c = colors_yr.get(yr, CHART_GRAY)
    fig1.add_trace(
        go.Bar(
            x=grp["days_to_event"], y=grp["articles"],
            name=str(yr), marker_color=c, legendgroup=str(yr),
        ),
        row=1, col=1,
    )
    fig1.add_trace(
        go.Bar(
            x=grp["days_to_event"], y=(grp["impressions"] / 1e6).round(3),
            name=str(yr), marker_color=c, legendgroup=str(yr), showlegend=False,
        ),
        row=2, col=1,
    )
fig1.add_vline(x=0, line_dash="dash", line_color=CHART_ACCENT, annotation_text=EVENT_NAME)
fig1.update_layout(
    title=f"Evergreen volume relative to {EVENT_LABEL}",
    barmode="overlay", height=500, legend_title="Year",
)

# Fig 2: Top evergreen topics
if not forecast_table_total.empty:
    top15_plot = forecast_table_total.head(15).sort_values("impressions_M")
    top15_plot = top15_plot.assign(
        desc=top15_plot["topic_label"].map(lambda lbl: _wrap_desc(desc_by_label.get(lbl, "")))
    )
    fig2 = go.Figure(go.Bar(
        x=top15_plot["impressions_M"], y=top15_plot["topic_label"],
        orientation="h", marker_color=CHART_ACCENT,
        text=[f"avg {event_offset_date(d)}" for d in top15_plot["avg_days_to_event"]],
        textposition="outside",
        customdata=top15_plot["desc"],
        hovertemplate="<b>%{y}</b><br>Impressions: %{x:.3f}M<br><br>%{customdata}<extra></extra>",
    ))
    fig2.update_layout(
        title=f"Top Evergreen Topics — {EVENT_LABEL}",
        xaxis_title="Impressions (M)", height=500, margin=dict(l=300),
    )
else:
    fig2 = go.Figure()
    fig2.update_layout(title="Top Evergreen Topics — no topic data available")

# Fig 3: Topic activity heatmap
if not heat.empty:
    heat_plot = heat.fillna(0)
    fig3 = go.Figure(go.Heatmap(
        z=heat_plot.values,
        x=[f"D{d:+d}" for d in heat_plot.columns],
        y=heat_plot.index.tolist(),
        colorscale=[
            [0.0, CHART_GRID],
            [0.5, CHART_ORANGE],
            [1.0, CHART_ACCENT],
        ],
        colorbar_title="Impressions",
    ))
    fig3.update_layout(title=f"Topic activity heatmap — {EVENT_LABEL}", height=520)
else:
    fig3 = go.Figure()
    fig3.update_layout(title="Topic heatmap — no topic data available")

# Fig 4: 2024 vs 2025 vs 2026 grouped bar
top15p = proj.head(15)
fig4 = go.Figure()
for col, yr, color, pat in [
    ("impressions_M_2024",      "2024",        CHART_GRAY, ""),
    ("impressions_M_2025",      "2025",        CHART_BLUE, ""),
    ("impressions_M_2026_proj", "2026 (proj)", CHART_ACCENT, "/"),
]:
    vals = top15p[col].fillna(0).tolist() if col in top15p.columns else [0] * len(top15p)
    fig4.add_trace(go.Bar(
        name=yr, x=top15p["topic_label"], y=vals,
        marker_color=color, marker_pattern_shape=pat,
    ))
fig4.update_layout(
    barmode="group",
    title="Topic impressions: 2024 vs 2025 vs 2026",
    xaxis_tickangle=-35, yaxis_title="Impressions (M)", height=500,
)

# Fig: topic distribution per content type
topic_counts = (
    df_src
    .dropna(subset=["topic_id", "topic_label"])
    .query("topic_id != -1")
    .groupby([LABEL_COL, "topic_label"], as_index=False)
    .size()
    .rename(columns={"size": "article_count"})
)
content_types_for_dist = sorted(topic_counts[LABEL_COL].unique())
fig_dist = make_subplots(
    rows=1, cols=len(content_types_for_dist),
    subplot_titles=content_types_for_dist,
    shared_yaxes=False,
)
for col_idx, ct_label in enumerate(content_types_for_dist, start=1):
    grp = topic_counts[topic_counts[LABEL_COL] == ct_label].sort_values("article_count")
    fig_dist.add_trace(
        go.Bar(
            x=grp["article_count"], y=grp["topic_label"],
            orientation="h",
            marker_color=CONTENT_COLORS.get(ct_label, CHART_ACCENT),
            name=ct_label,
        ),
        row=1, col=col_idx,
    )
fig_dist.update_layout(
    title=f"Topic Distribution by Content Type — {EVENT_LABEL}",
    height=max(400, 30 * topic_counts["topic_label"].nunique()),
    showlegend=False,
    margin=dict(l=220),
)

# Fig: daily forecast by content type (from step 3)
fig_fc_type = go.Figure()
for ct_label, grp in forecast_by_type.groupby(LABEL_COL):
    color = CONTENT_COLORS.get(ct_label, CHART_ACCENT)
    fig_fc_type.add_trace(go.Scatter(
        x=pd.to_datetime(grp["date"]), y=grp[f"forecast_impressions_{FORECAST_YEAR}"],
        mode="lines+markers", name=ct_label, line_color=color,
        hovertemplate="%{x|%d %b %Y}<br>Impressions: %{y:,.0f}<extra></extra>",
    ))
fig_fc_type.add_shape(
    type="rect",
    x0=BF_2026.strftime("%Y-%m-%d"),
    x1=event_range_x1(),
    y0=0,
    y1=1,
    xref="x",
    yref="paper",
    fillcolor=CHART_ACCENT,
    opacity=0.10,
    line=dict(width=0),
)
fig_fc_type.add_annotation(
    x=BF_2026 + (FW_END_2026 - BF_2026) / 2,
    y=1,
    xref="x",
    yref="paper",
    text=f"{EVENT_NAME}<br>{event_range_label()}",
    showarrow=False,
    xanchor="center",
    yanchor="bottom",
)
fig_fc_type.update_layout(
    title=f"{EVENT_NAME} {FORECAST_YEAR} — Daily Impression Forecast by Content Type",
    xaxis_title="Date",
    yaxis_title="Forecast Impressions",
    height=420,
)

# Fig: total impressions per main topic for 2026 (Evergreen only, no outliers)
if not forecast_by_topic_df.empty and "topic_label" in forecast_by_topic_df.columns:
    topic_totals = (
        forecast_by_topic_df[
            (forecast_by_topic_df[LABEL_COL] == "Evergreen") &
            (forecast_by_topic_df["topic_label"] != "_outlier")
        ]
        .groupby("topic_label")[f"forecast_impressions_{FORECAST_YEAR}"]
        .sum()
        .reset_index()
    )
    topic_threshold = (
        topic_totals[f"forecast_impressions_{FORECAST_YEAR}"].sum()
        * MIN_TOPIC_IMPRESSION_SHARE
    )
    topic_totals = (
        topic_totals[
            topic_totals[f"forecast_impressions_{FORECAST_YEAR}"] >= topic_threshold
        ]
        .sort_values(f"forecast_impressions_{FORECAST_YEAR}", ascending=True)
        .tail(MAX_TOPICS)
    )
    topic_totals["desc"] = topic_totals["topic_label"].map(
        lambda lbl: _wrap_desc(desc_by_label.get(lbl, ""))
    )
    fig_fc_topic = go.Figure(go.Bar(
        x=topic_totals[f"forecast_impressions_{FORECAST_YEAR}"],
        y=topic_totals["topic_label"],
        orientation="h",
        marker_color=CHART_ACCENT,
        customdata=topic_totals["desc"],
        hovertemplate="<b>%{y}</b><br>Impressions: %{x:,.0f}<br><br>%{customdata}<extra></extra>",
    ))
    fig_fc_topic.update_layout(
        title=f"{EVENT_NAME} {FORECAST_YEAR} — Main Evergreen Topics by Forecast Impressions",
        xaxis_title="Forecast Impressions",
        height=420,
        margin=dict(l=260),
    )
else:
    fig_fc_topic = None

# Fig: daily topic timing curves, normalized so the arrival pattern is readable
fig_topic_timing_by_type = {}
_topic_imps_col = f"forecast_impressions_{FORECAST_YEAR}"
_topic_arts_col = f"forecast_articles_{FORECAST_YEAR}"
if (
    not forecast_by_topic_df.empty
    and {
        "date",
        LABEL_COL,
        "topic_key",
        "topic_label",
        _topic_imps_col,
        _topic_arts_col,
    }.issubset(forecast_by_topic_df.columns)
):
    timing_src = forecast_by_topic_df[
        forecast_by_topic_df["topic_label"].notna()
        & (forecast_by_topic_df["topic_label"] != "_outlier")
    ].copy()
    timing_src["date"] = pd.to_datetime(timing_src["date"], errors="coerce")
    timing_src = timing_src.dropna(subset=["date"])

    timing_topic_totals = (
        timing_src
        .groupby([LABEL_COL, "topic_key", "topic_label"], as_index=False)
        .agg(total_impressions=(_topic_imps_col, "sum"))
        .query("total_impressions > 0")
    )
    timing_thresholds = (
        timing_topic_totals
        .groupby(LABEL_COL)["total_impressions"]
        .transform("sum")
        * MIN_TOPIC_IMPRESSION_SHARE
    )
    top_timing_topics = (
        timing_topic_totals[timing_topic_totals["total_impressions"] >= timing_thresholds]
        .sort_values([LABEL_COL, "total_impressions"], ascending=[True, False])
        .groupby(LABEL_COL)
        .head(MAX_TOPICS)
    )

    if not top_timing_topics.empty:
        timing_plot = timing_src.merge(
            top_timing_topics[[LABEL_COL, "topic_key", "topic_label"]],
            on=[LABEL_COL, "topic_key", "topic_label"],
            how="inner",
        )
        daily_topic_timing = (
            timing_plot
            .groupby([LABEL_COL, "topic_key", "topic_label", "date"], as_index=False)
            .agg(
                forecast_impressions=(_topic_imps_col, "sum"),
                forecast_articles=(_topic_arts_col, "sum"),
            )
            .sort_values("date")
        )
        daily_topic_timing["topic_peak"] = (
            daily_topic_timing
            .groupby([LABEL_COL, "topic_key"])["forecast_impressions"]
            .transform("max")
        )
        daily_topic_timing["interest"] = np.where(
            daily_topic_timing["topic_peak"] > 0,
            daily_topic_timing["forecast_impressions"] / daily_topic_timing["topic_peak"] * 100,
            0,
        )

        palette = CHART_PALETTE
        for ct_label in ["Evergreen"]:
            ct_timing = daily_topic_timing[daily_topic_timing[LABEL_COL] == ct_label]
            if ct_timing.empty:
                continue

            fig_topic_timing = go.Figure()
            _topic_order = (
                top_timing_topics[top_timing_topics[LABEL_COL] == ct_label]
                .sort_values("total_impressions", ascending=False)
                [["topic_key", "topic_label"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            for idx, (topic_key, topic_label) in enumerate(_topic_order):
                grp = (
                    ct_timing[
                        (ct_timing["topic_key"] == topic_key)
                        & (ct_timing["topic_label"] == topic_label)
                    ]
                    .sort_values("date")
                )
                if grp.empty:
                    continue
                custom = grp[["forecast_impressions", "forecast_articles"]].to_numpy()
                fig_topic_timing.add_trace(
                    go.Scatter(
                        x=grp["date"],
                        y=grp["interest"],
                        mode="lines+markers",
                        name=topic_label,
                        line=dict(
                            color=palette[idx % len(palette)],
                            width=2.4,
                        ),
                        marker=dict(size=5),
                        customdata=custom,
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            "%{x|%d %b %Y}<br>"
                            "Interest: %{y:.0f}/100<br>"
                            "Impressions: %{customdata[0]:,.0f}<br>"
                            "Articles: %{customdata[1]:,.0f}"
                            "<extra></extra>"
                        ),
                    )
                )
            fig_topic_timing.add_shape(
                type="rect",
                x0=BF_2026.strftime("%Y-%m-%d"),
                x1=event_range_x1(),
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                fillcolor=CHART_ACCENT,
                opacity=0.10,
                line=dict(width=0),
            )
            fig_topic_timing.add_annotation(
                x=BF_2026 + (FW_END_2026 - BF_2026) / 2,
                y=1,
                xref="x",
                yref="paper",
                text=f"{EVENT_NAME}<br>{event_range_label()}",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
            )
            fig_topic_timing.update_layout(
                title=f"{EVENT_NAME} {FORECAST_YEAR} — {ct_label} Topic Timing",
                xaxis_title="Date",
                yaxis_title="Normalized Interest",
                yaxis=dict(range=[0, 105], ticksuffix="/100"),
                height=520,
                hovermode="x unified",
                legend=dict(orientation="h", y=1.16, x=0),
                margin=dict(t=120),
            )
            fig_topic_timing_by_type[ct_label] = fig_topic_timing

# Fig: Evergreen forecast by page_categories_tier1 for 2026.
# Allocate the Evergreen forecast total across categories by average historical
# impression share, so categories reconcile to the Evergreen KPI.
CAT_COL = "page_categories_tier1"
if CAT_COL in df_src.columns:
    cat_src = df_src[
        (df_src[LABEL_COL] == "Evergreen")
        & df_src[CAT_COL].notna()
    ].copy()
    cat_yearly = (
        cat_src
        .groupby(["year", CAT_COL])["ClientCreativeImpression"]
        .sum()
        .unstack(fill_value=0)
    )
    cat_shares = cat_yearly.div(cat_yearly.sum(axis=1), axis=0).mean()
    cat_shares = cat_shares / cat_shares.sum()
    cat_forecast = (cat_shares * total_ev_proj_imps).round().astype(int)
    cat_delta = total_ev_proj_imps - int(cat_forecast.sum())
    if cat_delta:
        cat_forecast.loc[cat_forecast.idxmax()] += cat_delta

    cat_both = (
        cat_forecast
        .rename(f"forecast_impressions_{FORECAST_YEAR}")
        .reset_index()
        .sort_values(f"forecast_impressions_{FORECAST_YEAR}", ascending=True)
    )
    cat_top10 = cat_both.tail(10)
    cat_extra = cat_both.iloc[:-10].sort_values(
        f"forecast_impressions_{FORECAST_YEAR}", ascending=False
    )
    cat_extra_html = ""

    fig_cat = go.Figure()
    fig_cat.add_trace(go.Bar(
        x=cat_top10[f"forecast_impressions_{FORECAST_YEAR}"],
        y=cat_top10[CAT_COL],
        orientation="h",
        marker_color=CHART_ACCENT,
        hovertemplate="%{y}<br>Impressions: %{x:,.0f}<extra></extra>",
    ))
    fig_cat.update_layout(
        title=f"{EVENT_NAME} {FORECAST_YEAR} — Evergreen Impression Forecast by Category (Tier 1)",
        xaxis_title="Forecast Impressions",
        height=max(400, len(cat_top10) * 28 + 120),
        margin=dict(l=220),
    )
else:
    fig_cat = None
    cat_extra_html = ""

# Fig 5: Spontaneous uncertainty budget
budget_chart_base = spont_base.copy()
if not projected_budget_rows.empty:
    budget_chart_base = pd.concat([budget_chart_base, projected_budget_rows], ignore_index=True)

fig5 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("% of impressions", "Absolute impressions"),
)
for ct, color in CONTENT_COLORS.items():
    sub = budget_chart_base[budget_chart_base[LABEL_COL] == ct]
    fig5.add_trace(
        go.Bar(
            name=ct, x=sub["year"].astype(str), y=sub["pct_impressions"],
            marker_color=color, legendgroup=ct,
        ),
        row=1, col=1,
    )
    fig5.add_trace(
        go.Bar(
            name=ct, x=sub["year"].astype(str), y=sub["impressions"],
            marker_color=color, legendgroup=ct, showlegend=False,
        ),
        row=1, col=2,
    )
fig5.update_layout(
    barmode="stack",
    title=f"Evergreen vs Spontaneous Mix — {EVENT_LABEL}",
    height=400,
    xaxis=dict(type="category"),
    xaxis2=dict(type="category"),
)

# Fig: projected impressions per day 2026
fig_b = go.Figure(go.Bar(
    x=daily["date_2026"], y=daily["imps_2026"],
    marker_color=CHART_ACCENT,
    hovertemplate="%{x|%d %b %Y}<br>Impressions: %{y:,.0f}<extra></extra>",
))
fig_b.add_vrect(
    x0=BF_2026.strftime("%Y-%m-%d"),
    x1=event_range_x1(),
    fillcolor=CHART_ACCENT,
    opacity=0.10,
    line_width=0,
)
fig_b.add_annotation(
    x=BF_2026 + (FW_END_2026 - BF_2026) / 2,
    y=1,
    xref="x",
    yref="paper",
    text=f"{EVENT_NAME}<br>{event_range_label()}",
    showarrow=False,
    xanchor="center",
    yanchor="bottom",
    font=dict(color=CHART_TEXT),
)
fig_b.update_layout(
    title=f"Projected Impressions per Day — {EVENT_NAME} {FORECAST_YEAR}",
    xaxis_title="Date", yaxis_title="Impressions", height=420,
)

# Fig: projected articles per day
fig_a = go.Figure(go.Bar(
    x=daily["date_2026"], y=daily["arts_2026"],
    marker_color=CHART_BLUE,
    hovertemplate="%{x|%d %b %Y}<br>Articles: %{y:,.0f}<extra></extra>",
))
fig_a.add_vrect(
    x0=BF_2026.strftime("%Y-%m-%d"),
    x1=event_range_x1(),
    fillcolor=CHART_ACCENT,
    opacity=0.10,
    line_width=0,
)
fig_a.add_annotation(
    x=BF_2026 + (FW_END_2026 - BF_2026) / 2,
    y=1,
    xref="x",
    yref="paper",
    text=f"{EVENT_NAME}<br>{event_range_label()}",
    showarrow=False,
    xanchor="center",
    yanchor="bottom",
    font=dict(color=CHART_TEXT),
)
fig_a.update_layout(
    title=f"Projected Articles per Day — {EVENT_NAME} {FORECAST_YEAR}",
    xaxis_title="Date", yaxis_title="Articles", height=420,
)

# Fig: combined articles + impressions per day 2026 (like the 2025 volume chart)
_fc_2026 = forecast_by_type.copy()
_fc_2026["date"] = pd.to_datetime(_fc_2026["date"])
_fc_2026["date_str"] = _fc_2026["date"].dt.strftime("%Y-%m-%d")
_ct_colors = CONTENT_COLORS

daily_article_examples = {}
daily_example_cols = [
    "days_to_event",
    LABEL_COL,
    URL_COL,
    TEXT_COL,
    BODY_COL,
    "ClientCreativeImpression",
    "date",
    "topic_label",
]
for _optional_col in ["original_content_type", "classification_adjusted"]:
    if _optional_col in df_temporal.columns:
        daily_example_cols.append(_optional_col)


def _article_snippet(value, limit=170):
    if pd.isna(value):
        return ""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


if set(daily_example_cols).issubset(df_temporal.columns):
    daily_examples_src = (
        df_temporal.dropna(subset=["days_to_event", LABEL_COL, URL_COL, TEXT_COL])
        .loc[:, daily_example_cols]
    )
    daily_examples_src = (
        daily_examples_src
        .drop_duplicates(["days_to_event", LABEL_COL, URL_COL])
        .sort_values(
            ["days_to_event", LABEL_COL, "ClientCreativeImpression"],
            ascending=[True, True, False],
        )
    )
    for (event_day, content_type), group in daily_examples_src.groupby(["days_to_event", LABEL_COL]):
        daily_article_examples[(int(event_day), content_type)] = [
            {
                "title": str(row[TEXT_COL]),
                "url": str(row[URL_COL]),
                "description": _article_snippet(row[BODY_COL]),
                "impressions": int(row["ClientCreativeImpression"]) if pd.notna(row["ClientCreativeImpression"]) else 0,
                "source_date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d") if pd.notna(row["date"]) else "",
                "topic": str(row.get("topic_label", "")) if pd.notna(row.get("topic_label", "")) else "",
            }
            for _, row in group.head(5).iterrows()
        ]


def daily_point_payloads(frame):
    payloads = []
    for _, point in frame.iterrows():
        event_day = int(point["event_day"]) if pd.notna(point.get("event_day")) else None
        content_type = str(point["content_type"])
        payloads.append(
            {
                "date": pd.to_datetime(point["date"]).strftime("%Y-%m-%d"),
                "event_day": event_day,
                "content_type": content_type,
                "forecast_articles": int(point.get(f"forecast_articles_{FORECAST_YEAR}", 0) or 0),
                "forecast_impressions": int(point.get(f"forecast_impressions_{FORECAST_YEAR}", 0) or 0),
            }
        )
    return payloads

fig_vol_2026 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    subplot_titles=("Articles", "Impressions"),
    vertical_spacing=0.08,
)
for ct, color in _ct_colors.items():
    sub = _fc_2026[_fc_2026["content_type"] == ct].sort_values("date")
    point_payloads = daily_point_payloads(sub)
    line_color = CONTENT_LINE_COLORS.get(ct, CHART_TEXT)
    fig_vol_2026.add_trace(
        go.Bar(
            x=sub["date_str"], y=sub[f"forecast_articles_{FORECAST_YEAR}"],
            name=ct, marker_color=color, legendgroup=ct,
            customdata=point_payloads,
            hoverinfo="none",
        ),
        row=1, col=1,
    )
    fig_vol_2026.add_trace(
        go.Bar(
            x=sub["date_str"], y=sub[f"forecast_impressions_{FORECAST_YEAR}"],
            name=ct, marker_color=color, legendgroup=ct, showlegend=False,
            customdata=point_payloads,
            hoverinfo="none",
        ),
        row=2, col=1,
    )
    fig_vol_2026.add_trace(
        go.Scatter(
            x=sub["date_str"],
            y=sub[f"forecast_articles_{FORECAST_YEAR}"],
            mode="lines+markers",
            name=f"{ct} line",
            legendgroup=ct,
            showlegend=False,
            line=dict(color=line_color, width=2.6),
            marker=dict(size=5, color=line_color),
            customdata=point_payloads,
            hoverinfo="none",
        ),
        row=1, col=1,
    )
    fig_vol_2026.add_trace(
        go.Scatter(
            x=sub["date_str"],
            y=sub[f"forecast_impressions_{FORECAST_YEAR}"],
            mode="lines+markers",
            name=f"{ct} line",
            legendgroup=ct,
            showlegend=False,
            line=dict(color=line_color, width=2.6),
            marker=dict(size=5, color=line_color),
            customdata=point_payloads,
            hoverinfo="none",
        ),
        row=2, col=1,
    )
for row in [1, 2]:
    fig_vol_2026.add_vrect(
        x0=BF_2026.strftime("%Y-%m-%d"),
        x1=event_range_x1(),
        fillcolor=CHART_ACCENT,
        opacity=0.10,
        line_width=0,
        row=row,
        col=1,
    )
fig_vol_2026.update_layout(
    title=f"Daily Forecast — {EVENT_LABEL} {FORECAST_YEAR}",
    barmode="stack",
    xaxis=dict(type="date"),
    xaxis2=dict(type="date", title="Date"),
    yaxis_title="Articles",
    yaxis2_title="Impressions",
    height=650,
    legend=dict(orientation="h", y=1.04, x=0),
)

# Outlier toggle — impressions subplot
# Identify outlier ARTICLES (not days) so that "Hide outliers" removes only
# the high-impression articles and still shows the remaining impressions.
_imp_col = f"forecast_impressions_{FORECAST_YEAR}"

# Build per-article-per-day impression totals from historical aligned data
_art_src = df_temporal.copy()
_art_src["date_str"] = _art_src["forecast_date"].dt.strftime("%Y-%m-%d")
_art_daily = (
    _art_src.groupby(["date_str", LABEL_COL, URL_COL])["ClientCreativeImpression"]
    .sum()
    .reset_index()
    .rename(columns={LABEL_COL: "content_type", "ClientCreativeImpression": "art_imps"})
)

# Outlier articles: per-article-per-day impressions above 95th percentile
_art_outlier_threshold = float(_art_daily["art_imps"].quantile(0.95))
_outlier_contribution = (
    _art_daily[_art_daily["art_imps"] > _art_outlier_threshold]
    .groupby(["date_str", "content_type"])["art_imps"]
    .sum()
    .reset_index()
    .rename(columns={"art_imps": "outlier_imps"})
)

# Historical daily totals per content type
_hist_daily_total = (
    _art_daily.groupby(["date_str", "content_type"])["art_imps"]
    .sum()
    .reset_index()
    .rename(columns={"art_imps": "hist_total"})
)

# Non-outlier fraction = (total - outlier) / total, clamped to [0, 1]
_hist_fractions = _hist_daily_total.merge(
    _outlier_contribution, on=["date_str", "content_type"], how="left"
)
_hist_fractions["outlier_imps"] = _hist_fractions["outlier_imps"].fillna(0)
_hist_fractions["non_outlier_frac"] = (
    (_hist_fractions["hist_total"] - _hist_fractions["outlier_imps"])
    / _hist_fractions["hist_total"].replace(0, float("nan"))
).fillna(1.0).clip(0, 1)

# Apply the fraction to forecast values
_fc_filtered = _fc_2026.merge(
    _hist_fractions[["date_str", "content_type", "non_outlier_frac"]],
    on=["date_str", "content_type"],
    how="left",
)
_fc_filtered["non_outlier_frac"] = _fc_filtered["non_outlier_frac"].fillna(1.0)
_fc_filtered["filtered_impressions"] = (
    _fc_filtered[_imp_col] * _fc_filtered["non_outlier_frac"]
).round(0).fillna(0).astype(int)

_imp_trace_indices = []
_y_all_imp = []
_y_filt_imp = []
for _i, _ct in enumerate(_ct_colors):
    _sub = _fc_2026[_fc_2026["content_type"] == _ct].sort_values("date").reset_index(drop=True)
    _sub_filt = _fc_filtered[_fc_filtered["content_type"] == _ct].sort_values("date").reset_index(drop=True)
    _y_imp = _sub[_imp_col].tolist()
    _y_filt = _sub_filt["filtered_impressions"].tolist()
    for _offset in [1, 3]:   # bar_impressions, line_impressions traces
        _imp_trace_indices.append(4 * _i + _offset)
        _y_all_imp.append(_y_imp)
        _y_filt_imp.append(_y_filt)

fig_vol_2026.update_layout(
    updatemenus=[dict(
        type="buttons",
        direction="right",
        x=0.0, y=-0.08,
        xanchor="left",
        showactive=True,
        bgcolor="#F7F4F2",
        bordercolor="#D4D0CE",
        font=dict(size=12),
        buttons=[
            dict(
                label="Show all",
                method="restyle",
                args=[{"y": _y_all_imp}, _imp_trace_indices],
            ),
            dict(
                label="Hide outliers",
                method="restyle",
                args=[{"y": _y_filt_imp}, _imp_trace_indices],
            ),
        ],
    )]
)

# ── Per-topic activation windows (from daily forecast) ──────
# For each topic find the date range where daily forecast impressions are at
# least 15 % of that topic's peak — gives a focused, differentiated window.
topic_activation_windows = {}
if (
    not forecast_by_topic_df.empty
    and {LABEL_COL, "topic_label", "date", f"forecast_impressions_{FORECAST_YEAR}"}.issubset(
        forecast_by_topic_df.columns
    )
):
    _ev_fc = forecast_by_topic_df[
        (forecast_by_topic_df[LABEL_COL] == "Evergreen")
        & forecast_by_topic_df["topic_label"].notna()
        & (forecast_by_topic_df["topic_label"] != "_outlier")
    ].copy()
    _ev_fc["date"] = pd.to_datetime(_ev_fc["date"], errors="coerce")
    for _lbl, _grp in _ev_fc.groupby("topic_label"):
        _peak = _grp[f"forecast_impressions_{FORECAST_YEAR}"].max()
        if _peak <= 0:
            continue
        _active = _grp[_grp[f"forecast_impressions_{FORECAST_YEAR}"] >= _peak * 0.15]
        if not _active.empty:
            topic_activation_windows[_lbl] = (
                _active["date"].min().strftime("%Y-%m-%d"),
                _active["date"].max().strftime("%Y-%m-%d"),
            )


def activation_window_to_dates(window_tuple):
    return f"{report_date(window_tuple[0])} to {report_date(window_tuple[1])}"


# ── Per-day recommendation table ─────────────────────────────
reco_rows = []
topic_impression_threshold = (
    proj["impressions_M_2026_proj"].fillna(0).sum()
    * 1e6
    * MIN_TOPIC_IMPRESSION_SHARE
)
for _, row in proj.iterrows():
    if pd.isna(row.get("avg_days_to_event")) or pd.isna(row.get("window")):
        continue
    if row.get("impressions_M_2026_proj", 0) * 1e6 < topic_impression_threshold:
        continue
    pub_date = BF_2026 + pd.Timedelta(days=int(row["avg_days_to_event"]))
    _win = topic_activation_windows.get(
        row["topic_label"],
        (pub_date.strftime("%Y-%m-%d"), pub_date.strftime("%Y-%m-%d")),
    )
    reco_rows.append({
        "publish_date":    pub_date.strftime("%Y-%m-%d"),
        "activation_date": report_date(pub_date),
        "topic":           row["topic_label"],
        "est_impressions": int(row["impressions_M_2026_proj"] * 1e6),
        "window":          activation_window_to_dates(_win),
        "window_start":    _win[0],
        "window_end":      _win[1],
    })

reco_df = (
    pd.DataFrame(reco_rows)
    .sort_values("publish_date")
    .reset_index(drop=True)
)
# Keep only the top MAX_TOPICS topics by forecast impressions.
if len(reco_df) > MAX_TOPICS:
    keep = reco_df.nlargest(MAX_TOPICS, "est_impressions")["topic"].tolist()
    reco_df = reco_df[reco_df["topic"].isin(keep)].sort_values("publish_date").reset_index(drop=True)
reco_html = df_to_html(reco_df)

# ── Commercial strategy recommendations ──────────────────────
daily_strategy = (
    _fc_2026
    .groupby(["date", "event_day"], as_index=False)
    .agg(
        forecast_impressions=(f"forecast_impressions_{FORECAST_YEAR}", "sum"),
        forecast_articles=(f"forecast_articles_{FORECAST_YEAR}", "sum"),
    )
    .sort_values("date")
)
peak_row = daily_strategy.sort_values("forecast_impressions", ascending=False).iloc[0]
peak_date = pd.to_datetime(peak_row["date"])
peak_label = report_date(peak_date, with_weekday=True)
peak_imps = int(peak_row["forecast_impressions"])
peak_driver_type = ""
if {LABEL_COL, f"forecast_impressions_{FORECAST_YEAR}"}.issubset(_fc_2026.columns):
    peak_split = _fc_2026[_fc_2026["date"] == peak_date]
    if not peak_split.empty:
        peak_driver_type = str(
            peak_split
            .sort_values(f"forecast_impressions_{FORECAST_YEAR}", ascending=False)
            .iloc[0][LABEL_COL]
        )

if not reco_df.empty:
    rec_dates = pd.to_datetime(reco_df["publish_date"])
    evergreen_start = rec_dates.min()
    evergreen_end = rec_dates.max()
    evergreen_window = f"{report_date(evergreen_start)} to {report_date(evergreen_end)}"
else:
    evergreen_start = BF_2026 - pd.Timedelta(days=14)
    evergreen_end = BF_2026 - pd.Timedelta(days=6)
    evergreen_window = f"{report_date(evergreen_start)} to {report_date(evergreen_end)}"

event_push_start = peak_date - pd.Timedelta(days=1)
event_push_end = peak_date + pd.Timedelta(days=1)
event_push_window = f"{report_date(event_push_start)} to {report_date(event_push_end)}"
event_period_window = event_range_label()

strategy_topics = (
    proj[
        (proj["impressions_M_2026_proj"] * 1e6 >= topic_impression_threshold)
        & (proj["topic_label"] != "_outlier")
    ]
    .sort_values("impressions_M_2026_proj", ascending=False)
    .head(4)
)
topic_focus_text = ", ".join(str(t) for t in strategy_topics["topic_label"])

if not reco_df.empty:
    activation_plot = reco_df.copy()
    activation_plot["publish_date_dt"] = pd.to_datetime(activation_plot["publish_date"])
    activation_plot["window_start_dt"] = pd.to_datetime(activation_plot["window_start"])
    activation_plot["window_end_dt"]   = pd.to_datetime(activation_plot["window_end"])
    activation_plot = activation_plot.sort_values(
        ["publish_date_dt", "est_impressions"],
        ascending=[True, False],
    ).reset_index(drop=True)
    activation_topic_order = activation_plot["topic"].tolist()
    activation_plot["reach_label"] = activation_plot["est_impressions"].map(lambda v: f"{int(v):,} imps")
    activation_plot["win_label"] = (
        activation_plot["window_start_dt"].dt.strftime("%d %b")
        + " – "
        + activation_plot["window_end_dt"].dt.strftime("%d %b")
    )
    # Convert to ms timestamps for Plotly date-axis bar chart
    _base_ms  = (activation_plot["window_start_dt"].astype("int64") // 10**6).tolist()
    _width_ms = (
        (activation_plot["window_end_dt"] - activation_plot["window_start_dt"])
        .dt.total_seconds() * 1000
    ).astype("int64").tolist()
    fig_strategy_activation = go.Figure()
    # Gantt bars — width = activation duration, color = forecast reach
    fig_strategy_activation.add_trace(go.Bar(
        x=_width_ms,
        y=activation_plot["topic"].tolist(),
        base=_base_ms,
        orientation="h",
        marker=dict(
            color=activation_plot["est_impressions"].tolist(),
            colorscale=[[0, CHART_BLUE], [1, CHART_ACCENT]],
            colorbar=dict(title="Reach"),
            line=dict(color=CHART_TEXT, width=0.5),
            opacity=0.85,
        ),
        customdata=activation_plot[["activation_date", "est_impressions", "win_label"]].to_numpy(),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Activation window: %{customdata[2]}<br>"
            "Peak activation: %{customdata[0]}<br>"
            "Forecast reach: %{customdata[1]:,.0f} imps"
            "<extra></extra>"
        ),
        showlegend=False,
    ))
    # Marker at the peak activation date
    fig_strategy_activation.add_trace(go.Scatter(
        x=activation_plot["publish_date_dt"].tolist(),
        y=activation_plot["topic"].tolist(),
        mode="markers",
        marker=dict(
            symbol="line-ns",
            size=16,
            color="white",
            line=dict(color=CHART_TEXT, width=2),
        ),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig_strategy_activation.add_shape(
        type="rect",
        x0=BF_2026.strftime("%Y-%m-%d"),
        x1=event_range_x1(),
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        fillcolor=CHART_ACCENT,
        opacity=0.10,
        line=dict(width=0),
    )
    fig_strategy_activation.add_annotation(
        x=BF_2026 + (FW_END_2026 - BF_2026) / 2,
        y=1,
        xref="x",
        yref="paper",
        text=f"{EVENT_NAME}<br>{event_range_label()}",
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font=dict(color=CHART_TEXT),
    )
    fig_strategy_activation.update_xaxes(type="date", title="Date")
    fig_strategy_activation.update_layout(
        title=f"When to Activate Topics — {EVENT_NAME} {FORECAST_YEAR}",
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=activation_topic_order,
            autorange="reversed",
        ),
        height=380,
        margin=dict(l=230, r=40, t=70, b=50),
        bargap=0.35,
    )
    strategy_activation_graph_html = fig_to_html(fig_strategy_activation)

    topic_activation_df = activation_plot.assign(
        **{
            "Activate": activation_plot.apply(
                lambda r: report_date(r["publish_date_dt"]),
                axis=1,
            ),
            "Topic to activate": activation_plot["topic"],
            "Forecast reach": activation_plot["est_impressions"].map(lambda v: f"{int(v):,} imps"),
        }
    )[["Activate", "Topic to activate", "Forecast reach"]]
else:
    strategy_activation_graph_html = "<p>No topic activation timing available.</p>"
    topic_activation_df = pd.DataFrame(columns=["Activate", "Topic to activate", "Forecast reach"])

topic_activation_html = df_to_html(topic_activation_df, no_toggle=True)

strategy_reco_rows = [
    {
        "Action": "Start evergreen campaign activation",
        "When": evergreen_window,
        "Recommendation": f"Launch prospecting and contextual packages around the main predictable topics: {topic_focus_text}.",
    },
    {
        "Action": "Increase delivery pressure",
        "When": event_push_window,
        "Recommendation": f"Shift budget toward high-reach inventory around the forecast peak on {peak_label} ({peak_imps:,} imps).",
    },
    {
        "Action": "Balance evergreen and spontaneous delivery",
        "When": f"{report_date(BF_2026)} to {report_date(BF_2026 + pd.Timedelta(days=3))}",
        "Recommendation": f"Plan against a {100 - spont_avg_pct:.1f}% / {spont_avg_pct:.1f}% evergreen-to-spontaneous reach mix, with spontaneous kept as a flexible reserve.",
    },
    {
        "Action": "Post-event capture",
        "When": f"{report_date(BF_2026 + pd.Timedelta(days=2))} to {report_date(BF_2026 + pd.Timedelta(days=6))}",
        "Recommendation": "Use retargeting and recap/entertainment inventory after the live-event peak, then taper once daily volume normalizes.",
    },
]
strategy_reco_df = pd.DataFrame(strategy_reco_rows)
strategy_reco_html = df_to_html(strategy_reco_df, no_toggle=True)
strategy_summary_html = f"""
<div class="strategy-cards">
  <div class="strategy-card"><div class="strategy-label">Event period</div><div class="strategy-value">{event_period_window}</div><div class="strategy-note">{EVENT_NAME} {FORECAST_YEAR}</div></div>
  <div class="strategy-card"><div class="strategy-label">Peak attention</div><div class="strategy-value">{peak_label}</div><div class="strategy-note">{peak_imps:,} imps · organic editorial reach</div></div>
  <div class="strategy-card"><div class="strategy-label">High-pressure window</div><div class="strategy-value">{event_push_window}</div><div class="strategy-note">Concentrate delivery around the forecast peak</div></div>
  <div class="strategy-card"><div class="strategy-label">Reach mix</div><div class="strategy-value">{100 - spont_avg_pct:.1f}% / {spont_avg_pct:.1f}%</div><div class="strategy-note">Evergreen / spontaneous impressions</div></div>
</div>
"""
strategy_html = f"""
{strategy_summary_html}
<div class="strategy-graph">{strategy_activation_graph_html}</div>
%%TLDR%%
"""

# ── HTML tables ──────────────────────────────────────────────
proj_disp = proj[
    (proj["impressions_M_2026_proj"] * 1e6 >= topic_impression_threshold)
].copy()
# Convert millions back to absolute impressions for readability
proj_disp["forecast_impressions"] = (proj_disp["impressions_M_2026_proj"] * 1e6).round(0).astype(int).map("{:,}".format)
proj_disp["articles_2026_proj"]   = proj_disp["articles_2026_proj"].round(0).astype(int).map("{:,}".format)
# Add an actual publish date for readability.
proj_disp["publish_day"] = proj_disp["avg_days_to_event"].apply(
    lambda d: event_offset_date(d, with_weekday=True)
    if pd.notna(d) else ""
)
proj_disp["activation_window"] = proj_disp.apply(
    lambda row: activation_window_to_dates(
        topic_activation_windows.get(
            row["topic_label"],
            (
                (BF_2026 + pd.Timedelta(days=int(row["avg_days_to_event"]))).strftime("%Y-%m-%d"),
                (BF_2026 + pd.Timedelta(days=int(row["avg_days_to_event"]))).strftime("%Y-%m-%d"),
            ),
        )
    ) if pd.notna(row.get("avg_days_to_event")) else "",
    axis=1,
)
proj_disp = proj_disp.rename(columns={
    "topic_label":          "Topic",
    "articles_2026_proj":   "Forecast Articles",
    "forecast_impressions": "Forecast Impressions",
    "publish_day":          "Peak Activation Date",
    "activation_window":     "Window",
})
forecast_html = df_to_html(
    proj_disp.head(MAX_TOPICS),
    ["Topic", "Forecast Articles", "Forecast Impressions", "Peak Activation Date", "Window"],
)

# YoY breakdown — top Evergreen topics
def _fmt_growth(v):
    if pd.isna(v): return "—"
    return f"{v:+.1f}%"

def _build_yoy_table(content_type: str) -> pd.DataFrame:
    src = df_temporal[
        (df_temporal[LABEL_COL] == content_type)
        & df_temporal["topic_label"].notna()
        & (df_temporal["topic_label"] != "_outlier")
    ].copy()
    if src.empty:
        return pd.DataFrame()

    main_topics = set(
        proj.loc[
            proj["impressions_M_2026_proj"] * 1e6 >= topic_impression_threshold,
            "topic_label",
        ]
    )
    src = src[src["topic_label"].isin(main_topics)]
    if src.empty:
        return pd.DataFrame()

    grouped = (
        src.groupby(["topic_label", "year"], as_index=False)
        .agg(
            articles=(URL_COL, "count"),
            impressions=("ClientCreativeImpression", "sum"),
        )
    )
    table = grouped.pivot_table(
        index="topic_label",
        columns="year",
        values=["articles", "impressions"],
        aggfunc="sum",
        fill_value=0,
    )
    table.columns = [f"{metric}_{int(year)}" for metric, year in table.columns]
    table = table.reset_index()

    for metric in ["articles", "impressions"]:
        c24, c25 = f"{metric}_2024", f"{metric}_2025"
        if c24 not in table.columns:
            table[c24] = 0
        if c25 not in table.columns:
            table[c25] = 0
        table[f"{metric}_growth"] = np.where(
            table[c24] > 0,
            (table[c25] - table[c24]) / table[c24] * 100,
            np.nan,
        )

    table = table.sort_values("impressions_2025", ascending=False).head(MAX_TOPICS)
    table["impressions_2024"] = table["impressions_2024"].round(0).astype(int).map("{:,}".format)
    table["impressions_2025"] = table["impressions_2025"].round(0).astype(int).map("{:,}".format)
    table["articles_2024"] = table["articles_2024"].round(0).astype(int).map("{:,}".format)
    table["articles_2025"] = table["articles_2025"].round(0).astype(int).map("{:,}".format)
    table["articles_growth"] = table["articles_growth"].apply(_fmt_growth)
    table["impressions_growth"] = table["impressions_growth"].apply(_fmt_growth)

    return table.rename(columns={
        "topic_label": "Topic",
        "articles_2024": "Articles 2024",
        "articles_2025": "Articles 2025",
        "impressions_2024": "Impressions 2024",
        "impressions_2025": "Impressions 2025",
        "articles_growth": "Articles YoY",
        "impressions_growth": "Impressions YoY",
    })[
        ["Topic", "Articles 2024", "Articles 2025", "Articles YoY",
         "Impressions 2024", "Impressions 2025", "Impressions YoY"]
    ]

yoy_sections = []
for _ct in ["Evergreen"]:
    _table = _build_yoy_table(_ct)
    if _table.empty:
        continue
    yoy_sections.append(
        f'<h3 style="margin:18px 0 8px;color:#0f3460;font-size:1rem">{_ct}</h3>'
        + df_to_html(_table, no_toggle=True)
    )
yoy_html = "\n".join(yoy_sections) if yoy_sections else "<p>No YoY topic data available.</p>"
_spont_rows = spont_base[["year", LABEL_COL, "articles", "impressions_M", "pct_impressions"]].copy()
# 2026 projected rows are intentionally excluded from the display table: the model
# produces a flat projection (same mix as 2025), so adding a duplicate row adds no
# information. Absolute 2026 volume targets are shown in the KPI cards above.
spont_disp = _spont_rows.copy()
spont_disp["impressions"] = (spont_disp["impressions_M"] * 1e6).round(0).astype(int).map("{:,}".format)
spont_disp["articles"]    = spont_disp["articles"].map("{:,}".format)
spont_disp = spont_disp.drop(columns=["impressions_M"]).rename(columns={
    "year":            "Year",
    LABEL_COL:         "Content Type",
    "articles":        "Articles",
    "impressions":     "Impressions",
    "pct_impressions": "% Impressions",
})
spont_html = df_to_html(spont_disp)

# ── Main topic cards (clickable, with descriptions) ──────────
# Cards show 2026 forecast values (matching the "2026 Forecast per Topic" table).
import html as _html_mod, json as _json_mod

# Build a topic_key lookup so we can attach descriptions to forecast rows
key_by_label = (
    df_src.dropna(subset=["topic_label", "topic_key"])
    .drop_duplicates("topic_label")
    .set_index("topic_label")["topic_key"]
    .to_dict()
)

topic_examples = {}
example_cols = ["topic_key", "url", TEXT_COL, BODY_COL, "ClientCreativeImpression"]
for _optional_col in [LABEL_COL, "original_content_type", "classification_adjusted"]:
    if _optional_col in df_src.columns:
        example_cols.append(_optional_col)
event_example_terms = (
    EVENT_NAME,
    "halloween",
    "hallowe'en",
    "citrouille",
    "citrouilles",
    "potiron",
    "déguisement",
    "deguisement",
    "costume",
    "horreur",
    "hanté",
    "hantee",
    "hantée",
    "fantôme",
    "fantome",
    "sorcière",
    "sorciere",
    "monstre",
    "momie",
    "bonbons",
)
if set(example_cols).issubset(df_src.columns):
    def _event_relevance(row):
        haystack = " ".join(
            str(row.get(col, "")).lower()
            for col in [TEXT_COL, BODY_COL, "url"]
        )
        return sum(term.lower() in haystack for term in event_example_terms)

    def _presentation_relevance(row):
        cfg = topic_example_config(row.get("topic_key", ""))
        terms = cfg.get("example_terms", ())
        if not terms:
            return 0
        haystack = " ".join(
            str(row.get(col, "")).lower()
            for col in [TEXT_COL, BODY_COL, "url"]
        )
        return sum(term.lower() in haystack for term in terms)

    examples_src = (
        df_src.dropna(subset=["topic_key", "url", TEXT_COL])
        .loc[:, example_cols]
    )
    examples_src["event_relevance"] = examples_src.apply(_event_relevance, axis=1)
    examples_src["presentation_relevance"] = examples_src.apply(_presentation_relevance, axis=1)
    examples_src = examples_src.sort_values(
        ["topic_key", "presentation_relevance", "event_relevance", "ClientCreativeImpression"],
        ascending=[True, False, False, False],
    ).drop_duplicates(["topic_key", "url"])
    for topic_key, g in examples_src.groupby("topic_key"):
        _items = [
            {
                "title": str(row[TEXT_COL]),
                "url": str(row["url"]),
                "impressions": int(row["ClientCreativeImpression"]) if pd.notna(row["ClientCreativeImpression"]) else 0,
            }
            for _, row in g.iterrows()
        ]
        preferred_terms = topic_example_config(topic_key).get("preferred_url_terms", ())
        if preferred_terms:
            def _preferred_rank(item):
                haystack = f"{item.get('title', '')} {item.get('url', '')}".lower()
                for idx, term in enumerate(preferred_terms):
                    if term.lower() in haystack:
                        return idx
                return len(preferred_terms)

            _items = sorted(
                _items,
                key=lambda item: (_preferred_rank(item), -int(item.get("impressions") or 0)),
            )
            diversified, used_urls, used_ranks = [], set(), set()
            for item in _items:
                rank = _preferred_rank(item)
                if rank >= len(preferred_terms) or rank in used_ranks:
                    continue
                diversified.append(item)
                used_urls.add(item.get("url"))
                used_ranks.add(rank)
            for item in _items:
                if item.get("url") in used_urls:
                    continue
                diversified.append(item)
                used_urls.add(item.get("url"))
                if len(diversified) >= 5:
                    break
            _items = diversified
        topic_examples[topic_key] = _items[:5]

# Use exact daily Evergreen topic forecasts for the cards, including small
# nonzero topics. Spontaneous remains visible in summary/charts, but not here.
if (
    not forecast_by_topic_df.empty
    and {
        LABEL_COL,
        "topic_key",
        "topic_label",
        "event_day",
        f"forecast_impressions_{FORECAST_YEAR}",
        f"forecast_articles_{FORECAST_YEAR}",
    }.issubset(forecast_by_topic_df.columns)
):
    fc_topics = (
        forecast_by_topic_df
        .groupby([LABEL_COL, "topic_key", "topic_label"], as_index=False)
        .agg(
            forecast_impressions=(f"forecast_impressions_{FORECAST_YEAR}", "sum"),
            forecast_articles=(f"forecast_articles_{FORECAST_YEAR}", "sum"),
            avg_days_to_event=("event_day", "mean"),
            earliest_day=("event_day", "min"),
            latest_day=("event_day", "max"),
        )
    )
    fc_topics["window"] = fc_topics.apply(
        lambda r: f"{event_offset_date(r['earliest_day'])} to {event_offset_date(r['latest_day'])}",
        axis=1,
    )
else:
    fc_topics = proj.copy()
    fc_topics["topic_key"] = fc_topics["topic_label"].map(key_by_label).fillna("")
    fc_topics["forecast_impressions"] = (fc_topics["impressions_M_2026_proj"] * 1e6).round(0).astype(int)
    fc_topics["forecast_articles"] = fc_topics["articles_2026_proj"].round(0).astype(int)
    if LABEL_COL not in fc_topics.columns:
        fc_topics[LABEL_COL] = "Evergreen"

fc_topics["avg_impressions_per_article"] = np.where(
    fc_topics["forecast_articles"] > 0,
    fc_topics["forecast_impressions"] / fc_topics["forecast_articles"],
    0,
)

fc_topics = fc_topics[
    (fc_topics["topic_label"] != "_outlier")
    & (fc_topics[LABEL_COL] == "Evergreen")
].copy()
card_topic_impression_threshold = (
    fc_topics["forecast_impressions"].sum() * MIN_TOPIC_IMPRESSION_SHARE
)
fc_topics = fc_topics[
    fc_topics["forecast_impressions"] >= card_topic_impression_threshold
]

topic_cards_html_parts = []
for ct in sorted(fc_topics[LABEL_COL].dropna().unique()):
    sub = (
        fc_topics[fc_topics[LABEL_COL] == ct]
        .sort_values("forecast_impressions", ascending=False)
        .head(MAX_TOPICS)
    )
    topic_cards_html_parts.append('<div class="topic-grid">')
    for _, r in sub.iterrows():
        topic_key = r.get("topic_key") or key_by_label.get(r["topic_label"], "")
        payload = {
            "content_type": ct,
            "label": r["topic_label"],
            "description": desc_by_key.get(topic_key, desc_by_label.get(r["topic_label"], "")),
            "forecast_articles":    int(r["forecast_articles"]),
            "forecast_impressions": int(r["forecast_impressions"]),
            "avg_impressions_per_article": round(float(r["avg_impressions_per_article"]), 1),
            "window": window_to_dates(r.get("window", "")),
            "examples": topic_examples.get(topic_key, []),
        }
        data_attr = _html_mod.escape(_json_mod.dumps(payload), quote=True)
        description = desc_by_key.get(topic_key, desc_by_label.get(r["topic_label"], ""))
        description_html = (
            f'  <div class="tc-desc">{_html_mod.escape(str(description))}</div>'
            if description
            else ""
        )
        visible_examples = topic_examples.get(topic_key, [])[:3]
        if visible_examples:
            example_items = []
            for ex in visible_examples:
                href = str(ex.get("url", ""))
                if href and not href.lower().startswith(("http://", "https://")):
                    href = "https://" + href
                title = str(ex.get("title") or ex.get("url") or "Untitled article")
                _words = title.split()
                title_short = " ".join(_words[:7]) + ("…" if len(_words) > 7 else "")
                imps = int(ex.get("impressions") or 0)
                example_items.append(
                    f'<li><a href="{_html_mod.escape(href, quote=True)}" target="_blank" '
                    f'rel="noopener noreferrer" onclick="event.stopPropagation()">'
                    f'{_html_mod.escape(title_short)}</a>'
                    f'</li>'
                )
            examples_html = (
                '<div class="tc-examples"><div class="tc-examples-title">Top articles</div>'
                f'<ol>{"".join(example_items)}</ol></div>'
            )
        else:
            examples_html = ""
        topic_cards_html_parts.append(
            f'<div class="topic-card" data-topic="{data_attr}" onclick="openTopic(this)">'
            f'  <div class="tc-label">{_html_mod.escape(str(r["topic_label"]))}</div>'
            f'{description_html}'
            f'  <div class="tc-kpi-row">'
            f'    <div class="tc-kpi"><div class="tc-kpi-val">{int(r["forecast_articles"]):,}</div><div class="tc-kpi-lbl">Articles</div></div>'
            f'    <div class="tc-kpi"><div class="tc-kpi-val">{int(r["forecast_impressions"]):,}</div><div class="tc-kpi-lbl">Total imps</div></div>'
            f'    <div class="tc-kpi"><div class="tc-kpi-val">{float(r["avg_impressions_per_article"]):,.0f}</div><div class="tc-kpi-lbl">Avg/article</div></div>'
            f'  </div>'
            f'{examples_html}'
            f'</div>'
        )
    topic_cards_html_parts.append('</div>')
topic_cards_html = "\n".join(topic_cards_html_parts)

# ── Chart takeaway captions ─────────────────────────────────
_top_proj = proj.iloc[0] if not proj.empty else None
_top_topic_name  = str(_top_proj["topic_label"]) if _top_proj is not None else "The top topic"
_top_topic_imps  = int(_top_proj["impressions_M_2026_proj"] * 1e6) if _top_proj is not None else 0
_ev_total_imps      = int(total_ev_proj_M * 1e6)
_all_total_imps     = int(total_proj_M * 1e6)
_top_share_pct      = round(_top_topic_imps / _all_total_imps * 100) if _all_total_imps > 0 else 0
_top_share_ev_pct   = round(_top_topic_imps / _ev_total_imps * 100) if _ev_total_imps > 0 else 0
_peak_days_before   = (BF_2026 - peak_date).days

takeaway_volume = (
    f"Attention builds before the {EVENT_NAME} period and peaks on <b>{peak_label}</b> "
    f"({peak_imps:,} forecast impressions). "
    f"The shaded band marks the event period: <b>{event_range_label()}</b>. "
    f"The peak is led by the {peak_driver_type.lower() if peak_driver_type else 'highest-reach'} layer, "
    f"so keep budget flexible around this date."
)
takeaway_category = (
    "Entertainment and celebrity categories carry the bulk of impressions, but the reach "
    "spreads across film, music, lifestyle, and culture \u2014 giving brands "
    "a credible entry point into the cultural moment."
)
takeaway_topic = (
    f"<b>{_top_topic_name}</b> is the largest evergreen topic, with {_top_topic_imps:,}\u202fimps "
    f"({_top_share_ev_pct}% of evergreen reach and {_top_share_pct}% of total forecast reach). "
    "Use it as one anchor layer, then extend coverage with the remaining evergreen topics."
)
takeaway_timing = (
    "Topics don\u2019t all peak at the same time. Stagger activation to ride each topic\u2019s "
    "natural content curve \u2014 activating too early wastes budget, too late misses the attention window."
)
takeaway_mix = (
    f"The split shown here is based on <b>impressions</b>: Evergreen represents "
    f"<b>{100 - spont_avg_pct:.0f}%</b> of forecast reach and Spontaneous represents "
    f"<b>{spont_avg_pct:.0f}%</b>. Article counts are less useful for budget planning because "
    "a small number of high-reach articles can carry most of the audience."
)

topic_timing_html_parts = []
for ct_label in ["Evergreen"]:
    fig = fig_topic_timing_by_type.get(ct_label)
    if fig is None:
        continue
    topic_timing_html_parts.append(
        '<div class="section">'
        f'<h2>&#128202; Daily Topic Timing \u2014 {ct_label}</h2>'
        '<div class="inner">'
        f'<p class="chart-takeaway">{takeaway_timing}</p>'
        f'{fig_to_html(fig)}'
        '<p class="chart-hint">&#8599; Hover over any point to see daily forecast values</p>'
        '</div></div>'
    )
topic_timing_html = "\n".join(topic_timing_html_parts)

_second_topic      = proj.iloc[1] if len(proj) > 1 else None
_second_topic_name = str(_second_topic["topic_label"]) if _second_topic is not None else ""
_peak_offset_days = (peak_date - BF_2026).days
_peak_timing_phrase = (
    f"{abs(_peak_offset_days)} days after {EVENT_NAME} begins"
    if _peak_offset_days >= 0
    else f"{abs(_peak_offset_days)} days before {EVENT_NAME} begins"
)

tldr_html = (
    '<div class="tldr">'
    '<div class="tldr-label">The short version</div>'
    f'<div class="tldr-headline">Everything you need to know about {EVENT_NAME} {FORECAST_YEAR} in four points.</div>'
    '<div class="tldr-items">'
    f'<div class="tldr-item"><div class="tldr-num">1</div><div class="tldr-text">'
    f'<b>Reach mix: {100 - spont_avg_pct:.0f}% evergreen / {spont_avg_pct:.0f}% spontaneous</b> by forecast impressions. '
    f'That equals {_ev_total_imps:,} evergreen impressions and {_all_total_imps - _ev_total_imps:,} spontaneous impressions. '
    f'Use this as the budget split; article counts are shown as volume context, not the planning mix.'
    '</div></div>'
    f'<div class="tldr-item"><div class="tldr-num">2</div><div class="tldr-text">'
    f'<b>{_top_topic_name}</b> is the largest evergreen topic, with <b>{_top_topic_imps:,}\u202fimps</b> '
    f'(<b>{_top_share_ev_pct}% of evergreen reach</b>, {_top_share_pct}% of total forecast reach). '
    f'Use it as an anchor layer, while {_second_topic_name} and the remaining topics add audience breadth across film, music, and celebrity coverage.'
    '</div></div>'
    f'<div class="tldr-item"><div class="tldr-num">3</div><div class="tldr-text">'
    f'Content volume peaks <b>{_peak_timing_phrase}</b>, on <b>{peak_label}</b> '
    f'({peak_imps:,}\u202fimps in a single day), driven mainly by the {peak_driver_type.lower() if peak_driver_type else "highest-reach"} layer. '
    f'This is historical editorial demand \u2014 audience attention on relevant articles, not campaign delivery. '
    f'The event period runs <b>{event_period_window}</b>; topic activation windows can start before it and extend after it. '
    f'Use the activation chart and table to stagger launches across that full attention window.'
    '</div></div>'
    '<div class="tldr-item"><div class="tldr-num">4</div><div class="tldr-text">'
    '<b>Do not launch all topics at once.</b> '
    f'Each topic has a different natural content rhythm \u2014 some peak during the {EVENT_NAME} period, others build around adjacent coverage moments. '
    'The activation schedule in this report assigns each topic its optimal start date. '
    'Staggering the launch maximises reach across the full window without increasing budget.'
    '</div></div>'
    '</div>'
    '</div>'
)

strategy_html = strategy_html.replace("%%TLDR%%", tldr_html)

# ── CSS / JS ─────────────────────────────────────────────────
HTML_CSS = """
  body{font-family:'Poppins',sans-serif;background:#EBE6E4;color:#000000;margin:0;padding:0}
  .hero{background:#EBE6E4;color:#000000;padding:48px 40px 0}
  .hero-inner{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;border-bottom:0.5px solid #D4D0CE;padding-bottom:28px}
  .hero-marker{display:block;font-size:.62rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#948A8A;margin-bottom:10px}
  .hero h1{margin:0;font-size:3.4rem;line-height:1.05;color:#FF6B7C}.hero-title-line{display:block}.hero p{margin:0;opacity:1;font-size:.95rem}
  .hero-meta{display:flex;flex-direction:column;align-items:flex-end;gap:6px;text-align:right}
  .hero-meta .hero-pill{font-size:.62rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;background:#FFFFFF;color:#948A8A;padding:5px 10px;border-radius:2px}
  .pill{display:inline-block;background:rgba(255,255,255,.12);border-radius:8px;padding:10px 16px;font-size:.82rem;margin-top:16px;line-height:1.6}
  .kpi-row{display:flex;gap:16px;padding:24px 40px 16px;flex-wrap:wrap}
  .content-defs{display:flex;gap:12px;padding:8px 40px 24px;flex-wrap:wrap}
  .content-def{display:flex;align-items:flex-start;gap:10px;flex:1;min-width:260px;background:white;border-radius:8px;padding:12px 16px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
  .content-def-dot{min-width:10px;height:10px;border-radius:50%;margin-top:4px}
  .content-def-term{font-size:.82rem;font-weight:700;color:#2F2E2E;white-space:nowrap}
  .content-def-desc{font-size:.78rem;color:#666;line-height:1.5}
  .kpi-group{display:flex;gap:16px;padding:0 40px 16px;flex-wrap:wrap;align-items:flex-start}
  .kpi-group-label{width:100%;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#999;padding:0 0 4px}
  .kpi{background:white;border-radius:10px;padding:20px 28px;flex:1;min-width:160px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
  .kpi .val{font-size:2rem;font-weight:700;color:#FF6B7C}.kpi .lbl{font-size:.82rem;color:#666;margin-top:4px}
  .section{background:white;margin:0 40px 24px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.08);overflow:hidden}
  .section h2{margin:0;padding:20px 24px 0;font-size:1.1rem;color:#FF6B7C}
  .section .inner{padding:16px 24px 24px}
  .data-table{width:100%;border-collapse:collapse;font-size:.82rem}
  .data-table th{background:#FF6B7C;color:white;padding:8px 12px;text-align:left}
  .data-table td{padding:7px 12px;border-bottom:1px solid #D4D0CE}
  .data-table tr:last-child td{border-bottom:none}
  .data-table tr:hover td{background:#EBE6E4}
  footer{text-align:center;padding:24px;font-size:.78rem;color:#999}
  .tbl-wrap.collapsed tbody tr{display:none}.tbl-wrap.collapsed tbody tr:nth-child(-n+5){display:table-row}
  .tbl-toggle{display:inline-flex;align-items:center;gap:6px;margin-top:10px;background:#FF6B7C;color:white;border:none;border-radius:6px;padding:6px 14px;font-size:.8rem;cursor:pointer}
  .tbl-toggle:hover{background:#E05565}
  .more-cats{margin-top:12px}
  .more-cats summary{display:inline-flex;align-items:center;gap:8px;background:#FF6B7C;color:white;border-radius:6px;padding:7px 14px;font-size:.82rem;cursor:pointer;user-select:none}
  .more-cats summary::-webkit-details-marker{display:none}
  .more-cats .plus{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:50%;background:rgba(255,255,255,.18);font-weight:700}
  .more-cats[open] .plus{transform:rotate(45deg)}
  .more-cats table{margin-top:12px}
  /* Topic browser */
  .topic-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;margin-top:12px}
  .topic-card{background:#FFFFFF;border:1px solid #D4D0CE;border-radius:8px;padding:14px 16px;cursor:pointer;transition:all .15s}
  .topic-card:hover{transform:translateY(-2px);box-shadow:0 4px 14px rgba(84,118,255,.18);border-color:#5476FF}
  .topic-card .tc-type{font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;color:#888;font-weight:700}
  .topic-card .tc-label{font-size:.95rem;font-weight:600;color:#FF6B7C;margin-top:4px;line-height:1.25}
  .topic-card .tc-desc{font-size:.78rem;color:#333;line-height:1.45;margin-top:8px}
  .topic-card .tc-examples{margin-top:10px;padding-top:10px;border-top:1px solid #EBE6E4}
  .topic-card .tc-examples-title{font-size:.68rem;text-transform:uppercase;color:#948A8A;font-weight:700;letter-spacing:.04em;margin-bottom:5px}
  .topic-card .tc-examples ol{margin:0;padding-left:18px;display:grid;gap:5px}
  .topic-card .tc-examples li{font-size:.72rem;color:#333;line-height:1.35}
  .topic-card .tc-examples a{color:#2F2E2E;text-decoration:none;font-weight:600}
  .topic-card .tc-examples a:hover{color:#FF6B7C;text-decoration:underline}
  .topic-card .tc-examples span{display:block;color:#777;font-size:.66rem;margin-top:1px}
  .topic-card .tc-kpi-row{display:flex;gap:6px;margin-top:12px;margin-bottom:2px;padding-bottom:10px;border-bottom:1px solid #EBE6E4}
  .topic-card .tc-kpi{flex:1;text-align:center;background:#F7F4F2;border-radius:6px;padding:7px 4px}
  .topic-card .tc-kpi-val{font-size:.92rem;font-weight:700;color:#FF6B7C}
  .topic-card .tc-kpi-lbl{font-size:.6rem;color:#888;text-transform:uppercase;letter-spacing:.04em;margin-top:2px}
  .forecast-popover{position:fixed;z-index:120;width:min(520px,calc(100vw - 32px));background:#FFFFFF;border:2px solid #5476FF;border-radius:7px;box-shadow:0 12px 34px rgba(47,46,46,.22);padding:20px 24px 18px;text-align:center}
  .forecast-popover .fp-article{font-size:1rem;line-height:1.35;color:#000;font-weight:600}
  .forecast-popover .fp-link{display:inline-block;margin-top:10px;color:#FF6B7C;text-decoration:none;font-size:1rem;font-weight:600}
  .forecast-popover .fp-link:hover{text-decoration:underline}
  .forecast-popover .fp-close{position:absolute;top:6px;right:8px;border:0;background:transparent;color:#948A8A;font-size:1.1rem;line-height:1;cursor:pointer}
  .strategy-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-bottom:16px}
  .strategy-card{background:#EBE6E4;border:1px solid #D4D0CE;border-radius:8px;padding:14px 16px}
  .strategy-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.06em;color:#948A8A;font-weight:700}
  .strategy-value{font-size:1.15rem;color:#FF6B7C;font-weight:700;margin-top:5px;line-height:1.25}
  .strategy-value.strategy-small{font-size:.88rem;color:#2F2E2E;font-weight:600}
  .strategy-note{font-size:.75rem;color:#666;margin-top:3px}
  .strategy-intro{margin:0 0 20px;padding:18px 22px;background:linear-gradient(135deg,#FF6B7C18 0%,#5476FF10 100%);border:1.5px solid #FF6B7C55;border-radius:10px;font-size:1rem;color:#2F2E2E;line-height:1.65}
  .strategy-intro .si-headline{font-size:1.15rem;font-weight:700;color:#FF6B7C;margin-bottom:6px}
  .strategy-intro .si-body{font-size:.92rem;color:#2F2E2E}
  .strategy-graph{margin:8px 0 18px}
  .strategy-subhead{margin:18px 0 10px;color:#FF6B7C;font-size:.95rem}
  .tldr{margin:0;padding:0}
  .tldr-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;color:#999;font-weight:700;margin-bottom:10px}
  .tldr-headline{font-size:1.1rem;font-weight:700;color:#FF6B7C;margin-bottom:18px;line-height:1.3}
  .tldr-items{display:grid;gap:12px}
  .tldr-item{display:flex;gap:14px;align-items:flex-start}
  .tldr-num{min-width:28px;height:28px;border-radius:50%;background:#FF6B7C;color:white;font-size:.78rem;font-weight:700;display:flex;align-items:center;justify-content:center;margin-top:1px}
  .tldr-text{font-size:.88rem;color:#333;line-height:1.6}
  .tldr-text b{color:#000}
  .chart-takeaway{margin:0 0 14px;padding:12px 16px;background:#F7F4F2;border-left:3px solid #FF6B7C;border-radius:0 6px 6px 0;font-size:.88rem;color:#333;line-height:1.6}
  .chart-hint{margin:4px 0 0;font-size:.76rem;color:#aaa;font-style:italic;text-align:right}
  .modal-bg{display:none;position:fixed;inset:0;background:rgba(15,30,55,.55);z-index:99;align-items:center;justify-content:center;padding:20px}
  .modal-bg.open{display:flex}
  .modal-box{background:white;max-width:560px;width:100%;border-radius:12px;padding:28px 32px;box-shadow:0 12px 40px rgba(0,0,0,.3);max-height:80vh;overflow-y:auto}
  .modal-box h3{margin:0 0 4px;color:#FF6B7C}
  .modal-box .mb-type{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:#888;font-weight:700}
  .modal-box .mb-desc{font-size:.95rem;line-height:1.55;color:#333;margin-top:14px}
  .modal-box .mb-stats{display:flex;gap:14px;margin-top:18px;flex-wrap:wrap}
  .modal-box .mb-stat{background:#EBE6E4;padding:8px 14px;border-radius:6px;font-size:.82rem}
  .modal-box .mb-stat b{color:#FF6B7C}
  .modal-box .mb-examples{margin-top:20px}
  .modal-box .mb-examples h4{margin:0 0 8px;color:#FF6B7C;font-size:.9rem}
  .modal-box .mb-examples ul{list-style:none;margin:0;padding:0;display:grid;gap:8px}
  .modal-box .mb-examples li{background:#FFFFFF;border:1px solid #D4D0CE;border-radius:6px;padding:9px 11px}
  .modal-box .mb-examples a{color:#FF6B7C;text-decoration:none;font-weight:600;font-size:.86rem;line-height:1.35}
  .modal-box .mb-examples a:hover{text-decoration:underline}
  .modal-box .mb-examples .ex-meta{display:block;color:#777;font-size:.72rem;margin-top:3px}
  .modal-box .mb-close{margin-top:20px;background:#FF6B7C;color:white;border:none;border-radius:6px;padding:8px 18px;cursor:pointer;font-size:.85rem}
  @media(max-width:900px){.hero{padding:28px 18px 0}.hero-inner{display:block}.hero-meta{align-items:flex-start;text-align:left;margin-top:18px}.kpi-row,.content-defs,.kpi-group{padding-left:18px;padding-right:18px}.section{margin-left:18px;margin-right:18px}}

"""
HTML_JS = """
function toggleTable(btn){
  const wrap=btn.previousElementSibling;
  const collapsed=wrap.classList.toggle('collapsed');
  const total=wrap.querySelectorAll('tbody tr').length;
  btn.querySelector('span').textContent=collapsed?'Show all '+total+' rows':'Show less';
}
function openTopic(card){
  const data=JSON.parse(card.dataset.topic);
  document.getElementById('mb-type').textContent=data.content_type;
  document.getElementById('mb-label').textContent=data.label;
  document.getElementById('mb-desc').textContent=data.description||'No description available.';
  const stats=document.getElementById('mb-stats');
  stats.innerHTML='';
  if(data.forecast_articles!=null) stats.insertAdjacentHTML('beforeend','<div class="mb-stat"><b>'+data.forecast_articles.toLocaleString()+'</b> articles (FORECAST_YEAR_PLACEHOLDER forecast)</div>');
  if(data.forecast_impressions!=null) stats.insertAdjacentHTML('beforeend','<div class="mb-stat"><b>'+data.forecast_impressions.toLocaleString()+'</b> total impressions (FORECAST_YEAR_PLACEHOLDER forecast)</div>');
  if(data.avg_impressions_per_article!=null) stats.insertAdjacentHTML('beforeend','<div class="mb-stat"><b>'+Math.round(data.avg_impressions_per_article).toLocaleString()+'</b> avg impressions/article</div>');
  if(data.window) stats.insertAdjacentHTML('beforeend','<div class="mb-stat">Window: <b>'+data.window+'</b></div>');
  const examples=document.getElementById('mb-examples');
  examples.innerHTML='';
  if(data.examples&&data.examples.length){
    const items=data.examples.map(ex=>{
      const href=/^https?:\\/\\//i.test(ex.url)?ex.url:'https://'+ex.url;
      const title=(ex.title||ex.url);
      const imps=ex.impressions!=null?Number(ex.impressions).toLocaleString()+' imps':'';
      return '<li><a href="'+href+'" target="_blank" rel="noopener noreferrer">'+title+'</a><span class="ex-meta">'+ex.url+(imps?' · '+imps:'')+'</span></li>';
    }).join('');
    examples.innerHTML='<h4>Example URLs</h4><ul>'+items+'</ul>';
  }
  document.getElementById('topic-modal').classList.add('open');
}
function closeTopic(){document.getElementById('topic-modal').classList.remove('open');}
function closeForecastPopover(){
  const old=document.getElementById('forecast-popover');
  if(old)old.remove();
}
function openForecastPoint(raw, e){
  const data=typeof raw==='string'?JSON.parse(raw):raw;
  if(!data)return;
  const article=(data.examples&&data.examples.length)?data.examples[0]:null;
  if(!article)return;
  closeForecastPopover();
  const href=/^https?:\/\//i.test(article.url)?article.url:'https://'+article.url;
  const pop=document.createElement('div');
  pop.id='forecast-popover';
  pop.className='forecast-popover';
  const close=document.createElement('button');
  close.className='fp-close';
  close.type='button';
  close.innerHTML='&times;';
  close.onclick=closeForecastPopover;
  const articleTitle=document.createElement('div');
  articleTitle.className='fp-article';
  articleTitle.textContent=article.title||article.url;
  const link=document.createElement('a');
  link.className='fp-link';
  link.href=href;
  link.target='_blank';
  link.rel='noopener noreferrer';
  link.textContent='Read more';
  pop.append(close,articleTitle,link);
  document.body.appendChild(pop);
  const cx=e&&e.clientX?e.clientX:window.innerWidth/2;
  const cy=e&&e.clientY?e.clientY:window.innerHeight/2;
  const rect=pop.getBoundingClientRect();
  let left=cx-rect.width/2;
  let top=cy-rect.height-22;
  if(top<12)top=cy+22;
  left=Math.max(12,Math.min(left,window.innerWidth-rect.width-12));
  top=Math.max(12,Math.min(top,window.innerHeight-rect.height-12));
  pop.style.left=left+'px';
  pop.style.top=top+'px';
}
document.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('.data-table').forEach(tbl=>{
    const rows=tbl.querySelectorAll('tbody tr').length;
    if(tbl.classList.contains('no-toggle'))return;
    if(rows<=5)return;
    const wrap=document.createElement('div');wrap.className='tbl-wrap collapsed';
    tbl.parentNode.insertBefore(wrap,tbl);wrap.appendChild(tbl);
    const btn=document.createElement('button');btn.className='tbl-toggle';
    btn.innerHTML='<span>Show all '+rows+' rows</span>';
    btn.setAttribute('onclick','toggleTable(this)');
    wrap.parentNode.insertBefore(btn,wrap.nextSibling);
  });
  document.getElementById('topic-modal').addEventListener('click',e=>{
    if(e.target.id==='topic-modal')closeTopic();
  });
  document.querySelectorAll('.plotly-graph-div').forEach(graph=>{
    graph.on('plotly_hover',eventData=>{
      const point=eventData&&eventData.points&&eventData.points[0];
      if(point&&point.customdata&&point.customdata.examples){
        openForecastPoint(point.customdata,eventData.event);
      }
    });
  });
  document.addEventListener('click',e=>{
    const pop=document.getElementById('forecast-popover');
    if(!pop)return;
    if(pop.contains(e.target))return;
    if(e.target.closest&&e.target.closest('.plotly-graph-div'))return;
    closeForecastPopover();
  });
});
"""
HTML_JS = HTML_JS.replace("FORECAST_YEAR_PLACEHOLDER", str(FORECAST_YEAR))

# ── Assemble HTML ─────────────────────────────────────────────
_fc_topic_section_html = (
    '<div class="section"><h2>&#128202; Total Forecast by Topic \u2014 2026</h2>'
    '<div class="inner">'
    f'<p class="chart-takeaway">{takeaway_topic}</p>'
    f'{fig_to_html(fig_fc_topic)}'
    '<p class="chart-hint">&#8599; Hover over any bar to see example articles for that topic</p>'
    '</div></div>'
) if fig_fc_topic is not None else ""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<link rel="icon" href="https://www.seedtag.com/static/images/seedtag-icon-orange-57b3f29dfea458f7b74ac292e9a8f49a.svg" type="image/svg+xml">
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{EVENT_LABEL} 2026 Forecast Report</title>
<link href="https://fonts.googleapis.com/css?family=Poppins" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{HTML_CSS}</style>
<script>{HTML_JS}</script>
</head>
<body>
<div class="hero">
  <div class="hero-inner">
    <div>
      <span class="hero-marker">Seedtag · Forecast Report</span>
      <h1><span class="hero-title-line">{EVENT_LABEL} 2026</span><span class="hero-title-line">Forecast Report</span></h1>
    </div>
    <div class="hero-meta">
      <span class="hero-marker">Planning Intelligence</span>
      <span class="hero-pill">2026 forecast</span>
      <span class="hero-pill">Evergreen + spontaneous</span>
    </div>
  </div>
</div>
<div class="kpi-row">
  <div class="kpi"><div class="val">{total_proj_M * 1e6:,.0f}</div><div class="lbl">Total Projected Impressions</div></div>
  <div class="kpi"><div class="val">{total_all_arts_2026:,}</div><div class="lbl">Total Projected Articles</div></div>
  <div class="kpi"><div class="val">{total_ev_proj_M * 1e6:,.0f}</div><div class="lbl">Evergreen Impressions</div></div>
  <div class="kpi"><div class="val">{total_ev_arts_2026:,}</div><div class="lbl">Evergreen Articles</div></div>
</div>
<div class="content-defs">
  <div class="content-def">
    <span class="content-def-dot" style="background:#5476FF"></span>
    <span class="content-def-term">Evergreen</span>
    <span class="content-def-desc">Plannable contextual inventory assigned to stable {EVENT_NAME} topic clusters. For this forecast, that includes recurring entertainment themes plus high-reach event-adjacent stories that can be packaged against predictable audience interest.</span>
  </div>
  <div class="content-def">
    <span class="content-def-dot" style="background:#FF6B7C"></span>
    <span class="content-def-term">Spontaneous</span>
    <span class="content-def-desc">Flexible reserve inventory for fast-moving stories that are not assigned to stable planning clusters. This layer captures the remaining one-off news, live moments, and context that is harder to forecast in advance.</span>
  </div>
</div>

<div class="section"><h2>Key Takeaways</h2><div class="inner">{strategy_html}</div></div>
<div class="section"><h2>&#128200; Forecast Volume \u2014 2026</h2><div class="inner"><p class="chart-takeaway">{takeaway_volume}</p>{fig_to_html(fig_vol_2026)}</div></div>
<div class="section"><h2>&#128202; Evergreen Forecast by Category \u2014 2026</h2><div class="inner"><p class="chart-takeaway">{takeaway_category}</p>{(fig_to_html(fig_cat) + cat_extra_html) if fig_cat is not None else '<p>No tier-1 evergreen category data available.</p>'}</div></div>
{_fc_topic_section_html}
<div class="section"><h2>&#128269; Main Topics</h2><div class="inner"><p style="margin:0 0 8px;color:#666;font-size:.88rem">Topic descriptions and top articles are shown below; click a card for full stats.</p>{topic_cards_html}</div></div>
<div class="section"><h2>&#128193; 2026 Forecast per Topic</h2><div class="inner">{forecast_html}</div></div>
<div class="section"><h2>&#9889; Evergreen vs Spontaneous Mix</h2><div class="inner"><p class="chart-takeaway">{takeaway_mix}</p>{spont_html}<p style="margin:10px 0 0;font-size:.78rem;color:#999">The adjusted 2025 reach mix is used to allocate the 2026 forecast. Absolute 2026 volume targets are shown in the KPIs above.</p></div></div>
{topic_timing_html}

<div id="topic-modal" class="modal-bg">
  <div class="modal-box">
    <div id="mb-type" class="mb-type"></div>
    <h3 id="mb-label"></h3>
    <div id="mb-desc" class="mb-desc"></div>
    <div id="mb-stats" class="mb-stats"></div>
    <div id="mb-examples" class="mb-examples"></div>
    <button class="mb-close" onclick="closeTopic()">Close</button>
  </div>
</div>

<footer>{EVENT_LABEL} Forecast &nbsp;·&nbsp; {TODAY}</footer>
</body>
</html>"""

out_html = REPORT_PATH
Path(out_html).parent.mkdir(parents=True, exist_ok=True)
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html)

reco_df.to_csv(RECO_CSV_PATH, index=False)

print(f"HTML report : {out_html}")
print(f"Reco CSV    : {RECO_CSV_PATH}")
print(f"\n{'='*55}")
print(f"  {EVENT_NAME.upper()} {FORECAST_YEAR} — FORECAST SUMMARY")
print(f"{'='*55}")
print(f"  Evergreen impressions : {total_ev_proj_M * 1e6:,.0f}")
print(f"  Spontaneous budget    : ~{total_proj_M * spont_avg_pct / 100 * 1e6:,.0f}  ({spont_avg_pct:.1f}%)")
print(f"  Total projected       : ~{total_proj_M * 1e6:,.0f}")
print(f"\n  Top 5 topics:")
for _, r in proj.head(5).iterrows():
    print(f"    • {r['topic_label']:<45} {r['impressions_M_2026_proj'] * 1e6:,.0f} imps  |  {window_to_dates(r['window'])}")
