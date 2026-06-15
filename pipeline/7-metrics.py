from pathlib import Path
import html
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import (
    ARTICLE_LANGUAGE,
    BASE_YEAR,
    COMPARE_YEAR,
    EVENT_ANCHORS,
    EVENT_KEY,
    EVENT_NAME,
    FORECAST_WINDOW_END,
    FORECAST_WINDOW_START,
    FORECAST_YEAR,
)


TEXT_COL = "page_content_title_formalized"
URL_COL = "url"
LABEL_COL = "content_type"
IMPS_COL = "ClientCreativeImpression"

OUT_HTML = f"reports/{EVENT_KEY}_{FORECAST_YEAR}_metrics_dashboard.html"
OUT_CSV = f"data/{EVENT_KEY}_{FORECAST_YEAR}_model_metrics.csv"


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()


def fig_to_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False)


def df_to_html(df: pd.DataFrame, cols=None, max_rows=None) -> str:
    if df is None or df.empty:
        return '<p class="empty">No data available.</p>'
    sub = df.copy()
    if cols:
        sub = sub[cols]
    if max_rows:
        sub = sub.head(max_rows)
    return sub.to_html(index=False, classes="data-table", border=0, escape=False)


def metric_rows(section, rows):
    return [{"section": section, **row} for row in rows]


def safe_pct(numer, denom):
    return 0 if denom in (0, None) or pd.isna(denom) else numer / denom * 100


def regression_metrics(actual, predicted):
    actual = pd.Series(actual, dtype="float64")
    predicted = pd.Series(predicted, dtype="float64")
    mask = actual.notna() & predicted.notna()
    actual = actual[mask]
    predicted = predicted[mask]
    err = predicted - actual
    abs_err = err.abs()
    denom = actual.abs().replace(0, np.nan)
    return {
        "rows": int(len(actual)),
        "actual_total": float(actual.sum()),
        "predicted_total": float(predicted.sum()),
        "bias": float(err.mean()) if len(err) else np.nan,
        "mae": float(abs_err.mean()) if len(abs_err) else np.nan,
        "rmse": float(np.sqrt((err**2).mean())) if len(err) else np.nan,
        "mape_pct": float((abs_err / denom).mean() * 100) if denom.notna().any() else np.nan,
        "wape_pct": float(abs_err.sum() / actual.abs().sum() * 100) if actual.abs().sum() else np.nan,
        "forecast_to_actual": float(predicted.sum() / actual.sum()) if actual.sum() else np.nan,
    }


def fmt_num(v, digits=0):
    if pd.isna(v):
        return ""
    if isinstance(v, str):
        return v
    return f"{v:,.{digits}f}" if digits else f"{v:,.0f}"


def fmt_pct(v):
    return "" if pd.isna(v) else f"{v:.1f}%"


metrics = []

classified_path = Path("data/classified.csv")
topics_path = Path("data/with_topics.csv")
labels_path = Path("data/2025_claude.csv")
forecast_type_path = Path(f"data/{FORECAST_YEAR}_daily_impression_forecast_by_content_type.csv")
forecast_topic_path = Path(f"data/{FORECAST_YEAR}_daily_impression_forecast_by_topic.csv")

source_path = topics_path if topics_path.exists() else classified_path
df = clean_df(pd.read_csv(source_path))
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year
df["event_date"] = df["year"].map(EVENT_ANCHORS)
df["days_to_event"] = (df["date"] - df["event_date"]).dt.days
df["forecast_date"] = EVENT_ANCHORS[FORECAST_YEAR] + pd.to_timedelta(df["days_to_event"], unit="D")
df = df[df["forecast_date"].between(FORECAST_WINDOW_START, FORECAST_WINDOW_END)].copy()

metrics += metric_rows(
    "data_health",
    [
        {"metric": "source_rows", "value": len(df), "detail": str(source_path)},
        {"metric": "article_language", "value": ARTICLE_LANGUAGE, "detail": "from config.py"},
        {"metric": "missing_dates", "value": int(df["date"].isna().sum()), "detail": ""},
        {"metric": "missing_impressions", "value": int(df[IMPS_COL].isna().sum()), "detail": ""},
        {"metric": "duplicate_urls", "value": int(df[URL_COL].duplicated().sum()), "detail": ""},
        {"metric": "zero_impression_rows", "value": int((df[IMPS_COL].fillna(0) == 0).sum()), "detail": ""},
    ],
)

# ── Classification metrics ───────────────────────────────────
classification_summary = pd.DataFrame()
confusion_df = pd.DataFrame()
classification_note = ""
fig_confusion = None

if labels_path.exists() and classified_path.exists():
    labels_df = clean_df(pd.read_csv(labels_path))
    pred_df = clean_df(pd.read_csv(classified_path))
    if {URL_COL, TEXT_COL, LABEL_COL}.issubset(labels_df.columns) and {URL_COL, LABEL_COL}.issubset(pred_df.columns):
        # Reproduce the same deterministic train/test split used in 1-classify-content.py
        split_df = labels_df[[URL_COL, TEXT_COL, LABEL_COL]].dropna(subset=[TEXT_COL, LABEL_COL]).copy()
        split_df["label_int"] = split_df[LABEL_COL].map({"Evergreen": 0, "Spontaneous": 1})
        split_df = split_df.dropna(subset=["label_int"])
        _, test_df = train_test_split(
            split_df,
            test_size=0.2,
            random_state=42,
            stratify=split_df["label_int"],
        )
        # Deduplicate: one ground-truth label per URL
        test_df = test_df.drop_duplicates(subset=[URL_COL])
        # Look up model predictions for test-split URLs in classified.csv
        eval_df = test_df[[URL_COL, LABEL_COL]].rename(columns={LABEL_COL: "actual"}).merge(
            pred_df[[URL_COL, LABEL_COL]].drop_duplicates(subset=[URL_COL]).rename(columns={LABEL_COL: "predicted"}),
            on=URL_COL,
            how="inner",
        )
        eval_df = eval_df.dropna(subset=["actual", "predicted"])
        classes = ["Evergreen", "Spontaneous"]
        if not eval_df.empty:
            report = classification_report(
                eval_df["actual"],
                eval_df["predicted"],
                labels=classes,
                output_dict=True,
                zero_division=0,
            )
            classification_summary = (
                pd.DataFrame(report)
                .T.reset_index()
                .rename(columns={"index": "label"})
            )
            for col in ["precision", "recall", "f1-score", "support"]:
                if col in classification_summary.columns:
                    classification_summary[col] = classification_summary[col].round(3)
            cm = confusion_matrix(eval_df["actual"], eval_df["predicted"], labels=classes)
            confusion_df = pd.DataFrame(cm, index=classes, columns=classes)
            fig_confusion = go.Figure(
                go.Heatmap(
                    z=cm,
                    x=[f"Pred {c}" for c in classes],
                    y=[f"Actual {c}" for c in classes],
                    colorscale="Blues",
                    text=cm,
                    texttemplate="%{text}",
                    hovertemplate="%{y}<br>%{x}<br>Rows: %{z}<extra></extra>",
                )
            )
            fig_confusion.update_layout(
                title="Evergreen vs Spontaneous Confusion Matrix",
                height=380,
                margin=dict(l=120, r=40, t=60, b=60),
            )
            accuracy = report.get("accuracy", np.nan)
            metrics += metric_rows(
                "classification",
                [
                    {"metric": "evaluation_rows", "value": len(eval_df), "detail": "held-out test split (20% of 2025_claude.csv, same split as training)"},
                    {"metric": "accuracy", "value": round(accuracy, 4), "detail": ""},
                    {"metric": "macro_f1", "value": round(report["macro avg"]["f1-score"], 4), "detail": ""},
                    {"metric": "weighted_f1", "value": round(report["weighted avg"]["f1-score"], 4), "detail": ""},
                ],
            )
        else:
            classification_note = "No overlapping labeled/predicted rows were found."
else:
    classification_note = "Classification metrics need data/2025_claude.csv and data/classified.csv."

# ── Topic diagnostics ────────────────────────────────────────
topic_summary = pd.DataFrame()
topic_by_type = pd.DataFrame()
fig_topic_sizes = None
fig_outliers = None

if {"topic_key", "topic_label", "topic_id"}.issubset(df.columns):
    topic_df = df.dropna(subset=["topic_key", "topic_label"]).copy()
    topic_df["is_outlier"] = topic_df["topic_label"].eq("_outlier") | topic_df["topic_key"].astype(str).str.endswith("_-1")
    topic_by_type = (
        topic_df.groupby([LABEL_COL, "is_outlier"], as_index=False)
        .agg(articles=(URL_COL, "count"), impressions=(IMPS_COL, "sum"))
    )
    totals_by_type = topic_by_type.groupby(LABEL_COL)[["articles", "impressions"]].transform("sum")
    topic_by_type["article_share_pct"] = (topic_by_type["articles"] / totals_by_type["articles"] * 100).round(1)
    topic_by_type["impression_share_pct"] = (topic_by_type["impressions"] / totals_by_type["impressions"] * 100).round(1)

    topic_summary = (
        topic_df[~topic_df["is_outlier"]]
        .groupby([LABEL_COL, "topic_key", "topic_label"], as_index=False)
        .agg(
            articles=(URL_COL, "count"),
            impressions=(IMPS_COL, "sum"),
            avg_days_to_event=("days_to_event", "mean"),
        )
        .sort_values("impressions", ascending=False)
    )
    topic_summary["avg_days_to_event"] = topic_summary["avg_days_to_event"].round(1)

    fig_topic_sizes = go.Figure()
    for ct, color in [("Evergreen", "#4C72B0"), ("Spontaneous", "#DD8452")]:
        sub = topic_summary[topic_summary[LABEL_COL] == ct].sort_values("articles", ascending=False).head(15)
        fig_topic_sizes.add_trace(
            go.Bar(
                x=sub["articles"],
                y=sub["topic_label"],
                orientation="h",
                name=ct,
                marker_color=color,
                hovertemplate="%{y}<br>Articles: %{x:,.0f}<extra>" + ct + "</extra>",
            )
        )
    fig_topic_sizes.update_layout(
        title="Top Topic Sizes by Article Count",
        barmode="group",
        height=620,
        margin=dict(l=260),
        xaxis_title="Articles",
    )

    fig_outliers = go.Figure()
    for metric, title, col in [("article_share_pct", "Article Share", 1), ("impression_share_pct", "Impression Share", 2)]:
        pass
    fig_outliers = make_subplots(rows=1, cols=2, subplot_titles=("Outlier Article Share", "Outlier Impression Share"))
    outlier_view = topic_by_type[topic_by_type["is_outlier"]].copy()
    fig_outliers.add_trace(
        go.Bar(x=outlier_view[LABEL_COL], y=outlier_view["article_share_pct"], marker_color="#4C72B0", name="Articles"),
        row=1,
        col=1,
    )
    fig_outliers.add_trace(
        go.Bar(x=outlier_view[LABEL_COL], y=outlier_view["impression_share_pct"], marker_color="#DD8452", name="Impressions"),
        row=1,
        col=2,
    )
    fig_outliers.update_layout(title="Topic Outlier Share", height=360, showlegend=False)
    fig_outliers.update_yaxes(ticksuffix="%")

    non_outlier_count = int((~topic_df["is_outlier"]).sum())
    outlier_count = int(topic_df["is_outlier"].sum())
    metrics += metric_rows(
        "topics",
        [
            {"metric": "topic_rows_non_outlier", "value": non_outlier_count, "detail": ""},
            {"metric": "topic_rows_outlier", "value": outlier_count, "detail": ""},
            {"metric": "outlier_article_share_pct", "value": round(safe_pct(outlier_count, len(topic_df)), 2), "detail": ""},
            {"metric": "unique_topics_non_outlier", "value": topic_summary["topic_key"].nunique(), "detail": ""},
        ],
    )

# ── Forecast diagnostics ─────────────────────────────────────
forecast_metrics = pd.DataFrame()
fig_backtest_imps = None
fig_growth = None
fig_forecast_totals = None

valid_hist = df.dropna(subset=["date", "event_date", LABEL_COL]).copy()
daily_hist = (
    valid_hist.groupby([LABEL_COL, "year", "days_to_event"], as_index=False)
    .agg(impressions=(IMPS_COL, "sum"), articles=(URL_COL, "count"))
)

backtest_rows = []
for ct, g in daily_hist.groupby(LABEL_COL):
    pivot_imps = g.pivot(index="days_to_event", columns="year", values="impressions")
    pivot_arts = g.pivot(index="days_to_event", columns="year", values="articles")
    if BASE_YEAR in pivot_imps.columns and COMPARE_YEAR in pivot_imps.columns:
        aligned_imps = pivot_imps.dropna(subset=[BASE_YEAR, COMPARE_YEAR])
        aligned_arts = pivot_arts.reindex(aligned_imps.index)
        imps_metrics = regression_metrics(aligned_imps[COMPARE_YEAR], aligned_imps[BASE_YEAR])
        arts_metrics = regression_metrics(aligned_arts[COMPARE_YEAR], aligned_arts[BASE_YEAR])
        backtest_rows.append({"content_type": ct, "metric_target": "impressions", **imps_metrics})
        backtest_rows.append({"content_type": ct, "metric_target": "articles", **arts_metrics})

forecast_metrics = pd.DataFrame(backtest_rows)
if not forecast_metrics.empty:
    metrics += metric_rows(
        "forecast_backtest",
        [
            {
                "metric": f"{r.content_type}_{r.metric_target}_wape_pct",
                "value": round(r.wape_pct, 2) if pd.notna(r.wape_pct) else np.nan,
                "detail": f"naive backtest: {BASE_YEAR} same event-day predicts {COMPARE_YEAR}",
            }
            for _, r in forecast_metrics.iterrows()
        ],
    )

    imps_view = forecast_metrics[forecast_metrics["metric_target"] == "impressions"].copy()
    fig_backtest_imps = make_subplots(rows=1, cols=2, subplot_titles=("WAPE", "Forecast / Actual"))
    fig_backtest_imps.add_trace(
        go.Bar(x=imps_view["content_type"], y=imps_view["wape_pct"], marker_color="#4C72B0", name="WAPE"),
        row=1,
        col=1,
    )
    fig_backtest_imps.add_trace(
        go.Bar(x=imps_view["content_type"], y=imps_view["forecast_to_actual"], marker_color="#DD8452", name="Forecast / Actual"),
        row=1,
        col=2,
    )
    fig_backtest_imps.update_layout(
        title=f"Naive Backtest: {BASE_YEAR} Same Event-Day Predicts {COMPARE_YEAR}",
        height=380,
        showlegend=False,
    )
    fig_backtest_imps.update_yaxes(ticksuffix="%", row=1, col=1)

growth_rows = []
for ct, g in daily_hist.groupby(LABEL_COL):
    totals = g.groupby("year").agg(impressions=("impressions", "sum"), articles=("articles", "sum"))
    if BASE_YEAR in totals.index and COMPARE_YEAR in totals.index:
        growth_rows.append(
            {
                "content_type": ct,
                "impressions_growth_factor": totals.loc[COMPARE_YEAR, "impressions"] / totals.loc[BASE_YEAR, "impressions"] if totals.loc[BASE_YEAR, "impressions"] else np.nan,
                "articles_growth_factor": totals.loc[COMPARE_YEAR, "articles"] / totals.loc[BASE_YEAR, "articles"] if totals.loc[BASE_YEAR, "articles"] else np.nan,
            }
        )
growth_df = pd.DataFrame(growth_rows)
if not growth_df.empty:
    fig_growth = go.Figure()
    fig_growth.add_trace(go.Bar(x=growth_df["content_type"], y=growth_df["impressions_growth_factor"], name="Impressions", marker_color="#4C72B0"))
    fig_growth.add_trace(go.Bar(x=growth_df["content_type"], y=growth_df["articles_growth_factor"], name="Articles", marker_color="#DD8452"))
    fig_growth.add_hline(y=1, line_dash="dash", line_color="gray")
    fig_growth.update_layout(title=f"Observed Growth Factors: {BASE_YEAR} to {COMPARE_YEAR}", yaxis_title="Growth Factor", barmode="group", height=380)

if forecast_type_path.exists():
    fc_type = clean_df(pd.read_csv(forecast_type_path))
    imps_col = f"forecast_impressions_{FORECAST_YEAR}"
    arts_col = f"forecast_articles_{FORECAST_YEAR}"
    if {LABEL_COL, imps_col, arts_col}.issubset(fc_type.columns):
        fc_totals = (
            fc_type.groupby(LABEL_COL, as_index=False)
            .agg(forecast_impressions=(imps_col, "sum"), forecast_articles=(arts_col, "sum"))
        )
        fig_forecast_totals = make_subplots(rows=1, cols=2, subplot_titles=("Forecast Articles", "Forecast Impressions"))
        fig_forecast_totals.add_trace(
            go.Bar(x=fc_totals[LABEL_COL], y=fc_totals["forecast_articles"], marker_color="#4C72B0"),
            row=1,
            col=1,
        )
        fig_forecast_totals.add_trace(
            go.Bar(x=fc_totals[LABEL_COL], y=fc_totals["forecast_impressions"], marker_color="#DD8452"),
            row=1,
            col=2,
        )
        fig_forecast_totals.update_layout(title=f"{FORECAST_YEAR} Forecast Totals by Content Type", height=380, showlegend=False)

# ── Data health tables/charts ────────────────────────────────
year_content = (
    df.groupby(["year", LABEL_COL], as_index=False)
    .agg(articles=(URL_COL, "count"), impressions=(IMPS_COL, "sum"))
    .sort_values(["year", LABEL_COL])
)
fig_year_content = make_subplots(rows=1, cols=2, subplot_titles=("Articles", "Impressions"))
for ct, color in [("Evergreen", "#4C72B0"), ("Spontaneous", "#DD8452")]:
    sub = year_content[year_content[LABEL_COL] == ct]
    fig_year_content.add_trace(go.Bar(x=sub["year"], y=sub["articles"], name=ct, marker_color=color, legendgroup=ct), row=1, col=1)
    fig_year_content.add_trace(go.Bar(x=sub["year"], y=sub["impressions"], name=ct, marker_color=color, legendgroup=ct, showlegend=False), row=1, col=2)
fig_year_content.update_layout(title="Historical Data Volume by Year", barmode="group", height=380)

missing_summary = pd.DataFrame(
    [
        {"field": col, "missing_rows": int(df[col].isna().sum()), "missing_pct": round(df[col].isna().mean() * 100, 2)}
        for col in [URL_COL, TEXT_COL, "date", "page_categories_tier1", IMPS_COL, LABEL_COL]
        if col in df.columns
    ]
)

metrics_df = pd.DataFrame(metrics)
Path("data").mkdir(exist_ok=True)
metrics_df.to_csv(OUT_CSV, index=False)

classification_html = (
    f'<p class="note">{html.escape(classification_note)}</p>'
    if classification_note
    else (
        '<p class="note">Metrics below are computed on the held-out test split (20% of 2025_claude.csv, '
        "same deterministic split as model training). These URLs were not used for training.</p>"
        + fig_to_html(fig_confusion)
        + df_to_html(classification_summary)
    )
)

topic_html = ""
if fig_outliers is not None:
    topic_html += fig_to_html(fig_outliers)
if fig_topic_sizes is not None:
    topic_html += fig_to_html(fig_topic_sizes)
topic_html += df_to_html(topic_summary.assign(impressions=topic_summary.get("impressions", pd.Series(dtype=float)).map(lambda x: fmt_num(x))) if not topic_summary.empty else topic_summary, max_rows=20)

forecast_html = ""
for fig in [fig_backtest_imps, fig_growth, fig_forecast_totals]:
    if fig is not None:
        forecast_html += fig_to_html(fig)
forecast_display = forecast_metrics.copy()
if not forecast_display.empty:
    for col in ["actual_total", "predicted_total", "bias", "mae", "rmse"]:
        forecast_display[col] = forecast_display[col].map(lambda x: fmt_num(x))
    for col in ["mape_pct", "wape_pct"]:
        forecast_display[col] = forecast_display[col].map(fmt_pct)
    forecast_display["forecast_to_actual"] = forecast_display["forecast_to_actual"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}x")
forecast_html += df_to_html(forecast_display)

metrics_display = metrics_df.copy()
if not metrics_display.empty:
    metrics_display["value"] = metrics_display["value"].map(lambda x: fmt_num(x, 4) if isinstance(x, float) and not pd.isna(x) else x)

html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{EVENT_NAME} {FORECAST_YEAR} - Metrics Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f4f6f9;color:#1a1a2e;margin:0}}
.hero{{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);color:white;padding:42px 40px 32px}}
.hero h1{{margin:0;font-size:2rem}}.hero p{{margin:10px 0 0;opacity:.78}}
.section{{background:white;margin:24px 40px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.08);overflow:hidden}}
.section h2{{margin:0;padding:20px 24px 0;font-size:1.1rem;color:#0f3460}}
.inner{{padding:16px 24px 24px}}
.data-table{{width:100%;border-collapse:collapse;font-size:.82rem;margin-top:12px}}
.data-table th{{background:#0f3460;color:white;padding:8px 12px;text-align:left}}
.data-table td{{padding:7px 12px;border-bottom:1px solid #edf0f4;vertical-align:top}}
.data-table tr:hover td{{background:#f0f4ff}}
.note,.empty{{color:#666;font-size:.9rem}}
footer{{text-align:center;padding:24px;font-size:.78rem;color:#999}}
</style>
</head>
<body>
<div class="hero">
  <h1>{EVENT_NAME} {FORECAST_YEAR} - Metrics Dashboard</h1>
  <p>Diagnostics for classification, topics, forecast reliability, and source-data health.</p>
</div>

<div class="section"><h2>Classification Metrics</h2><div class="inner">{classification_html}</div></div>
<div class="section"><h2>Forecast Metrics</h2><div class="inner"><p class="note">Backtest shown here is a naive baseline: {BASE_YEAR} same event-day values are used to predict {COMPARE_YEAR}. This is a trust diagnostic, not the production forecast method.</p>{forecast_html}</div></div>
<div class="section"><h2>Topic Diagnostics</h2><div class="inner">{topic_html}</div></div>
<div class="section"><h2>Data Health</h2><div class="inner">{fig_to_html(fig_year_content)}{df_to_html(year_content)}{df_to_html(missing_summary)}</div></div>
<div class="section"><h2>Metric Export</h2><div class="inner">{df_to_html(metrics_display)}</div></div>

<footer>{EVENT_NAME} Metrics &nbsp;·&nbsp; source: {html.escape(str(source_path))}</footer>
</body>
</html>"""

Path(OUT_HTML).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_HTML).write_text(html_doc, encoding="utf-8")

print(f"Metrics HTML : {OUT_HTML}")
print(f"Metrics CSV  : {OUT_CSV}")
