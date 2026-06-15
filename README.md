# Event Forecasting Inputs

## Launch

Run the full pipeline from the project root:

```bash
python3 run_pipeline.py
```

Reports are written to `reports/`.

Project layout:

```text
pipeline/   Python scripts and config
notebooks/  Exploratory notebooks
docs/       Extra event-specific notes
data/       Input and generated CSV files
setfit evergreen/
            Language-specific SetFit models
reports/    Generated HTML reports
```

Send this information before asking for a new forecast.

## Required Brief

```text
Event name:
Language:

Forecast event dates:
- 2026: YYYY-MM-DD to YYYY-MM-DD

Forecast report window:
- YYYY-MM-DD to YYYY-MM-DD

Historical event dates:
- 2025: YYYY-MM-DD to YYYY-MM-DD

Files:
- data/2025.csv = how to use it
- data/2026_recent.csv = how to use it
```

## Important Difference

```text
Forecast event dates = when the event actually happens
Forecast report window = which dates the final report is allowed to show
```

Example:

```text
Event dates: 2026-09-28 to 2026-10-06
Report window: 2026-09-01 to 2026-10-31
```

This means the event starts on September 28, but the report only shows forecasts in September and October.

## CSV Columns

Each CSV should contain:

```text
url
page_content_title_formalized
page_content_body_formalized
date
page_categories_tier1
ClientCreativeImpression
```

## Example

```text
Event: Paris Fashion Week
Language: fr

Forecast event dates:
- 2026: 2026-09-28 to 2026-10-06

Forecast report window:
- 2026-09-01 to 2026-10-31

Historical event dates:
- 2025: 2025-09-29 to 2025-10-07

Files:
- data/2025.csv = historical Fashion Week data to project into 2026
- data/2026_recent.csv = recent topic/momentum data only, not forecast dates
```
