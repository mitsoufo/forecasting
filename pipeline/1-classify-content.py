# 1-classify-content.py — Classify event content as Evergreen / Spontaneous using SetFit
# Output: data/classified.csv

from pathlib import Path

import numpy as np
import pandas as pd
from langdetect import detect, LangDetectException
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss

from config import (
    ARTICLE_LANGUAGE,
    EVENT_ANCHORS,
    EVENT_KEY,
    EVENT_NAME,
    FORECAST_WINDOW_END,
    FORECAST_WINDOW_START,
    FORECAST_YEAR,
    MIN_EVERGREEN_ARTICLE_SHARE,
)  # noqa: E402

# ── Config ──────────────────────────────────────────────────
TEXT_COL         = "page_content_title_formalized"
BODY_COL         = "page_content_body_formalized"
URL_COL          = "url"
LABEL_COL        = "content_type"
BASE_MODEL       = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_ROOT       = Path("setfit evergreen")
SAVED_MODEL_PATH = MODEL_ROOT / (
    "setfit_evergreen_model"
    if ARTICLE_LANGUAGE == "en"
    else f"setfit_evergreen_model_{ARTICLE_LANGUAGE}"
)
TRAINING_CSV     = "data/2025_claude.csv"
NUM_EPOCHS       = 1
NUM_ITERATIONS   = 20

print("Imports OK")
print(f"Event: {EVENT_NAME} ({EVENT_KEY}) | article language: {ARTICLE_LANGUAGE}")

## Load & Classify Data with SetFit

def load_labeled_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df["text"]  = df[TEXT_COL]
    df["label"] = df[LABEL_COL].map({"Evergreen": 0, "Spontaneous": 1})
    return df[["text", "label"]].dropna()


def train(csv_path: str):
    df = load_labeled_data(csv_path)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_ds  = Dataset.from_pandas(eval_df.reset_index(drop=True))

    model_source = SAVED_MODEL_PATH if Path(SAVED_MODEL_PATH).exists() else BASE_MODEL
    print(f"Loading model from: {model_source}")
    model = SetFitModel.from_pretrained(model_source)

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss_class=CosineSimilarityLoss,
        num_iterations=NUM_ITERATIONS,
        num_epochs=NUM_EPOCHS,
        column_mapping={"text": "text", "label": "label"},
    )
    trainer.train()

    preds  = model.predict(eval_df["text"].tolist())
    labels = eval_df["label"].tolist()
    print("\n", classification_report(labels, preds, target_names=["Evergreen", "Spontaneous"]))

    model.save_pretrained(SAVED_MODEL_PATH)
    print(f"\nModel saved to: {SAVED_MODEL_PATH}")

    try:
        metrics = trainer.evaluate()
        print("\nEval metrics:", metrics)
    except Exception as e:
        print(f"\nSkipping SetFit eval metrics: {e}")

    return model


def predict(input_csv: str, model_path: str = SAVED_MODEL_PATH):
    model = SetFitModel.from_pretrained(model_path)
    df = pd.read_csv(input_csv)
    df["text"] = df[TEXT_COL]
    preds = model.predict(df["text"].tolist())
    try:
        proba = model.predict_proba(df["text"].tolist())
        df["evergreen_score"] = [float(row[0]) for row in proba]
        df["spontaneous_score"] = [float(row[1]) for row in proba]
    except Exception as e:
        print(f"Could not compute prediction probabilities: {e}")
        df["evergreen_score"] = pd.NA
        df["spontaneous_score"] = pd.NA
    label_map = {0: "Evergreen", 1: "Spontaneous"}
    df[LABEL_COL] = [label_map[int(p)] for p in preds]
    df = df.drop(columns=["text"])
    # Clean unnamed index columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    print(f"Predicted: {input_csv}")
    print(df[LABEL_COL].value_counts().to_string())
    return df


def model_is_stale(model_path: str, training_csv: str) -> bool:
    model_dir = Path(model_path)
    training_path = Path(training_csv)
    if not model_dir.exists():
        return True
    model_files = list(model_dir.rglob("*"))
    model_mtime = max((p.stat().st_mtime for p in model_files if p.is_file()), default=0)
    return training_path.exists() and training_path.stat().st_mtime > model_mtime


# ── Run predictions (uncomment train() to retrain) ──────────
if model_is_stale(SAVED_MODEL_PATH, TRAINING_CSV):
    print(f"Training from {TRAINING_CSV} because {SAVED_MODEL_PATH} is missing or stale")
    train(TRAINING_CSV)

historical_inputs = []
for year in sorted(y for y in EVENT_ANCHORS if y != FORECAST_YEAR):
    path = Path(f"data/{year}.csv")
    if path.exists():
        historical_inputs.append((year, path))
    else:
        print(f"Skipping missing historical input: {path}")

if not historical_inputs:
    raise FileNotFoundError(
        "No historical input CSVs found. Expected at least one file like data/2025.csv."
    )

predicted_frames = [predict(str(path)) for _, path in historical_inputs]

def detect_lang(text):
    try:
        return detect(str(text)[:300])
    except LangDetectException:
        return "unknown"


merge_df = pd.concat(predicted_frames, ignore_index=True)

_langs = merge_df[TEXT_COL].fillna("").apply(detect_lang)
merge_df = merge_df[_langs == ARTICLE_LANGUAGE].reset_index(drop=True)

print(f"Rows after {ARTICLE_LANGUAGE!r} language filter: {len(merge_df):,}")

def enforce_min_evergreen_share(df: pd.DataFrame, mask=None, label: str = "dataset") -> pd.DataFrame:
    if mask is None:
        mask = pd.Series(True, index=df.index)
    scoped = df[mask]
    if scoped.empty:
        return df

    target_rows = int(np.ceil(len(scoped) * MIN_EVERGREEN_ARTICLE_SHARE))
    current_rows = int((scoped[LABEL_COL] == "Evergreen").sum())
    needed = target_rows - current_rows
    if needed <= 0:
        print(
            f"Evergreen article share already meets floor for {label}: "
            f"{current_rows / len(scoped) * 100:.1f}%"
        )
        return df

    candidates = df[mask & (df[LABEL_COL] == "Spontaneous")].copy()
    if "evergreen_score" in candidates.columns and candidates["evergreen_score"].notna().any():
        candidates = candidates.sort_values("evergreen_score", ascending=False)
    else:
        candidates = candidates.sort_values("ClientCreativeImpression", ascending=True)

    flip_idx = candidates.head(needed).index
    df = df.copy()
    df.loc[flip_idx, "classification_adjusted"] = True
    df.loc[df["classification_adjusted"].isna(), "classification_adjusted"] = False
    df.loc[flip_idx, "original_content_type"] = "Spontaneous"
    df.loc[flip_idx, LABEL_COL] = "Evergreen"

    new_rows = int((df[LABEL_COL] == "Evergreen").sum())
    scoped_new_rows = int((df.loc[mask, LABEL_COL] == "Evergreen").sum())
    print(
        f"Applied Evergreen article floor for {label}: flipped {len(flip_idx):,} rows "
        f"({current_rows / len(scoped) * 100:.1f}% → {scoped_new_rows / len(scoped) * 100:.1f}%)."
    )
    return df


merge_df["classification_adjusted"] = False
merge_df["original_content_type"] = merge_df[LABEL_COL]
merge_df = enforce_min_evergreen_share(merge_df, label="all rows")

merge_df["date"] = pd.to_datetime(merge_df["date"], errors="coerce")
merge_df["year"] = merge_df["date"].dt.year
merge_df["event_date"] = merge_df["year"].map(EVENT_ANCHORS)
merge_df["days_to_event"] = (merge_df["date"] - merge_df["event_date"]).dt.days
merge_df["forecast_date"] = (
    EVENT_ANCHORS[FORECAST_YEAR]
    + pd.to_timedelta(merge_df["days_to_event"], unit="D")
)
forecast_window_mask = merge_df["forecast_date"].between(
    FORECAST_WINDOW_START,
    FORECAST_WINDOW_END,
)
merge_df = enforce_min_evergreen_share(
    merge_df,
    mask=forecast_window_mask,
    label="forecast window",
)
merge_df = merge_df.drop(columns=["year", "event_date", "days_to_event", "forecast_date"])

merge_df.to_csv('data/classified.csv')
print("Saved: data/classified.csv")
