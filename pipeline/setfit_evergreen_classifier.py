"""
SetFit model — Evergreen vs Spontaneous content classifier

Requirements:
    pip install setfit datasets pandas scikit-learn

Usage:
    python setfit_evergreen_classifier.py --train                  # train and save model
    python setfit_evergreen_classifier.py --predict input.csv      # predict on new data
"""

import argparse
from pathlib import Path
import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

LABELED_CSV      = "black_friday_classified.csv"    # your classified CSV
TEXT_COL         = "page_content_title_formalized"  # column used as input text
URL_COL          = "url"                            # appended to text for more signal
LABEL_COL        = "content_type"                  # "Evergreen" or "Spontaneous"
BASE_MODEL       = "sentence-transformers/paraphrase-MiniLM-L6-v2"
SAVED_MODEL_PATH = Path("setfit evergreen") / "setfit_evergreen_model"
NUM_EPOCHS       = 1
NUM_ITERATIONS   = 20                               # contrastive pairs per class


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def build_text(row):
    """Combine title + URL slug as a single input string."""
    return f"{row[TEXT_COL]} {row[URL_COL]}"


def load_labeled_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df["text"]  = df.apply(build_text, axis=1)
    df["label"] = df[LABEL_COL].map({"Evergreen": 0, "Spontaneous": 1})
    return df[["text", "label"]].dropna()


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def train(csv_path: str = LABELED_CSV):
    df = load_labeled_data(csv_path)

    train_df, eval_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

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

    metrics = trainer.evaluate()
    print("\nEval metrics:", metrics)

    preds  = model.predict(eval_df["text"].tolist())
    labels = eval_df["label"].tolist()
    print("\n", classification_report(
        labels, preds, target_names=["Evergreen", "Spontaneous"]
    ))

    model.save_pretrained(SAVED_MODEL_PATH)
    print(f"\nModel saved to: {SAVED_MODEL_PATH}")
    return model


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────

def predict(input_csv: str, model_path: str = SAVED_MODEL_PATH):
    model = SetFitModel.from_pretrained(model_path)

    df = pd.read_csv(input_csv)
    df["text"] = df.apply(build_text, axis=1)

    preds = model.predict(df["text"].tolist())

    label_map = {0: "Evergreen", 1: "Spontaneous"}
    df["content_type"] = [label_map[int(p)] for p in preds]

    out_path = input_csv.replace(".csv", "_predicted.csv")
    df.drop(columns=["text"]).to_csv(out_path, index=False)
    print(f"Predictions saved to: {out_path}")
    print(df["content_type"].value_counts())
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",   action="store_true",      help="Train and save model")
    parser.add_argument("--predict", type=str, metavar="CSV",  help="Run predictions on a new CSV")
    parser.add_argument("--csv",     type=str, default=LABELED_CSV, help="Labeled training CSV path")
    args = parser.parse_args()

    if args.train:
        train(args.csv)
    elif args.predict:
        predict(args.predict)
    else:
        parser.print_help()
