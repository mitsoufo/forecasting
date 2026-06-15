# 2-find-topics.py — Discover topics per content type using BERTopic
# Output: data/with_topics.csv

import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import nltk
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import time
import google.generativeai as genai

# ── Config ──────────────────────────────────────────────────
TEXT_COL         = "page_content_title_formalized"
BODY_COL         = "page_content_body_formalized"
URL_COL          = "url"
LABEL_COL        = "content_type"
BASE_MODEL       = "sentence-transformers/paraphrase-MiniLM-L6-v2"
SAVED_MODEL_PATH = Path("setfit evergreen") / "setfit_evergreen_model"
NUM_EPOCHS       = 1
NUM_ITERATIONS   = 20
NR_TOPICS_MIN    = 10
NR_TOPICS_MAX    = 18
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")  # set env var or paste key here
GEMINI_MODEL     = "gemini-2.5-flash-lite"  # 500 RPD on free tier vs 20 for 2.5-flash

from config import EVENT_NAME, EVENT_KEY, EVENT_ANCHORS, FORECAST_YEAR, DOMAIN_STOPWORDS

print("Imports OK")

merge_df = pd.read_csv('data/classified.csv')

from nltk.corpus import stopwords

_stop_langs = ["english", "dutch", "swedish", "norwegian", "danish",
               "german", "french", "spanish", "portuguese", "italian"]
all_stopwords = set()
for _lang in _stop_langs:
    try:
        all_stopwords.update(stopwords.words(_lang))
    except (LookupError, OSError):
        pass
all_stopwords.update(DOMAIN_STOPWORDS)

# ── LLM topic labeling (single batched API call) ────────────
def llm_label_topics(topic_payload: dict[str, dict]) -> dict[str, dict]:
    """
    topic_payload: {topic_key: {"keywords": "kw1, kw2, ...", "examples": [title1, ...]}}
    Returns:       {topic_key: {"label": "...", "description": "..."}}
    Single Gemini API call for all topics. Falls back to keyword strings on failure.
    """
    fallback = {
        k: {"label": v["keywords"], "description": "(LLM unavailable)"}
        for k, v in topic_payload.items()
    }
    if not GEMINI_API_KEY:
        print("[LLM] GEMINI_API_KEY not set — skipping LLM labeling")
        return fallback

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    payload_json = json.dumps(topic_payload, indent=2, ensure_ascii=False)
    prompt = (
        f"You are a senior content strategist analyzing {EVENT_NAME} article clusters.\n"
        "Below is a JSON object: each key is a topic id; each value contains BERTopic "
        "keywords plus 5 representative article titles for that cluster.\n\n"
        "Your job: produce a SHORT distinctive label (3-6 words, title case) AND a 1-2 "
        "sentence description for each topic. Rules:\n"
        "  • Labels must be DISTINCT — do not reuse generic phrases.\n"
        "  • Focus on the specific PRODUCT CATEGORY, BRAND, or AUDIENCE that distinguishes\n"
        "    each cluster (e.g. 'Premium TVs & Soundbars', 'UK Energy Bills & Heating',\n"
        "    'Beauty & Skincare Sets', 'Gaming Consoles & Accessories').\n"
        f"  • Avoid the words '{EVENT_NAME}', 'Deals', 'Best', 'Top' in the label — every\n"
        "    cluster is already about this event; the label must add information.\n"
        "  • Description: explain what makes this cluster interesting (audience, product\n"
        "    angle, why it matters editorially). 1-2 sentences, max 220 chars.\n\n"
        "Return JSON ONLY in this shape (no markdown, no commentary):\n"
        '{ "<topic_key>": { "label": "...", "description": "..." }, ... }\n\n'
        + payload_json
    )

    for attempt in range(2):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            labeled = json.loads(text)
            # Sanity-fill any missing keys
            for k in topic_payload:
                if k not in labeled or "label" not in labeled[k]:
                    labeled[k] = fallback[k]
            print(f"[LLM] Labeled {len(labeled)} topics via Gemini")
            return labeled
        except Exception as e:
            # Parse retry_delay from the error message if available
            retry_delay = None
            match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(e))
            if match:
                retry_delay = int(match.group(1)) + 5  # small buffer
            if attempt == 0 and retry_delay is not None:
                print(f"[LLM] Gemini call failed (429 quota). Retrying in {retry_delay}s…")
                time.sleep(retry_delay)
            else:
                print(f"[LLM] Gemini call failed: {e} — falling back to keyword labels")
                return fallback


topic_rows = []
topic_info_rows = []
raw_topic_payload: dict[str, dict] = {}  # topic_key -> {keywords, examples}

for content_type, group in merge_df.groupby(LABEL_COL):
    df_type = group.copy()
    text = (
        df_type[TEXT_COL].fillna("").astype(str)
        + ". "
        + df_type[BODY_COL].fillna("").astype(str)
    )

    if len(df_type) < 5:
        print(f"Skipping {content_type}: only {len(df_type)} articles")
        continue

    min_df = 3 if len(df_type) >= 500 else 2
    nr_topics = min(NR_TOPICS_MAX, max(NR_TOPICS_MIN, len(df_type) // 700))
    nr_topics = min(nr_topics, len(df_type))
    vectorizer_model = TfidfVectorizer(
        stop_words=list(all_stopwords),
        min_df=min_df,
        max_df=0.9,
        ngram_range=(1, 2),
        max_features=8000,
        strip_accents="unicode",
    )
    X = vectorizer_model.fit_transform(text)

    if X.shape[1] == 0:
        print(f"Skipping {content_type}: no usable vocabulary")
        continue

    topic_model = MiniBatchKMeans(
        n_clusters=nr_topics,
        random_state=42,
        n_init=10,
        batch_size=1024,
    )
    topics = topic_model.fit_predict(X)
    terms = np.array(vectorizer_model.get_feature_names_out())

    df_type["topic_id"] = topics
    df_type["topic_key"] = df_type[LABEL_COL] + "_" + df_type["topic_id"].astype(str)

    # Collect keywords + 5 example titles per topic for LLM batching.
    title_series = df_type[TEXT_COL].fillna("").astype(str)
    for tid in set(topics):
        tkey = f"{content_type}_{tid}"
        top_idx = topic_model.cluster_centers_[tid].argsort()[-8:][::-1]
        kws = ", ".join(terms[top_idx])
        # Sample up to 5 representative titles from this cluster
        in_topic = df_type["topic_id"] == tid
        examples = (
            title_series[in_topic]
            .loc[lambda s: s.str.len() > 0]
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raw_topic_payload[tkey] = {"keywords": kws, "examples": examples}

    keyword_labels = {
        tid: ", ".join(terms[topic_model.cluster_centers_[tid].argsort()[-4:][::-1]])
        for tid in set(topics)
    }
    df_type["topic_label"] = df_type["topic_id"].map(keyword_labels)

    topic_info = (
        df_type.groupby("topic_id")
        .agg(Count=(URL_COL, "count"), Name=("topic_label", "first"))
        .reset_index()
        .rename(columns={"topic_id": "Topic"})
    )
    topic_info[LABEL_COL] = content_type
    topic_info["topic_key"] = topic_info[LABEL_COL] + "_" + topic_info["Topic"].astype(str)

    topic_rows.append(df_type)
    topic_info_rows.append(topic_info)

    n_topics = len(set(topics))
    n_outliers = 0
    print(
        f"{content_type}: topics found: {n_topics} | "
        f"outlier articles: {n_outliers:,} | "
        f"min_df: {min_df}"
    )

topic_assignments = pd.concat(topic_rows, ignore_index=False)
topic_info_all = pd.concat(topic_info_rows, ignore_index=True)

# ── 1. Resolve LLM labels: use cache if all topic keys already exist ──────────
desc_cache_path = Path("data/topic_descriptions.csv")
current_keys = set(raw_topic_payload.keys())

if desc_cache_path.exists():
    cached_df = pd.read_csv(desc_cache_path)
    cached_keys = set(cached_df["topic_key"])
    cache_is_stale = Path("data/classified.csv").stat().st_mtime > desc_cache_path.stat().st_mtime
    cached_current = cached_df[cached_df["topic_key"].isin(current_keys)].copy()
    cached_has_fallback = (
        cached_current["topic_description"].fillna("").eq("(LLM unavailable)").any()
        or cached_current["topic_label"].fillna("").str.contains(r"\|").any()
    )
    if current_keys <= cached_keys and not cached_has_fallback and not cache_is_stale:
        print(f"[LLM] Cache hit — all {len(current_keys)} topic keys found in topic_descriptions.csv, skipping API call")
        label_map = dict(zip(cached_df["topic_key"], cached_df["topic_label"]))
        desc_map  = dict(zip(cached_df["topic_key"], cached_df["topic_description"].fillna("")))
    else:
        new_keys = current_keys - cached_keys
        stale_reasons = []
        if cached_has_fallback:
            stale_reasons.append("stale fallback labels")
        if cache_is_stale:
            stale_reasons.append("new classified input")
        stale_msg = " or " + " / ".join(stale_reasons) if stale_reasons else ""
        print(f"[LLM] {len(new_keys)} new topic keys{stale_msg} — calling API")
        llm_out = llm_label_topics(raw_topic_payload)
        label_map = {k: v["label"] for k, v in llm_out.items()}
        desc_map  = {k: v.get("description", "") for k, v in llm_out.items()}
else:
    llm_out = llm_label_topics(raw_topic_payload)
    label_map = {k: v["label"] for k, v in llm_out.items()}
    desc_map  = {k: v.get("description", "") for k, v in llm_out.items()}

# ── 2. Save topic_descriptions.csv FIRST (canonical source of truth) ─────────
desc_df = pd.DataFrame([
    {"topic_key": k, "topic_label": label_map.get(k, ""), "topic_description": desc_map.get(k, "")}
    for k in raw_topic_payload
])
desc_df.to_csv(desc_cache_path, index=False)
print("Saved: data/topic_descriptions.csv")

# ── 3. Populate with_topics.csv by mapping from topic_descriptions ────────────
topic_assignments["topic_label"] = topic_assignments["topic_key"].map(label_map).fillna(
    topic_assignments["topic_label"]
)
topic_assignments["topic_description"] = topic_assignments["topic_key"].map(desc_map).fillna("")

merge_df_with_topics = merge_df.copy()
merge_df_with_topics["topic_id"] = np.nan
merge_df_with_topics["topic_label"] = np.nan
merge_df_with_topics["topic_key"] = np.nan
merge_df_with_topics["topic_description"] = ""
merge_df_with_topics.loc[
    topic_assignments.index,
    ["topic_id", "topic_label", "topic_key", "topic_description"],
] = topic_assignments[["topic_id", "topic_label", "topic_key", "topic_description"]]
merge_df_with_topics.to_csv('data/with_topics.csv', index=False)
print("Saved: data/with_topics.csv")
