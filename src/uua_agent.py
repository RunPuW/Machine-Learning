"""
uua_agent.py  -  User Understanding Agent
Builds a structured user preference profile from positive review history + game metadata.
  data_profile    : pure Python stats, no LLM, no hallucination risk
  semantic_profile: GPT-4o infers themes from game titles, with confidence and evidence
"""

import os
import json
import time
import numpy as np
import pandas as pd
from openai import OpenAI
from typing import Optional

PROJECT_ROOT = r"C:\Users\vipuser\Desktop\ml"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_KEY_HERE")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1"
MODEL_NAME         = "openai/gpt-4o"

RATING_ORDER = {
    "Overwhelmingly Positive": 6,
    "Very Positive":           5,
    "Mostly Positive":         4,
    "Mixed":                   3,
    "Mostly Negative":         2,
    "Overwhelmingly Negative": 1,
    "Positive":                4,
    "Negative":                2,
}


def ratio_to_label(r: float) -> str:
    if r >= 95: return "overwhelmingly positive"
    if r >= 85: return "very positive"
    if r >= 70: return "mostly positive"
    if r >= 40: return "mixed"
    return "mostly negative"


def load_games_meta(data_dir: str = DATA_DIR) -> pd.DataFrame:
    path = os.path.join(data_dir, "games_cleaned.csv")
    df   = pd.read_csv(path)
    df["date_release"] = pd.to_datetime(df["date_release"], errors="coerce")
    df["release_year"] = df["date_release"].dt.year.fillna(0).astype(int)
    df["rating_score"] = df["rating"].map(RATING_ORDER).fillna(3)
    return df.set_index("app_id")


def compute_data_profile(
    user_history: list[dict],
    games_meta:   pd.DataFrame,
) -> dict:
    app_ids = [h["app_id"] for h in user_history]
    hours   = [h.get("hours", 0.0) for h in user_history]

    present = [a for a in app_ids if a in games_meta.index]
    if not present:
        return {"error": "no_history_in_games_meta"}

    meta = games_meta.loc[present]

    prices = meta["price_final"].dropna()
    price_profile = {
        "min":             round(float(prices.min()),    2) if len(prices) else 0.0,
        "median":          round(float(prices.median()), 2) if len(prices) else 0.0,
        "max":             round(float(prices.max()),    2) if len(prices) else 0.0,
        "free_game_ratio": round(float((prices == 0).mean()), 3),
        "discount_ratio":  round(float((meta["discount"] > 0).mean()), 3),
    }

    rating_scores = meta["rating_score"].dropna()
    rating_profile = {
        "median_rating_score": round(float(rating_scores.median()), 1) if len(rating_scores) else 3,
        "min_rating_label":    meta["rating"].value_counts().index[-1] if len(meta) else "unknown",
        "avg_positive_ratio":  round(float(meta["positive_ratio"].mean()), 1) if "positive_ratio" in meta else 0.0,
    }

    years = meta["release_year"].replace(0, np.nan).dropna()
    year_profile = {
        "min_year":       int(years.min())    if len(years) else 0,
        "median_year":    int(years.median()) if len(years) else 0,
        "max_year":       int(years.max())    if len(years) else 0,
        "prefers_classic": bool(years.median() < 2015) if len(years) else False,
    }

    platform_cols    = [c for c in ["win", "mac", "linux"] if c in meta.columns]
    platform_profile = {col: round(float(meta[col].mean()), 3) for col in platform_cols}

    reviews = meta["user_reviews"].dropna()
    niche_profile = {
        "median_user_reviews": int(reviews.median()) if len(reviews) else 0,
        "prefers_niche":       bool(reviews.median() < 500) if len(reviews) else False,
    }

    valid_hours      = [h for h in hours if h > 0]
    playtime_profile = {
        "median_hours":  round(float(np.median(valid_hours)), 1) if valid_hours else 0.0,
        "long_session":  bool(np.median(valid_hours) > 20) if valid_hours else False,
        "history_count": len(app_ids),
        "in_meta_count": len(present),
    }

    return {
        "price":    price_profile,
        "rating":   rating_profile,
        "year":     year_profile,
        "platform": platform_profile,
        "niche":    niche_profile,
        "playtime": playtime_profile,
    }


SEMANTIC_SYSTEM_PROMPT = """You are a Steam game preference analyst.
Given a list of games a user has positively reviewed, infer their gaming preferences.

Rules:
- Only infer what is clearly supported by the game list.
- Do NOT infer "dislikes" or "avoids" - you have no negative feedback data.
- Do NOT infer price sensitivity - that is computed separately from purchase data.
- For each theme or genre you infer, you must name at least one supporting game.
- Use "confidence": "high" only when 3+ games clearly support a theme.
- Use "confidence": "medium" for 1-2 supporting games.
- Output ONLY valid JSON, no markdown, no explanation outside the JSON.

Output schema:
{
  "inferred_themes": ["action", "rpg"],
  "playstyle_tags": ["story-driven", "single-player"],
  "confidence": "high" | "medium" | "low",
  "evidence": {
    "action": ["Dark Souls", "Prince of Persia"],
    "rpg": ["The Witcher 3"]
  },
  "warning": "inferred from game titles only, no tag/description data available"
}"""


def infer_semantic_profile(
    game_titles: list[str],
    client:      OpenAI,
    max_titles:  int = 20,
    retry:       int = 2,
) -> dict:
    titles   = game_titles[:max_titles]
    user_msg = (
        f"The user has positively reviewed these {len(titles)} games:\n"
        + "\n".join(f"- {t}" for t in titles)
        + "\n\nInfer their gaming preferences in JSON."
    )

    for attempt in range(retry + 1):
        try:
            resp = client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = [
                    {"role": "system", "content": SEMANTIC_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature = 0.0,
                max_tokens  = 400,
            )
            raw    = resp.choices[0].message.content.strip()
            raw    = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(raw)

        except json.JSONDecodeError as e:
            if attempt == retry:
                return {"error": f"json_parse_failed: {e}", "raw": raw[:200]}
            time.sleep(1)

        except Exception as e:
            if attempt == retry:
                return {"error": str(e)}
            time.sleep(2)

    return {"error": "max_retries_exceeded"}


def build_uua_profile(
    user_id:       int,
    user_history:  list[dict],
    games_meta:    pd.DataFrame,
    client:        Optional[OpenAI] = None,
    skip_semantic: bool = False,
) -> dict:
    """
    skip_semantic=True skips the LLM call and returns only data_profile.
    Used for ml_only and ranker_only ablation groups.
    """
    app_ids = [h["app_id"] for h in user_history]
    present = [a for a in app_ids if a in games_meta.index]
    titles  = games_meta.loc[present, "title"].tolist() if present else []

    data_profile = compute_data_profile(user_history, games_meta)

    if skip_semantic or not client or not titles:
        semantic_profile = {"skipped": True, "reason": "ablation or no history"}
    else:
        semantic_profile = infer_semantic_profile(titles, client)

    return {
        "user_id":          user_id,
        "data_profile":     data_profile,
        "semantic_profile": semantic_profile,
    }


if __name__ == "__main__":
    print("[TEST] uua_agent self-test")
    games_meta = load_games_meta()
    print(f"  games_meta loaded: {len(games_meta):,} rows")

    test_history = [
        {"app_id": 13500,  "score": 3.2, "hours": 45.0},
        {"app_id": 292030, "score": 4.1, "hours": 120.0},
        {"app_id": 570,    "score": 2.8, "hours": 30.0},
    ]

    profile = build_uua_profile(
        user_id       = 999,
        user_history  = test_history,
        games_meta    = games_meta,
        skip_semantic = True,
    )
    print("\n  data_profile:")
    print(json.dumps(profile["data_profile"], indent=2, ensure_ascii=False))
    print("\n[PASS] data_profile OK")