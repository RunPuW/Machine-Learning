"""
ranker_agent.py  -  Ranker Agent
Reranks KNN+SVD fusion candidates using GPT-4o, with fallback to knn_score ordering.
"""

import os
import json
import time
import pandas as pd
from openai import OpenAI
from typing import Optional

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_KEY_HERE")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1"
MODEL_NAME         = "openai/gpt-4o"

RATING_SHORT = {
    "Overwhelmingly Positive": "Overwhelm.Pos",
    "Very Positive":           "Very Pos",
    "Mostly Positive":         "Mostly Pos",
    "Mixed":                   "Mixed",
    "Mostly Negative":         "Mostly Neg",
    "Overwhelmingly Negative": "Overwhelm.Neg",
    "Positive":                "Positive",
    "Negative":                "Negative",
}

RANKER_SYSTEM_PROMPT = """You are a Steam game recommendation ranker.
Your task: rerank a candidate list to best match the user's preferences, then select top 10.

Guidelines:
- Prioritize games that match the user's inferred themes and playstyle.
- Respect data_profile hard constraints: do not recommend games outside the user's price range unless the game is very strongly matched thematically.
- Do NOT recommend games the user has already played (listed in played_history).
- The knn_score column shows retrieval confidence - use it as a prior, don't ignore it entirely.
- For semantic_profile, treat it as soft guidance only (it was inferred from titles, no tag data).
- Be specific in explanations - reference the user's actual history games when possible.

Output ONLY valid JSON:
{
  "ranked_ids": [app_id1, app_id2, ...],
  "explanations": {
    "app_id1": "one sentence reason",
    "app_id2": "one sentence reason"
  }
}
ranked_ids must contain exactly 10 app_ids from the candidates list."""


def build_ranker_prompt(
    uua_profile:   dict,
    candidates:    list[dict],
    played_titles: list[str],
    top_n:         int = 10,
) -> str:
    dp = uua_profile.get("data_profile", {})
    sp = uua_profile.get("semantic_profile", {})

    profile_summary = {
        "price_median":        dp.get("price", {}).get("median", "unknown"),
        "price_max":           dp.get("price", {}).get("max", "unknown"),
        "free_game_ratio":     dp.get("price", {}).get("free_game_ratio", 0),
        "avg_positive_ratio":  dp.get("rating", {}).get("avg_positive_ratio", "unknown"),
        "median_year":         dp.get("year", {}).get("median_year", "unknown"),
        "long_session_player": dp.get("playtime", {}).get("long_session", False),
        "prefers_niche":       dp.get("niche", {}).get("prefers_niche", False),
        "inferred_themes":     sp.get("inferred_themes", []),
        "playstyle_tags":      sp.get("playstyle_tags", []),
        "theme_confidence":    sp.get("confidence", "low"),
        "theme_evidence":      sp.get("evidence", {}),
        "semantic_warning":    sp.get("warning", ""),
    }

    header = "rank | app_id | title | rating | pos% | price | year | knn_score | source"
    rows   = []
    for i, c in enumerate(candidates, 1):
        title     = str(c.get("title", ""))[:35].ljust(35)
        rating    = RATING_SHORT.get(str(c.get("rating", "")), str(c.get("rating", ""))[:12])
        pos_ratio = f"{c.get('positive_ratio', 0):.0f}"
        price     = f"${c.get('price_final', 0):.2f}"
        year      = str(c.get("release_year", ""))
        knn_score = f"{c.get('knn_score', 0):.4f}"
        source    = c.get("source", "knn")
        rows.append(
            f"{i:>4} | {c['app_id']:>7} | {title} | {rating:<14} | "
            f"{pos_ratio:>3}% | {price:>7} | {year} | {knn_score} | {source}"
        )

    candidates_table = "\n".join([header] + rows)

    played_str = ", ".join(played_titles[:30])
    if len(played_titles) > 30:
        played_str += f" ... (+{len(played_titles)-30} more)"

    return f"""## User Profile
{json.dumps(profile_summary, indent=2, ensure_ascii=False)}

## Already Played (DO NOT recommend these)
{played_str}

## Candidates ({len(candidates)} games, select top {top_n})
{candidates_table}

Rerank the candidates and select the top {top_n}. Output JSON only."""


def rank_candidates(
    uua_profile:   dict,
    candidates:    list[dict],
    played_titles: list[str],
    client:        OpenAI,
    top_n:         int = 10,
    retry:         int = 2,
) -> dict:
    fallback_ids = [
        c["app_id"] for c in sorted(candidates, key=lambda x: -x.get("knn_score", 0))
    ][:top_n]

    user_msg   = build_ranker_prompt(uua_profile, candidates, played_titles, top_n)
    last_error = None

    for attempt in range(retry + 1):
        try:
            resp = client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = [
                    {"role": "system", "content": RANKER_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature = 0.0,
                max_tokens  = 600,
            )
            raw    = resp.choices[0].message.content.strip()
            raw    = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(raw)

            ranked_ids = result.get("ranked_ids", [])
            if not isinstance(ranked_ids, list) or len(ranked_ids) == 0:
                raise ValueError(f"ranked_ids empty or malformed: {ranked_ids}")

            valid_ids  = {c["app_id"] for c in candidates}
            ranked_ids = [int(x) for x in ranked_ids if int(x) in valid_ids][:top_n]

            if len(ranked_ids) < top_n:
                seen = set(ranked_ids)
                for fid in fallback_ids:
                    if fid not in seen:
                        ranked_ids.append(fid)
                    if len(ranked_ids) == top_n:
                        break

            return {
                "ranked_ids":   ranked_ids,
                "explanations": {str(k): str(v) for k, v in result.get("explanations", {}).items()},
                "source":       "llm",
                "error":        None,
            }

        except json.JSONDecodeError as e:
            last_error = f"json_parse: {e}"
            time.sleep(1)
        except Exception as e:
            last_error = str(e)
            time.sleep(2)

    return {
        "ranked_ids":   fallback_ids,
        "explanations": {},
        "source":       "fallback",
        "error":        last_error,
    }


if __name__ == "__main__":
    print("[TEST] ranker_agent prompt build test (no API call)")

    dummy_uua = {
        "data_profile": {
            "price":    {"min": 0, "median": 9.99, "max": 29.99, "free_game_ratio": 0.1, "discount_ratio": 0.3},
            "rating":   {"median_rating_score": 5, "avg_positive_ratio": 88.0},
            "year":     {"median_year": 2016, "prefers_classic": False},
            "playtime": {"long_session": True},
            "niche":    {"prefers_niche": False},
        },
        "semantic_profile": {
            "inferred_themes": ["action", "rpg"],
            "playstyle_tags":  ["story-driven"],
            "confidence":      "medium",
            "evidence":        {"action": ["Dark Souls"], "rpg": ["The Witcher 3"]},
            "warning":         "inferred from titles only",
        }
    }

    dummy_candidates = [
        {"app_id": 570,    "title": "Dota 2",          "rating": "Very Positive", "positive_ratio": 84, "price_final": 0.0,   "release_year": 2013, "knn_score": 0.823, "source": "knn"},
        {"app_id": 292030, "title": "The Witcher 3",    "rating": "Very Positive", "positive_ratio": 97, "price_final": 29.99, "release_year": 2015, "knn_score": 0.761, "source": "both"},
        {"app_id": 381210, "title": "Dead by Daylight", "rating": "Mixed",         "positive_ratio": 68, "price_final": 19.99, "release_year": 2016, "knn_score": 0.612, "source": "svd"},
    ]

    prompt = build_ranker_prompt(dummy_uua, dummy_candidates, ["Dark Souls"])
    print("\n--- prompt preview (first 800 chars) ---")
    print(prompt[:800])
    print("\n[PASS] prompt build OK")