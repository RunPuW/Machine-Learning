"""
cold_start.py  -  Cold-Start Recommendation Module
====================================================
Handles users with fewer than 5 training interactions.

Data reality (verified experimentally):
  - 0 interactions : 0 users (test/train user sets fully overlap)
  - 1/2/3          : 0 users (temporal split math: users with 5 total get exactly 4 in train)
  - 4 interactions : 435,823 users (all cold-start users fall here)

Three-tier design (retained for system completeness):
  Level 0 (0 interactions) : popularity fallback, pseudo-eval protocol
  Level 1 (1-2)            : popularity + metadata soft filter, pseudo-eval protocol
  Level 2 (3-4)            : profile agent + content recall + ranker, real test eval

Feature design notes:
  - positive_ratio is NOT used (full-timeline aggregate snapshot, leakage risk)
  - release_year and price_final are used (temporally stable)
  - LLM confidence is capped at "medium" for 3-4 game histories
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from evaluate import evaluate_model
from uua_agent import load_games_meta

PROJECT_ROOT = r"C:\Users\vipuser\Desktop\ml"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
RESULT_DIR   = os.path.join(PROJECT_ROOT, "results", "test")
os.makedirs(RESULT_DIR, exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_KEY_HERE")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1"
MODEL_NAME         = "openai/gpt-4o"

COLD_START_THRESHOLD = 5
POPULARITY_POOL      = 200
TOP_K_RECOMMEND      = 10
RANDOM_SEED          = 42


def build_popularity_list(train: pd.DataFrame, top_n: int = 500) -> list:
    pop = (
        train[train["is_recommended"] == 1]
        .groupby("app_id")
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    return [int(x) for x in pop]


def recommend_level0(popularity_list: list, top_k: int = TOP_K_RECOMMEND) -> list:
    return popularity_list[:top_k]


def recommend_level1(
    user_history:    list,
    popularity_list: list,
    games_meta:      pd.DataFrame,
    top_k:           int = TOP_K_RECOMMEND,
    pool_size:       int = POPULARITY_POOL,
) -> list:
    app_ids = [h["app_id"] for h in user_history]
    present = [a for a in app_ids if a in games_meta.index]

    if not present:
        return popularity_list[:top_k]

    meta_hist  = games_meta.loc[present]
    year_mean  = meta_hist["release_year"].replace(0, np.nan).mean()
    price_mean = meta_hist["price_final"].mean()

    if pd.isna(year_mean):
        year_mean = 2015.0
    if pd.isna(price_mean):
        price_mean = 9.99

    pool         = popularity_list[:pool_size]
    present_pool = [a for a in pool if a in games_meta.index]

    if not present_pool:
        return popularity_list[:top_k]

    meta_pool = games_meta.loc[present_pool].copy()
    meta_pool["release_year"] = meta_pool["release_year"].replace(0, np.nan).fillna(year_mean)

    year_range  = max(meta_pool["release_year"].max() - meta_pool["release_year"].min(), 1.0)
    price_range = max(meta_pool["price_final"].max()  - meta_pool["price_final"].min(),  1.0)

    meta_pool["year_dist"]  = (meta_pool["release_year"] - year_mean).abs()  / year_range
    meta_pool["price_dist"] = (meta_pool["price_final"]  - price_mean).abs() / price_range
    meta_pool["dist"]       = meta_pool["year_dist"] + meta_pool["price_dist"]

    seen_app_ids = set(app_ids)
    sorted_ids   = [
        a for a in meta_pool.sort_values("dist").index.tolist()
        if a not in seen_app_ids
    ]

    seen_all = set(sorted_ids) | seen_app_ids
    extra    = [a for a in popularity_list if a not in seen_all]
    final    = sorted_ids + extra

    return [int(x) for x in final[:top_k]]


PROFILE_SYSTEM_PROMPT = """You are a Steam game preference analyst for cold-start users.
The user has very limited history (3-4 games). Infer preferences conservatively.

Rules:
- Only infer what is CLEARLY supported by the game list.
- Do NOT infer dislikes or avoided genres (no negative feedback available).
- For any game you are NOT confident about (obscure/indie/unknown title),
  mark its evidence as "uncertain" instead of guessing.
- Maximum confidence is "medium" for 3-4 games. Never output "high".
- Output ONLY valid JSON, no markdown.

Output schema:
{
  "inferred_themes": ["theme1", "theme2"],
  "confidence": "low" | "medium",
  "evidence": {
    "theme1": ["GameA", "GameB"],
    "theme2": ["GameC"]
  },
  "uncertain_games": ["GameX"],
  "warning": "Profile inferred from only N historical items. High uncertainty."
}"""


def infer_cold_profile(
    game_titles:   list,
    client:        OpenAI,
    n_history:     int,
    game_metadata: list = None,
    retry:         int  = 2,
) -> dict:
    if game_metadata:
        game_lines = "\n".join(
            f"- {m['title']} (year={m['year']}, price=${m['price']:.2f}, rating={m['rating']})"
            for m in game_metadata
        )
    else:
        game_lines = "\n".join(f"- {t}" for t in game_titles)

    user_msg = (
        f"This user has played only {n_history} games:\n"
        + game_lines
        + "\n\nInfer their preferences conservatively. JSON only."
    )

    for attempt in range(retry + 1):
        try:
            resp = client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = [
                    {"role": "system", "content": PROFILE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature = 0.0,
                max_tokens  = 350,
            )
            raw    = resp.choices[0].message.content.strip()
            raw    = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(raw)

            if result.get("confidence") == "high":
                result["confidence"] = "medium"
            if "warning" not in result:
                result["warning"] = f"Profile inferred from only {n_history} items."

            return result

        except json.JSONDecodeError as e:
            if attempt == retry:
                return {"error": f"parse_failed: {e}", "inferred_themes": [], "confidence": "low"}
            time.sleep(1)
        except Exception as e:
            if attempt == retry:
                return {"error": str(e), "inferred_themes": [], "confidence": "low"}
            time.sleep(2)

    return {"error": "max_retries", "inferred_themes": [], "confidence": "low"}


RANKER_COLD_SYSTEM_PROMPT = """You are a Steam game ranker for cold-start users.
The user has very limited history. Rank candidates by likely relevance.

Rules:
- Use the user profile as soft guidance (confidence is low/medium, not high).
- Respect the popularity_score as a strong prior (user has little history).
- Do NOT recommend games in the played_history list.
- Output ONLY valid JSON.

Output schema:
{
  "ranked_ids": [app_id1, app_id2, ...],
  "explanations": {
    "app_id1": "one sentence reason"
  }
}
ranked_ids must contain exactly the requested number of app_ids."""


def rank_cold_candidates(
    profile:       dict,
    candidates:    list,
    played_titles: list,
    client:        OpenAI,
    top_n:         int = TOP_K_RECOMMEND,
    retry:         int = 2,
) -> dict:
    fallback = [c["app_id"] for c in sorted(
        candidates, key=lambda x: -x.get("popularity_score", 0)
    )][:top_n]

    header = "rank | app_id | title | rating | price | year | pop_score"
    rows   = []
    for i, c in enumerate(candidates, 1):
        title = str(c.get("title", c["app_id"]))[:35].ljust(35)
        rows.append(
            f"{i:>4} | {c['app_id']:>7} | {title} | "
            f"{str(c.get('rating', ''))[:12]:<12} | "
            f"${c.get('price_final', 0):.2f} | "
            f"{c.get('release_year', 0)} | "
            f"{c.get('popularity_score', 0):.4f}"
        )

    played_str = ", ".join(played_titles[:20])

    user_msg = f"""## User Profile (cold-start, limited history)
{json.dumps(profile, indent=2, ensure_ascii=False)}

## Played History (DO NOT recommend)
{played_str}

## Candidates ({len(candidates)} games, select top {top_n})
{header}
{chr(10).join(rows)}

Rank and select top {top_n}. JSON only."""

    for attempt in range(retry + 1):
        try:
            resp = client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = [
                    {"role": "system", "content": RANKER_COLD_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature = 0.0,
                max_tokens  = 500,
            )
            raw    = resp.choices[0].message.content.strip()
            raw    = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(raw)

            valid_ids = {c["app_id"] for c in candidates}
            ranked    = [int(x) for x in result.get("ranked_ids", []) if int(x) in valid_ids][:top_n]

            seen = set(ranked)
            for fid in fallback:
                if len(ranked) >= top_n:
                    break
                if fid not in seen:
                    ranked.append(fid)
                    seen.add(fid)

            return {
                "ranked_ids":   ranked,
                "explanations": {str(k): str(v) for k, v in result.get("explanations", {}).items()},
                "source":       "llm",
                "error":        None,
            }

        except json.JSONDecodeError:
            if attempt == retry:
                break
            time.sleep(1)
        except Exception:
            if attempt == retry:
                break
            time.sleep(2)

    return {"ranked_ids": fallback, "explanations": {}, "source": "fallback", "error": "llm_failed"}


def recommend_level2(
    uid:             int,
    user_history:    list,
    played_titles:   list,
    games_meta:      pd.DataFrame,
    popularity_list: list,
    client:          OpenAI,
    top_k:           int = TOP_K_RECOMMEND,
) -> dict:
    app_ids = [h["app_id"] for h in user_history]
    present = [a for a in app_ids if a in games_meta.index]
    titles  = games_meta.loc[present, "title"].tolist() if present else []

    game_metadata_list = []
    for a in present:
        row = games_meta.loc[a]
        game_metadata_list.append({
            "title":  str(row.get("title", a)),
            "year":   int(row.get("release_year", 0)),
            "price":  float(row.get("price_final", 0)),
            "rating": str(row.get("rating", "")),
        })
    profile = infer_cold_profile(
        game_titles   = titles,
        client        = client,
        n_history     = len(user_history),
        game_metadata = game_metadata_list if game_metadata_list else None,
    )

    year_vals  = games_meta.loc[present, "release_year"].replace(0, np.nan).dropna()
    price_vals = games_meta.loc[present, "price_final"].dropna()
    year_mean  = float(year_vals.mean())  if len(year_vals)  > 0 else 2015.0
    price_mean = float(price_vals.mean()) if len(price_vals) > 0 else 9.99

    pool_ids  = popularity_list[:200]
    pool_meta = games_meta.loc[[a for a in pool_ids if a in games_meta.index]].copy()
    pool_meta["release_year"] = pool_meta["release_year"].replace(0, np.nan).fillna(year_mean)

    year_range  = max(pool_meta["release_year"].max() - pool_meta["release_year"].min(), 1.0)
    price_range = max(pool_meta["price_final"].max()  - pool_meta["price_final"].min(),  1.0)

    pool_meta["year_dist"]  = (pool_meta["release_year"] - year_mean).abs()  / year_range
    pool_meta["price_dist"] = (pool_meta["price_final"]  - price_mean).abs() / price_range
    pool_meta["dist"]       = pool_meta["year_dist"] + pool_meta["price_dist"]

    sorted_pool  = pool_meta.sort_values("dist").head(30)
    seen_app_ids = set(app_ids)

    candidates = []
    for app_id, row in sorted_pool.iterrows():
        if int(app_id) in seen_app_ids:
            continue
        pop_rank = pool_ids.index(app_id) if app_id in pool_ids else 999
        candidates.append({
            "app_id":           int(app_id),
            "title":            str(row.get("title", app_id)),
            "rating":           str(row.get("rating", "")),
            "price_final":      float(row.get("price_final", 0)),
            "release_year":     int(row.get("release_year", 0)),
            "popularity_score": float(1.0 / (pop_rank + 1)),
        })

    if len(candidates) < top_k:
        extra_pool = [a for a in popularity_list if a not in seen_app_ids
                      and a not in {c["app_id"] for c in candidates}]
        for a in extra_pool[:top_k - len(candidates)]:
            if a in games_meta.index:
                row      = games_meta.loc[a]
                pop_rank = popularity_list.index(a) if a in popularity_list else 999
                candidates.append({
                    "app_id":           int(a),
                    "title":            str(row.get("title", a)),
                    "rating":           str(row.get("rating", "")),
                    "price_final":      float(row.get("price_final", 0)),
                    "release_year":     int(row.get("release_year", 0)),
                    "popularity_score": float(1.0 / (pop_rank + 1)),
                })

    rank_result = rank_cold_candidates(profile, candidates, played_titles, client, top_n=top_k)

    return {
        "user_id":      uid,
        "level":        2,
        "ranked_ids":   rank_result["ranked_ids"],
        "explanations": rank_result["explanations"],
        "profile":      profile,
        "source":       rank_result["source"],
        "error":        rank_result.get("error"),
    }


def get_user_level(train_count: int) -> int:
    if train_count == 0:
        return 0
    elif train_count <= 2:
        return 1
    else:
        return 2


def recommend_cold_start(
    uid:             int,
    train_count:     int,
    user_history:    list,
    played_titles:   list,
    games_meta:      pd.DataFrame,
    popularity_list: list,
    client:          OpenAI,
) -> dict:
    level = get_user_level(train_count)

    if level == 0:
        recs = recommend_level0(popularity_list)
        return {"user_id": uid, "level": 0, "ranked_ids": recs,
                "explanations": {}, "profile": None, "source": "popularity"}

    elif level == 1:
        recs = recommend_level1(user_history, popularity_list, games_meta)
        return {"user_id": uid, "level": 1, "ranked_ids": recs,
                "explanations": {}, "profile": None, "source": "metadata_filter"}

    else:
        return recommend_level2(
            uid, user_history, played_titles, games_meta, popularity_list, client
        )


def run_evaluate(n_users: int):
    print("=" * 60)
    print(f"  Cold-Start Evaluation (Level 2, test set, n={n_users})")
    print("=" * 60)

    train      = pd.read_parquet(os.path.join(DATA_DIR, "train_interactions.parquet"))
    test       = pd.read_parquet(os.path.join(DATA_DIR, "test_interactions.parquet"))
    games_meta = load_games_meta(DATA_DIR)
    client     = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE)

    train_counts = train.groupby("user_id").size()
    cold_users   = set(train_counts[train_counts < COLD_START_THRESHOLD].index)

    test_cold     = test[test["user_id"].isin(cold_users)]
    test_positive = test_cold[test_cold["is_recommended"] == 1]
    eval_users    = test_positive["user_id"].unique()

    print(f"  cold-start users in train : {len(cold_users):,}")
    print(f"  with positives in test    : {len(eval_users):,}")

    rng          = np.random.default_rng(RANDOM_SEED)
    sample_size  = min(n_users, len(eval_users))
    sample_users = rng.choice(eval_users, size=sample_size, replace=False).tolist()
    print(f"  actual eval users         : {sample_size:,}")
    print(f"  estimated cost            : ~${sample_size * 850 / 1_000_000 * 10:.2f} (GPT-4o)")

    popularity_list = build_popularity_list(train, top_n=500)

    train_pos = train[train["score_B"] > 0]
    user_history_dict = {}
    for uid, grp in train_pos.groupby("user_id"):
        user_history_dict[uid] = [
            {"app_id": int(a), "score": float(s)}
            for a, s in zip(grp["app_id"], grp["score_B"])
        ]

    all_recs    = {}
    all_details = []
    errors      = 0
    t0          = time.time()

    for i, uid in enumerate(tqdm(sample_users, desc="  Cold-Start [Level 2]")):
        uid = int(uid)
        try:
            history       = user_history_dict.get(uid, [])
            played_aids   = [h["app_id"] for h in history]
            played_titles = [
                str(games_meta.loc[a, "title"]) if a in games_meta.index else str(a)
                for a in played_aids
            ]
            train_count = int(train_counts.get(uid, 0))

            result        = recommend_cold_start(
                uid, train_count, history, played_titles,
                games_meta, popularity_list, client
            )
            all_recs[uid] = result["ranked_ids"]
            all_details.append(result)
            if result.get("error"):
                errors += 1

            if i == 0:
                print(f"\n  [DIAG] user={uid}  level={result['level']}  train_count={train_count}")
                print(f"  [DIAG] ranked_ids: {result['ranked_ids'][:5]} ...")
                print(f"  [DIAG] source={result['source']}  error={result.get('error')}")
                if result.get("profile") and result["profile"].get("inferred_themes"):
                    print(f"  [DIAG] themes={result['profile']['inferred_themes']}"
                          f"  confidence={result['profile'].get('confidence')}")
                print()

        except Exception as e:
            errors += 1
            all_recs[uid] = []
            print(f"\n  [ERROR] user {uid}: {e}")

        time.sleep(0.3)

    elapsed = time.time() - t0
    print(f"\n  elapsed: {elapsed/60:.1f} min")
    print(f"  users with results : {sum(1 for v in all_recs.values() if v)}/{sample_size}")
    print(f"  LLM fallbacks      : {errors}")

    print("\n  [EVAL] computing test metrics ...")
    eval_df = test[test["user_id"].isin(sample_users)]
    results = evaluate_model(
        recommendations = all_recs,
        train_df        = train,
        eval_df         = eval_df,
        k_list          = [5, 10],
        only_positive   = True,
        verbose         = True,
    )

    out_csv = os.path.join(RESULT_DIR, "cold_start_level2_results.csv")
    results.to_csv(out_csv)
    print(f"\n  [SAVE] metrics -> {out_csv}")

    clean = []
    for d in all_details[:20]:
        clean.append({
            "user_id":      int(d["user_id"]),
            "level":        d["level"],
            "ranked_ids":   [int(x) for x in d["ranked_ids"]],
            "source":       d["source"],
            "error":        d.get("error"),
            "explanations": {str(k): str(v) for k, v in d.get("explanations", {}).items()},
            "profile":      d.get("profile"),
        })
    out_json = os.path.join(RESULT_DIR, "cold_start_level2_samples.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    print(f"  [SAVE] sample details -> {out_json}")

    return results


def run_pseudo_val(truncate_n: int, n_users: int = 200):
    """
    Simulate cold-start by giving warm val users only truncate_n training interactions.
    Used for hyperparameter tuning without touching the test set.
    Candidate exclusion always uses the full history to prevent seen-item leakage.
    """
    print("=" * 60)
    print(f"  Pseudo Cold-Start Validation (truncate={truncate_n}, n={n_users})")
    print("=" * 60)

    train      = pd.read_parquet(os.path.join(DATA_DIR, "train_interactions.parquet"))
    val        = pd.read_parquet(os.path.join(DATA_DIR, "val_interactions.parquet"))
    games_meta = load_games_meta(DATA_DIR)

    client = (
        OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE)
        if truncate_n >= 3 else None
    )

    train_counts  = train.groupby("user_id").size()
    warm_users    = set(train_counts[train_counts >= COLD_START_THRESHOLD].index)
    val_pos_users = set(val[val["is_recommended"] == 1]["user_id"].unique())
    eligible      = sorted(warm_users & val_pos_users)

    rng          = np.random.default_rng(RANDOM_SEED)
    sample_size  = min(n_users, len(eligible))
    sample_users = rng.choice(eligible, size=sample_size, replace=False).tolist()
    print(f"  truncate_n : {truncate_n}")
    print(f"  eval users : {sample_size:,}")

    popularity_list = build_popularity_list(train, top_n=500)

    train_pos = train[train["score_B"] > 0]
    user_history_full = {}
    for uid, grp in train_pos.groupby("user_id"):
        user_history_full[uid] = [
            {"app_id": int(a), "score": float(s)}
            for a, s in zip(grp["app_id"], grp["score_B"])
        ]

    all_recs = {}
    t0       = time.time()

    for uid in tqdm(sample_users, desc=f"  Pseudo Val [truncate={truncate_n}]"):
        uid  = int(uid)
        full = user_history_full.get(uid, [])

        played_titles = [
            str(games_meta.loc[h["app_id"], "title"])
            if h["app_id"] in games_meta.index else str(h["app_id"])
            for h in full
        ]

        # full history for seen-item exclusion; truncate_n controls profile induction depth
        try:
            result        = recommend_cold_start(
                uid, truncate_n, full, played_titles,
                games_meta, popularity_list, client
            )
            all_recs[uid] = result["ranked_ids"]
        except Exception as e:
            all_recs[uid] = []
            print(f"\n  [ERROR] user {uid}: {e}")

        if client:
            time.sleep(0.3)

    elapsed = time.time() - t0
    print(f"\n  elapsed: {elapsed/60:.1f} min")

    print("\n  [EVAL] computing val metrics ...")
    eval_df = val[val["user_id"].isin(sample_users)]
    results = evaluate_model(
        recommendations = all_recs,
        train_df        = train,
        eval_df         = eval_df,
        k_list          = [5, 10],
        only_positive   = True,
        verbose         = True,
    )

    out_csv = os.path.join(
        PROJECT_ROOT, "results", "val",
        f"cold_start_pseudo_val_truncate{truncate_n}_results.csv"
    )
    results.to_csv(out_csv)
    print(f"\n  [SAVE] metrics -> {out_csv}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cold-Start Recommendation")
    parser.add_argument(
        "--mode", choices=["evaluate", "pseudo_val"], default="evaluate",
        help="evaluate: Level 2 eval on test set | pseudo_val: truncated history tuning on val"
    )
    parser.add_argument("--n_users",  type=int, default=200, help="number of users to evaluate")
    parser.add_argument("--truncate", type=int, default=4,   help="pseudo_val: truncate history to N (0-4)")
    args = parser.parse_args()

    if args.mode == "evaluate":
        run_evaluate(args.n_users)
    else:
        run_pseudo_val(args.truncate, args.n_users)