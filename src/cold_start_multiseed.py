"""
cold_start_multiseed.py  -  Cold-Start Multi-Seed Experiment
=============================================================
Samples 500 users per seed across 3 random seeds, runs Level 2 and Popularity,
and reports mean/std across seeds as an uncertainty estimate.

Cost estimate: 3 x 500 x $0.0085 ~= $12.75 (GPT-4o)
Reduce N_USERS_PER_SEED to 200 to cut cost to ~$5.

Output: results/test/cold_start_multiseed_results.csv
"""

import os, sys, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import evaluate_model
from uua_agent import load_games_meta
from cold_start import (
    build_popularity_list, recommend_cold_start,
    COLD_START_THRESHOLD,
)

PROJECT_ROOT = r"C:\Users\vipuser\Desktop\ml"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
RESULT_DIR   = os.path.join(PROJECT_ROOT, "results", "test")
os.makedirs(RESULT_DIR, exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_KEY_HERE")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1"
MODEL_NAME         = "openai/gpt-4o"

# seed=42 already has 500-user results; two new seeds for stability check
SEEDS            = [123, 777]
N_USERS_PER_SEED = 150          # 2 x 150 x $0.0085 ~= $2.55


def run_one_seed(seed, train, test, games_meta, popularity_list,
                 train_counts, eval_users, client):
    rng          = np.random.default_rng(seed)
    sample_size  = min(N_USERS_PER_SEED, len(eval_users))
    sample_users = rng.choice(eval_users, size=sample_size, replace=False).tolist()

    train_pos = train[train["score_B"] > 0]
    user_history_dict = {}
    for uid, grp in train_pos.groupby("user_id"):
        user_history_dict[uid] = [
            {"app_id": int(a), "score": float(s)}
            for a, s in zip(grp["app_id"], grp["score_B"])
        ]

    all_recs_llm = {}
    errors = 0
    for uid in tqdm(sample_users, desc=f"  [seed={seed}] Level2"):
        uid = int(uid)
        try:
            history       = user_history_dict.get(uid, [])
            played_titles = [
                str(games_meta.loc[a, "title"]) if a in games_meta.index else str(a)
                for a in [h["app_id"] for h in history]
            ]
            result = recommend_cold_start(
                uid, int(train_counts.get(uid, 0)),
                history, played_titles,
                games_meta, popularity_list, client
            )
            all_recs_llm[uid] = result["ranked_ids"]
            if result.get("error"):
                errors += 1
        except Exception:
            all_recs_llm[uid] = []
            errors += 1
        time.sleep(0.3)

    all_recs_pop = {int(uid): [int(x) for x in popularity_list[:10]]
                    for uid in sample_users}

    eval_df = test[test["user_id"].isin(sample_users)]

    res_llm = evaluate_model(
        recommendations=all_recs_llm, train_df=train, eval_df=eval_df,
        k_list=[10], only_positive=True, verbose=False,
    )
    res_pop = evaluate_model(
        recommendations=all_recs_pop, train_df=train, eval_df=eval_df,
        k_list=[10], only_positive=True, verbose=False,
    )

    r_llm = res_llm.loc[10].to_dict()
    r_pop = res_pop.loc[10].to_dict()

    return {
        "seed":         seed,
        "n_users":      sample_size,
        "llm_fallback": errors,
        "llm_ndcg10":   r_llm.get("ndcg@10",   float("nan")),
        "llm_recall10": r_llm.get("recall@10", float("nan")),
        "llm_hit10":    r_llm.get("hit@10",    float("nan")),
        "pop_ndcg10":   r_pop.get("ndcg@10",   float("nan")),
        "pop_recall10": r_pop.get("recall@10", float("nan")),
        "pop_hit10":    r_pop.get("hit@10",    float("nan")),
    }


def main():
    cost = len(SEEDS) * N_USERS_PER_SEED * 850 / 1e6 * 10
    print("=" * 65)
    print(f"  Cold-Start Multi-Seed Experiment  seeds={SEEDS}  n={N_USERS_PER_SEED}/seed")
    print(f"  estimated cost: ~${cost:.2f} (GPT-4o)")
    print("=" * 65)

    train      = pd.read_parquet(os.path.join(DATA_DIR, "train_interactions.parquet"))
    test       = pd.read_parquet(os.path.join(DATA_DIR, "test_interactions.parquet"))
    games_meta = load_games_meta(DATA_DIR)
    client     = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE)

    train_counts    = train.groupby("user_id").size()
    cold_users      = set(train_counts[train_counts < COLD_START_THRESHOLD].index)
    test_cold       = test[test["user_id"].isin(cold_users)]
    eval_users      = test_cold[test_cold["is_recommended"] == 1]["user_id"].unique()
    popularity_list = build_popularity_list(train, top_n=500)

    all_rows = []
    for seed in SEEDS:
        print(f"\n  === seed={seed} ===")
        row = run_one_seed(seed, train, test, games_meta, popularity_list,
                           train_counts, eval_users, client)
        all_rows.append(row)
        print(f"  Level2 NDCG@10={row['llm_ndcg10']:.4f}  "
              f"Pop NDCG@10={row['pop_ndcg10']:.4f}  "
              f"fallbacks={row['llm_fallback']}")

    df       = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULT_DIR, "cold_start_multiseed_results.csv")
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 65)
    print("  Multi-Seed Summary")
    print("=" * 65)
    for metric in ["ndcg10", "recall10", "hit10"]:
        llm_vals = df[f"llm_{metric}"].values
        pop_vals = df[f"pop_{metric}"].values
        consistent = "consistent" if all(l > p for l, p in zip(llm_vals, pop_vals)) else "inconsistent across seeds"
        print(f"  {metric}:")
        print(f"    Level2  mean={llm_vals.mean():.4f}  std={llm_vals.std():.4f}  values={[round(v, 4) for v in llm_vals]}")
        print(f"    Pop     mean={pop_vals.mean():.4f}  std={pop_vals.std():.4f}  values={[round(v, 4) for v in pop_vals]}")
        print(f"    delta   mean={llm_vals.mean() - pop_vals.mean():+.4f}  ({consistent})")

    print(f"\n  [SAVE] -> {out_path}")


if __name__ == "__main__":
    main()