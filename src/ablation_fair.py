"""
ablation_fair.py  -  Fair ablation experiment (four groups, same user cohort)
==============================================================================
Evaluates four configurations on the same 50 users with strict variable isolation:

  A. ml_only_knn   : KNN IDF top-10 direct output (no SVD, no LLM)
  B. fusion_no_llm : KNN+SVD fusion ranked by knn_score (no LLM)
  C. fusion_ranker : fusion + Ranker (no UUA)
  D. fusion_full   : fusion + UUA + Ranker

Variable isolation:
  A vs B : contribution of SVD fusion (no LLM involved, clean signal)
  B vs C : contribution of Ranker (same candidate pool)
  C vs D : contribution of UUA (same Ranker)
"""

import os, sys, time, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import evaluate_model
from uua_agent import load_games_meta, build_uua_profile
from ranker_agent import rank_candidates
from pipeline import (
    load_all_caches,
    get_knn_candidates_with_scores,
    get_svd_candidates_with_scores,
    fuse_candidates,
    KNN_TOP_K, SVD_TOP_K, RANKER_TOP_N,
)

PROJECT_ROOT = r"C:\Users\vipuser\Desktop\ml"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
RESULT_DIR   = os.path.join(PROJECT_ROOT, "results", "val")
os.makedirs(RESULT_DIR, exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_KEY_HERE")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1"

# fixed for all four groups -- do not change per-group
LLM_CONFIG = {
    "model":       "openai/gpt-4o",
    "temperature": 0.0,
    "max_tokens":  600,
}

N_USERS     = 50    # budget: 50 users x 2 LLM groups x ~1750 tokens ~= $1.75
RANDOM_SEED = 42

COST_PER_1K_INPUT  = 0.005   # $5 / 1M input tokens
COST_PER_1K_OUTPUT = 0.015   # $15 / 1M output tokens
EST_INPUT_TOKENS   = 1250
EST_OUTPUT_TOKENS  = 500

cost_log = {
    "llm_calls":      0,
    "est_input_tok":  0,
    "est_output_tok": 0,
    "est_cost_usd":   0.0,
    "fallbacks":      0,
}


def track_call(n_calls=1):
    cost_log["llm_calls"]      += n_calls
    cost_log["est_input_tok"]  += n_calls * EST_INPUT_TOKENS
    cost_log["est_output_tok"] += n_calls * EST_OUTPUT_TOKENS
    cost_log["est_cost_usd"]   += n_calls * (
        EST_INPUT_TOKENS  / 1000 * COST_PER_1K_INPUT +
        EST_OUTPUT_TOKENS / 1000 * COST_PER_1K_OUTPUT
    )


def save_cost_log():
    path = os.path.join(RESULT_DIR, "ablation_fair_cost_log.txt")
    with open(path, "w") as f:
        f.write("ablation_fair.py cost log\n")
        f.write("=" * 40 + "\n")
        f.write(f"total LLM calls     : {cost_log['llm_calls']}\n")
        f.write(f"fallbacks           : {cost_log['fallbacks']}\n")
        f.write(f"est input tokens    : {cost_log['est_input_tok']:,}\n")
        f.write(f"est output tokens   : {cost_log['est_output_tok']:,}\n")
        f.write(f"est cost (USD)      : ${cost_log['est_cost_usd']:.4f}\n")
        f.write("\nnote: token counts are estimates (input ~1250, output ~500 per call)\n")
        f.write(f"model       : {LLM_CONFIG['model']}\n")
        f.write(f"temperature : {LLM_CONFIG['temperature']}\n")
    print(f"  [SAVE] cost log -> {path}")


def get_sample_users(val, caches, n=N_USERS):
    val_positive       = val[val["is_recommended"] == 1]
    users_with_gt      = set(val_positive["user_id"].unique())
    users_with_history = set(caches["knn_user_items"].keys())
    valid_users        = sorted(users_with_gt & users_with_history)
    rng                = np.random.default_rng(RANDOM_SEED)
    return rng.choice(valid_users, size=min(n, len(valid_users)), replace=False).tolist()


def eval_recs(all_recs, train, val, sample_users):
    return evaluate_model(
        recommendations = all_recs,
        train_df        = train,
        eval_df         = val[val["user_id"].isin(sample_users)],
        k_list          = [5, 10],
        only_positive   = True,
        verbose         = False,
    )


def run_group_A(sample_users, caches, train, val):
    all_recs = {}
    for uid in tqdm(sample_users, desc="  [A] ml_only_knn   "):
        uid = int(uid)
        knn = get_knn_candidates_with_scores(uid, caches, top_k=RANKER_TOP_N)
        all_recs[uid] = [int(aid) for aid, _ in knn]
    return eval_recs(all_recs, train, val, sample_users), all_recs


def run_group_B(sample_users, caches, games_meta, train, val):
    all_recs = {}
    for uid in tqdm(sample_users, desc="  [B] fusion_no_llm "):
        uid        = int(uid)
        knn_cands  = get_knn_candidates_with_scores(uid, caches, top_k=KNN_TOP_K)
        svd_cands  = get_svd_candidates_with_scores(uid, caches, top_k=SVD_TOP_K)
        candidates = fuse_candidates(knn_cands, svd_cands, games_meta)
        seen       = set(caches["knn_user_items"].get(uid, {}).keys())
        all_recs[uid] = [int(c["app_id"]) for c in candidates
                         if int(c["app_id"]) not in seen][:RANKER_TOP_N]
    return eval_recs(all_recs, train, val, sample_users), all_recs


def run_group_C(sample_users, caches, games_meta, train, val, client):
    all_recs    = {}
    raw_outputs = []
    errors      = 0

    for uid in tqdm(sample_users, desc="  [C] fusion_ranker "):
        uid        = int(uid)
        knn_cands  = get_knn_candidates_with_scores(uid, caches, top_k=KNN_TOP_K)
        svd_cands  = get_svd_candidates_with_scores(uid, caches, top_k=SVD_TOP_K)
        candidates = fuse_candidates(knn_cands, svd_cands, games_meta)

        pos_train     = train[(train["user_id"] == uid) & (train["score_B"] > 0)]
        played_titles = [
            str(games_meta.loc[int(a), "title"]) if int(a) in games_meta.index else str(a)
            for a in pos_train["app_id"].tolist()
        ]

        # empty UUA so Ranker only sees candidate metadata and knn_score
        empty_uua = {"data_profile": {}, "semantic_profile": {"skipped": True,
                                                               "reason": "group_C_no_uua"}}
        result = rank_candidates(empty_uua, candidates, played_titles,
                                 client, top_n=RANKER_TOP_N)

        all_recs[uid] = [int(x) for x in result["ranked_ids"]]
        raw_outputs.append({
            "user_id":      uid,
            "ranked_ids":   [int(x) for x in result["ranked_ids"]],
            "explanations": {str(k): str(v) for k, v in result.get("explanations", {}).items()},
            "source":       result["source"],
            "error":        result.get("error"),
        })

        if result.get("error"):
            errors += 1
            cost_log["fallbacks"] += 1
        track_call(n_calls=1)
        time.sleep(0.3)

    print(f"  [C] fallbacks: {errors}/{len(sample_users)}")
    return eval_recs(all_recs, train, val, sample_users), all_recs, raw_outputs


def run_group_D(sample_users, caches, games_meta, train, val, client):
    all_recs    = {}
    raw_outputs = []
    errors      = 0

    for uid in tqdm(sample_users, desc="  [D] fusion_full   "):
        uid        = int(uid)
        knn_cands  = get_knn_candidates_with_scores(uid, caches, top_k=KNN_TOP_K)
        svd_cands  = get_svd_candidates_with_scores(uid, caches, top_k=SVD_TOP_K)
        candidates = fuse_candidates(knn_cands, svd_cands, games_meta)

        pos_train    = train[(train["user_id"] == uid) & (train["score_B"] > 0)]
        user_history = [
            {"app_id": int(a), "score": float(s), "hours": float(h)}
            for a, s, h in zip(pos_train["app_id"], pos_train["score_B"], pos_train["hours"])
        ]
        played_titles = [
            str(games_meta.loc[int(a), "title"]) if int(a) in games_meta.index else str(a)
            for a in pos_train["app_id"].tolist()
        ]

        uua_profile = build_uua_profile(uid, user_history, games_meta,
                                        client=client, skip_semantic=False)
        track_call(n_calls=1)   # UUA call

        result = rank_candidates(uua_profile, candidates, played_titles,
                                 client, top_n=RANKER_TOP_N)
        track_call(n_calls=1)   # Ranker call

        all_recs[uid] = [int(x) for x in result["ranked_ids"]]
        raw_outputs.append({
            "user_id":        uid,
            "ranked_ids":     [int(x) for x in result["ranked_ids"]],
            "explanations":   {str(k): str(v) for k, v in result.get("explanations", {}).items()},
            "source":         result["source"],
            "error":          result.get("error"),
            "uua_themes":     uua_profile.get("semantic_profile", {}).get("inferred_themes"),
            "uua_confidence": uua_profile.get("semantic_profile", {}).get("confidence"),
        })

        if result.get("error"):
            errors += 1
            cost_log["fallbacks"] += 1
        time.sleep(0.3)

    print(f"  [D] fallbacks: {errors}/{len(sample_users)}")
    return eval_recs(all_recs, train, val, sample_users), all_recs, raw_outputs


def main():
    est_cost = (
        N_USERS * 3 * EST_INPUT_TOKENS  / 1000 * COST_PER_1K_INPUT +
        N_USERS * 3 * EST_OUTPUT_TOKENS / 1000 * COST_PER_1K_OUTPUT
    )
    print("=" * 65)
    print("  Fair Ablation Experiment (4 groups, same user cohort)")
    print(f"  n_users={N_USERS}  seed={RANDOM_SEED}")
    print(f"  model={LLM_CONFIG['model']}  temperature={LLM_CONFIG['temperature']}")
    print(f"  estimated cost (C+D): ~${est_cost:.2f}")
    print("=" * 65)

    train      = pd.read_parquet(os.path.join(DATA_DIR, "train_interactions.parquet"))
    val        = pd.read_parquet(os.path.join(DATA_DIR, "val_interactions.parquet"))
    games_meta = load_games_meta(DATA_DIR)
    caches     = load_all_caches(train)
    client     = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE)

    sample_users = get_sample_users(val, caches, n=N_USERS)
    users_path   = os.path.join(RESULT_DIR, "ablation_fair_sample_users.json")
    with open(users_path, "w") as f:
        json.dump({"seed": RANDOM_SEED, "n_users": len(sample_users),
                   "user_ids": [int(u) for u in sample_users]}, f, indent=2)
    print(f"  user IDs saved -> {users_path}")
    print(f"  actual eval users: {len(sample_users)}\n")

    res_A, recs_A                = run_group_A(sample_users, caches, train, val)
    res_B, recs_B                = run_group_B(sample_users, caches, games_meta, train, val)
    res_C, recs_C, raw_C         = run_group_C(sample_users, caches, games_meta, train, val, client)
    res_D, recs_D, raw_D         = run_group_D(sample_users, caches, games_meta, train, val, client)

    for name, raw in [("C", raw_C), ("D", raw_D)]:
        raw_path = os.path.join(RESULT_DIR, f"ablation_fair_raw_{name}.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        print(f"  raw output saved -> {raw_path}")

    groups = {
        "A_ml_only_knn":   res_A,
        "B_fusion_no_llm": res_B,
        "C_fusion_ranker":  res_C,
        "D_fusion_full":    res_D,
    }

    print("\n" + "=" * 65)
    print(f"  Ablation Results (K=10, n={len(sample_users)} users)")
    print("=" * 65)
    print(f"  {'Group':<22} {'NDCG@10':>9} {'Recall@10':>10} {'Hit@10':>8} {'Prec@10':>9}")
    print(f"  {'-'*60}")

    summary_rows = []
    ndcg_vals    = {}
    for name, df in groups.items():
        row    = df.loc[10].to_dict()
        ndcg   = row.get("ndcg@10",      float("nan"))
        recall = row.get("recall@10",    float("nan"))
        hit    = row.get("hit@10",       float("nan"))
        prec   = row.get("precision@10", float("nan"))
        print(f"  {name:<22} {ndcg:>9.4f} {recall:>10.4f} {hit:>8.4f} {prec:>9.4f}")
        summary_rows.append({
            "group": name, "n_users": len(sample_users),
            "ndcg@10": ndcg, "recall@10": recall,
            "hit@10": hit, "precision@10": prec,
        })
        ndcg_vals[name] = ndcg

    def delta(a, b):
        va, vb = ndcg_vals.get(a, float("nan")), ndcg_vals.get(b, float("nan"))
        return f"{(vb - va) / va * 100:+.1f}%" if va > 0 else "N/A"

    print("\n  Variable decomposition (NDCG@10):")
    print(f"  SVD fusion  (A->B): {delta('A_ml_only_knn',  'B_fusion_no_llm')}")
    print(f"  Ranker      (B->C): {delta('B_fusion_no_llm', 'C_fusion_ranker')}")
    print(f"  UUA         (C->D): {delta('C_fusion_ranker', 'D_fusion_full')}")

    out_csv = os.path.join(RESULT_DIR, "ablation_fair_results.csv")
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\n  [SAVE] summary -> {out_csv}")

    save_cost_log()
    print(f"\n  estimated cost : ~${cost_log['est_cost_usd']:.4f}")
    print(f"  LLM calls      : {cost_log['llm_calls']}  fallbacks: {cost_log['fallbacks']}")


if __name__ == "__main__":
    main()