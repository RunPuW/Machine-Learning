"""
pipeline.py  -  Main recommendation pipeline
Full chain: KNN + SVD candidate fusion -> UUA -> Ranker -> top-10 + explanations

Ablation modes (--mode):
  ml_only     : ItemKNN IDF top-10 direct output, no LLM
  ranker_only : fusion top-45 + raw history -> Ranker, skip UUA
  full        : KNN+SVD fusion + UUA + Ranker
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from uua_agent    import load_games_meta, build_uua_profile
from ranker_agent import rank_candidates
from evaluate     import evaluate_model, build_seen_items

PROJECT_ROOT = r"C:\Users\vipuser\Desktop\ml"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "cache")
RESULT_DIR   = os.path.join(PROJECT_ROOT, "results", "val")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_KEY_HERE")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1"
MODEL_NAME         = "openai/gpt-4o"

KNN_TOP_K       = 30
SVD_TOP_K       = 20
RANKER_TOP_N    = 10
KNN_SCORE_COL   = "score_B"
KNN_K_NEIGHBORS = 500
SVD_SCORE_COL   = "score_A"
SVD_N_FACTORS   = 100
RANDOM_SEED     = 42


def load_all_caches(train: pd.DataFrame):
    print("[CACHE] loading KNN cache ...")
    knn_cache = np.load(
        os.path.join(CACHE_DIR, f"itemknn_idf_neighbors_{KNN_SCORE_COL}.npz")
    )
    nb_idx = knn_cache["indices"]
    nb_sim = knn_cache["sims"]

    print("[CACHE] loading SVD cache ...")
    svd_cache    = np.load(
        os.path.join(CACHE_DIR, f"svd_factors_{SVD_SCORE_COL}_f{SVD_N_FACTORS}.npz")
    )
    user_factors = svd_cache["user_factors"]
    item_factors = svd_cache["item_factors"]

    pos_knn         = train[train[KNN_SCORE_COL] > 0]
    knn_items       = pos_knn["app_id"].unique()
    knn_item_to_idx = {v: i for i, v in enumerate(knn_items)}
    knn_idx_to_item = {i: v for v, i in knn_item_to_idx.items()}
    knn_user_items  = {}
    for uid, grp in pos_knn.groupby("user_id"):
        knn_user_items[uid] = dict(zip(grp["app_id"], grp[KNN_SCORE_COL]))

    pos_svd         = train[train[SVD_SCORE_COL] > 0]
    svd_items       = pos_svd["app_id"].unique()
    svd_item_to_idx = {v: i for i, v in enumerate(svd_items)}
    svd_idx_to_item = {i: v for v, i in svd_item_to_idx.items()}
    svd_user_to_idx = {u: i for i, u in enumerate(pos_svd["user_id"].unique())}
    svd_seen        = pos_svd.groupby("user_id")["app_id"].apply(set).to_dict()

    print(f"  KNN items={len(knn_items):,}  SVD items={len(svd_items):,}")

    return {
        "nb_idx":           nb_idx,
        "nb_sim":           nb_sim,
        "knn_item_to_idx":  knn_item_to_idx,
        "knn_idx_to_item":  knn_idx_to_item,
        "knn_user_items":   knn_user_items,
        "user_factors":     user_factors,
        "item_factors":     item_factors,
        "svd_item_to_idx":  svd_item_to_idx,
        "svd_idx_to_item":  svd_idx_to_item,
        "svd_user_to_idx":  svd_user_to_idx,
        "svd_seen":         svd_seen,
    }


def get_knn_candidates_with_scores(uid, caches, top_k=KNN_TOP_K):
    u_items = caches["knn_user_items"].get(uid, {})
    if not u_items:
        return []

    item_to_idx = caches["knn_item_to_idx"]
    idx_to_item = caches["knn_idx_to_item"]
    nb_idx      = caches["nb_idx"]
    nb_sim      = caches["nb_sim"]

    candidate_scores = {}
    for app_id, user_score in u_items.items():
        if app_id not in item_to_idx:
            continue
        item_idx  = item_to_idx[app_id]
        neighbors = nb_idx[item_idx, :KNN_K_NEIGHBORS]
        sims      = nb_sim[item_idx, :KNN_K_NEIGHBORS]
        for n_idx, sim in zip(neighbors, sims):
            if sim <= 0:
                break
            n_app_id = int(idx_to_item[int(n_idx)])
            candidate_scores[n_app_id] = (
                candidate_scores.get(n_app_id, 0.0) + float(user_score) * float(sim)
            )

    seen   = set(u_items.keys())
    result = [(aid, s) for aid, s in candidate_scores.items() if aid not in seen]
    result.sort(key=lambda x: -x[1])
    return result[:top_k]


def get_svd_candidates_with_scores(uid, caches, top_k=SVD_TOP_K):
    svd_user_to_idx = caches["svd_user_to_idx"]
    if uid not in svd_user_to_idx:
        return []

    uidx         = svd_user_to_idx[uid]
    user_factors = caches["user_factors"]
    item_factors = caches["item_factors"]
    idx_to_item  = caches["svd_idx_to_item"]
    item_to_idx  = caches["svd_item_to_idx"]
    seen         = caches["svd_seen"].get(uid, set())

    scores      = user_factors[uidx] @ item_factors.T
    seen_idxs   = [item_to_idx[a] for a in seen if a in item_to_idx]
    scores_copy = scores.copy()
    if seen_idxs:
        scores_copy[seen_idxs] = -np.inf

    top_idx = np.argpartition(scores_copy, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(-scores_copy[top_idx])]
    return [(int(idx_to_item[int(i)]), float(scores_copy[i])) for i in top_idx]


def fuse_candidates(
    knn_cands:  list,
    svd_cands:  list,
    games_meta: pd.DataFrame,
) -> list[dict]:
    """
    Merge KNN and SVD candidates with game metadata.
    knn_score is normalized to [0,1] and passed to Ranker.
    SVD scores are not passed (different scale). Source field indicates origin.
    Result is sorted by knn_score descending (SVD-only items rank last).
    """
    knn_dict = {aid: score for aid, score in knn_cands}
    svd_set  = {aid for aid, _ in svd_cands}

    if knn_dict:
        max_s    = max(knn_dict.values())
        min_s    = min(knn_dict.values())
        denom    = max_s - min_s if max_s > min_s else 1.0
        knn_norm = {aid: (s - min_s) / denom for aid, s in knn_dict.items()}
    else:
        knn_norm = {}

    all_ids = set(knn_dict.keys()) | svd_set
    result  = []
    for app_id in all_ids:
        in_knn = app_id in knn_dict
        in_svd = app_id in svd_set
        source = "both" if (in_knn and in_svd) else ("knn" if in_knn else "svd")

        if app_id in games_meta.index:
            row  = games_meta.loc[app_id]
            meta = {
                "title":          str(row.get("title", app_id)),
                "rating":         str(row.get("rating", "")),
                "positive_ratio": float(row.get("positive_ratio", 0)),
                "price_final":    float(row.get("price_final", 0)),
                "release_year":   int(row.get("release_year", 0)),
            }
        else:
            meta = {
                "title":          str(app_id),
                "rating":         "unknown",
                "positive_ratio": 0.0,
                "price_final":    0.0,
                "release_year":   0,
            }

        result.append({
            "app_id":    int(app_id),
            "knn_score": round(knn_norm.get(app_id, 0.0), 4),
            "source":    source,
            **meta,
        })

    result.sort(key=lambda x: (-x["knn_score"], x["source"] != "knn"))
    return result


def process_single_user(
    uid:        int,
    caches:     dict,
    games_meta: pd.DataFrame,
    client:     OpenAI,
    train:      pd.DataFrame,
    mode:       str = "full",
) -> dict:
    pos_train = train[
        (train["user_id"] == uid) & (train[KNN_SCORE_COL] > 0)
    ][["app_id", KNN_SCORE_COL, "hours"]].rename(columns={KNN_SCORE_COL: "score"})

    user_history   = pos_train.to_dict("records")
    played_app_ids = pos_train["app_id"].tolist()
    played_titles  = [
        str(games_meta.loc[a, "title"]) if a in games_meta.index else str(a)
        for a in played_app_ids
    ]

    if mode == "ml_only":
        knn_cands = get_knn_candidates_with_scores(uid, caches, top_k=RANKER_TOP_N)
        return {
            "user_id":      uid,
            "ranked_ids":   [aid for aid, _ in knn_cands],
            "source":       "ml_only",
            "uua":          None,
            "explanations": {},
        }

    knn_cands  = get_knn_candidates_with_scores(uid, caches, top_k=KNN_TOP_K)
    svd_cands  = get_svd_candidates_with_scores(uid, caches, top_k=SVD_TOP_K)
    candidates = fuse_candidates(knn_cands, svd_cands, games_meta)

    if not candidates:
        return {"user_id": uid, "ranked_ids": [], "source": "no_candidates",
                "uua": None, "explanations": {}}

    if mode == "full":
        uua_profile = build_uua_profile(
            user_id       = uid,
            user_history  = user_history,
            games_meta    = games_meta,
            client        = client,
            skip_semantic = False,
        )
    else:  # ranker_only: skip semantic UUA, use data_profile only
        uua_profile = build_uua_profile(
            user_id       = uid,
            user_history  = user_history,
            games_meta    = games_meta,
            client        = None,
            skip_semantic = True,
        )

    rank_result = rank_candidates(
        uua_profile   = uua_profile,
        candidates    = candidates,
        played_titles = played_titles,
        client        = client,
        top_n         = RANKER_TOP_N,
    )

    return {
        "user_id":      uid,
        "ranked_ids":   rank_result["ranked_ids"],
        "source":       rank_result["source"],
        "uua":          uua_profile,
        "explanations": rank_result["explanations"],
        "llm_error":    rank_result.get("error"),
    }


def run_pipeline(mode: str, n_users: int):
    print("=" * 60)
    print(f"  Pipeline mode={mode}  n_users={n_users}")
    print("=" * 60)

    print("[LOAD] loading train / val ...")
    train = pd.read_parquet(os.path.join(DATA_DIR, "train_interactions.parquet"))
    val   = pd.read_parquet(os.path.join(DATA_DIR, "val_interactions.parquet"))
    print(f"  train: {len(train):,}  val: {len(val):,}")

    games_meta = load_games_meta(DATA_DIR)
    print(f"  games_meta: {len(games_meta):,} rows")

    caches = load_all_caches(train)
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE) if mode != "ml_only" else None

    val_positive       = val[val["is_recommended"] == 1]
    users_with_gt      = set(val_positive["user_id"].unique())
    users_with_history = set(caches["knn_user_items"].keys())
    valid_users        = sorted(users_with_gt & users_with_history)

    print(f"\n  val users with positives    : {len(users_with_gt):,}")
    print(f"  of which have train history : {len(valid_users):,}")

    if not valid_users:
        raise RuntimeError("[ERROR] no users satisfy both conditions, check data path and score_col")

    rng          = np.random.default_rng(RANDOM_SEED)
    sample_size  = min(n_users, len(valid_users))
    sample_users = rng.choice(valid_users, size=sample_size, replace=False).tolist()
    print(f"  actual eval users           : {sample_size:,}")

    if mode != "ml_only":
        est_cost = sample_size * 1750 / 1_000_000 * (5 + 15) / 2
        print(f"  estimated cost: ~${est_cost:.2f} (GPT-4o)")

    all_recs    = {}
    all_details = []
    errors      = 0
    t0          = time.time()
    delay       = 0.5 if mode != "ml_only" else 0.0

    for i, uid in enumerate(tqdm(sample_users, desc=f"  Pipeline [{mode}]")):
        try:
            result        = process_single_user(uid, caches, games_meta, client, train, mode=mode)
            all_recs[uid] = result["ranked_ids"]
            all_details.append(result)
            if result.get("llm_error"):
                errors += 1

            if i == 0:
                print(f"\n  [DIAG] user={uid}")
                print(f"  [DIAG] ranked_ids ({len(result['ranked_ids'])}): {result['ranked_ids'][:5]} ...")
                print(f"  [DIAG] source={result['source']}  llm_error={result.get('llm_error')}")
                if result.get("explanations"):
                    first_id = list(result["explanations"].keys())[0]
                    print(f"  [DIAG] explanation[{first_id}]: "
                          f"{list(result['explanations'].values())[0][:120]}")
                if result.get("uua") and result["uua"].get("semantic_profile"):
                    sp = result["uua"]["semantic_profile"]
                    print(f"  [DIAG] UUA themes={sp.get('inferred_themes')}  "
                          f"confidence={sp.get('confidence')}")
                print()

        except Exception as e:
            errors += 1
            all_recs[uid] = []
            print(f"\n  [ERROR] user {uid}: {e}")

        if delay > 0:
            time.sleep(delay)

    elapsed   = time.time() - t0
    non_empty = sum(1 for v in all_recs.values() if v)
    print(f"\n  elapsed: {elapsed/60:.1f} min")
    print(f"  users with results : {non_empty}/{sample_size}")
    print(f"  LLM fallbacks      : {errors}")

    print("\n[EVAL] computing metrics ...")
    results = evaluate_model(
        recommendations = all_recs,
        train_df        = train,
        eval_df         = val[val["user_id"].isin(sample_users)],
        k_list          = [5, 10],
        only_positive   = True,
        verbose         = True,
    )

    out_csv = os.path.join(RESULT_DIR, f"pipeline_{mode}_results.csv")
    results.to_csv(out_csv)
    print(f"\n  [SAVE] metrics -> {out_csv}")

    clean_details = []
    for d in all_details[:20]:
        clean_details.append({
            "user_id":               int(d["user_id"]),
            "ranked_ids":            [int(x) for x in d["ranked_ids"]],
            "source":                d["source"],
            "llm_error":             d.get("llm_error"),
            "explanations":          {str(k): str(v) for k, v in d.get("explanations", {}).items()},
            "uua_semantic":          d.get("uua", {}).get("semantic_profile", {}) if d.get("uua") else {},
            "uua_data_price_median": (
                float(d["uua"]["data_profile"]["price"]["median"])
                if d.get("uua") and d["uua"].get("data_profile", {}).get("price", {}).get("median") is not None
                else None
            ),
        })

    out_json = os.path.join(RESULT_DIR, f"pipeline_{mode}_samples.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(clean_details, f, ensure_ascii=False, indent=2)
    print(f"  [SAVE] sample details -> {out_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommendation Pipeline Evaluation")
    parser.add_argument(
        "--mode", choices=["ml_only", "ranker_only", "full"], default="full",
        help="ml_only | ranker_only | full"
    )
    parser.add_argument("--n_users", type=int, default=100, help="number of users to evaluate")
    args = parser.parse_args()
    run_pipeline(args.mode, args.n_users)