"""
svd_knn_overlap.py  -  SVD vs KNN candidate complementarity analysis
Answers: overlap rate, union recall gain, and SVD-unique hit share.
Decision threshold: if overlap > 50% AND union gain < 10% AND SVD unique hits < 5% -> skip fusion.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

PROJECT_ROOT = r"C:\Users\vipuser\Desktop\ml"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "cache")
RESULT_DIR   = os.path.join(PROJECT_ROOT, "results", "val")

SAMPLE_USERS  = 10000
TOP_K         = 20
KNN_SCORE_COL = "score_B"
KNN_K         = 500
SVD_N_FACTORS = 100
SVD_SCORE_COL = "score_A"
RANDOM_SEED   = 42

# module-level SVD index (set in run_overlap_analysis, used in get_svd_candidates)
item_to_idx_svd_global     = {}
idx_to_item_svd_global_inv = {}


def load_data():
    print("[LOAD] loading data ...")
    train = pd.read_parquet(os.path.join(DATA_DIR, "train_interactions.parquet"))
    val   = pd.read_parquet(os.path.join(DATA_DIR, "val_interactions.parquet"))
    print(f"  train: {len(train):,}  val: {len(val):,}")
    return train, val


def load_knn_cache():
    path = os.path.join(CACHE_DIR, f"itemknn_idf_neighbors_{KNN_SCORE_COL}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"KNN cache not found: {path}\nRun itemknn_idf.py first.")
    print(f"  [CACHE] KNN neighbors: {path}")
    c = np.load(path)
    return c["indices"], c["sims"]


def load_svd_cache():
    path = os.path.join(CACHE_DIR, f"svd_factors_{SVD_SCORE_COL}_f{SVD_N_FACTORS}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"SVD cache not found: {path}\nRun svd_batched.py first.")
    print(f"  [CACHE] SVD factors: {path}")
    c = np.load(path)
    return c["user_factors"], c["item_factors"]


def build_indices(train_df: pd.DataFrame, score_col: str):
    pos         = train_df[train_df[score_col] > 0]
    unique_items = pos["app_id"].unique()
    unique_users = pos["user_id"].unique()
    item_to_idx  = {v: i for i, v in enumerate(unique_items)}
    idx_to_item  = {i: v for v, i in item_to_idx.items()}
    user_to_idx  = {u: i for i, u in enumerate(unique_users)}
    return item_to_idx, idx_to_item, user_to_idx


def get_knn_candidates(uid, user_items_dict, item_to_idx, idx_to_item,
                       neighbor_indices, neighbor_sims, k_neighbors, top_n):
    u_items = user_items_dict.get(uid, {})
    if not u_items:
        return []

    candidate_scores = {}
    for app_id, user_score in u_items.items():
        if app_id not in item_to_idx:
            continue
        item_idx  = item_to_idx[app_id]
        neighbors = neighbor_indices[item_idx, :k_neighbors]
        sims      = neighbor_sims[item_idx, :k_neighbors]
        for n_idx, sim in zip(neighbors, sims):
            if sim <= 0:
                break
            n_app_id = idx_to_item[int(n_idx)]
            candidate_scores[n_app_id] = (
                candidate_scores.get(n_app_id, 0.0) + float(user_score) * float(sim)
            )

    seen       = set(u_items.keys())
    candidates = [(aid, s) for aid, s in candidate_scores.items() if aid not in seen]
    candidates.sort(key=lambda x: -x[1])
    return [aid for aid, _ in candidates[:top_n]]


def get_svd_candidates(uid, user_to_idx, idx_to_item_svd, seen_items,
                       user_factors, item_factors, top_n):
    if uid not in user_to_idx:
        return []

    uidx   = user_to_idx[uid]
    scores = user_factors[uidx] @ item_factors.T

    seen      = seen_items.get(uid, set())
    seen_idx  = [item_to_idx_svd_global.get(a) for a in seen
                 if a in item_to_idx_svd_global]
    seen_idx  = [i for i in seen_idx if i is not None]

    scores_copy = scores.copy()
    if seen_idx:
        scores_copy[seen_idx] = -np.inf

    top_idx = np.argpartition(scores_copy, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(-scores_copy[top_idx])]
    return [idx_to_item_svd_global_inv[i] for i in top_idx]


def run_overlap_analysis(train, val):
    global item_to_idx_svd_global, idx_to_item_svd_global_inv

    val_users_all = val["user_id"].unique()
    rng           = np.random.default_rng(RANDOM_SEED)
    sample_users  = (
        rng.choice(val_users_all, size=SAMPLE_USERS, replace=False).tolist()
        if len(val_users_all) > SAMPLE_USERS else val_users_all.tolist()
    )
    print(f"\n  sampled users: {len(sample_users):,}")

    gt = val[val["is_recommended"] == 1].groupby("user_id")["app_id"].apply(set).to_dict()

    print("\n[KNN] loading IDF KNN ...")
    item_to_idx_knn, idx_to_item_knn, _ = build_indices(train, KNN_SCORE_COL)
    nb_idx, nb_sim = load_knn_cache()

    pos_knn        = train[train[KNN_SCORE_COL] > 0]
    user_items_knn = {}
    for uid, grp in pos_knn.groupby("user_id"):
        user_items_knn[uid] = dict(zip(grp["app_id"], grp[KNN_SCORE_COL]))

    print("\n[SVD] loading SVD factors ...")
    item_to_idx_svd, idx_to_item_svd, user_to_idx_svd = build_indices(train, SVD_SCORE_COL)
    item_to_idx_svd_global     = item_to_idx_svd
    idx_to_item_svd_global_inv = idx_to_item_svd

    user_factors, item_factors = load_svd_cache()

    pos_svd  = train[train[SVD_SCORE_COL] > 0]
    seen_svd = pos_svd.groupby("user_id")["app_id"].apply(set).to_dict()

    print("\n[DIAG] running per-user analysis ...")
    t0      = time.time()
    rows    = []
    skipped = 0

    for uid in tqdm(sample_users, desc="  Overlap analysis"):
        gt_items = gt.get(uid, set())
        if not gt_items:
            skipped += 1
            continue

        knn_cands = set(get_knn_candidates(
            uid, user_items_knn, item_to_idx_knn, idx_to_item_knn,
            nb_idx, nb_sim, KNN_K, TOP_K
        ))

        if uid not in user_to_idx_svd:
            skipped += 1
            continue

        uidx        = user_to_idx_svd[uid]
        scores      = user_factors[uidx] @ item_factors.T
        seen        = seen_svd.get(uid, set())
        seen_idxs   = [item_to_idx_svd[a] for a in seen if a in item_to_idx_svd]
        scores_copy = scores.copy()
        if seen_idxs:
            scores_copy[seen_idxs] = -np.inf
        top_idx_svd = np.argpartition(scores_copy, -TOP_K)[-TOP_K:]
        top_idx_svd = top_idx_svd[np.argsort(-scores_copy[top_idx_svd])]
        svd_cands   = set(idx_to_item_svd[i] for i in top_idx_svd)

        union_cands     = knn_cands | svd_cands
        overlap         = len(knn_cands & svd_cands)
        knn_hits        = len(knn_cands   & gt_items)
        svd_hits        = len(svd_cands   & gt_items)
        union_hits      = len(union_cands & gt_items)
        svd_unique_hits = len((svd_cands - knn_cands) & gt_items)

        rows.append({
            "user_id":         uid,
            "gt_count":        len(gt_items),
            "knn_hits":        knn_hits,
            "svd_hits":        svd_hits,
            "union_hits":      union_hits,
            "svd_unique_hits": svd_unique_hits,
            "overlap":         overlap,
            "overlap_ratio":   overlap / TOP_K,
            "knn_recall":      knn_hits   / len(gt_items),
            "svd_recall":      svd_hits   / len(gt_items),
            "union_recall":    union_hits / len(gt_items),
        })

    print(f"  done in {time.time()-t0:.1f}s, skipped {skipped} invalid users")
    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame, out_txt: str):
    n = len(df)

    mean_overlap      = df["overlap_ratio"].mean()
    mean_knn_recall   = df["knn_recall"].mean()
    mean_svd_recall   = df["svd_recall"].mean()
    mean_union_recall = df["union_recall"].mean()
    recall_gain       = (mean_union_recall - mean_knn_recall) / (mean_knn_recall + 1e-9)

    total_knn_hits   = df["knn_hits"].sum()
    total_svd_hits   = df["svd_hits"].sum()
    total_union_hits = df["union_hits"].sum()
    svd_unique_hits  = df["svd_unique_hits"].sum()
    svd_unique_ratio = svd_unique_hits / (total_union_hits + 1e-9)

    p25 = df["overlap_ratio"].quantile(0.25)
    p50 = df["overlap_ratio"].quantile(0.50)
    p75 = df["overlap_ratio"].quantile(0.75)

    verdict_overlap = mean_overlap      > 0.5
    verdict_gain    = recall_gain       < 0.10
    verdict_unique  = svd_unique_ratio  < 0.05

    if verdict_overlap and verdict_gain and verdict_unique:
        verdict = "[no fusion] SVD candidates are highly redundant, negligible incremental value"
    elif not verdict_overlap or not verdict_gain:
        verdict = "[fuse] SVD candidates show significant incremental gain, proceed with fusion"
    else:
        verdict = "[borderline] run further experiments before deciding"

    lines = [
        "=" * 60,
        "  SVD vs KNN Candidate Complementarity Report",
        f"  KNN: IDF {KNN_SCORE_COL} K={KNN_K}   SVD: {SVD_SCORE_COL} n={SVD_N_FACTORS}",
        f"  eval users: {n:,}   Top-K: {TOP_K}",
        "=" * 60,
        "",
        "[A] Candidate overlap",
        f"  Overlap@{TOP_K} mean       : {mean_overlap:.3f}  ({mean_overlap*100:.1f}%)",
        f"  Overlap@{TOP_K} p25/p50/p75: {p25:.3f} / {p50:.3f} / {p75:.3f}",
        f"  -> avg {mean_overlap*TOP_K:.1f} / {TOP_K} candidates shared between models",
        "",
        "[B] Recall",
        f"  KNN   Recall@{TOP_K}: {mean_knn_recall:.4f}",
        f"  SVD   Recall@{TOP_K}: {mean_svd_recall:.4f}",
        f"  Union Recall@{TOP_K}: {mean_union_recall:.4f}",
        f"  Union vs KNN gain  : {recall_gain*100:+.1f}%",
        "",
        "[C] Hit counts (totals)",
        f"  KNN hits         : {total_knn_hits:,}",
        f"  SVD hits         : {total_svd_hits:,}",
        f"  union hits       : {total_union_hits:,}",
        f"  SVD unique hits  : {svd_unique_hits:,}  ({svd_unique_ratio*100:.1f}% of union hits)",
        "",
        "[D] Decision",
        f"  overlap > 50%          : {'yes' if verdict_overlap else 'no'}  ({mean_overlap*100:.1f}%)",
        f"  union recall gain < 10%: {'yes' if verdict_gain    else 'no'}  ({recall_gain*100:+.1f}%)",
        f"  SVD unique hits < 5%   : {'yes' if verdict_unique  else 'no'}  ({svd_unique_ratio*100:.1f}%)",
        f"\n  verdict: {verdict}",
        "=" * 60,
    ]

    report = "\n".join(lines)
    print(report)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  [SAVE] report -> {out_txt}")


def main():
    print("=" * 60)
    print("  SVD-KNN Candidate Complementarity Analysis")
    print("=" * 60)

    train, val = load_data()
    df         = run_overlap_analysis(train, val)

    out_csv = os.path.join(RESULT_DIR, "svd_knn_overlap_detail.csv")
    out_txt = os.path.join(RESULT_DIR, "svd_knn_overlap_report.txt")

    df.to_csv(out_csv, index=False)
    print(f"\n  [SAVE] detail -> {out_csv}")

    print_report(df, out_txt)


if __name__ == "__main__":
    main()