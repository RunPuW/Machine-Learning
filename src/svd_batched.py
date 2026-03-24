"""
svd_batched.py  -  Task 9: SVD (Matrix Factorization) Baseline
Truncated SVD via scipy.sparse.linalg.svds on score_A and score_B, comparing n_factors=50/100/200.
Recommendation uses batched GEMM in the main process; numpy MKL handles multi-threading automatically.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from tqdm import tqdm

from evaluate import evaluate_model

PROJECT_ROOT = r"C:\Users\vipuser\Desktop\ml"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "cache")
RESULT_DIR   = os.path.join(PROJECT_ROOT, "results", "val")

for _d in [CACHE_DIR, RESULT_DIR]:
    os.makedirs(_d, exist_ok=True)

N_FACTORS_LIST  = [50, 100, 200]
TOP_K_RECOMMEND = 20
REC_BATCH_SIZE  = 5000   # 5000 x 35325 x 4 ~= 707 MB/batch at n_factors=200


def load_splits():
    print("[LOAD] loading train / val ...")
    train = pd.read_parquet(os.path.join(DATA_DIR, "train_interactions.parquet"))
    val   = pd.read_parquet(os.path.join(DATA_DIR, "val_interactions.parquet"))
    print(f"  train: {len(train):,}  val: {len(val):,}")
    return train, val


def build_user_item_matrix(train_df: pd.DataFrame, score_col: str):
    pos = train_df[train_df[score_col] > 0][["user_id", "app_id", score_col]].copy()
    print(f"  {score_col} > 0: {len(pos):,} records")

    unique_users = pos["user_id"].unique()
    unique_items = pos["app_id"].unique()

    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    item_to_idx = {v: i for i, v in enumerate(unique_items)}
    idx_to_item = {i: v for v, i in item_to_idx.items()}

    row    = pos["user_id"].map(user_to_idx).values
    col    = pos["app_id"].map(item_to_idx).values
    data   = pos[score_col].values.astype(np.float32)
    matrix = csr_matrix(
        (data, (row, col)),
        shape=(len(unique_users), len(unique_items)),
        dtype=np.float32,
    )
    print(f"  user-item matrix: {matrix.shape}, nnz: {matrix.nnz:,}")
    return matrix, user_to_idx, item_to_idx, idx_to_item


def compute_svd(matrix, score_col: str, n_factors: int):
    cache_path = os.path.join(CACHE_DIR, f"svd_factors_{score_col}_f{n_factors}.npz")

    if os.path.exists(cache_path):
        print(f"  [CACHE HIT] {cache_path}")
        c = np.load(cache_path)
        return c["user_factors"], c["item_factors"]

    print(f"  [SVD] {score_col}, n_factors={n_factors}, matrix={matrix.shape}")
    t0 = time.time()

    U, s, Vt = svds(matrix.astype(np.float64), k=n_factors, which="LM")

    sqrt_s       = np.sqrt(s).astype(np.float32)
    user_factors = (U  * sqrt_s).astype(np.float32)
    item_factors = (Vt * sqrt_s[:, None]).T.astype(np.float32)

    print(f"  SVD done in {(time.time()-t0):.1f}s")
    print(f"  user_factors: {user_factors.shape}  item_factors: {item_factors.shape}")

    np.savez_compressed(cache_path, user_factors=user_factors, item_factors=item_factors)
    print(f"  [SAVE] factor cache -> {cache_path}")
    return user_factors, item_factors


def recommend_svd_batched(
    val_users:    list,
    user_to_idx:  dict,
    item_to_idx:  dict,
    idx_to_item:  dict,
    seen_items:   dict,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    top_n:        int = TOP_K_RECOMMEND,
    batch_size:   int = REC_BATCH_SIZE,
) -> dict:
    n_items = item_factors.shape[0]

    valid_pairs  = [(uid, user_to_idx[uid]) for uid in val_users if uid in user_to_idx]
    invalid_uids = [uid for uid in val_users if uid not in user_to_idx]
    print(f"  val users in train: {len(valid_pairs):,} / {len(val_users):,}")
    print(f"  no train history (empty recs): {len(invalid_uids):,}")

    recommendations = {uid: [] for uid in invalid_uids}

    print("  [PREP] preprocessing seen items ...")
    t0           = time.time()
    seen_idx_map = {}
    for uid, _ in valid_pairs:
        raw               = seen_items.get(uid, set())
        seen_idx_map[uid] = [item_to_idx[a] for a in raw if a in item_to_idx]
    print(f"  [PREP] done in {time.time()-t0:.1f}s")

    uids  = [p[0] for p in valid_pairs]
    uidxs = np.array([p[1] for p in valid_pairs], dtype=np.int32)

    n_batches = (len(uids) + batch_size - 1) // batch_size
    print(f"  [REC] {len(uids):,} users, batch={batch_size}, {n_batches} batches")
    print(f"  [REC] ~{batch_size * n_items * 4 / 1024**3:.2f} GB/batch "
          f"(n_items={n_items}, n_factors={item_factors.shape[1]})")

    t0 = time.time()
    for start in tqdm(range(0, len(uids), batch_size), desc="  Recs SVD"):
        end        = min(start + batch_size, len(uids))
        batch_uids = uids[start:end]
        batch_uidx = uidxs[start:end]

        batch_uf = user_factors[batch_uidx]
        scores   = (batch_uf @ item_factors.T).copy()

        for i, uid in enumerate(batch_uids):
            seen = seen_idx_map.get(uid)
            if seen:
                scores[i, seen] = -np.inf

        top_idx    = np.argpartition(scores, -top_n, axis=1)[:, -top_n:]
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        sort_ord   = np.argsort(-top_scores, axis=1)
        sorted_idx = np.take_along_axis(top_idx, sort_ord, axis=1)

        for i, uid in enumerate(batch_uids):
            recommendations[uid] = [idx_to_item[int(j)] for j in sorted_idx[i]]

    print(f"  [REC] done in {(time.time()-t0)/60:.1f} min")
    return recommendations


def run_score(score_col: str, train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  SVD - {score_col}")
    print(f"{'='*60}")

    matrix, user_to_idx, item_to_idx, idx_to_item = build_user_item_matrix(train, score_col)

    pos        = train[train[score_col] > 0]
    seen_items = pos.groupby("user_id")["app_id"].apply(set).to_dict()
    val_users  = val["user_id"].unique().tolist()

    all_rows = []
    for n_factors in N_FACTORS_LIST:
        print(f"\n  --- n_factors = {n_factors} ---")

        user_factors, item_factors = compute_svd(matrix, score_col, n_factors)

        recs = recommend_svd_batched(
            val_users, user_to_idx, item_to_idx, idx_to_item,
            seen_items, user_factors, item_factors,
            top_n      = TOP_K_RECOMMEND,
            batch_size = REC_BATCH_SIZE,
        )

        results = evaluate_model(
            recommendations = recs,
            train_df        = train,
            eval_df         = val,
            k_list          = [5, 10],
            only_positive   = True,
            verbose         = True,
        )

        for k_eval in [5, 10]:
            row              = results.loc[k_eval].to_dict()
            row["model"]     = f"SVD_{score_col}"
            row["n_factors"] = n_factors
            row["K_eval"]    = k_eval
            all_rows.append(row)

    summary  = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULT_DIR, f"svd_{score_col}_val_results.csv")
    summary.to_csv(out_path, index=False)
    print(f"\n  [SAVE] {score_col} -> {out_path}")
    return summary


def main():
    print("=" * 60)
    print("  Task 9: SVD Baseline")
    print(f"  DATA_DIR       : {DATA_DIR}")
    print(f"  CACHE_DIR      : {CACHE_DIR}")
    print(f"  RESULT_DIR     : {RESULT_DIR}")
    print(f"  REC_BATCH_SIZE : {REC_BATCH_SIZE}")
    print("=" * 60)

    train, val = load_splits()

    results_A = run_score("score_A", train, val)
    results_B = run_score("score_B", train, val)

    knn_path     = os.path.join(RESULT_DIR, "itemknn_compare_summary.csv")
    summary_rows = []

    if os.path.exists(knn_path):
        knn = pd.read_csv(knn_path)
        for _, row in knn.iterrows():
            summary_rows.append(row.to_dict())
    else:
        print("  [WARN] itemknn_compare_summary.csv not found")

    for df in [results_A, results_B]:
        for _, row in df.iterrows():
            summary_rows.append(row.to_dict())

    compare      = pd.DataFrame(summary_rows)
    compare_path = os.path.join(RESULT_DIR, "svd_compare_summary.csv")
    compare.to_csv(compare_path, index=False)

    print("\n" + "=" * 60)
    print("  SVD vs ItemKNN (K_eval=10)")
    print("=" * 60)
    cols     = ["model", "K_neighbors", "n_factors", "ndcg@10", "recall@10",
                "precision@10", "hit@10"]
    existing = [c for c in cols if c in compare.columns]
    k10      = compare[compare["K_eval"] == 10][existing]
    print(k10.to_string(index=False))
    print(f"\n  [SAVE] comparison table -> {compare_path}")


if __name__ == "__main__":
    main()