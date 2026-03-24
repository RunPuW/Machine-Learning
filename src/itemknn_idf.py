"""
itemknn_idf.py  -  ItemKNN + IDF user weighting variant
Weights each user column by log(1 + n_items / items_count(u)) before computing
cosine similarity, suppressing power-user dominance.
Runs score_B only (confirmed better than score_A), K=200/500.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse import csc_matrix
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from multiprocessing.shared_memory import SharedMemory

from evaluate import evaluate_model

PROJECT_ROOT = r"C:\Users\vipuser\Desktop\ml"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "cache")
RESULT_DIR   = os.path.join(PROJECT_ROOT, "results", "val")

for _d in [CACHE_DIR, RESULT_DIR]:
    os.makedirs(_d, exist_ok=True)

SCORE_COL       = "score_B"
K_NEIGHBOR_LIST = [200, 500]
TOP_K_RECOMMEND = 20
SIM_BATCH_SIZE  = 4000
N_WORKERS       = max(1, cpu_count() - 2)


def load_splits():
    print("[LOAD] loading train / val ...")
    train = pd.read_parquet(os.path.join(DATA_DIR, "train_interactions.parquet"))
    val   = pd.read_parquet(os.path.join(DATA_DIR, "val_interactions.parquet"))
    print(f"  train: {len(train):,}  val: {len(val):,}")
    return train, val


def build_idf_item_user_matrix(train_df: pd.DataFrame, score_col: str):
    pos          = train_df[train_df[score_col] > 0][["user_id", "app_id", score_col]].copy()
    unique_items = pos["app_id"].unique()
    unique_users = pos["user_id"].unique()
    n_items      = len(unique_items)

    print(f"  {score_col} > 0: {len(pos):,} records")

    item_to_idx = {item: i for i, item in enumerate(unique_items)}
    idx_to_item = {i: item for item, i in item_to_idx.items()}
    user_to_idx = {u: i for i, u in enumerate(unique_users)}

    row    = pos["app_id"].map(item_to_idx).values
    col    = pos["user_id"].map(user_to_idx).values
    data   = pos[score_col].values.astype(np.float32)
    matrix = csr_matrix((data, (row, col)),
                         shape=(n_items, len(unique_users)), dtype=np.float32)

    user_item_count = np.array(matrix.getnnz(axis=0), dtype=np.float32)
    idf_weights     = np.log1p(n_items / (user_item_count + 1)).astype(np.float32)

    print(f"  user item counts: min={user_item_count.min():.0f}  "
          f"median={np.median(user_item_count):.0f}  max={user_item_count.max():.0f}")
    print(f"  IDF weights:      min={idf_weights.min():.3f}  "
          f"median={np.median(idf_weights):.3f}  max={idf_weights.max():.3f}")

    matrix = csr_matrix(matrix.multiply(idf_weights), dtype=np.float32)
    print(f"  IDF-weighted item-user matrix: {matrix.shape}, nnz: {matrix.nnz:,}")
    return matrix, item_to_idx, idx_to_item, user_to_idx


def compute_item_similarity(item_user_matrix, k_max=500, batch_size=SIM_BATCH_SIZE):
    cache_path = os.path.join(CACHE_DIR, f"itemknn_idf_neighbors_{SCORE_COL}.npz")

    if os.path.exists(cache_path):
        print(f"  [CACHE HIT] {cache_path}")
        cache = np.load(cache_path)
        return cache["indices"], cache["sims"]

    n_items = item_user_matrix.shape[0]
    k_max   = min(k_max, n_items - 1)

    norms = np.sqrt(np.array(item_user_matrix.power(2).sum(axis=1))).ravel()
    norms[norms == 0] = 1.0
    norm_matrix = diags(1.0 / norms.astype(np.float32)) @ item_user_matrix

    neighbor_indices = np.zeros((n_items, k_max), dtype=np.int32)
    neighbor_sims    = np.zeros((n_items, k_max), dtype=np.float32)

    print(f"  [SIM] IDF+{SCORE_COL}: {n_items} items, batch={batch_size}, k_max={k_max}")
    t0 = time.time()

    for start in tqdm(range(0, n_items, batch_size), desc="  Similarity [IDF]"):
        end   = min(start + batch_size, n_items)
        batch = norm_matrix[start:end]
        sims  = batch @ norm_matrix.T
        sims  = np.asarray(
            sims.todense() if hasattr(sims, "todense") else sims,
            dtype=np.float32,
        )
        local_range = np.arange(end - start)
        sims[local_range, start + local_range] = 0.0

        actual_k = min(k_max, sims.shape[1])
        top_idx  = np.argpartition(sims, -actual_k, axis=1)[:, -actual_k:]
        top_sims = np.take_along_axis(sims, top_idx, axis=1)
        sort_ord = np.argsort(-top_sims, axis=1)
        neighbor_indices[start:end] = np.take_along_axis(top_idx,  sort_ord, axis=1)
        neighbor_sims[start:end]    = np.take_along_axis(top_sims, sort_ord, axis=1)

    print(f"  similarity done in {(time.time()-t0)/60:.1f} min")
    np.savez_compressed(cache_path, indices=neighbor_indices, sims=neighbor_sims)
    print(f"  [SAVE] neighbor cache -> {cache_path}")
    return neighbor_indices, neighbor_sims


_SHM_STATE = {}

def _worker_init_shm(idx_name, sim_name, nb_shape, idx_to_item):
    shm_idx = SharedMemory(name=idx_name)
    shm_sim = SharedMemory(name=sim_name)
    _SHM_STATE["shm_idx"]     = shm_idx
    _SHM_STATE["shm_sim"]     = shm_sim
    _SHM_STATE["nb_idx"]      = np.ndarray(nb_shape, dtype=np.int32,   buffer=shm_idx.buf)
    _SHM_STATE["nb_sim"]      = np.ndarray(nb_shape, dtype=np.float32, buffer=shm_sim.buf)
    _SHM_STATE["idx_to_item"] = idx_to_item

def _recommend_batch_shm(args):
    user_items_list, k_neighbors, top_n = args
    nb_idx      = _SHM_STATE["nb_idx"]
    nb_sim      = _SHM_STATE["nb_sim"]
    idx_to_item = _SHM_STATE["idx_to_item"]

    recommendations = {}
    for uid, u_items in user_items_list:
        if not u_items:
            recommendations[uid] = []
            continue
        candidate_scores = {}
        for item_idx, user_score in u_items:
            neighbors = nb_idx[item_idx, :k_neighbors]
            sims      = nb_sim[item_idx, :k_neighbors]
            for n_idx, sim in zip(neighbors, sims):
                if sim <= 0:
                    break
                n_app_id = idx_to_item[int(n_idx)]
                if n_app_id in candidate_scores:
                    candidate_scores[n_app_id] += user_score * float(sim)
                else:
                    candidate_scores[n_app_id]  = user_score * float(sim)
        sorted_cands         = sorted(candidate_scores.items(), key=lambda x: -x[1])
        recommendations[uid] = [aid for aid, _ in sorted_cands[:top_n]]
    return recommendations

def build_user_items_dict(train_df, score_col):
    pos        = train_df[train_df[score_col] > 0]
    user_items = {}
    for uid, grp in pos.groupby("user_id"):
        user_items[uid] = dict(zip(grp["app_id"], grp[score_col]))
    return user_items

def prepare_val_user_items(val_users, user_items_dict, item_to_idx):
    result = []
    for uid in val_users:
        raw           = user_items_dict.get(uid, {})
        items_indexed = [
            (item_to_idx[app_id], float(score))
            for app_id, score in raw.items()
            if app_id in item_to_idx
        ]
        result.append((uid, items_indexed))
    return result

def recommend_parallel(val_users_items, idx_to_item, neighbor_indices, neighbor_sims,
                       k_neighbors, top_n=TOP_K_RECOMMEND, n_workers=N_WORKERS):
    n          = len(val_users_items)
    chunk_size = max(1, (n + n_workers - 1) // n_workers)
    chunks     = [val_users_items[i:i+chunk_size] for i in range(0, n, chunk_size)]
    tasks      = [(chunk, k_neighbors, top_n) for chunk in chunks]
    nb_shape   = neighbor_indices.shape

    shm_idx = SharedMemory(create=True, size=neighbor_indices.nbytes)
    shm_sim = SharedMemory(create=True, size=neighbor_sims.nbytes)
    try:
        np.ndarray(nb_shape, dtype=np.int32,   buffer=shm_idx.buf)[:] = neighbor_indices
        np.ndarray(nb_shape, dtype=np.float32, buffer=shm_sim.buf)[:] = neighbor_sims
        print(f"  [REC] K={k_neighbors}, {n:,} users, {n_workers} workers")
        t0   = time.time()
        recs = {}
        with Pool(processes=n_workers,
                  initializer=_worker_init_shm,
                  initargs=(shm_idx.name, shm_sim.name, nb_shape, idx_to_item)) as pool:
            for partial in tqdm(pool.imap_unordered(_recommend_batch_shm, tasks),
                                total=len(tasks), desc=f"  Recs K={k_neighbors}"):
                recs.update(partial)
        print(f"  [REC] done in {(time.time()-t0)/60:.1f} min")
        return recs
    finally:
        shm_idx.close(); shm_idx.unlink()
        shm_sim.close(); shm_sim.unlink()


def main():
    print("=" * 60)
    print(f"  ItemKNN + IDF  score={SCORE_COL}  K={K_NEIGHBOR_LIST}")
    print("=" * 60)

    train, val = load_splits()

    matrix, item_to_idx, idx_to_item, _ = build_idf_item_user_matrix(train, SCORE_COL)
    neighbor_indices, neighbor_sims      = compute_item_similarity(matrix, k_max=max(K_NEIGHBOR_LIST))

    user_items_dict = build_user_items_dict(train, SCORE_COL)
    val_users       = val["user_id"].unique().tolist()
    print(f"\n  val users: {len(val_users):,}")

    t0              = time.time()
    val_users_items = prepare_val_user_items(val_users, user_items_dict, item_to_idx)
    print(f"  [PREP] done in {time.time()-t0:.1f}s")

    all_rows = []
    for k_neighbors in K_NEIGHBOR_LIST:
        print(f"\n  --- K_neighbors = {k_neighbors} ---")
        recs = recommend_parallel(val_users_items, idx_to_item,
                                  neighbor_indices, neighbor_sims,
                                  k_neighbors=k_neighbors)
        results = evaluate_model(
            recommendations=recs, train_df=train, eval_df=val,
            k_list=[5, 10], only_positive=True, verbose=True,
        )
        for k_eval in [5, 10]:
            row                = results.loc[k_eval].to_dict()
            row["model"]       = f"ItemKNN_IDF_{SCORE_COL}"
            row["K_neighbors"] = k_neighbors
            row["K_eval"]      = k_eval
            all_rows.append(row)

    summary  = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULT_DIR, "itemknn_idf_val_results.csv")
    summary.to_csv(out_path, index=False)
    print(f"\n  [SAVE] -> {out_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()