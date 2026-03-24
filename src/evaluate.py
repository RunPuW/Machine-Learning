"""
evaluate.py
===========
Task 6  : seen-item filtering
Task 10 : unified evaluation script

Functions:
  build_seen_items()          build per-user seen item sets from train
  filter_seen_items()         remove seen items from candidate lists
  build_ground_truth()        build per-user ground truth sets from val/test
  evaluate_recommendations()  compute Precision / Recall / NDCG / Hit @K
  evaluate_model()            all-in-one evaluation entry point
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def build_seen_items(train_df: pd.DataFrame) -> Dict[int, Set[int]]:
    seen = defaultdict(set)
    for uid, aid in zip(train_df['user_id'], train_df['app_id']):
        seen[uid].add(aid)
    return dict(seen)


def filter_seen_items(
    recommendations: Dict[int, List[int]],
    seen_items:      Dict[int, Set[int]],
    k:               int = None,
) -> Dict[int, List[int]]:
    filtered = {}
    for uid, item_list in recommendations.items():
        user_seen     = seen_items.get(uid, set())
        cleaned       = [i for i in item_list if i not in user_seen]
        filtered[uid] = cleaned[:k] if k is not None else cleaned
    return filtered


def build_ground_truth(
    eval_df:       pd.DataFrame,
    only_positive: bool = True,
) -> Dict[int, Set[int]]:
    if only_positive and 'is_recommended' in eval_df.columns:
        sub = eval_df[eval_df['is_recommended'] == 1]
    else:
        sub = eval_df
    gt = defaultdict(set)
    for uid, aid in zip(sub['user_id'], sub['app_id']):
        gt[uid].add(aid)
    return dict(gt)


def _dcg(relevances: List[int]) -> float:
    return sum(rel / np.log2(rank + 2) for rank, rel in enumerate(relevances))


def _ndcg_at_k(recommended: List[int], ground_truth: Set[int], k: int) -> float:
    top_k = recommended[:k]
    hits  = [1 if item in ground_truth else 0 for item in top_k]
    ideal = [1] * min(len(ground_truth), k)
    dcg   = _dcg(hits)
    idcg  = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(
    recommendations: Dict[int, List[int]],
    ground_truth:    Dict[int, Set[int]],
    k:               int,
) -> Dict[str, float]:
    precisions, recalls, ndcgs, hits = [], [], [], []

    eval_users = set(ground_truth.keys()) & set(recommendations.keys())

    for uid in eval_users:
        gt      = ground_truth[uid]
        rec     = recommendations[uid][:k]
        hit_set = set(rec) & gt

        precisions.append(len(hit_set) / k)
        recalls.append(len(hit_set) / len(gt) if gt else 0.0)
        ndcgs.append(_ndcg_at_k(rec, gt, k))
        hits.append(1.0 if hit_set else 0.0)

    n = len(eval_users)
    if n == 0:
        return {f'precision@{k}': 0, f'recall@{k}': 0,
                f'ndcg@{k}': 0,     f'hit@{k}': 0,
                'n_users': 0}

    return {
        f'precision@{k}': round(np.mean(precisions), 6),
        f'recall@{k}':    round(np.mean(recalls),    6),
        f'ndcg@{k}':      round(np.mean(ndcgs),      6),
        f'hit@{k}':       round(np.mean(hits),        6),
        'n_users':        n,
    }


def evaluate_recommendations(
    recommendations: Dict[int, List[int]],
    ground_truth:    Dict[int, Set[int]],
    k_list:          List[int] = [5, 10],
) -> pd.DataFrame:
    rows = []
    for k in k_list:
        metrics      = compute_metrics(recommendations, ground_truth, k)
        metrics['K'] = k
        rows.append(metrics)
    return pd.DataFrame(rows).set_index('K')


def evaluate_model(
    recommendations: Dict[int, List[int]],
    train_df:        pd.DataFrame,
    eval_df:         pd.DataFrame,
    k_list:          List[int] = [5, 10],
    only_positive:   bool = True,
    verbose:         bool = True,
) -> pd.DataFrame:
    seen_items    = build_seen_items(train_df)
    max_k         = max(k_list)
    recs_filtered = filter_seen_items(recommendations, seen_items, k=max_k)
    ground_truth  = build_ground_truth(eval_df, only_positive=only_positive)
    results       = evaluate_recommendations(recs_filtered, ground_truth, k_list)

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Evaluation results (only_positive={only_positive})")
        print(f"  eval users: {results['n_users'].iloc[0]:,}")
        print(f"{'='*55}")
        cols = [c for c in results.columns if c != 'n_users']
        print(results[cols].to_string())
        print(f"{'='*55}\n")

    return results


if __name__ == '__main__':
    import random
    random.seed(42)
    np.random.seed(42)

    print("=" * 55)
    print("  evaluate.py self-test")
    print("=" * 55)

    all_users = list(range(1, 11))
    all_items = list(range(1, 51))

    train_rows, val_rows = [], []
    for uid in all_users:
        train_items = random.sample(all_items, 15)
        val_items   = random.sample([i for i in all_items if i not in train_items], 5)
        for aid in train_items:
            train_rows.append({'user_id': uid, 'app_id': aid, 'is_recommended': 1})
        for aid in val_items:
            val_rows.append({'user_id': uid, 'app_id': aid, 'is_recommended': 1})

    train_df = pd.DataFrame(train_rows)
    val_df   = pd.DataFrame(val_rows)

    recs = {uid: random.sample(all_items, 20) for uid in all_users}

    seen = build_seen_items(train_df)
    filt = filter_seen_items(recs, seen, k=10)

    leak = 0
    for uid, item_list in filt.items():
        leak += len(set(item_list) & seen.get(uid, set()))
    print(f"  seen-item leakage after filtering: {leak} (expected 0)")
    assert leak == 0, "seen-item filter is broken"
    print("  [PASS] seen-item filtering correct")

    results = evaluate_model(recs, train_df, val_df, k_list=[5, 10])

    for k in [5, 10]:
        ndcg = results.loc[k, f'ndcg@{k}']
        assert 0.0 <= ndcg <= 1.0, f"NDCG@{k} out of range: {ndcg}"
    print("  [PASS] metric values in valid range")
    print("\n  all self-tests passed")