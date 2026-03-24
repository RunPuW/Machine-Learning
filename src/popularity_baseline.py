"""
popularity_baseline.py  -  Task 7: Popularity Baseline
Count positive ratings per game in the training set and recommend the global
top-K to all users, filtered by seen items.
"""

import os
import pandas as pd
import numpy as np
from evaluate import evaluate_model, build_seen_items, filter_seen_items

DATA_DIR   = r"C:\Users\vipuser\Desktop\ml\archive"
OUTPUT_DIR = r"C:\Users\vipuser\Desktop\ml\archive"
TOP_K      = 20


def load_splits():
    print("loading train / val / test ...")
    train = pd.read_parquet(os.path.join(DATA_DIR, 'train_interactions.parquet'))
    val   = pd.read_parquet(os.path.join(DATA_DIR, 'val_interactions.parquet'))
    test  = pd.read_parquet(os.path.join(DATA_DIR, 'test_interactions.parquet'))
    print(f"  train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    return train, val, test


def build_popularity_ranking(train_df: pd.DataFrame, top_k: int = TOP_K):
    pos = train_df[train_df['is_recommended'] == 1]
    ranking = (
        pos.groupby('app_id')['user_id']
        .count()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    return ranking[:top_k]


def make_popularity_recommendations(
    users:          list,
    global_ranking: list,
    seen_items:     dict,
    k:              int = TOP_K,
) -> dict:
    recs = {}
    for uid in users:
        user_seen = seen_items.get(uid, set())
        filtered  = [i for i in global_ranking if i not in user_seen]
        recs[uid] = filtered[:k]
    return recs


def main():
    print("=" * 55)
    print("  Task 7: Popularity Baseline")
    print("=" * 55)

    train, val, test = load_splits()

    print(f"\nbuilding global popularity ranking (top {TOP_K}) ...")
    global_ranking = build_popularity_ranking(train, top_k=TOP_K)
    print(f"  top 5 app_ids: {global_ranking[:5]}")

    seen_items = build_seen_items(train)

    val_users = val['user_id'].unique().tolist()
    print(f"\ngenerating recommendations for {len(val_users):,} val users ...")
    val_recs = make_popularity_recommendations(val_users, global_ranking, seen_items, k=TOP_K)

    print("\n[EVAL] val set")
    val_results = evaluate_model(
        recommendations=val_recs,
        train_df=train,
        eval_df=val,
        k_list=[5, 10],
        only_positive=True,
        verbose=True,
    )

    out_csv = os.path.join(OUTPUT_DIR, 'popularity_val_results.csv')
    val_results.to_csv(out_csv)
    print(f"  [SAVE] -> {out_csv}")

    test_users = test['user_id'].unique().tolist()
    print(f"\ngenerating recommendations for {len(test_users):,} test users (archive, no eval) ...")
    test_recs = make_popularity_recommendations(test_users, global_ranking, seen_items, k=TOP_K)

    test_rows   = [{'user_id': uid, 'recommendations': str(item_list)}
                   for uid, item_list in test_recs.items()]
    out_parquet = os.path.join(OUTPUT_DIR, 'popularity_test_recs.parquet')
    pd.DataFrame(test_rows).to_parquet(out_parquet, index=False)
    print(f"  [SAVE] -> {out_parquet}")


if __name__ == '__main__':
    main()