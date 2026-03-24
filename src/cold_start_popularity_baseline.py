"""
cold_start_popularity_baseline.py
Run popularity baseline on the same 500 cold-start users for comparison with Level 2.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from evaluate import evaluate_model

PROJECT_ROOT = r"C:\Users\vipuser\Desktop\ml"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
RESULT_DIR   = os.path.join(PROJECT_ROOT, "results", "test")
os.makedirs(RESULT_DIR, exist_ok=True)

print("[LOAD] loading data ...")
train = pd.read_parquet(os.path.join(DATA_DIR, "train_interactions.parquet"))
test  = pd.read_parquet(os.path.join(DATA_DIR, "test_interactions.parquet"))

train_counts = train.groupby("user_id").size()
cold_users   = set(train_counts[train_counts < 5].index)
test_cold    = test[test["user_id"].isin(cold_users)]
eval_users   = test_cold[test_cold["is_recommended"] == 1]["user_id"].unique()

rng          = np.random.default_rng(42)
sample_users = rng.choice(eval_users, size=500, replace=False).tolist()
print(f"  eval users: {len(sample_users)}")

pop_list = (
    train[train["is_recommended"] == 1]
    .groupby("app_id").size()
    .sort_values(ascending=False)
    .head(20)
    .index.tolist()
)
pop_recs = {int(uid): [int(x) for x in pop_list] for uid in sample_users}

print("[EVAL] computing metrics ...")
results = evaluate_model(
    recommendations = pop_recs,
    train_df        = train,
    eval_df         = test[test["user_id"].isin(sample_users)],
    k_list          = [5, 10],
    only_positive   = True,
    verbose         = True,
)

out_path = os.path.join(RESULT_DIR, "cold_start_popularity_baseline.csv")
results.to_csv(out_path)
print(f"\n[SAVE] -> {out_path}")