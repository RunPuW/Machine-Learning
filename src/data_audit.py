"""
data_audit.py  -  Data filtering audit trail
Traces the full record count chain from raw 41M to train/val/test 22M.
"""
import os
import pandas as pd
import numpy as np

PROJECT_ROOT = r"C:\Users\vipuser\Desktop\ml"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")

print("=" * 65)
print("  Data Filtering Audit Trail")
print("=" * 65)

core = pd.read_parquet(os.path.join(DATA_DIR, "interactions_core.parquet"))
print(f"\n[S1] interactions_core.parquet")
print(f"     records : {len(core):,}")
print(f"     users   : {core['user_id'].nunique():,}")
print(f"     games   : {core['app_id'].nunique():,}")

train = pd.read_parquet(os.path.join(DATA_DIR, "train_interactions.parquet"))
val   = pd.read_parquet(os.path.join(DATA_DIR, "val_interactions.parquet"))
test  = pd.read_parquet(os.path.join(DATA_DIR, "test_interactions.parquet"))

# [A] cold-start defined by train record count (matches cold_start.py)
train_counts          = train.groupby("user_id").size()
cold_users_by_train   = set(train_counts[train_counts < 5].index)
normal_users_by_train = set(train_counts[train_counts >= 5].index)

# [B] cold-start defined by total interaction count (explains the 41M vs 22M gap)
user_total_counts = core.groupby("user_id").size()
excluded_users    = set(user_total_counts[user_total_counts < 5].index)
core_excluded     = core[core["user_id"].isin(excluded_users)]
core_in_splits    = core[~core["user_id"].isin(excluded_users)]

print(f"\n[S2] User stratification (two definitions, different populations)")
print(f"  [A: by train record count -- definition used in cold_start.py]")
print(f"     cold-start (train < 5) : {len(cold_users_by_train):,}")
print(f"     warm       (train >= 5): {len(normal_users_by_train):,}")
cold_start_train_recs = int(train_counts[train_counts < 5].sum())
print(f"     {len(cold_users_by_train):,} cold-start users have {cold_start_train_recs:,} train records total")

cold_train_recs = train[train["user_id"].isin(cold_users_by_train)]
cold_test_recs  = test[test["user_id"].isin(cold_users_by_train)]
print(f"     cold-start records in train : {len(cold_train_recs):,}")
print(f"     cold-start records in test  : {len(cold_test_recs):,}")
print(f"     cold-start positives in test: {(cold_test_recs['is_recommended'] == 1).sum():,}")

print(f"\n  [B: by total interaction count -- explains 41M vs 22M gap]")
print(f"     total < 5 (fully excluded from splits): {len(excluded_users):,} users, {len(core_excluded):,} records")
print(f"     total >= 5 (enter splits)             : {len(user_total_counts) - len(excluded_users):,} users, {len(core_in_splits):,} records")

cold_users   = cold_users_by_train
normal_users = normal_users_by_train
core_cold    = core[core["user_id"].isin(cold_users)]
core_normal  = core[core["user_id"].isin(normal_users)]

train_normal = train[train["user_id"].isin(normal_users)]
val_normal   = val[val["user_id"].isin(normal_users)]
test_normal  = test[test["user_id"].isin(normal_users)]
main_total   = len(train_normal) + len(val_normal) + len(test_normal)

print(f"\n[S3] Main experiment temporal split (warm users only)")
print(f"     train : {len(train_normal):,}  ({len(train_normal)/main_total*100:.1f}%)")
print(f"     val   : {len(val_normal):,}  ({len(val_normal)/main_total*100:.1f}%)")
print(f"     test  : {len(test_normal):,}  ({len(test_normal)/main_total*100:.1f}%)")
print(f"     total : {main_total:,}")

print(f"\n[S4] Parquet file sizes (include cold-start users)")
print(f"     train_interactions.parquet : {len(train):,}")
print(f"     val_interactions.parquet   : {len(val):,}")
print(f"     test_interactions.parquet  : {len(test):,}")
print(f"     combined                   : {len(train)+len(val)+len(test):,}")

split_total = len(train) + len(val) + len(test)
gap         = len(core) - split_total

print(f"\n{'='*65}")
print("  Full Audit Trail")
print(f"{'='*65}")
print(f"  {'Stage':<40} {'Records':>12}  Notes")
print(f"  {'-'*65}")
print(f"  {'interactions_core (all users)':<40} {len(core):>12,}  after cleaning")
print(f"  {'  cold-start user records':<40} {len(core_cold):>12,}  train history < 5")
print(f"  {'  warm user records':<40} {len(core_normal):>12,}  train history >= 5, main experiment")
print(f"  {'main train (warm, first 70%)':<40} {len(train_normal):>12,}  per-user percentile split")
print(f"  {'main val   (warm, middle 10%)':<40} {len(val_normal):>12,}  hyperparameter tuning")
print(f"  {'main test  (warm, last 20%)':<40} {len(test_normal):>12,}  final report")
print(f"  {'cold-start test records':<40} {len(test[test['user_id'].isin(cold_users)]):>12,}  evaluated separately")
print(f"  {'full train parquet':<40} {len(train):>12,}  includes partial cold-start history")
print(f"  {'full val parquet':<40} {len(val):>12,}")
print(f"  {'full test parquet':<40} {len(test):>12,}")
print(f"  {'-'*65}")
print(f"  main experiment train+val+test : {main_total:,}")
print(f"  full split train+val+test      : {split_total:,}")
print(f"  interactions_core total        : {len(core):,}")
print(f"\n  gap (core - full split)        : {gap:,} records")
print(f"  these {gap:,} records belong to cold-start users whose val window is empty")
print("=" * 65)