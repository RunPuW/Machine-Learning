import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "."

CORE_COLS = ['user_id', 'app_id', 'date', 'is_recommended', 'hours']
OPTIONAL_COLS = ['review_id']

COLD_START_THRESHOLD = 5
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10


def save_parquet_or_csv(df: pd.DataFrame, path_no_ext: str, label: str):
    try:
        out_path = path_no_ext + '.parquet'
        df.to_parquet(out_path, index=False)
        print(f"  {label} -> {out_path}  (parquet, {df.shape})")
    except ImportError:
        out_path = path_no_ext + '.csv'
        df.to_csv(out_path, index=False)
        print(f"  {label} -> {out_path}  (csv fallback, {df.shape})")
        print("  [INFO] install pyarrow to use parquet: pip install pyarrow")


def load_raw_data():
    print("=" * 60)
    print("[Step 1] Load raw data")
    print("=" * 60)

    df_games = pd.read_csv('games.csv')
    print(f"  games.csv           shape: {df_games.shape}")

    df_users = pd.read_csv('users.csv')
    print(f"  users.csv           shape: {df_users.shape}")

    header = pd.read_csv('recommendations.csv', nrows=0)
    available = set(header.columns)
    load_cols = [c for c in CORE_COLS + OPTIONAL_COLS if c in available]
    missing_required = [c for c in CORE_COLS if c not in available]
    if missing_required:
        raise ValueError(
            f"[FATAL] recommendations.csv is missing required columns: {missing_required}\n"
            f"  actual columns: {list(header.columns)}"
        )

    dtype_map = {
        'user_id':        'int64',
        'app_id':         'int64',
        'is_recommended': 'object',
        'hours':          'float64',
    }
    use_dtype = {k: v for k, v in dtype_map.items() if k in load_cols}

    df_rec = pd.read_csv('recommendations.csv', usecols=load_cols, dtype=use_dtype)
    print(f"  recommendations.csv shape: {df_rec.shape}  (loaded columns: {load_cols})")

    return df_games, df_users, df_rec


def clean_games(df_games: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("[Step 2] Clean games table")
    print("=" * 60)

    before = len(df_games)
    df_games = df_games.dropna(subset=['app_id'])
    df_games = df_games.drop_duplicates(subset=['app_id'])
    after = len(df_games)

    print(f"  app_id null/duplicate removal: {before} -> {after} rows (removed {before - after})")
    return df_games


def clean_users(df_users: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("[Step 3] Clean users table")
    print("=" * 60)

    before = len(df_users)
    df_users = df_users.dropna(subset=['user_id'])
    df_users = df_users.drop_duplicates(subset=['user_id'])
    after = len(df_users)

    print(f"  user_id null/duplicate removal: {before} -> {after} rows (removed {before - after})")
    return df_users


def clean_recommendations(df_rec: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("[Step 4] Clean recommendations table")
    print("=" * 60)

    required = ['user_id', 'app_id', 'is_recommended', 'hours']
    missing  = [c for c in required if c not in df_rec.columns]
    if missing:
        raise ValueError(f"[FATAL] DataFrame is missing required columns: {missing}")

    before = len(df_rec)
    df_rec = df_rec.dropna(subset=required)
    after  = len(df_rec)
    print(f"  Drop rows with nulls in core fields: {before} -> {after} (removed {before - after})")

    if 'date' in df_rec.columns:
        df_rec['date'] = pd.to_datetime(df_rec['date'], errors='coerce')
        bad_dates = df_rec['date'].isna().sum()
        if bad_dates > 0:
            print(f"  [WARN] {bad_dates} rows have unparseable dates, dropping them")
            df_rec = df_rec.dropna(subset=['date'])
        print(f"  date range: {df_rec['date'].min().date()} ~ {df_rec['date'].max().date()}")
    else:
        print("  [WARN] no 'date' column found, temporal split will be skipped")

    df_rec['hours'] = pd.to_numeric(df_rec['hours'], errors='coerce')
    neg_hours = (df_rec['hours'] < 0).sum()
    if neg_hours > 0:
        print(f"  [WARN] {neg_hours} rows have negative hours, dropping them")
        df_rec = df_rec[df_rec['hours'] >= 0]

    q = df_rec['hours'].quantile([0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    print(f"  hours quantiles (0/25/50/75/95/99/100):\n  {q.round(2).to_dict()}")

    raw_vals = df_rec['is_recommended'].unique()
    print(f"\n  is_recommended raw unique values: {raw_vals}")

    bool_map = {
        True: 1,  False: 0,
        'True': 1, 'False': 0,
        'true': 1, 'false': 0,
        '1': 1, '0': 0,
        1: 1, 0: 0,
        'yes': 1, 'no': 0,
        'Yes': 1, 'No': 0,
    }
    unmapped = [v for v in raw_vals if v not in bool_map]
    if unmapped:
        raise ValueError(
            f"  is_recommended contains unrecognized values: {unmapped}\n"
            f"  add them to bool_map manually."
        )
    df_rec['is_recommended'] = df_rec['is_recommended'].map(bool_map).astype(int)
    print(f"  is_recommended value counts after mapping: {df_rec['is_recommended'].value_counts().to_dict()}")

    print("\n  user_id x app_id pair frequency (top 5):")
    dup_dist = df_rec.groupby(['user_id', 'app_id']).size().value_counts().head()
    print(f"  {dup_dist.to_dict()}")

    before_dedup = len(df_rec)
    if 'date' in df_rec.columns:
        df_rec = (
            df_rec
            .sort_values('date', ascending=True)
            .drop_duplicates(subset=['user_id', 'app_id'], keep='first')
            .reset_index(drop=True)
        )
    else:
        df_rec = df_rec.drop_duplicates(subset=['user_id', 'app_id'], keep='first')

    after_dedup = len(df_rec)
    print(f"  Deduplicate user-game pairs (keep earliest): {before_dedup} -> {after_dedup} (removed {before_dedup - after_dedup})")
    print(f"\n  Final interactions shape: {df_rec.shape}")
    return df_rec


def run_eda(df_rec: pd.DataFrame):
    print("\n" + "=" * 60)
    print("[Step 5] EDA")
    print("=" * 60)

    total   = len(df_rec)
    n_users = df_rec['user_id'].nunique()
    n_games = df_rec['app_id'].nunique()

    print(f"  total interactions : {total:,}")
    print(f"  unique users       : {n_users:,}")
    print(f"  unique games       : {n_games:,}")

    pos = (df_rec['is_recommended'] == 1).sum()
    neg = (df_rec['is_recommended'] == 0).sum()
    print(f"\n  positive: {pos:,} ({pos/total*100:.2f}%)  negative: {neg:,} ({neg/total*100:.2f}%)")

    user_counts = df_rec['user_id'].value_counts()
    print(f"\n  Cold-start user breakdown:")
    for thr in [1, 3, 5, 10]:
        n = (user_counts < thr).sum()
        print(f"    < {thr:>2} interactions: {n:>10,} users ({n/n_users*100:.2f}%)")

    game_counts = df_rec['app_id'].value_counts()
    print(f"\n  Top 5 games by interaction count:")
    print(f"  {game_counts.head().to_dict()}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Recommender System EDA', fontsize=16)

    axes[0, 0].hist(np.log1p(user_counts.values), bins=50, color='steelblue', edgecolor='white')
    axes[0, 0].set_title('User interaction count (log1p)')
    axes[0, 0].set_xlabel('log1p(count)')
    axes[0, 0].set_ylabel('num users')

    axes[0, 1].hist(np.log1p(game_counts.values), bins=50, color='coral', edgecolor='white')
    axes[0, 1].set_title('Game interaction count (log1p)')
    axes[0, 1].set_xlabel('log1p(count)')
    axes[0, 1].set_ylabel('num games')

    axes[1, 0].hist(np.log1p(df_rec['hours'].values), bins=60, color='mediumseagreen', edgecolor='white')
    axes[1, 0].set_title('Playtime distribution (log1p hours)')
    axes[1, 0].set_xlabel('log1p(hours)')
    axes[1, 0].set_ylabel('num records')

    axes[1, 1].pie([pos, neg], labels=['positive(1)', 'negative(0)'],
                   autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=90)
    axes[1, 1].set_title('is_recommended distribution')

    plt.tight_layout()
    out_img = os.path.join(OUTPUT_DIR, 'eda_distributions.png')
    plt.savefig(out_img, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"\n  EDA figure saved to {out_img}")


def build_scores(df_rec: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("[Step 6] Build preference scores A / B")
    print("=" * 60)

    df_rec['score_A'] = df_rec['is_recommended'].astype(float)
    df_rec['score_B'] = (
        df_rec['is_recommended'].astype(float)
        * np.log1p(df_rec['hours'].clip(lower=0))
    )

    print(f"  score_A unique values: {sorted(df_rec['score_A'].unique())}")

    neg_mask    = df_rec['is_recommended'] == 0
    weak_mask   = (df_rec['is_recommended'] == 1) & (df_rec['hours'] <= 1)
    strong_mask = (df_rec['is_recommended'] == 1) & (df_rec['hours'] > 10)

    neg_score    = df_rec.loc[neg_mask,    'score_B']
    weak_score   = df_rec.loc[weak_mask,   'score_B']
    strong_score = df_rec.loc[strong_mask, 'score_B']

    print(f"\n  score_B sanity check:")
    print(f"    negative (0)         mean/median: {neg_score.mean():.4f} / {neg_score.median():.4f}")
    print(f"    weak positive (<=1h) mean/median: {weak_score.mean():.4f} / {weak_score.median():.4f}")
    print(f"    strong positive (>10h) mean/median: {strong_score.mean():.4f} / {strong_score.median():.4f}")

    if neg_score.mean() == weak_score.mean():
        print("\n  [WARN] negative and weak positive samples have identical score_B.")
        print("    The current formula cannot distinguish them.")
        print("    Consider assigning negative scores to dislikes or adding other signals.")
    else:
        print("\n  negative and weak positive score_B differ -- OK")

    if len(strong_score) > 0 and len(weak_score) > 0:
        if strong_score.mean() > weak_score.mean():
            print("  strong positive score_B > weak positive -- formula direction correct")
        else:
            print("  [WARN] strong positive score_B <= weak positive, check hours data quality")

    return df_rec


def temporal_split(df_rec: pd.DataFrame):
    if 'date' not in df_rec.columns:
        print("\n[SKIP] no 'date' column, temporal split skipped")
        return None, None, None

    print("\n" + "=" * 60)
    print("[Step 7] Temporal split (train / val / test) + cold-start labels")
    print("=" * 60)

    user_counts = df_rec['user_id'].value_counts()
    cold_users  = set(user_counts[user_counts < COLD_START_THRESHOLD].index)
    n_users     = len(user_counts)
    print(f"  Cold-start users (< {COLD_START_THRESHOLD} interactions): {len(cold_users):,} ({len(cold_users)/n_users*100:.2f}%)")

    df_rec = df_rec.copy()
    df_rec['is_cold_start'] = df_rec['user_id'].isin(cold_users).astype(int)

    df_normal = df_rec[df_rec['is_cold_start'] == 0].copy()
    df_normal = df_normal.sort_values(['user_id', 'date']).reset_index(drop=True)

    df_normal['_rank'] = df_normal.groupby('user_id').cumcount()
    df_normal['_cnt']  = df_normal.groupby('user_id')['user_id'].transform('count')
    df_normal['_pct']  = df_normal['_rank'] / df_normal['_cnt']

    df_normal['split'] = 'test'
    df_normal.loc[df_normal['_pct'] < TRAIN_RATIO, 'split'] = 'train'
    df_normal.loc[
        (df_normal['_pct'] >= TRAIN_RATIO) &
        (df_normal['_pct'] <  TRAIN_RATIO + VAL_RATIO),
        'split'
    ] = 'val'

    df_normal = df_normal.drop(columns=['_rank', '_cnt', '_pct'])

    train_df = df_normal[df_normal['split'] == 'train']
    val_df   = df_normal[df_normal['split'] == 'val']
    test_df  = df_normal[df_normal['split'] == 'test']

    n = len(df_normal)
    print(f"\n  Split result (warm users, {n:,} total):")
    print(f"    train : {len(train_df):>10,} ({len(train_df)/n*100:.1f}%)")
    print(f"    val   : {len(val_df):>10,} ({len(val_df)/n*100:.1f}%)")
    print(f"    test  : {len(test_df):>10,} ({len(test_df)/n*100:.1f}%)")

    print("\n  Temporal sanity check (5 sampled users: train max date <= test min date)")
    sample_uids = (
        df_normal['user_id']
        .drop_duplicates()
        .sample(min(5, df_normal['user_id'].nunique()), random_state=42)
    )
    ok = 0
    for uid in sample_uids:
        u = df_normal[df_normal['user_id'] == uid]
        u_train_max = u[u['split'] == 'train']['date'].max()
        u_test_min  = u[u['split'] == 'test']['date'].min()
        if pd.notna(u_train_max) and pd.notna(u_test_min):
            if u_train_max <= u_test_min:
                ok += 1
    print(f"  {ok} / {len(sample_uids)} sampled users pass (others may have only train or test records)")

    return train_df, val_df, test_df


def save_results(df_games, df_users, df_rec, train_df, val_df, test_df):
    print("\n" + "=" * 60)
    print("[Step 8] Save outputs")
    print("=" * 60)

    df_games.to_csv(os.path.join(OUTPUT_DIR, 'games_cleaned.csv'), index=False)
    print(f"  games_cleaned.csv  {df_games.shape}")

    df_users.to_csv(os.path.join(OUTPUT_DIR, 'users_cleaned.csv'), index=False)
    print(f"  users_cleaned.csv  {df_users.shape}")

    save_parquet_or_csv(df_rec, os.path.join(OUTPUT_DIR, 'interactions_core'), 'interactions_core')

    if train_df is not None:
        save_parquet_or_csv(train_df, os.path.join(OUTPUT_DIR, 'train_interactions'), 'train_interactions')
        save_parquet_or_csv(val_df,   os.path.join(OUTPUT_DIR, 'val_interactions'),   'val_interactions')
        save_parquet_or_csv(test_df,  os.path.join(OUTPUT_DIR, 'test_interactions'),  'test_interactions')


def main():
    print("=" * 60)
    print("  Steam Game Recommender -- Data Cleaning & Preprocessing")
    print("=" * 60)

    df_games, df_users, df_rec = load_raw_data()

    df_games = clean_games(df_games)
    df_users = clean_users(df_users)
    df_rec   = clean_recommendations(df_rec)

    run_eda(df_rec)
    df_rec = build_scores(df_rec)

    train_df, val_df, test_df = temporal_split(df_rec)

    save_results(df_games, df_users, df_rec, train_df, val_df, test_df)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == '__main__':
    main()