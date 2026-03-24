"""
EDA plot
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR   = r"C:\Users\vipuser\Desktop\ml\archive"
OUTPUT_DIR = r"C:\Users\vipuser\Desktop\ml\archive"

def load_data():
    parquet_path = os.path.join(DATA_DIR, 'interactions_core.parquet')
    csv_path     = os.path.join(DATA_DIR, 'interactions_core.csv')

    if os.path.exists(parquet_path):
        print(f"reading {parquet_path}")
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        print(f"reading {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            "error"
        )

    print(f"OK {df.shape}")
    return df


def plot_user_dist(df):
    user_counts = df['user_id'].value_counts().values

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(np.log1p(user_counts), bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
    ax.set_title('User Interaction Count Distribution (log1p)', fontsize=14)
    ax.set_xlabel('log1p(interaction count)', fontsize=12)
    ax.set_ylabel('Number of Users', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for thr, color in [(1, 'red'), (3, 'orange'), (5, 'green')]:
        n = (user_counts < thr).sum()
        pct = n / len(user_counts) * 100
        ax.axvline(np.log1p(thr), color=color, linestyle='--', linewidth=1.2,
                   label=f'< {thr} interactions: {pct:.1f}% users')
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'eda_01_user_dist.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_game_dist(df):
    game_counts = df['app_id'].value_counts().values

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(np.log1p(game_counts), bins=60, color='coral', edgecolor='white', linewidth=0.4)
    ax.set_title('Game Interaction Count Distribution (log1p)', fontsize=14)
    ax.set_xlabel('log1p(interaction count)', fontsize=12)
    ax.set_ylabel('Number of Games', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    median_val = np.median(game_counts)
    ax.axvline(np.log1p(median_val), color='darkred', linestyle='--', linewidth=1.2,
               label=f'Median: {median_val:.0f} interactions')
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'eda_02_game_dist.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_hours_dist(df):
    hours = df['hours'].clip(lower=0).values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax0 = axes[0]
    clipped = hours[hours <= 500]
    ax0.hist(clipped, bins=80, color='mediumseagreen', edgecolor='white', linewidth=0.3)
    ax0.set_title('Hours Distribution (raw, clipped at 500h)', fontsize=12)
    ax0.set_xlabel('hours', fontsize=11)
    ax0.set_ylabel('Count', fontsize=11)
    ax0.grid(axis='y', linestyle='--', alpha=0.5)


    ax1 = axes[1]
    ax1.hist(np.log1p(hours), bins=80, color='mediumseagreen', edgecolor='white', linewidth=0.3)
    ax1.set_title('Hours Distribution (log1p transformed)', fontsize=12)
    ax1.set_xlabel('log1p(hours)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)


    for q, color in [(0.5, 'navy'), (0.95, 'red')]:
        val = np.quantile(hours, q)
        ax1.axvline(np.log1p(val), color=color, linestyle='--', linewidth=1.2,
                    label=f'p{int(q*100)}: {val:.1f}h')
    ax1.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'eda_03_hours_dist.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)



def plot_label_ratio(df):
    pos = (df['is_recommended'] == 1).sum()
    neg = (df['is_recommended'] == 0).sum()
    total = pos + neg

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax0 = axes[0]
    ax0.pie(
        [pos, neg],
        labels=[f'Recommended (1)\n{pos:,}  ({pos/total*100:.1f}%)',
                f'Not Recommended (0)\n{neg:,}  ({neg/total*100:.1f}%)'],
        colors=['#4CAF50', '#F44336'],
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    ax0.set_title('is_recommended Ratio (Pie)', fontsize=13)

    ax1 = axes[1]
    bars = ax1.bar(['Recommended (1)', 'Not Recommended (0)'],
                   [pos, neg],
                   color=['#4CAF50', '#F44336'],
                   edgecolor='white', linewidth=0.8, width=0.5)
    ax1.set_title('is_recommended Absolute Count (Bar)', fontsize=13)
    ax1.set_ylabel('Number of Interactions', fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, [pos, neg]):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.01,
                 f'{val:,}\n({val/total*100:.1f}%)',
                 ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'eda_04_label_ratio.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ 已保存：{out}")



def main():
    df = load_data()

    plot_user_dist(df)
    plot_game_dist(df)
    plot_hours_dist(df)
    plot_label_ratio(df)

    print(f"  {OUTPUT_DIR}\\eda_01_user_dist.png")
    print(f"  {OUTPUT_DIR}\\eda_02_game_dist.png")
    print(f"  {OUTPUT_DIR}\\eda_03_hours_dist.png")
    print(f"  {OUTPUT_DIR}\\eda_04_label_ratio.png")


if __name__ == '__main__':
    main()
