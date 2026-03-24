"""
ablation_case_study.py  -  Ablation experiment candidate set case analysis
===========================================================================
Randomly picks 3 users from existing ablation results and shows side-by-side
recommendation lists for all four configurations, to help explain:
  - Why B = A  (SVD-only candidates cut off by knn_score ordering)
  - Why C drops (Ranker has no semantic context, displaces high-score KNN candidates)
  - Why D > C  (UUA provides a semantic user profile, Ranker decisions improve)
"""

import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (
    load_all_caches,
    get_knn_candidates_with_scores,
    get_svd_candidates_with_scores,
    fuse_candidates,
    KNN_TOP_K, SVD_TOP_K, RANKER_TOP_N,
)
from uua_agent import load_games_meta

PROJECT_ROOT = r"C:\Users\vipuser\Desktop\ml"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
RESULT_DIR   = os.path.join(PROJECT_ROOT, "results", "val")

N_CASE_USERS = 3
RANDOM_SEED  = 99   # different from ablation seed=42 to avoid cherry-picking


def load_ablation_outputs():
    users_path = os.path.join(RESULT_DIR, "ablation_fair_sample_users.json")
    raw_c_path = os.path.join(RESULT_DIR, "ablation_fair_raw_C.json")
    raw_d_path = os.path.join(RESULT_DIR, "ablation_fair_raw_D.json")

    for p in [users_path, raw_c_path, raw_d_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"File not found: {p}\nRun ablation_fair.py first."
            )

    with open(users_path) as f:
        user_data = json.load(f)
    with open(raw_c_path, encoding="utf-8") as f:
        raw_c = {d["user_id"]: d for d in json.load(f)}
    with open(raw_d_path, encoding="utf-8") as f:
        raw_d = {d["user_id"]: d for d in json.load(f)}

    return user_data["user_ids"], raw_c, raw_d


def get_title(app_id, games_meta):
    app_id = int(app_id)
    if app_id in games_meta.index:
        return str(games_meta.loc[app_id, "title"])[:40]
    return f"[app_id={app_id}]"


def format_rec_list(recs, games_meta, gt_set, label_width=42):
    lines = []
    for rank, app_id in enumerate(recs[:10], 1):
        title = get_title(app_id, games_meta)
        hit   = " [HIT]" if int(app_id) in gt_set else ""
        lines.append(f"  {rank:>2}. {title:<{label_width}}{hit}")
    return lines


def main():
    print("[LOAD] Loading data and ablation results ...")
    train      = pd.read_parquet(os.path.join(DATA_DIR, "train_interactions.parquet"))
    val        = pd.read_parquet(os.path.join(DATA_DIR, "val_interactions.parquet"))
    games_meta = load_games_meta(DATA_DIR)
    caches     = load_all_caches(train)

    all_user_ids, raw_c, raw_d = load_ablation_outputs()

    val_pos_dict = val[val["is_recommended"] == 1].groupby("user_id")["app_id"].apply(set).to_dict()

    eligible = [
        uid for uid in all_user_ids
        if uid in raw_c and uid in raw_d and val_pos_dict.get(uid)
    ]

    def has_any_hit(uid):
        gt    = val_pos_dict.get(uid, set())
        c_hit = any(int(x) in gt for x in raw_c[uid]["ranked_ids"])
        d_hit = any(int(x) in gt for x in raw_d[uid]["ranked_ids"])
        return c_hit or d_hit

    with_hits = [u for u in eligible if has_any_hit(u)]

    rng        = np.random.default_rng(RANDOM_SEED)
    case_pool  = with_hits if len(with_hits) >= N_CASE_USERS else eligible
    case_users = rng.choice(case_pool,
                            size=min(N_CASE_USERS, len(case_pool)),
                            replace=False).tolist()

    print(f"  Selected {len(case_users)} case users (prefer users with at least one hit)")

    def get_A(uid):
        knn = get_knn_candidates_with_scores(int(uid), caches, top_k=RANKER_TOP_N)
        return [int(aid) for aid, _ in knn]

    def get_B(uid):
        uid        = int(uid)
        knn_cands  = get_knn_candidates_with_scores(uid, caches, top_k=KNN_TOP_K)
        svd_cands  = get_svd_candidates_with_scores(uid, caches, top_k=SVD_TOP_K)
        candidates = fuse_candidates(knn_cands, svd_cands, games_meta)
        seen       = set(caches["knn_user_items"].get(uid, {}).keys())
        return [int(c["app_id"]) for c in candidates
                if int(c["app_id"]) not in seen][:RANKER_TOP_N]

    output_lines = []
    csv_rows     = []

    header = (
        "=" * 100 + "\n"
        "  Ablation Experiment -- Candidate Set Case Analysis\n"
        "  Goal: show recommendation list differences across A/B/C/D to explain\n"
        "        why B=A, why C drops, and why D recovers.\n"
        "=" * 100
    )
    output_lines.append(header)

    for case_idx, uid in enumerate(case_users, 1):
        uid = int(uid)
        gt  = {int(x) for x in val_pos_dict.get(uid, set())}

        history_aids   = list(caches["knn_user_items"].get(uid, {}).keys())
        history_titles = [get_title(a, games_meta) for a in history_aids[:5]]

        knn_cands_raw = get_knn_candidates_with_scores(uid, caches, top_k=KNN_TOP_K)
        svd_cands_raw = get_svd_candidates_with_scores(uid, caches, top_k=SVD_TOP_K)
        knn_set       = {int(a) for a, _ in knn_cands_raw}
        svd_set       = {int(a) for a, _ in svd_cands_raw}
        svd_only      = svd_set - knn_set

        recs_A = get_A(uid)
        recs_B = get_B(uid)
        recs_C = [int(x) for x in raw_c[uid]["ranked_ids"]]
        recs_D = [int(x) for x in raw_d[uid]["ranked_ids"]]

        svd_only_in_B = [r for r in recs_B if r in svd_only]
        svd_only_in_C = [r for r in recs_C if r in svd_only]
        svd_only_in_D = [r for r in recs_D if r in svd_only]

        uua_themes = raw_d[uid].get("uua_themes") or []
        uua_conf   = raw_d[uid].get("uua_confidence", "N/A")

        hits = {
            "A": len([r for r in recs_A if r in gt]),
            "B": len([r for r in recs_B if r in gt]),
            "C": len([r for r in recs_C if r in gt]),
            "D": len([r for r in recs_D if r in gt]),
        }

        block = [
            f"\n{'─'*100}",
            f"  Case {case_idx}: user_id = {uid}",
            f"{'─'*100}",
            f"  val ground truth positives : {len(gt)}",
            f"  train history (top 5)      : {', '.join(history_titles)}",
            f"  UUA inferred themes        : {uua_themes}  (confidence: {uua_conf})",
            f"",
            f"  Candidate pool stats:",
            f"    KNN top-{KNN_TOP_K}              : {len(knn_cands_raw)} games",
            f"    SVD top-{SVD_TOP_K}              : {len(svd_cands_raw)} games",
            f"    SVD-only (knn_score=0)   : {len(svd_only)} games",
            f"    SVD-only in B top-10     : {len(svd_only_in_B)}  <- {'none, explains B=A' if not svd_only_in_B else 'some, knn_score high enough'}",
            f"    SVD-only in C top-10     : {len(svd_only_in_C)}",
            f"    SVD-only in D top-10     : {len(svd_only_in_D)}",
        ]

        block.append(f"")
        block.append(
            f"  {'Rank':<6} {'A: KNN direct':<44} {'B: fusion no-LLM':<44} "
            f"{'C: +Ranker no-UUA':<44} {'D: +UUA+Ranker':<44}"
        )
        block.append(f"  {'':─<6} {'':─<44} {'':─<44} {'':─<44} {'':─<44}")

        for rank in range(10):
            def fmt(recs, idx):
                if idx >= len(recs):
                    return "-"
                app_id = recs[idx]
                title  = get_title(app_id, games_meta)[:38]
                src    = "[SVD]" if app_id in svd_only else ""
                hit    = "[HIT]" if app_id in gt else ""
                return f"{title:<38} {src:<5}{hit:<5}"

            block.append(
                f"  {rank+1:<6} {fmt(recs_A, rank):<44} {fmt(recs_B, rank):<44} "
                f"{fmt(recs_C, rank):<44} {fmt(recs_D, rank):<44}"
            )

        block.append(f"")
        block.append(f"  Hits: A={hits['A']}  B={hits['B']}  C={hits['C']}  D={hits['D']}")
        block.append(f"")
        block.append(f"  [Analysis]")

        if recs_A == recs_B:
            block.append(
                f"  * B=A: all SVD-only candidates fall outside top-10 cutoff (knn_score=0 "
                f"ranks last in the fused pool), fusion does not change top-10 order"
            )
        else:
            diff_ab = [r for r in recs_B if r not in set(recs_A)]
            block.append(
                f"  * B!=A: {len(diff_ab)} SVD-only candidate(s) entered top-10: "
                f"{[get_title(r, games_meta) for r in diff_ab[:2]]}"
            )

        if hits["C"] < hits["A"]:
            dropped  = [r for r in recs_A[:10] if r not in set(recs_C)]
            promoted = [r for r in recs_C[:10] if r not in set(recs_A)]
            block.append(
                f"  * C drops: Ranker without UUA displaced {len(dropped)} high-score KNN "
                f"candidate(s) from top-10, replacing with lower-hit alternatives"
            )
            if dropped:
                block.append(f"    displaced : {[get_title(r, games_meta) for r in dropped[:2]]}")
            if promoted:
                block.append(f"    inserted  : {[get_title(r, games_meta) for r in promoted[:2]]}")

        if hits["D"] > hits["C"]:
            block.append(
                f"  * D>C: UUA profile ({uua_themes}) gives Ranker a semantic basis "
                f"for ordering decisions, improving hit rate"
            )
        elif hits["D"] == hits["C"]:
            block.append(
                f"  * D=C: UUA did not change hit count for this user, "
                f"but ranking order may still differ"
            )

        output_lines.extend(block)

        for rank in range(10):
            def safe_get(recs, idx):
                return int(recs[idx]) if idx < len(recs) else None

            row = {
                "case_idx":       case_idx,
                "user_id":        uid,
                "rank":           rank + 1,
                "A_app_id":       safe_get(recs_A, rank),
                "A_title":        get_title(safe_get(recs_A, rank), games_meta) if safe_get(recs_A, rank) else "",
                "A_hit":          int(safe_get(recs_A, rank) in gt) if safe_get(recs_A, rank) else 0,
                "B_app_id":       safe_get(recs_B, rank),
                "B_title":        get_title(safe_get(recs_B, rank), games_meta) if safe_get(recs_B, rank) else "",
                "B_hit":          int(safe_get(recs_B, rank) in gt) if safe_get(recs_B, rank) else 0,
                "B_is_svd_only":  int(safe_get(recs_B, rank) in svd_only) if safe_get(recs_B, rank) else 0,
                "C_app_id":       safe_get(recs_C, rank),
                "C_title":        get_title(safe_get(recs_C, rank), games_meta) if safe_get(recs_C, rank) else "",
                "C_hit":          int(safe_get(recs_C, rank) in gt) if safe_get(recs_C, rank) else 0,
                "D_app_id":       safe_get(recs_D, rank),
                "D_title":        get_title(safe_get(recs_D, rank), games_meta) if safe_get(recs_D, rank) else "",
                "D_hit":          int(safe_get(recs_D, rank) in gt) if safe_get(recs_D, rank) else 0,
                "uua_themes":     str(uua_themes),
                "gt_count":       len(gt),
            }
            csv_rows.append(row)

    output_lines.append(f"\n{'='*100}")
    output_lines.append("  Summary")
    output_lines.append(f"{'='*100}")
    output_lines.append("  Why B = A:")
    output_lines.append("    fusion_no_llm ranks candidates by knn_score descending and takes top-10.")
    output_lines.append("    SVD-only candidates have knn_score=0, so they rank last in the 45-item")
    output_lines.append("    fused pool and are cut off before top-10. SVD candidates need LLM")
    output_lines.append("    semantic reasoning to enter the final list.")
    output_lines.append("    This is a limitation of the fusion strategy, not evidence that SVD")
    output_lines.append("    candidates are useless -- overlap experiments show SVD-only hits")
    output_lines.append("    account for 22.8% of union hits.")
    output_lines.append("")
    output_lines.append("  Why C < A/B:")
    output_lines.append("    Without UUA, the Ranker only has game metadata (title/rating/price/year)")
    output_lines.append("    and knn_score to work with. Without a user profile, the LLM tends to")
    output_lines.append("    rerank by game popularity and knn_score, but sometimes displaces")
    output_lines.append("    high-hit KNN candidates with games that look semantically plausible")
    output_lines.append("    but have lower actual hit rates.")
    output_lines.append("")
    output_lines.append("  Why D > C:")
    output_lines.append("    The structured UUA profile (price preference, playtime, theme affinity)")
    output_lines.append("    gives the Ranker a semantic basis for decisions, reducing arbitrary")
    output_lines.append("    reordering and aligning recommendations more closely with the user's")
    output_lines.append("    historical preference patterns.")
    output_lines.append("")
    output_lines.append("  Note: with n=50 users, hit-count differences are on the order of 2-3.")
    output_lines.append("        These findings are exploratory and not statistically significant.")
    output_lines.append("=" * 100)

    report  = "\n".join(output_lines)
    out_txt = os.path.join(RESULT_DIR, "ablation_case_study.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)

    out_csv = os.path.join(RESULT_DIR, "ablation_case_study.csv")
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n[SAVE] text report -> {out_txt}")
    print(f"[SAVE] CSV data    -> {out_csv}")


if __name__ == "__main__":
    main()