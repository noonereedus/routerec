#!/usr/bin/env python3
"""
Step 2: Displacement Analysis
Measures how much the LLM changes the SASRec ranking and whether
displacement magnitude correlates with quality changes.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats

# ── Configs (gpt-oss excluded: 0% valid, no displacement to measure) ──
MODEL_CONFIGS = [
    {'label': 'Qwen_t0.0',    'model_name': 'qwen',     'temperature': 0.0},
    {'label': 'Qwen_t0.1',    'model_name': 'qwen',     'temperature': 0.1},
    {'label': 'Qwen_t0.7',    'model_name': 'qwen',     'temperature': 0.7},
    {'label': 'Nemotron_t0.1', 'model_name': 'nemotron', 'temperature': 0.1},
    {'label': 'Nemotron_t0.7', 'model_name': 'nemotron', 'temperature': 0.7},
]

RECOVERABLE_ERRORS = {'duplicates_removed', 'out_of_range_removed'}


def load_sasrec_data(path):
    with open(path) as f:
        data = json.load(f)
    sasrec_rankings = {}
    ground_truth = {}
    for user_id, info in data.items():
        sasrec_rankings[user_id] = info['items'][:20]  # top 20 candidates
        ground_truth[user_id] = info['ground_truth']
    return sasrec_rankings, ground_truth


def load_jsonl(path):
    records = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            records[str(rec['user_id'])] = rec
    return records


def ndcg_at_k(ranked_list, gt, k=10):
    top_k = ranked_list[:k]
    if gt in top_k:
        rank = top_k.index(gt) + 1
        return 1.0 / np.log2(rank + 1)
    return 0.0


def mrr_at_k(ranked_list, gt, k=10):
    top_k = ranked_list[:k]
    if gt in top_k:
        rank = top_k.index(gt) + 1
        return 1.0 / rank
    return 0.0


def compute_displacement(sasrec_top20, llm_top10, k=10):
    """Compute displacement metrics between SASRec top-10 and LLM top-10."""
    sasrec_top10 = sasrec_top20[:k]

    sasrec_set = set(sasrec_top10)
    llm_set = set(llm_top10[:k])

    common = sasrec_set & llm_set
    overlap = len(common)

    # Items promoted from SASRec positions 11-20 into LLM top-10
    promoted = llm_set & set(sasrec_top20[k:])
    num_promoted = len(promoted)

    # Average original SASRec rank of promoted items
    if num_promoted > 0:
        promo_ranks = [sasrec_top20.index(item) + 1 for item in promoted]
        promotion_depth = np.mean(promo_ranks)
    else:
        promotion_depth = 0.0

    # Among common items: rank displacement and Kendall's tau
    if len(common) >= 2:
        sasrec_ranks = [sasrec_top10.index(item) + 1 for item in common]
        llm_ranks = [llm_top10.index(item) + 1 for item in common]

        displacements = [abs(s - l) for s, l in zip(sasrec_ranks, llm_ranks)]
        mean_displacement = np.mean(displacements)
        max_displacement = float(np.max(displacements))

        tau, tau_p = stats.kendalltau(sasrec_ranks, llm_ranks)
    else:
        mean_displacement = float('nan')
        max_displacement = float('nan')
        tau = float('nan')
        tau_p = float('nan')

    return {
        'overlap': overlap,
        'kendall_tau': tau,
        'tau_pvalue': tau_p,
        'mean_displacement': mean_displacement,
        'max_displacement': max_displacement,
        'num_promoted': num_promoted,
        'promotion_depth': promotion_depth,
    }


def analyze_config(label, jsonl_path, sasrec_rankings, ground_truth, k=10):
    """Run displacement analysis for one model config."""
    records = load_jsonl(jsonl_path)
    results = []

    for user_id, rec in records.items():
        if user_id not in sasrec_rankings:
            continue

        # Include valid + recoverable; skip catastrophic
        is_valid = rec.get('is_valid', False)
        error_type = rec.get('error_type', '') or ''
        is_recoverable = error_type in RECOVERABLE_ERRORS

        if not is_valid and not is_recoverable:
            continue

        sasrec_top20 = sasrec_rankings[user_id]
        llm_top10 = rec['filled_item_ids'][:k]
        gt = ground_truth[user_id]

        # Quality deltas
        sasrec_ndcg = ndcg_at_k(sasrec_top20[:k], gt, k)
        llm_ndcg = ndcg_at_k(llm_top10, gt, k)
        delta_ndcg = llm_ndcg - sasrec_ndcg

        sasrec_mrr = mrr_at_k(sasrec_top20[:k], gt, k)
        llm_mrr = mrr_at_k(llm_top10, gt, k)
        delta_mrr = llm_mrr - sasrec_mrr

        # Displacement
        disp = compute_displacement(sasrec_top20, llm_top10, k)

        results.append({
            'user_id': user_id,
            'is_valid': is_valid,
            'is_recoverable': is_recoverable,
            'delta_ndcg': delta_ndcg,
            'delta_mrr': delta_mrr,
            'sasrec_ndcg': sasrec_ndcg,
            'llm_ndcg': llm_ndcg,
            **disp,
        })

    return results


def print_analysis(label, results):
    """Print displacement results for one config."""
    n = len(results)

    print(f"\n{'='*70}")
    print(f"  DISPLACEMENT ANALYSIS: {label}")
    print(f"  ({n} users with valid/recoverable LLM output)")
    print(f"{'='*70}")

    # ── Population-level displacement stats ──
    overlaps      = [r['overlap'] for r in results]
    taus          = [r['kendall_tau'] for r in results if not np.isnan(r['kendall_tau'])]
    mean_disps    = [r['mean_displacement'] for r in results if not np.isnan(r['mean_displacement'])]
    max_disps     = [r['max_displacement'] for r in results if not np.isnan(r['max_displacement'])]
    promotions    = [r['num_promoted'] for r in results]

    print(f"\n  DISPLACEMENT STATISTICS")
    print(f"  {'Top-10 Overlap:':<30} {np.mean(overlaps):.2f} +/- {np.std(overlaps):.2f}  "
          f"(min={min(overlaps)}, max={max(overlaps)})")
    if taus:
        print(f"  {'Kendall tau (common items):':<30} {np.mean(taus):.4f} +/- {np.std(taus):.4f}")
    if mean_disps:
        print(f"  {'Mean Rank Displacement:':<30} {np.mean(mean_disps):.2f} +/- {np.std(mean_disps):.2f}")
    if max_disps:
        print(f"  {'Max Rank Displacement:':<30} {np.mean(max_disps):.2f} +/- {np.std(max_disps):.2f}")
    print(f"  {'Promoted from 11-20:':<30} {np.mean(promotions):.2f} +/- {np.std(promotions):.2f}")

    # ── Quality delta distribution ──
    deltas = [r['delta_ndcg'] for r in results]
    n_improved = sum(1 for d in deltas if d > 0)
    n_hurt     = sum(1 for d in deltas if d < 0)
    n_tied     = sum(1 for d in deltas if d == 0)

    print(f"\n  QUALITY CHANGE DISTRIBUTION (delta-NDCG@10)")
    print(f"  {'Improved:':<20} {n_improved:4d}  ({100*n_improved/n:.1f}%)")
    print(f"  {'Hurt:':<20} {n_hurt:4d}  ({100*n_hurt/n:.1f}%)")
    print(f"  {'Tied:':<20} {n_tied:4d}  ({100*n_tied/n:.1f}%)")
    print(f"  {'Mean delta-NDCG:':<20} {np.mean(deltas):+.4f}")

    # ── Correlations ──
    print(f"\n  CORRELATION: DISPLACEMENT vs QUALITY")

    disp_metrics = {
        'overlap':           np.array([r['overlap'] for r in results]),
        'kendall_tau':       np.array([r['kendall_tau'] for r in results]),
        'mean_displacement': np.array([r['mean_displacement'] for r in results]),
        'num_promoted':      np.array([r['num_promoted'] for r in results]),
    }

    delta_ndcg_arr = np.array([r['delta_ndcg'] for r in results])
    delta_mrr_arr  = np.array([r['delta_mrr'] for r in results])

    for mname, vals in disp_metrics.items():
        mask = ~np.isnan(vals) & ~np.isnan(delta_ndcg_arr)
        if mask.sum() < 10:
            print(f"  {mname:<25} -- insufficient data")
            continue

        v      = vals[mask]
        d_ndcg = delta_ndcg_arr[mask]
        d_mrr  = delta_mrr_arr[mask]

        if np.std(v) == 0 or np.std(d_ndcg) == 0:
            print(f"  {mname:<25} -- no variance")
            continue

        rho_n, p_n = stats.spearmanr(v, d_ndcg)
        rho_m, p_m = stats.spearmanr(v, d_mrr)

        sig_n = "***" if p_n < .001 else "**" if p_n < .01 else "*" if p_n < .05 else ""
        sig_m = "***" if p_m < .001 else "**" if p_m < .01 else "*" if p_m < .05 else ""

        print(f"  {mname:<25} rho(dNDCG)={rho_n:+.4f}  p={p_n:.4f} {sig_n}")
        print(f"  {'':<25} rho(dMRR) ={rho_m:+.4f}  p={p_m:.4f} {sig_m}")

    # ── Quartile analysis on mean displacement ──
    valid_r = [r for r in results if not np.isnan(r['mean_displacement'])]
    if len(valid_r) >= 20:
        print(f"\n  QUARTILE ANALYSIS: Mean Displacement -> Quality")
        sorted_r = sorted(valid_r, key=lambda r: r['mean_displacement'])
        q_size = len(sorted_r) // 4

        print(f"  {'Quartile':<15} {'N':>5} {'Avg Disp':>10} {'Avg dNDCG':>12} "
              f"{'Avg dMRR':>12} {'Overlap':>10}")
        print(f"  {'-'*67}")

        for qi in range(4):
            start = qi * q_size
            end = (qi + 1) * q_size if qi < 3 else len(sorted_r)
            qr = sorted_r[start:end]

            print(f"  Q{qi+1} {'(lowest)' if qi==0 else '(highest)' if qi==3 else '':>9}"
                  f" {len(qr):5d}"
                  f" {np.mean([r['mean_displacement'] for r in qr]):10.2f}"
                  f" {np.mean([r['delta_ndcg'] for r in qr]):+12.4f}"
                  f" {np.mean([r['delta_mrr'] for r in qr]):+12.4f}"
                  f" {np.mean([r['overlap'] for r in qr]):10.1f}")

    # ── Focus: users where quality actually changed ──
    changed = [r for r in results if r['delta_ndcg'] != 0]
    if len(changed) >= 5:
        improved = [r for r in changed if r['delta_ndcg'] > 0]
        hurt     = [r for r in changed if r['delta_ndcg'] < 0]

        print(f"\n  AMONG USERS WHERE QUALITY CHANGED (n={len(changed)}):")

        if improved:
            imp_d = [r['mean_displacement'] for r in improved if not np.isnan(r['mean_displacement'])]
            imp_o = [r['overlap'] for r in improved]
            imp_p = [r['num_promoted'] for r in improved]
            print(f"  Improved (n={len(improved):3d}):  "
                  f"avg_disp={np.mean(imp_d):.2f}  avg_overlap={np.mean(imp_o):.1f}  "
                  f"avg_promoted={np.mean(imp_p):.1f}")

        if hurt:
            h_d = [r['mean_displacement'] for r in hurt if not np.isnan(r['mean_displacement'])]
            h_o = [r['overlap'] for r in hurt]
            h_p = [r['num_promoted'] for r in hurt]
            print(f"  Hurt     (n={len(hurt):3d}):  "
                  f"avg_disp={np.mean(h_d):.2f}  avg_overlap={np.mean(h_o):.1f}  "
                  f"avg_promoted={np.mean(h_p):.1f}")

        # Mann-Whitney: do improved and hurt users have different displacement?
        if len(improved) >= 5 and len(hurt) >= 5:
            imp_disps  = [r['mean_displacement'] for r in improved
                          if not np.isnan(r['mean_displacement'])]
            hurt_disps = [r['mean_displacement'] for r in hurt
                          if not np.isnan(r['mean_displacement'])]
            if imp_disps and hurt_disps:
                u, p = stats.mannwhitneyu(imp_disps, hurt_disps, alternative='two-sided')
                sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
                print(f"  Mann-Whitney U (improved vs hurt disp): U={u:.1f}, p={p:.4f} ({sig})")


def main():
    parser = argparse.ArgumentParser(description='Displacement analysis')
    parser.add_argument('--data_dir', default='evaluation_data')
    parser.add_argument('--log_dir',  default='evaluation_data/reranking_logs')
    parser.add_argument('--k',        type=int, default=10)
    parser.add_argument('--output',   default='analysis_results/displacement_analysis.json')
    args = parser.parse_args()

    sasrec_rankings, ground_truth = load_sasrec_data(
        Path(args.data_dir) / 'sasrec_candidates.json')

    all_results = {}

    for config in MODEL_CONFIGS:
        label = config['label']
        temp_str = f"t{config['temperature']}"
        jsonl_name = f"{config['model_name']}_run0_main_{temp_str}.jsonl"
        jsonl_path = Path(args.log_dir) / jsonl_name

        if not jsonl_path.exists():
            print(f"  Skipping {label}: {jsonl_path} not found")
            continue

        results = analyze_config(label, str(jsonl_path),
                                 sasrec_rankings, ground_truth, args.k)
        print_analysis(label, results)
        all_results[label] = results

    # Save
    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  All results saved to {out}")


if __name__ == '__main__':
    main()