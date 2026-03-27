#!/usr/bin/env python3
"""
Perturbation Analysis: Compare original vs perturbed LLM outputs.
Tests whether sensitivity to input changes predicts quality.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats
from collections import Counter


RECOVERABLE_ERRORS = {'duplicates_removed', 'out_of_range_removed'}


def load_sasrec_data(path):
    with open(path) as f:
        data = json.load(f)
    sasrec_rankings = {}
    ground_truth = {}
    for user_id, info in data.items():
        sasrec_rankings[user_id] = info['items'][:20]
        ground_truth[user_id] = info['ground_truth']
    return sasrec_rankings, ground_truth


def load_jsonl(path):
    records = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            records[str(rec['user_id'])] = rec
    return records


def is_usable(rec):
    return rec.get('is_valid', False) or (rec.get('error_type', '') or '') in RECOVERABLE_ERRORS


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


def kendalltau_items(list_a, list_b):
    """Kendall's tau on common items between two ranked lists."""
    common = list(set(list_a) & set(list_b))
    if len(common) < 2:
        return float('nan'), float('nan')
    ranks_a = [list_a.index(item) for item in common]
    ranks_b = [list_b.index(item) for item in common]
    return stats.kendalltau(ranks_a, ranks_b)


def analyze_perturbation(orig_path, pert_path, sasrec_rankings, ground_truth, k=10):
    orig_recs = load_jsonl(orig_path)
    pert_recs = load_jsonl(pert_path)

    common_users = set(orig_recs.keys()) & set(pert_recs.keys()) & set(sasrec_rankings.keys())

    results = []
    n_orig_only = n_pert_only = n_neither = 0

    for uid in sorted(common_users):
        orig = orig_recs[uid]
        pert = pert_recs[uid]

        orig_ok = is_usable(orig)
        pert_ok = is_usable(pert)

        if not orig_ok and not pert_ok:
            n_neither += 1
            continue
        if not pert_ok:
            n_orig_only += 1
            continue
        if not orig_ok:
            n_pert_only += 1
            continue

        orig_items = orig['filled_item_ids'][:k]
        pert_items = pert['filled_item_ids'][:k]
        sasrec_top10 = sasrec_rankings[uid][:k]
        gt = ground_truth[uid]

        # Overlap
        overlap = len(set(orig_items) & set(pert_items))

        # Kendall's tau between outputs
        tau, tau_p = kendalltau_items(orig_items, pert_items)
        sensitivity = 1.0 - tau if not np.isnan(tau) else float('nan')

        # Quality
        orig_ndcg = ndcg_at_k(orig_items, gt, k)
        pert_ndcg = ndcg_at_k(pert_items, gt, k)
        sas_ndcg  = ndcg_at_k(sasrec_top10, gt, k)
        orig_mrr  = mrr_at_k(orig_items, gt, k)
        pert_mrr  = mrr_at_k(pert_items, gt, k)
        sas_mrr   = mrr_at_k(sasrec_top10, gt, k)

        results.append({
            'user_id': uid,
            'overlap': overlap,
            'tau': tau,
            'sensitivity': sensitivity,
            'orig_ndcg': orig_ndcg,
            'pert_ndcg': pert_ndcg,
            'sas_ndcg': sas_ndcg,
            'gain_orig': orig_ndcg - sas_ndcg,
            'gain_pert': pert_ndcg - sas_ndcg,
            'delta_pert_vs_orig': pert_ndcg - orig_ndcg,
            'orig_mrr': orig_mrr,
            'pert_mrr': pert_mrr,
            'sas_mrr': sas_mrr,
        })

    validity = {
        'common_users': len(common_users),
        'both_valid': len(results),
        'orig_only': n_orig_only,
        'pert_only': n_pert_only,
        'neither': n_neither,
    }
    return results, validity


def print_results(label, results, validity):
    n = len(results)
    print(f"\n{'='*70}")
    print(f"  PERTURBATION ANALYSIS: {label}")
    print(f"  {n} users with both outputs valid")
    print(f"  (of {validity['common_users']} common users)")
    if validity['orig_only'] or validity['pert_only'] or validity['neither']:
        print(f"  [orig_only={validity['orig_only']}, "
              f"pert_only={validity['pert_only']}, "
              f"neither={validity['neither']}]")
    print(f"{'='*70}")

    if n == 0:
        print("  No valid pairs to analyze.")
        return

    # ── Output stability ──
    overlaps = [r['overlap'] for r in results]
    taus = [r['tau'] for r in results if not np.isnan(r['tau'])]
    sens = [r['sensitivity'] for r in results if not np.isnan(r['sensitivity'])]

    print(f"\n  OUTPUT STABILITY (original vs perturbed)")
    print(f"  {'Top-10 Overlap:':<30} {np.mean(overlaps):.2f} +/- {np.std(overlaps):.2f}  "
          f"(min={min(overlaps)}, max={max(overlaps)})")
    if taus:
        print(f"  {'Kendall tau:':<30} {np.mean(taus):.4f} +/- {np.std(taus):.4f}")
    if sens:
        print(f"  {'Sensitivity (1-tau):':<30} {np.mean(sens):.4f} +/- {np.std(sens):.4f}")

    # ── Overlap distribution ──
    dist = Counter(overlaps)
    print(f"\n  OVERLAP DISTRIBUTION")
    for ov in sorted(dist.keys()):
        bar = '#' * int(dist[ov] / max(dist.values()) * 40)
        print(f"  {ov:2d}/10: {dist[ov]:4d} ({100*dist[ov]/n:5.1f}%) {bar}")

    # ── Quality comparison ──
    delta_pert = [r['delta_pert_vs_orig'] for r in results]
    n_better = sum(1 for d in delta_pert if d > 0)
    n_worse  = sum(1 for d in delta_pert if d < 0)
    n_same   = sum(1 for d in delta_pert if d == 0)

    print(f"\n  QUALITY: PERTURBED vs ORIGINAL")
    print(f"  {'Mean NDCG (original):':<30} {np.mean([r['orig_ndcg'] for r in results]):.4f}")
    print(f"  {'Mean NDCG (perturbed):':<30} {np.mean([r['pert_ndcg'] for r in results]):.4f}")
    print(f"  {'Mean ΔNDCG (pert-orig):':<30} {np.mean(delta_pert):+.4f}")
    print(f"  {'Perturbed better:':<30} {n_better:4d} ({100*n_better/n:.1f}%)")
    print(f"  {'Perturbed worse:':<30} {n_worse:4d} ({100*n_worse/n:.1f}%)")
    print(f"  {'Same:':<30} {n_same:4d} ({100*n_same/n:.1f}%)")

    # Wilcoxon signed-rank
    if n_better + n_worse >= 5:
        try:
            stat, p = stats.wilcoxon(
                [r['orig_ndcg'] for r in results],
                [r['pert_ndcg'] for r in results],
                zero_method='wilcox')
            sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
            print(f"  {'Wilcoxon signed-rank:':<30} stat={stat:.1f}, p={p:.4f} ({sig})")
        except ValueError:
            pass

    # ── CRITICAL: Sensitivity vs quality correlations ──
    print(f"\n  CORRELATION: SENSITIVITY vs QUALITY")
    print(f"  (Does perturbation-sensitivity predict whether LLM helps?)")

    sens_arr = np.array([r['sensitivity'] for r in results])
    gain_arr = np.array([r['gain_orig'] for r in results])
    ndcg_arr = np.array([r['orig_ndcg'] for r in results])
    qvar_arr = np.abs(np.array([r['delta_pert_vs_orig'] for r in results]))

    mask = ~np.isnan(sens_arr)
    if mask.sum() >= 10 and np.std(sens_arr[mask]) > 0:
        tests = [
            ('sensitivity vs gain(orig-SASRec)', gain_arr),
            ('sensitivity vs orig NDCG',         ndcg_arr),
            ('sensitivity vs |ΔNDCG(pert-orig)|', qvar_arr),
        ]
        for name, target in tests:
            if np.std(target[mask]) > 0:
                rho, p = stats.spearmanr(sens_arr[mask], target[mask])
                sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
                print(f"  {name:<40} rho={rho:+.4f}  p={p:.4f} {sig}")
            else:
                print(f"  {name:<40} -- no variance in target")
    else:
        print(f"  Insufficient variance in sensitivity scores")

    # ── Quartile analysis ──
    valid_r = [r for r in results if not np.isnan(r['sensitivity'])]
    if len(valid_r) >= 20:
        sorted_r = sorted(valid_r, key=lambda r: r['sensitivity'])
        q_size = len(sorted_r) // 4

        print(f"\n  QUARTILE ANALYSIS: Sensitivity -> Quality")
        print(f"  {'Quartile':<15} {'N':>5} {'Sens':>8} {'OrigNDCG':>10} "
              f"{'PertNDCG':>10} {'GainVsSAS':>10} {'Overlap':>8}")
        print(f"  {'-'*72}")

        for qi in range(4):
            start = qi * q_size
            end = (qi + 1) * q_size if qi < 3 else len(sorted_r)
            qr = sorted_r[start:end]

            print(f"  Q{qi+1} {'(stable)' if qi==0 else '(volatile)' if qi==3 else '':>9}"
                  f" {len(qr):5d}"
                  f" {np.mean([r['sensitivity'] for r in qr]):8.4f}"
                  f" {np.mean([r['orig_ndcg'] for r in qr]):10.4f}"
                  f" {np.mean([r['pert_ndcg'] for r in qr]):10.4f}"
                  f" {np.mean([r['gain_orig'] for r in qr]):+10.4f}"
                  f" {np.mean([r['overlap'] for r in qr]):8.1f}")

    # ── Among users where perturbation changed quality ──
    changed = [r for r in results if r['delta_pert_vs_orig'] != 0]
    if len(changed) >= 5:
        better = [r for r in changed if r['delta_pert_vs_orig'] > 0]
        worse  = [r for r in changed if r['delta_pert_vs_orig'] < 0]

        print(f"\n  AMONG USERS WHERE PERTURBATION CHANGED QUALITY (n={len(changed)}):")
        for grp, name in [(better, 'Pert better'), (worse, 'Pert worse')]:
            if grp:
                s = [r['sensitivity'] for r in grp if not np.isnan(r['sensitivity'])]
                o = [r['overlap'] for r in grp]
                print(f"  {name:<15} (n={len(grp):3d}):  "
                      f"avg_sens={np.mean(s):.4f}  avg_overlap={np.mean(o):.1f}")

        if len(better) >= 3 and len(worse) >= 3:
            s_b = [r['sensitivity'] for r in better if not np.isnan(r['sensitivity'])]
            s_w = [r['sensitivity'] for r in worse  if not np.isnan(r['sensitivity'])]
            if s_b and s_w:
                u, p = stats.mannwhitneyu(s_b, s_w, alternative='two-sided')
                sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
                print(f"  Mann-Whitney (better vs worse sens): U={u:.1f}, p={p:.4f} ({sig})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='evaluation_data')
    parser.add_argument('--log_dir',  default='evaluation_data/reranking_logs')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--output', default='analysis_results/perturbation_analysis.json')
    args = parser.parse_args()

    sasrec_rankings, ground_truth = load_sasrec_data(
        Path(args.data_dir) / 'sasrec_candidates.json')

    # ── Define comparisons ──
    comparisons = [
        {
            'label': 'Prompt Rephrase: Qwen t=0.1 (main vs rephrase_v1)',
            'original': 'qwen_run0_main_t0.1.jsonl',
            'perturbed': 'qwen_run0_rephrase_v1_t0.1.jsonl',
        },
        # Multi-run comparisons (same prompt, same temp — pure stochastic variation)
        {
            'label': 'Run Variance: Qwen t=0.1 (run0 vs run1)',
            'original': 'qwen_run0_main_t0.1.jsonl',
            'perturbed': 'qwen_run1_main_t0.1.jsonl',
        },
        {
            'label': 'Run Variance: Qwen t=0.7 (run0 vs run1)',
            'original': 'qwen_run0_main_t0.7.jsonl',
            'perturbed': 'qwen_run1_main_t0.7.jsonl',
        },
        {
            'label': 'Run Variance: Nemotron t=0.7 (run0 vs run1)',
            'original': 'nemotron_run0_main_t0.7.jsonl',
            'perturbed': 'nemotron_run1_main_t0.7.jsonl',
        },
        
                # Candidate order perturbations
        {
            'label': 'Candidate Reverse: Qwen t=0.1',
            'original': 'qwen_run0_main_t0.1.jsonl',
            'perturbed': 'qwen_run0_main_reverse_t0.1.jsonl',
        },
        {
            'label': 'Candidate Random: Qwen t=0.1',
            'original': 'qwen_run0_main_t0.1.jsonl',
            'perturbed': 'qwen_run0_main_random_t0.1.jsonl',
        },
        {
            'label': 'Candidate Reverse: Qwen t=0.0',
            'original': 'qwen_run0_main_t0.0.jsonl',
            'perturbed': 'qwen_run0_main_reverse_t0.0.jsonl',
        },
        
        # Diversity prompt comparisons
        {
            'label': 'Diversity v1: Qwen t=0.1',
            'original': 'qwen_run0_main_t0.1.jsonl',
            'perturbed': 'qwen_run0_diversity_v1_t0.1.jsonl',
        },
        {
            'label': 'Diversity v2: Qwen t=0.1',
            'original': 'qwen_run0_main_t0.1.jsonl',
            'perturbed': 'qwen_run0_diversity_v2_t0.1.jsonl',
        },
    ]

    all_results = {}

    for comp in comparisons:
        orig_path = Path(args.log_dir) / comp['original']
        pert_path = Path(args.log_dir) / comp['perturbed']

        if not orig_path.exists():
            print(f"\n  Skipping {comp['label']}: {orig_path} not found")
            continue
        if not pert_path.exists():
            print(f"\n  Skipping {comp['label']}: {pert_path} not found")
            continue

        results, validity = analyze_perturbation(
            str(orig_path), str(pert_path),
            sasrec_rankings, ground_truth, args.k)

        print_results(comp['label'], results, validity)
        all_results[comp['label']] = {'validity': validity, 'results': results}

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  All results saved to {out}")


if __name__ == '__main__':
    main()