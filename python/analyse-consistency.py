"""
analyze_consistency.py

Core analysis:
  1. Load K runs for each model
  2. Compute per-query self-consistency (mean pairwise Kendall tau)
  3. Compute per-query quality delta (LLM NDCG - SASRec NDCG)
  4. Correlate the two
  5. Calibrate fallback threshold on validation split
  6. Produce the stability-quality tradeoff curve
"""

import json
import os
import glob
import argparse
import numpy as np
from scipy import stats
from itertools import combinations
from collections import defaultdict
from metrics import ndcg_at_k, hit_rate_at_k


def load_logs(log_dir, model_name, prompt_version="main", temperature=0.1):
    """Load all run logs for a given model/prompt/temperature condition."""
    pattern = os.path.join(
        log_dir,
        f"{model_name}_run*_{prompt_version}_t{temperature}.jsonl"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")

    # Organise: user_id -> run_index -> record
    by_user = defaultdict(dict)
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                rec = json.loads(line)
                by_user[rec['user_id']][rec['run_index']] = rec

    print(f"Loaded {len(files)} run files for {model_name}, "
          f"{len(by_user)} users")
    return by_user


def kendall_tau_lists(list_a, list_b):
    """
    Kendall tau between two ranked item lists.
    Items not in both lists are dropped. Returns tau, or NaN if
    fewer than 2 common items.
    """
    common = set(list_a) & set(list_b)
    if len(common) < 2:
        return float('nan')

    # Build rank maps for common items only
    rank_a = {item: i for i, item in enumerate(list_a) if item in common}
    rank_b = {item: i for i, item in enumerate(list_b) if item in common}

    items = sorted(common, key=lambda x: rank_a[x])
    ranks_a = [rank_a[item] for item in items]
    ranks_b = [rank_b[item] for item in items]

    tau, _ = stats.kendalltau(ranks_a, ranks_b)
    return tau


def jaccard_at_k(list_a, list_b, k=10):
    """Top-k Jaccard overlap."""
    set_a = set(list_a[:k])
    set_b = set(list_b[:k])
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_self_consistency(by_user):
    """
    For each user, compute mean pairwise Kendall tau across runs
    using the RAW LLM output (llm_item_ids), not the filled version.
    """
    results = {}

    for user_id, runs in by_user.items():
        run_indices = sorted(runs.keys())
        if len(run_indices) < 2:
            continue

        # Collect valid LLM outputs
        valid_lists = []
        for ri in run_indices:
            rec = runs[ri]
            if rec['is_valid'] and len(rec['llm_item_ids']) >= 2:
                valid_lists.append(rec['llm_item_ids'])

        if len(valid_lists) < 2:
            # Cannot compute consistency — all or most runs failed
            results[user_id] = {
                'tau_mean': float('nan'),
                'tau_min': float('nan'),
                'jaccard_mean': float('nan'),
                'num_valid_runs': len(valid_lists),
                'num_total_runs': len(run_indices),
            }
            continue

        taus = []
        jaccards = []
        for la, lb in combinations(valid_lists, 2):
            taus.append(kendall_tau_lists(la, lb))
            jaccards.append(jaccard_at_k(la, lb, k=10))

        taus = [t for t in taus if not np.isnan(t)]

        results[user_id] = {
            'tau_mean': float(np.mean(taus)) if taus else float('nan'),
            'tau_min': float(np.min(taus)) if taus else float('nan'),
            'jaccard_mean': float(np.mean(jaccards)),
            'num_valid_runs': len(valid_lists),
            'num_total_runs': len(run_indices),
        }

    return results


def compute_quality_deltas(by_user, ground_truths, sasrec_rankings, k=10):
    """
    For each user, compute NDCG for the LLM output (averaged across runs)
    and for SASRec. Return delta = LLM - SASRec.
    Uses filled_item_ids since that is what the user would actually see.
    """
    results = {}

    for user_id, runs in by_user.items():
        if user_id not in ground_truths:
            continue
        gt = ground_truths[user_id]
        sasrec_list = sasrec_rankings[user_id][:k]

        sasrec_ndcg = ndcg_at_k(sasrec_list, gt, k)
        sasrec_hr = hit_rate_at_k(sasrec_list, gt, k)

        # Average LLM quality across runs
        llm_ndcgs = []
        llm_hrs = []
        for ri, rec in runs.items():
            filled = rec['filled_item_ids'][:k]
            llm_ndcgs.append(ndcg_at_k(filled, gt, k))
            llm_hrs.append(hit_rate_at_k(filled, gt, k))

        results[user_id] = {
            'sasrec_ndcg': sasrec_ndcg,
            'sasrec_hr': sasrec_hr,
            'llm_ndcg_mean': float(np.mean(llm_ndcgs)),
            'llm_hr_mean': float(np.mean(llm_hrs)),
            'delta_ndcg': float(np.mean(llm_ndcgs)) - sasrec_ndcg,
            'delta_hr': float(np.mean(llm_hrs)) - sasrec_hr,
            'llm_better': int(np.mean(llm_ndcgs) > sasrec_ndcg),
        }

    return results


def correlate_consistency_and_quality(consistency, quality):
    """
    The central analysis: does self-consistency predict quality gain?
    """
    common = set(consistency.keys()) & set(quality.keys())
    common = [u for u in common if not np.isnan(consistency[u]['tau_mean'])]

    taus = [consistency[u]['tau_mean'] for u in common]
    deltas = [quality[u]['delta_ndcg'] for u in common]
    llm_better = [quality[u]['llm_better'] for u in common]

    print(f"\n{'='*60}")
    print(f"  CONSISTENCY-QUALITY CORRELATION ({len(common)} users)")
    print(f"{'='*60}")

    # Spearman correlation
    rho, p_val = stats.spearmanr(taus, deltas)
    print(f"\n  Spearman rho (tau vs delta_NDCG): {rho:.4f}  (p={p_val:.6f})")

    # Point-biserial: does consistency predict LLM-wins?
    rpb, p_pb = stats.pointbiserialr(llm_better, taus)
    print(f"  Point-biserial (tau vs LLM_better): {rpb:.4f}  (p={p_pb:.6f})")

    # Bucketed analysis
    print(f"\n  {'Consistency bucket':<25} {'N':>5} {'Mean ΔNDCG':>12} "
          f"{'LLM wins %':>12} {'Mean τ':>10}")
    print(f"  {'-'*65}")

    buckets = [
        ('High (τ > 0.8)', lambda t: t > 0.8),
        ('Medium (0.5 < τ ≤ 0.8)', lambda t: 0.5 < t <= 0.8),
        ('Low (τ ≤ 0.5)', lambda t: t <= 0.5),
    ]

    for label, pred in buckets:
        mask = [i for i, t in enumerate(taus) if pred(t)]
        if not mask:
            print(f"  {label:<25} {'0':>5}")
            continue
        b_deltas = [deltas[i] for i in mask]
        b_wins = [llm_better[i] for i in mask]
        b_taus = [taus[i] for i in mask]
        print(f"  {label:<25} {len(mask):>5} {np.mean(b_deltas):>12.4f} "
              f"{np.mean(b_wins)*100:>11.1f}% {np.mean(b_taus):>10.3f}")

    return common, taus, deltas, llm_better


def fallback_threshold_sweep(consistency, quality, ground_truths,
                             sasrec_rankings, by_user, k=10):
    """
    Sweep the self-consistency threshold from 0.0 to 1.0.
    At each threshold: if tau >= threshold, use LLM; else use SASRec.
    Report system NDCG and fallback rate.

    This produces the stability-quality tradeoff curve.
    """
    common = set(consistency.keys()) & set(quality.keys())
    common = [u for u in common if not np.isnan(consistency[u]['tau_mean'])]

    thresholds = np.arange(0.0, 1.05, 0.05)

    print(f"\n{'='*60}")
    print(f"  FALLBACK THRESHOLD SWEEP ({len(common)} users)")
    print(f"{'='*60}")
    print(f"  {'Threshold':>10} {'System NDCG':>12} {'Fallback %':>12} "
          f"{'Accepted NDCG':>14} {'Rejected NDCG':>14}")
    print(f"  {'-'*65}")

    sweep_results = []

    for thresh in thresholds:
        system_ndcgs = []
        accepted_ndcgs = []
        rejected_ndcgs = []
        fallback_count = 0

        for user_id in common:
            gt = ground_truths[user_id]
            sasrec_ndcg = quality[user_id]['sasrec_ndcg']
            llm_ndcg = quality[user_id]['llm_ndcg_mean']
            tau = consistency[user_id]['tau_mean']

            if tau >= thresh:
                # Accept LLM
                system_ndcgs.append(llm_ndcg)
                accepted_ndcgs.append(llm_ndcg)
            else:
                # Fallback to SASRec
                system_ndcgs.append(sasrec_ndcg)
                rejected_ndcgs.append(sasrec_ndcg)
                fallback_count += 1

        ftr = fallback_count / len(common)
        sys_ndcg = np.mean(system_ndcgs)
        acc_ndcg = np.mean(accepted_ndcgs) if accepted_ndcgs else 0.0
        rej_ndcg = np.mean(rejected_ndcgs) if rejected_ndcgs else 0.0

        print(f"  {thresh:>10.2f} {sys_ndcg:>12.4f} {ftr*100:>11.1f}% "
              f"{acc_ndcg:>14.4f} {rej_ndcg:>14.4f}")

        sweep_results.append({
            'threshold': float(thresh),
            'system_ndcg': float(sys_ndcg),
            'fallback_rate': float(ftr),
            'accepted_ndcg': float(acc_ndcg),
            'rejected_ndcg': float(rej_ndcg),
        })

    return sweep_results


def report_dependability_summary(by_user):
    """
    Report basic dependability metrics across all runs.
    """
    total_records = 0
    valid_count = 0
    error_counts = defaultdict(int)
    latencies = []

    for user_id, runs in by_user.items():
        for ri, rec in runs.items():
            total_records += 1
            if rec['is_valid']:
                valid_count += 1
            if rec['error_type']:
                error_counts[rec['error_type']] += 1
            if rec['latency_seconds'] is not None:
                latencies.append(rec['latency_seconds'])

    print(f"\n{'='*60}")
    print(f"  DEPENDABILITY SUMMARY ({total_records} total records)")
    print(f"{'='*60}")
    print(f"  Valid output rate:  {valid_count/total_records*100:.1f}%")
    print(f"  Invalid output rate: {(total_records-valid_count)/total_records*100:.1f}%")

    if error_counts:
        print(f"\n  Error type breakdown:")
        for etype, count in sorted(error_counts.items(),
                                    key=lambda x: -x[1]):
            print(f"    {etype:<35} {count:>5} "
                  f"({count/total_records*100:.1f}%)")

    if latencies:
        print(f"\n  Latency (seconds):")
        print(f"    p50:  {np.percentile(latencies, 50):.2f}")
        print(f"    p95:  {np.percentile(latencies, 95):.2f}")
        print(f"    p99:  {np.percentile(latencies, 99):.2f}")
        print(f"    mean: {np.mean(latencies):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='evaluation_data', type=str)
    parser.add_argument('--model_name', required=True, type=str,
                        help='Model output_name used during reranking, e.g. qwen')
    parser.add_argument('--prompt_version', default='main', type=str)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--k', default=10, type=int)
    parser.add_argument('--output_dir', default='analysis_results', type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log_dir = os.path.join(args.data_dir, 'reranking_logs')

    # Load run logs
    by_user = load_logs(
        log_dir, args.model_name, args.prompt_version, args.temperature
    )

    # Load ground truths and SASRec rankings
    with open(os.path.join(args.data_dir, 'sasrec_candidates.json')) as f:
        candidates = json.load(f)

    ground_truths = {uid: data['ground_truth']
                     for uid, data in candidates.items()}
    sasrec_rankings = {uid: data['items']
                       for uid, data in candidates.items()}

    # ---- Dependability summary ----
    report_dependability_summary(by_user)

    # ---- Self-consistency ----
    consistency = compute_self_consistency(by_user)
    valid_consistency = {u: c for u, c in consistency.items()
                         if not np.isnan(c['tau_mean'])}
    taus = [c['tau_mean'] for c in valid_consistency.values()]

    print(f"\n{'='*60}")
    print(f"  SELF-CONSISTENCY SUMMARY ({len(valid_consistency)} users)")
    print(f"{'='*60}")
    print(f"  Mean Kendall τ:   {np.mean(taus):.4f}")
    print(f"  Median Kendall τ: {np.median(taus):.4f}")
    print(f"  Std Kendall τ:    {np.std(taus):.4f}")
    print(f"  τ > 0.8:          {sum(1 for t in taus if t > 0.8)} "
          f"({sum(1 for t in taus if t > 0.8)/len(taus)*100:.1f}%)")
    print(f"  τ ≤ 0.5:          {sum(1 for t in taus if t <= 0.5)} "
          f"({sum(1 for t in taus if t <= 0.5)/len(taus)*100:.1f}%)")

    # ---- Quality deltas ----
    quality = compute_quality_deltas(
        by_user, ground_truths, sasrec_rankings, args.k
    )

    wins = sum(1 for q in quality.values() if q['delta_ndcg'] > 0)
    losses = sum(1 for q in quality.values() if q['delta_ndcg'] < 0)
    ties = sum(1 for q in quality.values() if q['delta_ndcg'] == 0)
    print(f"\n  Quality summary:")
    print(f"    LLM wins: {wins}  losses: {losses}  ties: {ties}")

    # ---- Core finding: correlation ----
    common, tau_vals, delta_vals, better_vals = \
        correlate_consistency_and_quality(consistency, quality)

    # ---- Fallback threshold sweep ----
    sweep = fallback_threshold_sweep(
        consistency, quality, ground_truths, sasrec_rankings, by_user, args.k
    )

    # ---- Save everything for plotting ----
    output = {
        'consistency': {u: consistency[u] for u in common},
        'quality': {u: quality[u] for u in common},
        'sweep': sweep,
    }
    out_path = os.path.join(
        args.output_dir,
        f"{args.model_name}_{args.prompt_version}_t{args.temperature}_analysis.json"
    )
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nAnalysis saved to {out_path}")


if __name__ == '__main__':
    main()