import json
import os
import glob
import numpy as np
from metrics import *
import argparse


def evaluate_ranking(ranked_lists, ground_truths, item_popularity,
                     movie_metadata, user_demographics, total_items, k=10):
    """Compute all metrics for a set of rankings."""

    results = {}

    hr_scores = {}
    ndcg_scores = {}
    mrr_scores = {}
    arp_scores = {}
    pop_ratio_scores = {}
    ild_scores = {}
    genre_cov_scores = {}

    all_recommended = []

    for user_id in ranked_lists:
        if user_id not in ground_truths:
            continue

        ranked = ranked_lists[user_id]
        gt = ground_truths[user_id]

        hr_scores[user_id] = hit_rate_at_k(ranked, gt, k)
        ndcg_scores[user_id] = ndcg_at_k(ranked, gt, k)
        mrr_scores[user_id] = mrr(ranked, gt, k)
        arp_scores[user_id] = average_recommendation_popularity(ranked, item_popularity, k)
        pop_ratio_scores[user_id] = popularity_ratio(ranked, item_popularity, 80, k)
        ild_scores[user_id] = intra_list_diversity(ranked, movie_metadata, k)
        genre_cov_scores[user_id] = genre_coverage(ranked, movie_metadata, k)

        all_recommended.extend(ranked[:k])

    results['HR@10'] = np.mean(list(hr_scores.values()))
    results['NDCG@10'] = np.mean(list(ndcg_scores.values()))
    results['MRR'] = np.mean(list(mrr_scores.values()))
    results['ARP'] = np.mean(list(arp_scores.values()))
    results['PopRatio@80'] = np.mean(list(pop_ratio_scores.values()))
    results['Gini'] = gini_index(all_recommended)
    results['Coverage'] = coverage(all_recommended, total_items)
    results['ILD'] = np.mean(list(ild_scores.values()))
    results['GenreCov'] = np.mean(list(genre_cov_scores.values()))

    gender_hr, hr_gap = group_metric_disparity(hr_scores, user_demographics, 'gender')
    gender_ndcg, ndcg_gap = group_metric_disparity(ndcg_scores, user_demographics, 'gender')

    results['HR@10_M'] = gender_hr.get('M', 0)
    results['HR@10_F'] = gender_hr.get('F', 0)
    results['HR_Gender_Gap'] = hr_gap
    results['NDCG@10_M'] = gender_ndcg.get('M', 0)
    results['NDCG@10_F'] = gender_ndcg.get('F', 0)
    results['NDCG_Gender_Gap'] = ndcg_gap

    results['Num_Users'] = len(hr_scores)

    return results


def print_results(results, method_name="SASRec"):
    print(f"\n{'='*60}")
    print(f"  Results: {method_name} ({results['Num_Users']} users)")
    print(f"{'='*60}")

    print(f"\n  ACCURACY")
    print(f"  {'HR@10':<20} {results['HR@10']:.4f}")
    print(f"  {'NDCG@10':<20} {results['NDCG@10']:.4f}")
    print(f"  {'MRR':<20} {results['MRR']:.4f}")

    print(f"\n  POPULARITY BIAS")
    print(f"  {'ARP':<20} {results['ARP']:.2f}")
    print(f"  {'PopRatio@80':<20} {results['PopRatio@80']:.4f}")
    print(f"  {'Gini':<20} {results['Gini']:.4f}")
    print(f"  {'Coverage':<20} {results['Coverage']:.4f}")

    print(f"\n  DIVERSITY")
    print(f"  {'ILD':<20} {results['ILD']:.4f}")
    print(f"  {'GenreCov':<20} {results['GenreCov']:.2f}")

    print(f"\n  GENDER FAIRNESS")
    print(f"  {'HR@10 (Male)':<20} {results['HR@10_M']:.4f}")
    print(f"  {'HR@10 (Female)':<20} {results['HR@10_F']:.4f}")
    print(f"  {'HR Gap':<20} {results['HR_Gender_Gap']:.4f}")
    print(f"  {'NDCG@10 (Male)':<20} {results['NDCG@10_M']:.4f}")
    print(f"  {'NDCG@10 (Female)':<20} {results['NDCG@10_F']:.4f}")
    print(f"  {'NDCG Gap':<20} {results['NDCG_Gender_Gap']:.4f}")
    print(f"{'='*60}\n")


def compare_methods(all_results):
    methods = list(all_results.keys())
    metrics = [m for m in all_results[methods[0]].keys() if m != 'Num_Users']

    # Dynamic column width based on longest method name
    col_width = max(len(m) for m in methods) + 2

    print(f"\n{'='*120}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*120}")

    header = f"{'Metric':<22}" + "".join(f"{m:<{col_width}}" for m in methods)
    print(header)
    print("-" * 120)

    for metric in metrics:
        row = f"{metric:<22}"
        for method in methods:
            val = all_results[method][metric]
            if isinstance(val, float):
                row += f"{val:<{col_width}.4f}"
            else:
                row += f"{val:<{col_width}}"
        print(row)
    print(f"{'='*120}\n")


def load_llm_run(log_dir, model_name, run_index=0,
                 prompt_version="main", temperature=0.1):
    """
    Load a single run's JSONL results.
    Returns dict: user_id (str) -> record.
    """
    fname = f"{model_name}_run{run_index}_{prompt_version}_t{temperature}.jsonl"
    fpath = os.path.join(log_dir, fname)

    if not os.path.exists(fpath):
        pattern = os.path.join(
            log_dir,
            f"{model_name}_run{run_index}*t{temperature}*.jsonl"
        )
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No file found: {fpath} (also tried glob: {pattern})"
            )
        fpath = matches[0]

    results = {}
    with open(fpath) as f:
        for line in f:
            rec = json.loads(line)
            results[str(rec['user_id'])] = rec

    print(f"  Loaded {len(results)} users from {os.path.basename(fpath)}")
    return results


# Errors where the cleaned output is still usable (10 valid items after cleanup)
RECOVERABLE_ERRORS = {'duplicates_removed', 'out_of_range_removed'}


def build_four_systems(llm_run, sasrec_rankings, k=10):
    """
    Construct rankings for the four-way comparison.

    System A (SASRec):          SASRec top-k unchanged.
    System B (LLM reranked):    filled_item_ids always (naive deployment).
    System C (LLM + hard FB):   filled_item_ids when is_valid=True,
                                full SASRec revert on ANY error.
    System D (LLM + tiered FB): filled_item_ids when is_valid=True OR
                                error is recoverable. Full SASRec revert
                                only on catastrophic failure.
    """
    system_a = {}
    system_b = {}
    system_c = {}
    system_d = {}
    n_valid = 0
    n_recoverable = 0
    n_catastrophic = 0

    for user_id, rec in llm_run.items():
        if user_id not in sasrec_rankings:
            continue

        sasrec_list = sasrec_rankings[user_id][:k]
        filled_list = rec['filled_item_ids'][:k]

        system_a[user_id] = sasrec_list
        system_b[user_id] = filled_list

        if rec['is_valid']:
            # Clean output — all systems use LLM
            system_c[user_id] = filled_list
            system_d[user_id] = filled_list
            n_valid += 1
        elif rec.get('error_type') in RECOVERABLE_ERRORS:
            # Recoverable: LLM output usable after cleanup
            system_c[user_id] = sasrec_list   # Hard FB: reverts (too aggressive)
            system_d[user_id] = filled_list   # Tiered FB: keeps LLM output
            n_recoverable += 1
        else:
            # Catastrophic: no usable LLM output
            system_c[user_id] = sasrec_list   # Hard FB: reverts (correct)
            system_d[user_id] = sasrec_list   # Tiered FB: also reverts (correct)
            n_catastrophic += 1

    total = n_valid + n_recoverable + n_catastrophic
    print(f"  {total} users | Valid: {n_valid} | "
          f"Recoverable: {n_recoverable} | "
          f"Catastrophic: {n_catastrophic}")

    return (system_a, system_b, system_c, system_d,
            n_valid, n_recoverable, n_catastrophic)


# ---- Configuration ----
MODEL_CONFIGS = [
    {'label': 'Qwen_t0.0',     'model_name': 'qwen',     'temperature': 0.0},
    {'label': 'Qwen_t0.1',     'model_name': 'qwen',     'temperature': 0.1},
    {'label': 'Qwen_t0.7',     'model_name': 'qwen',     'temperature': 0.7},
    {'label': 'Nemotron_t0.1', 'model_name': 'nemotron',  'temperature': 0.1},
    {'label': 'Nemotron_t0.7', 'model_name': 'nemotron',  'temperature': 0.7},
    {'label': 'Gpt-oss_t0.1',  'model_name': 'gpt',      'temperature': 0.1},
    {'label': 'Qwen_div1_t0.1', 'model_name': 'qwen',    'temperature': 0.1, 'prompt_version': 'diversity_v1'},
    {'label': 'Qwen_div2_t0.1', 'model_name': 'qwen',    'temperature': 0.1, 'prompt_version': 'diversity_v2'},
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',       default='evaluation_data',  type=str)
    parser.add_argument('--k',              default=10,                 type=int)
    parser.add_argument('--output_dir',     default='analysis_results', type=str)
    parser.add_argument('--run_index',      default=0,                  type=int,
                        help='Which run to use (default: 0)')
    parser.add_argument('--prompt_version', default='main',             type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load shared data ----
    with open(os.path.join(args.data_dir, 'sasrec_candidates.json')) as f:
        candidates = json.load(f)
    with open(os.path.join(args.data_dir, 'movie_metadata.json')) as f:
        movie_metadata = json.load(f)
    with open(os.path.join(args.data_dir, 'user_demographics.json')) as f:
        user_demographics = json.load(f)
    with open(os.path.join(args.data_dir, 'item_popularity.json')) as f:
        item_popularity = {int(k): v for k, v in json.load(f).items()}

    total_items = len(movie_metadata)

    sasrec_rankings = {uid: data['items']  for uid, data in candidates.items()}
    ground_truths   = {uid: data['ground_truth'] for uid, data in candidates.items()}

    log_dir = os.path.join(args.data_dir, 'reranking_logs')

    # ---- Per-model four-way evaluation ----
    cross_model = {}
    json_export  = {}
    ref_users = None  # Will be set by first model to ensure consistent SASRec eval

    for config in MODEL_CONFIGS:
        label = config['label']
        print(f"\n{'#'*70}")
        print(f"  MODEL: {label}  (run {args.run_index}, t={config['temperature']})")
        print(f"{'#'*70}")

        try:
            llm_run = load_llm_run(
                log_dir, config['model_name'],
                run_index=args.run_index,
                prompt_version=config.get('prompt_version', args.prompt_version),
                temperature=config['temperature'],
            )
        except FileNotFoundError as e:
            print(f"  SKIPPED — {e}\n")
            continue

        # Build Systems A / B / C / D
        system_a, system_b, system_c, system_d, \
            n_valid, n_recoverable, n_catastrophic = \
            build_four_systems(llm_run, sasrec_rankings, args.k)

        # Capture reference user set from first model
        if ref_users is None:
            ref_users = set(system_a.keys())

        gt_subset = {uid: ground_truths[uid] for uid in system_a}

        # Evaluate all four systems
        res_a = evaluate_ranking(system_a, gt_subset, item_popularity,
                                 movie_metadata, user_demographics,
                                 total_items, args.k)
        res_b = evaluate_ranking(system_b, gt_subset, item_popularity,
                                 movie_metadata, user_demographics,
                                 total_items, args.k)
        res_c = evaluate_ranking(system_c, gt_subset, item_popularity,
                                 movie_metadata, user_demographics,
                                 total_items, args.k)
        res_d = evaluate_ranking(system_d, gt_subset, item_popularity,
                                 movie_metadata, user_demographics,
                                 total_items, args.k)

        name_a = "A: SASRec"
        name_b = f"B: {label}"
        name_c = f"C: {label}+HardFB"
        name_d = f"D: {label}+TierFB"

        print_results(res_a, name_a)
        print_results(res_b, name_b)
        print_results(res_c, name_c)
        print_results(res_d, name_d)

        compare_methods({name_a: res_a, name_b: res_b,
                         name_c: res_c, name_d: res_d})

        total = n_valid + n_recoverable + n_catastrophic
        hard_fb_rate = (n_recoverable + n_catastrophic) / total * 100
        tiered_fb_rate = n_catastrophic / total * 100

        print(f"  >>> Valid: {n_valid} | Recoverable: {n_recoverable} | "
              f"Catastrophic: {n_catastrophic}")
        print(f"  >>> Hard fallback rate:   {hard_fb_rate:.1f}%  "
              f"({n_recoverable + n_catastrophic} of {total} users)")
        print(f"  >>> Tiered fallback rate: {tiered_fb_rate:.1f}%  "
              f"({n_catastrophic} of {total} users)")

        # Accumulate for cross-model view
        cross_model[name_b] = res_b
        cross_model[name_c] = res_c
        cross_model[name_d] = res_d

        json_export[label] = {
            'system_a': res_a,
            'system_b': res_b,
            'system_c': res_c,
            'system_d': res_d,
            'n_valid':       n_valid,
            'n_recoverable': n_recoverable,
            'n_catastrophic': n_catastrophic,
            'hard_fallback_rate':   hard_fb_rate / 100,
            'tiered_fallback_rate': tiered_fb_rate / 100,
        }

    # ---- SASRec baseline on the same 500 users ----
    if ref_users is not None:
        sasrec_ref = evaluate_ranking(
            {uid: sasrec_rankings[uid] for uid in ref_users},
            {uid: ground_truths[uid] for uid in ref_users},
            item_popularity, movie_metadata, user_demographics,
            total_items, args.k
        )
    else:
        sasrec_ref = evaluate_ranking(
            {uid: sasrec_rankings[uid] for uid in ground_truths},
            ground_truths, item_popularity, movie_metadata,
            user_demographics, total_items, args.k
        )

    cross_model = {"A: SASRec": sasrec_ref, **cross_model}
    json_export['SASRec'] = sasrec_ref

        # ---- Cross-model comparison ----
    if len(cross_model) > 2:
        print(f"\n{'#'*70}")
        print(f"  CROSS-MODEL COMPARISON")
        print(f"{'#'*70}")
        compare_methods(cross_model)

    # ---- Save ----
    out_path = os.path.join(args.output_dir, 'full_evaluation_results.json')
    with open(out_path, 'w') as f:
        json.dump(json_export, f, indent=2)
    print(f"\nAll results saved to {out_path}")