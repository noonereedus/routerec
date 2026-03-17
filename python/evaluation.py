import json
import os
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
        mrr_scores[user_id] = mrr(ranked, gt)
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
    
    print(f"\n{'='*90}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*90}")
    
    header = f"{'Metric':<22}" + "".join(f"{m:<15}" for m in methods)
    print(header)
    print("-" * 90)
    
    for metric in metrics:
        row = f"{metric:<22}"
        for method in methods:
            val = all_results[method][metric]
            if isinstance(val, float):
                row += f"{val:<15.4f}"
            else:
                row += f"{val:<15}"
        print(row)
    print(f"{'='*90}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='evaluation_data', type=str)
    parser.add_argument('--k', default=10, type=int)
    args = parser.parse_args()
    
    # Load all data
    with open(f'{args.data_dir}/sasrec_candidates.json') as f:
        candidates = json.load(f)
    with open(f'{args.data_dir}/movie_metadata.json') as f:
        movie_metadata = json.load(f)
    with open(f'{args.data_dir}/user_demographics.json') as f:
        user_demographics = json.load(f)
    with open(f'{args.data_dir}/item_popularity.json') as f:
        item_popularity = {int(k): v for k, v in json.load(f).items()}
    
    total_items = len(movie_metadata)
    
    # Prepare SASRec rankings and ground truths
    sasrec_rankings = {}
    ground_truths = {}
    for user_id, data in candidates.items():
        sasrec_rankings[user_id] = data['items']
        ground_truths[user_id] = data['ground_truth']
    
    # Evaluate SASRec on all users
    sasrec_results_all = evaluate_ranking(
        sasrec_rankings, ground_truths,
        item_popularity, movie_metadata,
        user_demographics, total_items, args.k
    )
    print_results(sasrec_results_all, "SASRec (All Users)")
    
    # Evaluate LLM reranked results
    comparison = {}
    
    for llm_name, filename in [('Qwen', 'qwen_reranked.json'),
                                ('Nemotron', 'nemotron_reranked.json')]:
        filepath = f'{args.data_dir}/{filename}'
        if not os.path.exists(filepath):
            print(f"Skipping {llm_name}: {filepath} not found")
            continue
        
        with open(filepath) as f:
            llm_rankings = json.load(f)
        
        # Get overlapping users
        common_users = set(llm_rankings.keys()) & set(sasrec_rankings.keys())
        
        # SASRec on same subset (fair comparison)
        sasrec_subset = {uid: sasrec_rankings[uid] for uid in common_users}
        gt_subset = {uid: ground_truths[uid] for uid in common_users}
        llm_subset = {uid: llm_rankings[uid] for uid in common_users}
        
        sasrec_sub_results = evaluate_ranking(
            sasrec_subset, gt_subset,
            item_popularity, movie_metadata,
            user_demographics, total_items, args.k
        )
        
        llm_results = evaluate_ranking(
            llm_subset, gt_subset,
            item_popularity, movie_metadata,
            user_demographics, total_items, args.k
        )
        
        print_results(sasrec_sub_results, f"SASRec (same {len(common_users)} users)")
        print_results(llm_results, f"SASRec + {llm_name}")
        
        comparison[f'SASRec'] = sasrec_sub_results
        comparison[llm_name] = llm_results
    
    if len(comparison) > 1:
        compare_methods(comparison)