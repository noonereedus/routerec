import numpy as np
from collections import Counter


# ==================== ACCURACY METRICS ====================

def hit_rate_at_k(ranked_list, ground_truth, k=10):
    """1 if ground truth item is in top-k, else 0."""
    return 1.0 if ground_truth in ranked_list[:k] else 0.0


def ndcg_at_k(ranked_list, ground_truth, k=10):
    """NDCG@K for a single ground truth item."""
    top_k = ranked_list[:k]
    if ground_truth in top_k:
        rank = top_k.index(ground_truth)
        return 1.0 / np.log2(rank + 2)
    return 0.0


def mrr(ranked_list, ground_truth, k=10):
    """Mean Reciprocal Rank."""
    top_k = ranked_list[:k]
    if ground_truth in top_k:
        rank = top_k.index(ground_truth) + 1
        return 1.0 / rank
    return 0.0


# ==================== POPULARITY BIAS METRICS ====================

def average_recommendation_popularity(ranked_list, item_popularity, k=10):
    """Average popularity of recommended items."""
    top_k = ranked_list[:k]
    pops = [item_popularity.get(item, 0) for item in top_k]
    return np.mean(pops)


def popularity_ratio(ranked_list, item_popularity, threshold_percentile=80, k=10):
    """Fraction of top-k items that are 'popular' (above threshold)."""
    all_pops = sorted(item_popularity.values())
    threshold = np.percentile(all_pops, threshold_percentile)
    top_k = ranked_list[:k]
    popular_count = sum(1 for item in top_k if item_popularity.get(item, 0) >= threshold)
    return popular_count / k


def gini_index(all_recommended_items):
    """
    Gini index over all recommended items across all users.
    0 = perfectly equal, 1 = perfectly unequal.
    """
    counts = Counter(all_recommended_items)
    values = sorted(counts.values())
    n = len(values)
    if n == 0:
        return 0
    cumulative = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n+1) * values)) - (n + 1) * cumulative[-1]) / (n * cumulative[-1])


def coverage(all_recommended_items, total_items):
    """Fraction of total items that appear in any recommendation."""
    unique_items = set(all_recommended_items)
    return len(unique_items) / total_items


# ==================== DIVERSITY METRICS ====================

def intra_list_diversity(ranked_list, movie_metadata, k=10):
    """Average genre dissimilarity between all pairs in top-k."""
    top_k = ranked_list[:k]
    if len(top_k) < 2:
        return 0.0
    
    genre_sets = []
    for item in top_k:
        meta = movie_metadata.get(str(item), {})
        genres = set(meta.get('genres', []))
        genre_sets.append(genres)
    
    total_dissimilarity = 0.0
    pairs = 0
    for i in range(len(genre_sets)):
        for j in range(i + 1, len(genre_sets)):
            if len(genre_sets[i] | genre_sets[j]) > 0:
                jaccard = len(genre_sets[i] & genre_sets[j]) / len(genre_sets[i] | genre_sets[j])
                total_dissimilarity += (1 - jaccard)
            pairs += 1
    
    return total_dissimilarity / pairs if pairs > 0 else 0.0


def genre_coverage(ranked_list, movie_metadata, k=10):
    """Number of unique genres in top-k."""
    top_k = ranked_list[:k]
    all_genres = set()
    for item in top_k:
        meta = movie_metadata.get(str(item), {})
        all_genres.update(meta.get('genres', []))
    return len(all_genres)


# ==================== FAIRNESS METRICS ====================

def group_metric_disparity(metric_by_user, user_demographics, group_key='gender'):
    """
    Compute metric disparity between demographic groups.
    Returns per-group averages and the gap.
    """
    groups = {}
    for user_id, metric_val in metric_by_user.items():
        demo = user_demographics.get(str(user_id), {})
        group = demo.get(group_key, 'unknown')
        if group not in groups:
            groups[group] = []
        groups[group].append(metric_val)
    
    group_means = {g: np.mean(vals) for g, vals in groups.items()}
    
    if len(group_means) >= 2:
        values = list(group_means.values())
        gap = max(values) - min(values)
    else:
        gap = 0.0
    
    return group_means, gap