import torch
import numpy as np
import json
import os
import pandas as pd
from model import SASRec
from utils import data_partition
import argparse


def load_metadata(ml1m_path):
    """Load movie titles, genres, and user demographics."""
    
    # Movie info
    movies = pd.read_csv(
        os.path.join(ml1m_path, 'movies.dat'),
        sep='::',
        names=['movie_id', 'title', 'genres'],
        engine='python',
        encoding='latin-1'
    )
    
    # User info
    users = pd.read_csv(
        os.path.join(ml1m_path, 'users.dat'),
        sep='::',
        names=['user_id', 'gender', 'age', 'occupation', 'zipcode'],
        engine='python'
    )
    
    # Ratings for item popularity
    ratings = pd.read_csv(
        os.path.join(ml1m_path, 'ratings.dat'),
        sep='::',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        engine='python'
    )
    
    # Rebuild the same ID mapping used in preprocessing
    ratings = ratings.sort_values(['user_id', 'timestamp'])
    user_counts = ratings.groupby('user_id').size()
    valid_users = user_counts[user_counts >= 5].index
    ratings = ratings[ratings['user_id'].isin(valid_users)]
    
    user_map = {u: i+1 for i, u in enumerate(sorted(ratings['user_id'].unique()))}
    item_map = {it: i+1 for i, it in enumerate(sorted(ratings['item_id'].unique()))}
    reverse_item_map = {v: k for k, v in item_map.items()}
    reverse_user_map = {v: k for k, v in user_map.items()}
    
    # Item popularity (number of interactions per item)
    item_popularity = ratings['item_id'].value_counts().to_dict()
    remapped_popularity = {}
    for orig_id, count in item_popularity.items():
        if orig_id in item_map:
            remapped_popularity[item_map[orig_id]] = count
    
    # Movie metadata
    movie_dict = {}
    for _, row in movies.iterrows():
        if row['movie_id'] in item_map:
            remapped_id = item_map[row['movie_id']]
            movie_dict[remapped_id] = {
                'title': row['title'],
                'genres': row['genres'].split('|')
            }
    
    # User demographics
    user_demo = {}
    for _, row in users.iterrows():
        if row['user_id'] in user_map:
            remapped_id = user_map[row['user_id']]
            user_demo[remapped_id] = {
                'gender': row['gender'],
                'age': int(row['age']),
                'occupation': int(row['occupation'])
            }
    
    return movie_dict, user_demo, remapped_popularity, reverse_item_map, reverse_user_map


def generate_all_candidates(model, dataset, args, top_k=50):
    """Generate top-K SASRec candidates for all test users."""
    
    user_train, user_valid, user_test, usernum, itemnum = dataset
    all_candidates = {}
    
    for user_id in range(1, usernum + 1):
        if len(user_train.get(user_id, [])) < 1:
            continue
        if len(user_test.get(user_id, [])) < 1:
            continue
        
        # Build sequence
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        
        if len(user_valid.get(user_id, [])) > 0:
            seq[idx] = user_valid[user_id][0]
            idx -= 1
        
        for i in reversed(user_train[user_id]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        # Already rated
        rated = set(user_train[user_id])
        for lst in [user_valid.get(user_id, []), user_test.get(user_id, [])]:
            rated.update(lst)
        
        # Score all items
        item_indices = list(range(1, itemnum + 1))
        
        with torch.no_grad():
            predictions = model.predict(
                *[np.array(l) for l in [[user_id], [seq], item_indices]]
            )
        predictions = predictions[0]
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        
        # Mask rated
        for item in rated:
            if 1 <= item <= itemnum:
                predictions[item - 1] = -float('inf')
        
        # Top-K
        top_k_idx = np.argsort(-predictions)[:top_k]
        top_k_items = [item_indices[i] for i in top_k_idx]
        top_k_scores = [float(predictions[i]) for i in top_k_idx]
        
        #making sure ground truth is a candidate
        gt = user_test[user_id][0]
        if gt not in top_k_items:
            gt_score = float(predictions[gt - 1])
            # Insert at a random position (not first, not last)
            insert_pos = np.random.randint(1, len(top_k_items))
            top_k_items.insert(insert_pos, gt)
            top_k_scores.insert(insert_pos, gt_score)

        all_candidates[user_id] = {
            'items': top_k_items,
            'scores': top_k_scores,
            'ground_truth': gt
        }
        
        if user_id % 500 == 0:
            print(f"Generated candidates for {user_id}/{usernum} users")
    
    return all_candidates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m')
    parser.add_argument('--train_dir', default='default')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=200, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--inference_only', default=False, type=bool)
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true', default=False)
    parser.add_argument('--top_k', default=50, type=int)
    parser.add_argument('--ml1m_path', default='../../ml-1m', type=str)
    parser.add_argument('--output_dir', default='evaluation_data', type=str)
    args = parser.parse_args()
    
    # Load data
    dataset = data_partition(args.dataset)
    user_train, user_valid, user_test, usernum, itemnum = dataset
    
    # Load model
    model = SASRec(usernum, itemnum, args).to('cpu')
    fname = (f'SASRec.epoch={args.num_epochs}.lr={args.lr}'
             f'.layer={args.num_blocks}.head={args.num_heads}'
             f'.hidden={args.hidden_units}.maxlen={args.maxlen}.pth')
    model_path = os.path.join(f'{args.dataset}_{args.train_dir}', fname)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    print(f"Model loaded from: {model_path}")
    
    # Load metadata
    movie_dict, user_demo, item_popularity, reverse_item_map, reverse_user_map = \
        load_metadata(args.ml1m_path)
    
    # Generate candidates
    print(f"\nGenerating top-{args.top_k} candidates for all test users...")
    candidates = generate_all_candidates(model, dataset, args, args.top_k)
    
    # Save everything
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save candidates
    with open(os.path.join(args.output_dir, 'sasrec_candidates.json'), 'w') as f:
        json.dump(candidates, f)
    
    # Save metadata
    # Convert int keys to strings for JSON
    movie_dict_str = {str(k): v for k, v in movie_dict.items()}
    user_demo_str = {str(k): v for k, v in user_demo.items()}
    item_pop_str = {str(k): v for k, v in item_popularity.items()}
    
    with open(os.path.join(args.output_dir, 'movie_metadata.json'), 'w') as f:
        json.dump(movie_dict_str, f)
    
    with open(os.path.join(args.output_dir, 'user_demographics.json'), 'w') as f:
        json.dump(user_demo_str, f)
    
    with open(os.path.join(args.output_dir, 'item_popularity.json'), 'w') as f:
        json.dump(item_pop_str, f)
    
    print(f"\nSaved {len(candidates)} user candidates to {args.output_dir}/")
    print(f"Saved movie metadata for {len(movie_dict)} items")
    print(f"Saved demographics for {len(user_demo)} users")