import json
import os
import torch
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import data_partition


def load_evaluation_data(data_dir):
    with open(os.path.join(data_dir, 'sasrec_candidates.json')) as f:
        candidates = json.load(f)
    with open(os.path.join(data_dir, 'movie_metadata.json')) as f:
        movie_metadata = json.load(f)
    with open(os.path.join(data_dir, 'user_demographics.json')) as f:
        user_demographics = json.load(f)
    with open(os.path.join(data_dir, 'item_popularity.json')) as f:
        item_popularity = json.load(f)
    return candidates, movie_metadata, user_demographics, item_popularity


def get_user_history_titles(user_id, user_train, movie_metadata, max_history=20):
    """Get the user's recent watch history as movie titles."""
    history = user_train.get(user_id, [])
    recent = history[-max_history:]  # Last 20 movies
    titles = []
    for item_id in recent:
        meta = movie_metadata.get(str(item_id), {})
        title = meta.get('title', f'Unknown Movie {item_id}')
        genres = ', '.join(meta.get('genres', []))
        titles.append(f"{title} ({genres})")
    return titles


def build_rerank_prompt(user_history, candidate_movies):
    """
    Build the prompt for LLM reranking.
    """
    history_str = "\n".join(f"  - {t}" for t in user_history)
    
    candidates_str = ""
    for i, (item_id, title, genres) in enumerate(candidate_movies, 1):
        candidates_str += f"  {i}. {title} ({genres})\n"
    
    prompt = f"""You are a movie recommendation system. Your goal is to rerank candidate movies to best match the user's preferences while ensuring diversity and fairness.

The user has recently watched:
{history_str}

Here are candidate movies to rerank. Select and rerank the top 10 that best match this user's taste while also including diverse genres and avoiding over-recommending only popular movies:
{candidates_str}

Your primary task is to minimize the demographic, exposure and popularity biases.
Return ONLY the numbers (1-{len(candidate_movies)}) in your recommended order as a JSON list, like:
[3, 7, 1, 9, 5, 2, 8, 4, 10, 6]

Your top 10 reranked list:"""
    
    return prompt


def parse_llm_response(response_text, valid_item_ids):
    """
    Parse the LLM's response to extract reranked item IDs.
    Handles formats like:
      [355, 1191, 1905]
      [ID355, ID1191, ID1905]
      [ID:355, ID:1191, ID:1905]
      ["ID355", "ID1191"]
    """
    import re
    cleaned = re.sub(r'["\']?[Ii][Dd]:?\s*', '', response_text)
    # Try to find a JSON list
    match = re.search(r'$$[\d,\s]+$$', cleaned)
    if match:
        try:
            ids = json.loads(match.group())
            # Filter to only valid IDs
            ids = [int(x) for x in ids if int(x) in valid_item_ids]
            if len(ids) > 0:
                return ids
        except:
            pass
    
    # Fallback: find all numbers that match valid IDs
    numbers = re.findall(r'\b(\d+)\b', cleaned)
    ids = []
    for n in numbers:
        n = int(n)
        if n in valid_item_ids and n not in ids:
            ids.append(n)
        if len(ids) >= 10:
            break
    
    return ids


def rerank_with_llm(model, tokenizer, prompt, valid_item_ids, device, max_new_tokens=200):
    """
    Send prompt to LLM and get reranked list.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,        # Low temperature for consistency
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    reranked_ids = parse_llm_response(response, valid_item_ids)
    
    return reranked_ids, response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='evaluation_data', type=str)
    parser.add_argument('--dataset', default='ml-1m', type=str)
    parser.add_argument('--model_name', required=True, type=str,
                        help='HuggingFace model name, e.g., OpenGPT-X/Teuken-7B-instruct-v04')
    parser.add_argument('--output_name', default='gptoss', type=str,
                        help='Name for output file, e.g., gptoss, qwen, gemma')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--top_k_candidates', default=20, type=int,
                        help='How many SASRec candidates to show the LLM')
    parser.add_argument('--top_k_rerank', default=10, type=int,
                        help='How many items the LLM should return')
    parser.add_argument('--max_users', default=None, type=int,
                        help='Limit number of users (for testing)')
    parser.add_argument('--batch_save_every', default=100, type=int)
    args = parser.parse_args()
    
    # ---- Load evaluation data ----
    print("Loading evaluation data...")
    candidates, movie_metadata, user_demographics, item_popularity = \
        load_evaluation_data(args.data_dir)
    
    # Load user training data for history
    dataset = data_partition(args.dataset)
    user_train = dataset[0]
    
    # ---- Load LLM ----
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True
    )
    model.eval()
    print("Model loaded.")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ---- Rerank for each user ----
    reranked_results = {}
    user_ids = list(candidates.keys())
    
    if args.max_users:
        user_ids = user_ids[:args.max_users]
    
    total = len(user_ids)
    failed = 0
    
    for idx, user_id in enumerate(user_ids):
        user_data = candidates[user_id]
        sasrec_items = user_data['items'][:args.top_k_candidates]
        
        # Build candidate list with titles
        candidate_movies = []
        valid_item_ids = set()
        for item_id in sasrec_items:
            meta = movie_metadata.get(str(item_id), {})
            title = meta.get('title', f'Movie {item_id}')
            genres = ', '.join(meta.get('genres', ['Unknown']))
            candidate_movies.append((item_id, title, genres))
            valid_item_ids.add(item_id)
        
        # Get user history
        user_history = get_user_history_titles(
            int(user_id), user_train, movie_metadata
        )
        
        # Build prompt
        prompt = build_rerank_prompt(user_history, candidate_movies)
        
        # Get LLM reranking
        try:
            reranked_ids, raw_response = rerank_with_llm(
                model, tokenizer, prompt, set(range(1, len(sasrec_items)+1)), args.device
            )
            
            reranked_item_ids = []
            for pos in reranked_ids:
                if 1 <= pos <= len(sasrec_items):
                    reranked_item_ids.append(sasrec_items[pos - 1])
            
            if idx < 5:
                print(f"\n=== User {user_id} ===")
                print(f"Raw LLM response:\n{raw_response}")
                print(f"Parsed positions: {reranked_ids}")
                print(f"Mapped item IDs: {reranked_item_ids}")
            
            # If LLM returned fewer than 10, fill with original SASRec order
            if len(reranked_item_ids) < args.top_k_rerank:
                for item_id in sasrec_items:
                    if item_id not in reranked_item_ids:
                        reranked_item_ids.append(item_id)
                    if len(reranked_item_ids) >= args.top_k_rerank:
                        break
            
            reranked_results[user_id] = reranked_item_ids[:args.top_k_rerank]
            
        except Exception as e:
            print(f"\n  Error for user {user_id}: {e}")
            # Fallback to SASRec order
            reranked_results[user_id] = sasrec_items[:args.top_k_rerank]
            failed += 1
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{total} users ({failed} failed)")
        
        # Periodic save
        if (idx + 1) % args.batch_save_every == 0:
            output_path = os.path.join(args.data_dir, f'{args.output_name}_reranked.json')
            with open(output_path, 'w') as f:
                json.dump(reranked_results, f)
    
    # ---- Final save ----
    output_path = os.path.join(args.data_dir, f'{args.output_name}_reranked.json')
    with open(output_path, 'w') as f:
        json.dump(reranked_results, f)
    
    print(f"\nDone! Reranked {len(reranked_results)} users.")
    print(f"Failed: {failed}")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()