"""
rerank.py — LLM reranking with full logging for non-determinism experiments.

Each call saves one JSONL record per user containing:
  - raw LLM output (irreplaceable, non-reproducible)
  - parsed positions before any filling
  - validity classification
  - latency
  - all metadata needed for later analysis

Usage:
  # Standard run
  python rerank.py --model_name Qwen/Qwen2.5-7B-Instruct --output_name qwen --run_index 0

  # Repeated runs for self-consistency
  python rerank.py --model_name Qwen/Qwen2.5-7B-Instruct --output_name qwen --run_index 1
  python rerank.py --model_name Qwen/Qwen2.5-7B-Instruct --output_name qwen --run_index 2
"""

import json
import os
import re
import time
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
    history = user_train.get(user_id, [])
    recent = history[-max_history:]
    titles = []
    for item_id in recent:
        meta = movie_metadata.get(str(item_id), {})
        title = meta.get('title', f'Unknown Movie {item_id}')
        genres = ', '.join(meta.get('genres', []))
        titles.append(f"{title} ({genres})")
    return titles


def build_rerank_prompt(user_history, candidate_movies, prompt_version="main"):
    """
    Build reranking prompt. prompt_version enables perturbation experiments.
    """
    history_str = "\n".join(f"  - {t}" for t in user_history)

    candidates_str = ""
    for i, (item_id, title, genres) in enumerate(candidate_movies, 1):
        candidates_str += f"  {i}. {title} ({genres})\n"

    num = len(candidate_movies)

    if prompt_version == "main":
        prompt = f"""You are a movie recommendation system. Your goal is to rerank candidate movies to best match the user's preferences while ensuring diversity and fairness.

The user has recently watched:
{history_str}

Here are candidate movies to rerank. Select and rerank the top 10 that best match this user's taste while also including diverse genres and avoiding over-recommending only popular movies:
{candidates_str}

Your primary task is to minimize the demographic, exposure and popularity biases.
Return ONLY the numbers (1-{num}) in your recommended order as a JSON list, like:
[3, 7, 1, 9, 5, 2, 8, 4, 10, 6]

Your top 10 reranked list:"""

    elif prompt_version == "rephrase_v1":
        prompt = f"""As a film recommendation engine, reorder the following candidates to suit this viewer's preferences. Prioritise variety and reduce bias toward popular titles.

Viewing history:
{history_str}

Candidates to reorder (pick your best 10):
{candidates_str}

Output the candidate numbers (1-{num}) as a JSON list in your preferred order.

Reordered list:"""

    elif prompt_version == "diversity_v1":
        prompt = f"""You are a movie recommendation system. Your goal is to rerank candidate movies to maximise DIVERSITY in the final list.

The user has recently watched:
{history_str}

Here are candidate movies to rerank. Select the top 10 that:
1. SPREAD ACROSS AS MANY DIFFERENT GENRES as possible (maximise genre coverage)
2. MIX popular and niche/obscure titles (avoid all-blockbuster lists)
3. Still reasonably match the user's taste

Diversity is MORE IMPORTANT than pure relevance. A good list covers 5+ distinct genres.

Candidates:
{candidates_str}

Return ONLY the numbers (1-{num}) in your recommended order as a JSON list, like:
[3, 7, 1, 9, 5, 2, 8, 4, 10, 6]

Your top 10 diversity-optimised list:"""

    elif prompt_version == "diversity_v2":
        prompt = f"""You are a diversity-focused movie recommender. Your PRIMARY objective is to maximise the variety of genres in your recommendations. Your SECONDARY objective is relevance to the user.

The user has recently watched:
{history_str}

Candidates:
{candidates_str}

INSTRUCTIONS:
- Select exactly 10 movies
- Maximise GENRE COVERAGE: include as many different genres as possible (aim for 8+ unique genres across your 10 picks)
- Maximise INTRA-LIST DIVERSITY: consecutive movies should NOT share genres. Each pair of movies in your list should be as genre-dissimilar as possible
- Include at least one movie from a genre the user has NOT watched before
- Aim to improve the Gini index score
- Prefer lesser-known titles over blockbusters when relevance is similar

Return ONLY the numbers (1-{num}) as a JSON list.

Your diversity-maximised list:"""

    else:
        raise ValueError(f"Unknown prompt version: {prompt_version}")

    return prompt


def parse_llm_response(response_text, num_candidates):
    """
    Parse the LLM response to extract 1-based position indices.

    Returns:
        positions: list of ints (1-based, deduplicated, in valid range)
        error_type: None if clean 10-item output, else a string classifying the problem
    """
    cleaned = re.sub(r'["\']?[Ii][Dd]:?\s*', '', response_text)

    positions = []

    # Try JSON list first
    match = re.search(r'$$[\d,\s]+$$', cleaned)
    if match:
        try:
            raw = json.loads(match.group())
            positions = [int(x) for x in raw]
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: extract all bare numbers
    if not positions:
        numbers = re.findall(r'\b(\d+)\b', cleaned)
        positions = [int(n) for n in numbers]

    # --- Classify validity ---
    valid_range = set(range(1, num_candidates + 1))
    original_count = len(positions)

    # Filter to valid range
    positions = [p for p in positions if p in valid_range]
    out_of_range_count = original_count - len(positions)

    # Deduplicate preserving order
    seen = set()
    deduped = []
    for p in positions:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    duplicate_count = len(positions) - len(deduped)
    positions = deduped[:10]

    # Classify
    if original_count == 0:
        error_type = "no_parseable_output"
    elif len(positions) == 0:
        error_type = "all_out_of_range"
    elif duplicate_count > 0 and len(positions) < 10:
        error_type = "duplicates_and_incomplete"
    elif duplicate_count > 0:
        error_type = "duplicates_removed"
    elif out_of_range_count > 0 and len(positions) < 10:
        error_type = "out_of_range_and_incomplete"
    elif out_of_range_count > 0:
        error_type = "out_of_range_removed"
    elif len(positions) < 10:
        error_type = "incomplete"
    else:
        error_type = None  # Clean output

    return positions, error_type


def call_llm(model, tokenizer, prompt, device, max_new_tokens=200,
             temperature=0.1):
    """
    Raw LLM call. Returns the decoded response string and latency.
    No parsing here — keep generation and interpretation separate.
    """
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            top_p=0.9 if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    latency = time.time() - start

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    return response, latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='evaluation_data', type=str)
    parser.add_argument('--dataset', default='ml-1m', type=str)
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--output_name', default='qwen', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--top_k_candidates', default=20, type=int)
    parser.add_argument('--top_k_rerank', default=10, type=int)
    parser.add_argument('--max_users', default=None, type=int)
    parser.add_argument('--batch_save_every', default=100, type=int)
    # New: non-determinism experiment controls
    parser.add_argument('--run_index', default=0, type=int,
                        help='Run index for repeated experiments (0, 1, 2, ...)')
    parser.add_argument('--prompt_version', default='main', type=str,
                        help='Prompt variant: main, rephrase_v1')
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--candidate_order', default='original', type=str,
                    choices=['original', 'reverse', 'random'],
                    help='Candidate presentation order: original, reverse, random')
    parser.add_argument('--shuffle_seed', default=67, type=int,
                    help='Random seed for shuffle (reproducibility)')
    args = parser.parse_args()

    # ---- Load data ----
    print("Loading evaluation data...")
    candidates, movie_metadata, user_demographics, item_popularity = \
        load_evaluation_data(args.data_dir)

    dataset = data_partition(args.dataset)
    user_train = dataset[0]

    # ---- Load LLM ----
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded.")

    # ---- Output setup ----
    log_dir = os.path.join(args.data_dir, 'reranking_logs')
    os.makedirs(log_dir, exist_ok=True)

    order_suffix = f"_{args.candidate_order}" if args.candidate_order != 'original' else ""
    log_filename = (f"{args.output_name}_run{args.run_index}"
                f"_{args.prompt_version}{order_suffix}_t{args.temperature}.jsonl")
    log_path = os.path.join(log_dir, log_filename)

    # Resume support: skip already-completed users
    completed_users = set()
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                completed_users.add(record['user_id'])
        print(f"Resuming: {len(completed_users)} users already completed.")

    # ---- Process users ----
    user_ids = list(candidates.keys())
    if args.max_users:
        user_ids = user_ids[:args.max_users]

    total = len(user_ids)
    processed = 0
    errors = 0

    with open(log_path, 'a') as log_file:
        for idx, user_id in enumerate(user_ids):
            if user_id in completed_users:
                continue

            user_data = candidates[user_id]
            sasrec_items = user_data['items'][:args.top_k_candidates]
            num_candidates = len(sasrec_items)

            # Build candidate movie tuples
            candidate_movies = []
            for item_id in sasrec_items:
                meta = movie_metadata.get(str(item_id), {})
                title = meta.get('title', f'Movie {item_id}')
                genres = ', '.join(meta.get('genres', ['Unknown']))
                candidate_movies.append((item_id, title, genres))
                
            original_order = list(range(len(candidate_movies)))
            if args.candidate_order == 'reverse':
                candidate_movies = candidate_movies[::-1]
            elif args.candidate_order == 'random':
                rng = np.random.RandomState(args.shuffle_seed + hash(user_id) % 10000)
                indices = list(range(len(candidate_movies)))
                rng.shuffle(indices)
                candidate_movies = [candidate_movies[i] for i in indices]

            user_history = get_user_history_titles(
                int(user_id), user_train, movie_metadata
            )

            prompt = build_rerank_prompt(
                user_history, candidate_movies, args.prompt_version
            )

            # ---- Build the record ----
            record = {
                'user_id': user_id,
                'run_index': args.run_index,
                'model_name': args.output_name,
                'prompt_version': args.prompt_version,
                'temperature': args.temperature,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'input_candidate_ids': sasrec_items,
                'num_candidates': num_candidates,
                'candidate_order': args.candidate_order,
            }

            try:
                raw_response, latency = call_llm(
                    model, tokenizer, prompt, args.device,
                    temperature=args.temperature
                )

                positions, error_type = parse_llm_response(
                    raw_response, num_candidates
                )

                # Map positions to actual item IDs
                llm_item_ids = []
                for pos in positions:
                    if 1 <= pos <= num_candidates:
                        llm_item_ids.append(sasrec_items[pos - 1])

                # Validity: clean 10-item output with no errors
                is_valid = (
                    error_type is None
                    and len(llm_item_ids) == args.top_k_rerank
                )

                # Build filled list SEPARATELY — never overwrite llm_item_ids
                filled_item_ids = list(llm_item_ids)
                num_filled = 0
                if len(filled_item_ids) < args.top_k_rerank:
                    for item_id in sasrec_items:
                        if item_id not in filled_item_ids:
                            filled_item_ids.append(item_id)
                            num_filled += 1
                        if len(filled_item_ids) >= args.top_k_rerank:
                            break
                filled_item_ids = filled_item_ids[:args.top_k_rerank]

                record.update({
                    'raw_llm_output': raw_response,
                    'parsed_positions': positions,
                    'llm_item_ids': llm_item_ids,
                    'filled_item_ids': filled_item_ids,
                    'num_items_from_llm': len(llm_item_ids),
                    'num_items_filled': num_filled,
                    'is_valid': is_valid,
                    'error_type': error_type,
                    'latency_seconds': latency,
                    'exception': None,
                })

            except Exception as e:
                record.update({
                    'raw_llm_output': None,
                    'parsed_positions': [],
                    'llm_item_ids': [],
                    'filled_item_ids': sasrec_items[:args.top_k_rerank],
                    'num_items_from_llm': 0,
                    'num_items_filled': args.top_k_rerank,
                    'is_valid': False,
                    'error_type': 'exception',
                    'latency_seconds': None,
                    'exception': str(e),
                })
                errors += 1

            # Write immediately — each record survives crashes
            log_file.write(json.dumps(record) + '\n')
            log_file.flush()

            processed += 1
            if processed % 10 == 0:
                print(f"  [{args.output_name} run{args.run_index}] "
                      f"{processed}/{total} ({errors} errors)")

    print(f"\nDone. {processed} records saved to {log_path}")
    print(f"Errors: {errors}")


if __name__ == '__main__':
    main()