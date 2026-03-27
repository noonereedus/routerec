import json

with open('evaluation_data/sasrec_candidates.json') as f:
    candidates = json.load(f)

# Check first 5 users
for user_id in list(candidates.keys())[:5]:
    data = candidates[user_id]
    gt = data['ground_truth']
    items = data['items']
    
    print(f"\nUser: {user_id} (type: {type(user_id)})")
    print(f"  Ground truth: {gt} (type: {type(gt)})")
    print(f"  Num candidates: {len(items)}")
    print(f"  First 5 items: {items[:5]} (type: {type(items[0])})")
    print(f"  GT in top 10? {gt in items[:10]}")
    print(f"  GT in top 50? {gt in items[:50]}")
    print(f"  GT in all items? {gt in items}")
    
    # Check with type conversion
    gt_int = int(gt)
    items_int = [int(x) for x in items]
    print(f"  GT(int) in items(int)? {gt_int in items_int[:10]}")

# Also check Qwen output
print("\n\n=== QWEN RERANKED ===")
try:
    with open('evaluation_data/qwen_reranked.json') as f:
        qwen = json.load(f)
    
    first_user = list(qwen.keys())[0]
    print(f"First user: {first_user} (type: {type(first_user)})")
    print(f"Rankings: {qwen[first_user]}")
    print(f"Type of items: {type(qwen[first_user][0])}")
    
    # Compare with SASRec candidates
    sasrec_items = candidates[first_user]['items'][:10]
    print(f"\nSASRec top 10: {sasrec_items}")
    print(f"Qwen top 10:   {qwen[first_user][:10]}")
    print(f"Are they the same? {sasrec_items == qwen[first_user][:10]}")
except Exception as e:
    print(f"Error: {e}")