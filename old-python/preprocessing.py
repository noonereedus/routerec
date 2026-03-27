import pandas as pd

# Load MovieLens-1M
df = pd.read_csv(
    '../ml-1m/ratings.dat',
    sep='::',
    names=['user_id', 'item_id', 'rating', 'timestamp'],
    engine='python'
)

# Sort by timestamp
df = df.sort_values(['user_id', 'timestamp'])

# Filter users with < 5 interactions
user_counts = df.groupby('user_id').size()
valid_users = user_counts[user_counts >= 5].index
df = df[df['user_id'].isin(valid_users)]

# Remap IDs to contiguous integers (starting from 1)
user_map = {u: i+1 for i, u in enumerate(sorted(df['user_id'].unique()))}
item_map = {it: i+1 for i, it in enumerate(sorted(df['item_id'].unique()))}

df['user_id'] = df['user_id'].map(user_map)
df['item_id'] = df['item_id'].map(item_map)

# Write as user-item pairs (one per line)
with open('SASRec.pytorch/python/data/ml-1m.txt', 'w') as f:
    for _, row in df.iterrows():
        f.write(f"{row['user_id']} {row['item_id']}\n")

print(f"Users: {df['user_id'].nunique()}")
print(f"Items: {df['item_id'].nunique()}")
print(f"Interactions: {len(df)}")