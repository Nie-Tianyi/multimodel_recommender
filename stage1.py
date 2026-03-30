import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from collections import Counter
import random
from tqdm import tqdm
import os
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import CLIPModel, CLIPProcessor
import warnings
import pickle
warnings.filterwarnings('ignore')

def compute_counts(dataset, sample_fraction=1.0):
    """Compute user and item interaction counts."""
    user_counter = Counter()
    item_counter = Counter()
    
    # If sample_fraction < 1.0, sample rows
    if sample_fraction < 1.0:
        total = len(dataset)
        sample_size = int(total * sample_fraction)
        indices = random.sample(range(total), sample_size)
        dataset = dataset.select(indices)
    
    print("Counting interactions...")
    for batch in tqdm(dataset.iter(batch_size=10000), total=len(dataset)//10000):
        user_counter.update(batch['user_id'])
        item_counter.update(batch['asin'])
    
    return user_counter, item_counter

def filter_dataset(dataset, min_interactions=3, max_users=50000, max_items=100000):
    """Filter dataset to keep only users and items with at least min_interactions."""
    print("Computing counts...")
    user_counter, item_counter = compute_counts(dataset, sample_fraction=1.0)
    
    # Get users and items with enough interactions
    active_users = {user for user, count in user_counter.items() if count >= min_interactions}
    active_items = {item for item, count in item_counter.items() if count >= min_interactions}
    
    print(f"Active users: {len(active_users)}, Active items: {len(active_items)}")
    
    # Limit to top users and items by count
    top_users = sorted(active_users, key=lambda u: user_counter[u], reverse=True)[:max_users]
    top_items = sorted(active_items, key=lambda i: item_counter[i], reverse=True)[:max_items]
    
    top_users_set = set(top_users)
    top_items_set = set(top_items)
    
    # Filter dataset
    print("Filtering dataset...")
    def filter_fn(example):
        return example['user_id'] in top_users_set and example['asin'] in top_items_set
    
    filtered = dataset.filter(filter_fn, batched=False)  # batched=False for per-example filtering
    print(f"Filtered dataset size: {len(filtered)}")
    
    # Collect item metadata
    print("Collecting item metadata...")
    item_meta = {}
    for batch in tqdm(filtered.iter(batch_size=10000), total=len(filtered)//10000):
        batch_size = len(batch['asin'])
        for i in range(batch_size):
            asin = batch['asin'][i]
            if asin not in item_meta:
                item_meta[asin] = {
                    'title': batch['title'][i],
                    'text': batch['text'][i],
                    'images': batch['images'][i],
                    'parent_asin': batch['parent_asin'][i]
                }
    
    # Convert to DataFrame
    item_df = pd.DataFrame.from_dict(item_meta, orient='index').reset_index().rename(columns={'index': 'asin'})
    
    # Convert filtered dataset to pandas for sequence building
    df = filtered.to_pandas()
    
    return df, item_df

def extract_primary_image_url(images):
    """Extract primary image URL from images list."""
    if isinstance(images, list) and len(images) > 0:
        # Randomly select one image as per requirement
        img_dict = random.choice(images)
        return img_dict.get('large_image_url')
    return None

def load_models():
    """Load CLIP model."""
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device)
    
    return clip_model, clip_processor, device

def download_image_from_url(url, timeout=10):
    """Download image from URL and return PIL Image."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return None

def extract_image_features(image_urls, clip_model, clip_processor, device):
    """Extract image features from URLs."""
    features = []
    failed = 0
    
    for url in tqdm(image_urls, desc="Extracting image features"):
        if url is None:
            features.append(np.zeros(512, dtype=np.float32))
            failed += 1
            continue
            
        image = download_image_from_url(url)
        if image is None:
            features.append(np.zeros(512, dtype=np.float32))
            failed += 1
            continue
            
        try:
            inputs = clip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = clip_model.get_image_features(**inputs)
                # Handle different return types
                if isinstance(outputs, torch.Tensor):
                    image_features = outputs
                elif hasattr(outputs, 'image_embeds'):
                    image_features = outputs.image_embeds
                elif hasattr(outputs, 'pooler_output'):
                    image_features = outputs.pooler_output
                else:
                    # Try to get the first element
                    image_features = outputs[0]
                features.append(image_features.cpu().numpy().flatten().astype(np.float32))
        except Exception as e:
            print(f"Error processing image {url}: {e}")
            features.append(np.zeros(512, dtype=np.float32))
            failed += 1
    
    if failed > 0:
        print(f"Failed to process {failed} images")
    
    return np.array(features)

def combine_text(title, text):
    """Combine title and text, take first 100 words."""
    title_str = title if isinstance(title, str) else ''
    text_str = text if isinstance(text, str) else ''
    combined = title_str + ' ' + text_str
    words = combined.split()[:100]
    return ' '.join(words)

def extract_text_features_clip(texts, clip_processor, clip_model, device):
    """Extract text features using CLIP text encoder."""
    features = []
    batch_size = 32
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting text features"):
        batch_texts = texts[i:i+batch_size]
        try:
            inputs = clip_processor(text=batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = clip_model.get_text_features(**inputs)
                # Handle different return types
                if isinstance(outputs, torch.Tensor):
                    batch_features = outputs
                elif hasattr(outputs, 'text_embeds'):
                    batch_features = outputs.text_embeds
                elif hasattr(outputs, 'pooler_output'):
                    batch_features = outputs.pooler_output
                else:
                    batch_features = outputs[0]
                batch_features = batch_features.cpu().numpy()
            features.append(batch_features)
        except Exception as e:
            print(f"Error processing text batch: {e}")
            # Append zeros for failed batch
            features.append(np.zeros((len(batch_texts), 512), dtype=np.float32))
    
    return np.vstack(features)

def generate_id_embeddings(item_ids, embedding_dim=128):
    """Generate random ID embeddings."""
    np.random.seed(42)
    embeddings = np.random.randn(len(item_ids), embedding_dim).astype(np.float32)
    id_to_idx = {id: i for i, id in enumerate(item_ids)}
    return id_to_idx, embeddings

def build_user_sequences(df, max_len=20):
    """Build user interaction sequences."""
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df = df.sort_values(['user_id', 'timestamp'])
    
    sequences = {}
    for user_id, group in tqdm(df.groupby('user_id'), desc="Building user sequences"):
        item_seq = group['asin'].tolist()
        if len(item_seq) > max_len:
            item_seq = item_seq[-max_len:]
        sequences[user_id] = item_seq
    
    return sequences

def main():
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(
        'McAuley-Lab/Amazon-Reviews-2023',
        'raw_review_Clothing_Shoes_and_Jewelry',
        split='full',
        trust_remote_code=True
    )
    print(f"Dataset loaded: {len(dataset)} rows")
    
    # For testing, sample a fraction of the dataset
    sample_fraction = 0.2  # 20% for larger sample
    if sample_fraction < 1.0:
        total = len(dataset)
        sample_size = int(total * sample_fraction)
        indices = random.sample(range(total), sample_size)
        dataset = dataset.select(indices)
        print(f"Sampled dataset: {len(dataset)} rows")
    
    # Step 1: Filter dataset
    df, item_df = filter_dataset(dataset, min_interactions=3, max_users=50000, max_items=100000)
    
    # Save filtered data
    df.to_csv('filtered_interactions.csv', index=False)
    item_df.to_csv('item_metadata.csv', index=False)
    print(f"Filtered interactions: {len(df)} rows, {df['user_id'].nunique()} users, {df['asin'].nunique()} items")
    
    # Step 2: Prepare item metadata
    item_df['primary_image_url'] = item_df['images'].apply(extract_primary_image_url)
    item_df = item_df[item_df['primary_image_url'].notna()].reset_index(drop=True)
    print(f"Items with image URL: {len(item_df)}")
    
    # Limit to 100k items if more
    if len(item_df) > 100000:
        item_df = item_df.head(100000)
    
    # Step 3: Load models
    clip_model, clip_processor, device = load_models()
    
    # Step 4: Extract image features
    image_urls = item_df['primary_image_url'].tolist()
    image_features = extract_image_features(image_urls, clip_model, clip_processor, device)
    
    # Step 5: Extract text features
    texts = [combine_text(row['title'], row['text']) for _, row in item_df.iterrows()]
    text_features = extract_text_features_clip(texts, clip_processor, clip_model, device)
    
    # Step 6: Generate ID embeddings
    item_ids = item_df['asin'].tolist()
    id_to_idx, id_embeddings = generate_id_embeddings(item_ids)
    
    # Step 7: Save features
    np.save('image_features.npy', image_features)
    np.save('text_features.npy', text_features)
    np.save('id_embeddings.npy', id_embeddings)
    np.save('item_ids.npy', np.array(item_ids))
    
    # Save mapping
    with open('id_to_idx.pkl', 'wb') as f:
        pickle.dump(id_to_idx, f)
    
    # Step 8: Build user sequences
    sequences = build_user_sequences(df)
    with open('user_sequences.pkl', 'wb') as f:
        pickle.dump(sequences, f)
    
    # Save sequence data in numpy format as well (for easy loading)
    max_seq_len = 20
    user_ids = list(sequences.keys())
    seq_array = np.zeros((len(user_ids), max_seq_len), dtype=np.int32)
    for i, user_id in enumerate(user_ids):
        seq = sequences[user_id]
        # Convert item ids to indices
        seq_indices = [id_to_idx.get(item, -1) for item in seq]
        seq_indices = [idx for idx in seq_indices if idx != -1]
        if len(seq_indices) > max_seq_len:
            seq_indices = seq_indices[-max_seq_len:]
        seq_array[i, :len(seq_indices)] = seq_indices
    
    np.save('user_ids.npy', np.array(user_ids))
    np.save('user_sequences.npy', seq_array)
    
    print("Stage 1 completed successfully!")
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    print(f"ID embeddings shape: {id_embeddings.shape}")
    print(f"Sequences shape: {seq_array.shape}")

if __name__ == '__main__':
    main()