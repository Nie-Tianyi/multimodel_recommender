import pandas as pd
import numpy as np
from datasets import load_dataset
from collections import Counter
import random
from tqdm import tqdm
import os
import requests
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, DistilBertTokenizer, DistilBertModel
import warnings
warnings.filterwarnings('ignore')

def load_and_filter_dataset():
    print("Loading Clothing_Shoes_and_Jewelry dataset...")
    dataset = load_dataset(
        'McAuley-Lab/Amazon-Reviews-2023',
        'raw_review_Clothing_Shoes_and_Jewelry',
        split='full',
        trust_remote_code=True
    )
    print(f"Total rows: {len(dataset)}")
    
    # Convert to pandas for easier manipulation (if fits in memory)
    # We'll process in chunks due to large size
    # For now, let's sample a subset for development
    # We'll take 1% for prototyping
    sample_size = int(len(dataset) * 0.01)
    indices = random.sample(range(len(dataset)), sample_size)
    sampled = dataset.select(indices)
    df = sampled.to_pandas()
    
    # Count interactions per user and per item
    user_counts = df['user_id'].value_counts()
    item_counts = df['asin'].value_counts()
    
    # Filter users and items with at least 3 interactions
    active_users = user_counts[user_counts >= 3].index.tolist()
    active_items = item_counts[item_counts >= 3].index.tolist()
    
    print(f"Active users (>=3 interactions): {len(active_users)}")
    print(f"Active items (>=3 interactions): {len(active_items)}")
    
    # Filter dataframe
    filtered_df = df[df['user_id'].isin(active_users) & df['asin'].isin(active_items)]
    
    # Further limit to 50k users and 100k items (if more)
    # Keep top users and items by interaction count
    top_users = user_counts[user_counts.index.isin(active_users)].nlargest(50000).index.tolist()
    top_items = item_counts[item_counts.index.isin(active_items)].nlargest(100000).index.tolist()
    
    final_df = filtered_df[filtered_df['user_id'].isin(top_users) & filtered_df['asin'].isin(top_items)]
    print(f"Final dataset size: {len(final_df)} rows, {final_df['user_id'].nunique()} users, {final_df['asin'].nunique()} items")
    
    # Deduplicate items to get item metadata (title, text, images)
    # For each item, take the first occurrence (or aggregate)
    item_meta = final_df.groupby('asin').agg({
        'title': 'first',
        'text': 'first',
        'images': 'first',
        'parent_asin': 'first'
    }).reset_index()
    
    # For images, we need to extract a primary image URL
    def extract_primary_image(img_list):
        if isinstance(img_list, list) and len(img_list) > 0:
            # Randomly select one image as per requirement
            img_dict = random.choice(img_list)
            # Use large image URL
            return img_dict.get('large_image_url')
        return None
    
    item_meta['primary_image_url'] = item_meta['images'].apply(extract_primary_image)
    # Keep only items with image URL
    item_meta = item_meta[item_meta['primary_image_url'].notna()]
    print(f"Items with image URL: {len(item_meta)}")
    
    return final_df, item_meta

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False

def extract_image_features(image_paths, model, processor):
    features = []
    for path in tqdm(image_paths, desc="Extracting image features"):
        try:
            image = Image.open(path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            features.append(image_features.cpu().numpy().flatten())
        except Exception as e:
            print(f"Error processing {path}: {e}")
            features.append(np.zeros(512))
    return np.array(features)

def extract_text_features(texts, tokenizer, model):
    features = []
    for text in tqdm(texts, desc="Extracting text features"):
        try:
            inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                # Use pooler output or last hidden state mean
                text_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            features.append(text_features)
        except Exception as e:
            print(f"Error processing text: {e}")
            features.append(np.zeros(512))
    return np.array(features)

def generate_id_embeddings(item_ids, embedding_dim=128):
    np.random.seed(42)
    id_to_idx = {id: i for i, id in enumerate(item_ids)}
    embeddings = np.random.randn(len(item_ids), embedding_dim).astype(np.float32)
    return id_to_idx, embeddings

def build_user_sequences(df, max_len=20):
    # Sort by timestamp
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df = df.sort_values(['user_id', 'timestamp'])
    
    sequences = {}
    for user_id, group in tqdm(df.groupby('user_id'), desc="Building user sequences"):
        item_seq = group['asin'].tolist()
        if len(item_seq) > max_len:
            item_seq = item_seq[-max_len:]  # keep most recent
        sequences[user_id] = item_seq
    return sequences

def main():
    # Step 1: Load and filter dataset
    df, item_meta = load_and_filter_dataset()
    
    # Save filtered data for later use
    df.to_csv('filtered_interactions.csv', index=False)
    item_meta.to_csv('item_metadata.csv', index=False)
    
    # Step 2: Download images (optional: we can directly extract features from URL)
    # For simplicity, we'll download images to a temp directory
    os.makedirs('images', exist_ok=True)
    image_paths = []
    for idx, row in tqdm(item_meta.iterrows(), total=len(item_meta), desc="Downloading images"):
        item_id = row['asin']
        url = row['primary_image_url']
        if url:
            ext = url.split('.')[-1].split('?')[0]
            if ext not in ['jpg', 'jpeg', 'png', 'webp']:
                ext = 'jpg'
            save_path = f'images/{item_id}.{ext}'
            if not os.path.exists(save_path):
                download_image(url, save_path)
            if os.path.exists(save_path):
                image_paths.append(save_path)
            else:
                image_paths.append(None)
        else:
            image_paths.append(None)
    
    # Step 3: Load models
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print("Loading DistilBERT model...")
    text_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device)
    text_model.to(device)
    
    # Step 4: Extract image features
    valid_indices = [i for i, path in enumerate(image_paths) if path is not None]
    valid_paths = [image_paths[i] for i in valid_indices]
    valid_items = item_meta.iloc[valid_indices]
    
    image_features = extract_image_features(valid_paths, clip_model, clip_processor)
    
    # Step 5: Extract text features
    # Combine title and text, take first 100 words
    def combine_text(row):
        title = row['title'] if isinstance(row['title'], str) else ''
        text = row['text'] if isinstance(row['text'], str) else ''
        combined = title + ' ' + text
        words = combined.split()[:100]
        return ' '.join(words)
    
    texts = valid_items.apply(combine_text, axis=1).tolist()
    text_features = extract_text_features(texts, text_tokenizer, text_model)
    
    # Step 6: Generate ID embeddings
    item_ids = valid_items['asin'].tolist()
    id_to_idx, id_embeddings = generate_id_embeddings(item_ids)
    
    # Step 7: Save features
    np.save('image_features.npy', image_features)
    np.save('text_features.npy', text_features)
    np.save('id_embeddings.npy', id_embeddings)
    np.save('item_ids.npy', np.array(item_ids))
    
    # Step 8: Build user sequences
    sequences = build_user_sequences(df)
    # Save sequences as dictionary
    import pickle
    with open('user_sequences.pkl', 'wb') as f:
        pickle.dump(sequences, f)
    
    print("Data preprocessing completed!")

if __name__ == '__main__':
    main()