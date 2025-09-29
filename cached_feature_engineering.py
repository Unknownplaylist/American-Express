"""
Cached Feature Engineering
Save/load engineered features to avoid recomputation
"""

import pandas as pd
import numpy as np
import os
import pickle
import hashlib
from datetime import datetime
from simple_feature_engineering import process_and_feature_engineer

class CachedFeatureEngine:
    def __init__(self, cache_dir="feature_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_data_hash(self, file_path):
        """Get hash of data file for cache validation"""
        if not os.path.exists(file_path):
            return None

        # Hash file size and modification time (fast)
        stat = os.stat(file_path)
        hash_input = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def get_cache_filename(self, data_type, data_hash):
        """Get cache filename"""
        return os.path.join(self.cache_dir, f"{data_type}_features_{data_hash}.pkl")

    def save_features(self, features_df, data_type, data_hash):
        """Save engineered features to cache"""
        cache_file = self.get_cache_filename(data_type, data_hash)

        cache_data = {
            'features': features_df,
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'data_hash': data_hash,
            'shape': features_df.shape
        }

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"   Cached features: {cache_file}")
            print(f"   Shape: {features_df.shape}")
            return True
        except Exception as e:
            print(f"   Cache save failed: {e}")
            return False

    def load_features(self, data_type, data_hash):
        """Load engineered features from cache"""
        cache_file = self.get_cache_filename(data_type, data_hash)

        if not os.path.exists(cache_file):
            return None

        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            # Validate cache
            if cache_data['data_hash'] != data_hash:
                print(f"   Cache hash mismatch, regenerating...")
                return None

            features_df = cache_data['features']
            timestamp = cache_data['timestamp']

            print(f"   Loaded cached features: {cache_file}")
            print(f"   Shape: {features_df.shape}")
            print(f"   Created: {timestamp}")

            return features_df

        except Exception as e:
            print(f"   Cache load failed: {e}")
            return None

    def get_train_features(self, train_path, labels_path, force_rebuild=False):
        """Get training features (cached or fresh)"""
        print("Getting training features...")

        # Calculate data hash
        train_hash = self.get_data_hash(train_path)
        labels_hash = self.get_data_hash(labels_path)
        combined_hash = hashlib.md5(f"{train_hash}_{labels_hash}".encode()).hexdigest()[:12]

        # Try to load from cache
        if not force_rebuild:
            print("   Checking cache...")
            cached_features = self.load_features("train", combined_hash)
            if cached_features is not None:
                return cached_features

        # Generate fresh features
        print("   Cache miss - generating features...")
        print("   Loading training data...")
        try:
            train = pd.read_csv(train_path, engine='c', low_memory=False)
        except:
            print("   Retrying with python engine...")
            train = pd.read_csv(train_path, engine='python')

        print("   Loading labels...")
        try:
            labels = pd.read_csv(labels_path, engine='c', low_memory=False)
        except:
            print("   Retrying labels with python engine...")
            labels = pd.read_csv(labels_path, engine='python')
        # Handle customer_ID conversion more robustly
        def safe_hex_convert(x):
            try:
                # Take last 16 characters and convert to int
                hex_part = str(x)[-16:]
                return int(hex_part, 16)
            except:
                # Fallback: use hash of full string
                return hash(str(x)) % (2**63)

        labels['customer_ID'] = labels['customer_ID'].apply(safe_hex_convert)

        print("   Feature engineering...")
        features_df = process_and_feature_engineer(train)

        print("   Merging with labels...")
        # Ensure customer_ID types match
        features_df['customer_ID'] = features_df['customer_ID'].astype('uint64')
        labels['customer_ID'] = labels['customer_ID'].astype('uint64')
        model_data = features_df.merge(labels, on='customer_ID', how='inner')

        # Save to cache
        print("   Saving to cache...")
        self.save_features(model_data, "train", combined_hash)

        return model_data

    def get_test_features(self, test_path, force_rebuild=False):
        """Get test features (cached or fresh)"""
        print("Getting test features...")

        # Calculate data hash
        test_hash = self.get_data_hash(test_path)

        # Try to load from cache
        if not force_rebuild:
            print("   Checking cache...")
            cached_features = self.load_features("test", test_hash)
            if cached_features is not None:
                return cached_features

        # Generate fresh features
        print("   Cache miss - generating features...")
        print("   Loading test data...")
        try:
            test = pd.read_csv(test_path, engine='c', low_memory=False)
        except:
            print("   Retrying with python engine...")
            test = pd.read_csv(test_path, engine='python')

        # Store original customer_IDs before feature engineering
        original_customer_ids = test['customer_ID'].copy()

        print("   Feature engineering...")
        features_df = process_and_feature_engineer(test)

        # Create mapping from converted to original customer_IDs
        def safe_hex_convert(x):
            try:
                hex_part = str(x)[-16:]
                return int(hex_part, 16)
            except:
                return hash(str(x)) % (2**63)

        # Map converted customer_IDs back to original
        id_mapping = {}
        for orig_id in original_customer_ids:
            converted_id = safe_hex_convert(orig_id)
            id_mapping[converted_id] = orig_id

        # Add original customer_IDs to features using the mapping
        features_df['original_customer_ID'] = features_df['customer_ID'].map(id_mapping)

        # Save to cache
        print("   Saving to cache...")
        self.save_features(features_df, "test", test_hash)

        return features_df

    def clear_cache(self):
        """Clear all cached features"""
        if os.path.exists(self.cache_dir):
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            for cache_file in cache_files:
                os.remove(os.path.join(self.cache_dir, cache_file))
            print(f"Cleared {len(cache_files)} cache files")
        else:
            print("No cache directory found")

    def list_cache(self):
        """List cached features"""
        if not os.path.exists(self.cache_dir):
            print("No cache directory found")
            return

        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]

        if not cache_files:
            print("No cached features found")
            return

        print(f"Found {len(cache_files)} cached feature files:")

        for cache_file in cache_files:
            cache_path = os.path.join(self.cache_dir, cache_file)
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)

                print(f"  {cache_file}")
                print(f"    Type: {cache_data['data_type']}")
                print(f"    Shape: {cache_data['shape']}")
                print(f"    Created: {cache_data['timestamp']}")
                print(f"    Size: {os.path.getsize(cache_path) / 1024 / 1024:.1f} MB")

            except Exception as e:
                print(f"  {cache_file} (corrupted: {e})")

def main():
    """Test the caching system"""
    from config import TRAIN_PATH, TEST_PATH, LABELS_PATH

    engine = CachedFeatureEngine()

    print("CACHED FEATURE ENGINEERING TEST")
    print("="*50)

    # Test training features
    print("\n1. Testing training features...")
    train_features = engine.get_train_features(TRAIN_PATH, LABELS_PATH)
    print(f"Train features shape: {train_features.shape}")

    # Test loading from cache (should be fast)
    print("\n2. Testing cache loading...")
    train_features_cached = engine.get_train_features(TRAIN_PATH, LABELS_PATH)
    print(f"Cached features shape: {train_features_cached.shape}")

    # Test test features
    print("\n3. Testing test features...")
    test_features = engine.get_test_features(TEST_PATH)
    print(f"Test features shape: {test_features.shape}")

    # Show cache status
    print("\n4. Cache status...")
    engine.list_cache()

if __name__ == "__main__":
    main()