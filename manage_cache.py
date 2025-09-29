"""
Cache Management Utility
Manage cached feature engineering files
"""

import os
import sys
from cached_feature_engineering import CachedFeatureEngine

def main():
    """Cache management interface"""
    engine = CachedFeatureEngine()

    if len(sys.argv) < 2:
        print("CACHE MANAGEMENT UTILITY")
        print("="*40)
        print("Commands:")
        print("  python manage_cache.py list     - List cached features")
        print("  python manage_cache.py clear    - Clear all cache")
        print("  python manage_cache.py test     - Test caching system")
        print("  python manage_cache.py rebuild  - Force rebuild cache")
        return

    command = sys.argv[1].lower()

    if command == "list":
        print("CACHED FEATURES")
        print("="*40)
        engine.list_cache()

    elif command == "clear":
        print("CLEARING CACHE")
        print("="*40)
        confirm = input("Are you sure? This will delete all cached features (y/n): ")
        if confirm.lower() == 'y':
            engine.clear_cache()
            print("Cache cleared!")
        else:
            print("Cancelled.")

    elif command == "test":
        print("TESTING CACHE SYSTEM")
        print("="*40)
        from config import TRAIN_PATH, TEST_PATH, LABELS_PATH

        # Test training cache
        print("\n1. Testing training cache (first run)...")
        train_data = engine.get_train_features(TRAIN_PATH, LABELS_PATH)
        print(f"   Result: {train_data.shape}")

        print("\n2. Testing training cache (second run - should be fast)...")
        train_data = engine.get_train_features(TRAIN_PATH, LABELS_PATH)
        print(f"   Result: {train_data.shape}")

        # Test test cache
        print("\n3. Testing test cache...")
        test_data = engine.get_test_features(TEST_PATH)
        print(f"   Result: {test_data.shape}")

        print("\n4. Cache status:")
        engine.list_cache()

    elif command == "rebuild":
        print("REBUILDING CACHE")
        print("="*40)
        from config import TRAIN_PATH, TEST_PATH, LABELS_PATH

        print("Rebuilding training features...")
        train_data = engine.get_train_features(TRAIN_PATH, LABELS_PATH, force_rebuild=True)
        print(f"Training features: {train_data.shape}")

        print("Rebuilding test features...")
        test_data = engine.get_test_features(TEST_PATH, force_rebuild=True)
        print(f"Test features: {test_data.shape}")

        print("Cache rebuilt successfully!")

    else:
        print(f"Unknown command: {command}")
        print("Use 'python manage_cache.py' for help")

if __name__ == "__main__":
    main()