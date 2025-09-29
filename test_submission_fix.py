"""
Test Submission Fix with Small Batch
Verify that customer_ID format is correct for Kaggle
"""

import pandas as pd
from cached_feature_engineering import CachedFeatureEngine
from config import TEST_PATH

def test_small_batch_submission(n_rows=1000):
    """Test the submission fix with a small batch"""
    print("TESTING SUBMISSION FIX")
    print("="*50)

    # Create small test subset
    print(f"Loading first {n_rows} rows of test data...")
    try:
        test_small = pd.read_csv(TEST_PATH, nrows=n_rows, engine='c', low_memory=False)
        print(f"   Loaded {len(test_small)} rows")
        print(f"   Sample customer_ID: {test_small['customer_ID'].iloc[0]}")
        print(f"   Customer_ID length: {len(test_small['customer_ID'].iloc[0])}")

        # Save small test file
        small_test_path = 'data/test_small_batch.csv'
        test_small.to_csv(small_test_path, index=False)
        print(f"   Saved to: {small_test_path}")

    except Exception as e:
        print(f"   Error loading test data: {e}")
        return False

    # Test feature engineering with original customer_IDs
    print(f"\nTesting feature engineering...")
    try:
        engine = CachedFeatureEngine()

        # Force rebuild to test the fix
        test_features = engine.get_test_features(small_test_path, force_rebuild=True)
        print(f"   Features shape: {test_features.shape}")
        print(f"   Columns: {list(test_features.columns)}")

        # Check if original customer_IDs are preserved
        if 'original_customer_ID' in test_features.columns:
            print("   Original customer_IDs preserved!")
            print(f"   Sample original: {test_features['original_customer_ID'].iloc[0]}")
            print(f"   Sample converted: {test_features['customer_ID'].iloc[0]}")
        else:
            print("   Original customer_IDs missing!")
            return False

    except Exception as e:
        print(f"   Error in feature engineering: {e}")
        return False

    # Test submission format
    print(f"\nTesting submission format...")
    try:
        # Simulate predictions (random for testing)
        import numpy as np
        fake_predictions = np.random.random(len(test_features))

        # Create submission like the pipeline does
        feature_cols = [col for col in test_features.columns
                       if col not in ['customer_ID', 'original_customer_ID']]

        if 'original_customer_ID' in test_features.columns:
            customer_ids = test_features['original_customer_ID']
            print("   Using original customer_ID format")
        else:
            customer_ids = test_features['customer_ID'].apply(
                lambda x: hex(x)[2:].upper().zfill(16)
            )
            print("   Using converted customer_ID format")

        submission = pd.DataFrame({
            'customer_ID': customer_ids,
            'prediction': fake_predictions
        })

        # Save test submission
        submission.to_csv('test_submission_small.csv', index=False)
        print(f"   Test submission saved: test_submission_small.csv")
        print(f"   Submission shape: {submission.shape}")

        # Show samples
        print(f"\nSubmission sample:")
        print(submission.head(3).to_string(index=False))

        # Verify customer_ID format
        sample_id = submission['customer_ID'].iloc[0]
        print(f"\nCustomer_ID verification:")
        print(f"   Sample ID: {sample_id}")
        print(f"   ID length: {len(sample_id)}")
        print(f"   Is original format: {len(sample_id) > 16}")

        if len(sample_id) > 16:
            print("   SUCCESS: Using original long customer_ID format!")
            return True
        else:
            print("   ISSUE: Still using short format")
            return False

    except Exception as e:
        print(f"   Error creating submission: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    import os
    test_files = [
        'data/test_small_batch.csv',
        'test_submission_small.csv'
    ]

    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up: {file}")

if __name__ == "__main__":
    success = test_small_batch_submission()

    print(f"\n{'='*50}")
    if success:
        print("TEST PASSED: Submission fix is working!")
        print("Original customer_IDs will be used for Kaggle submission")
        print("Ready to run full pipeline")
    else:
        print("TEST FAILED: Submission fix needs more work")
        print("Check the errors above")

    # Ask if user wants to clean up
    cleanup = input(f"\nClean up test files? (y/n): ").strip().lower()
    if cleanup == 'y':
        cleanup_test_files()
        print("Test files cleaned up")