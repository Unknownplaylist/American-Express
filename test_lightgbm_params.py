"""
Test LightGBM with the exact parameters from config
"""

import pandas as pd
import numpy as np
from config import LGBM_PARAMS, SEED
from model_trainer import ModelTrainer, amex_metric

def test_lightgbm_params():
    """Test LightGBM with configured parameters"""
    print("Testing LightGBM with configured parameters...")
    print(f"Parameters: {LGBM_PARAMS}")

    # Create synthetic data
    np.random.seed(SEED)
    n_samples = 1000
    n_features = 20

    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.binomial(1, 0.269, n_samples))

    print(f"\nData shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Test with configured parameters
    trainer = ModelTrainer(model_type='lightgbm')

    try:
        print("\nTraining with configured LightGBM parameters...")
        models, oof_predictions = trainer.train_cv(X, y, n_folds=3,
                                                 params=LGBM_PARAMS,
                                                 num_rounds=100)

        cv_score = amex_metric(y.values, oof_predictions)
        print(f"CV Score: {cv_score:.6f}")

        # Test predictions
        test_pred = trainer.predict(X.head(100))
        print(f"Predictions range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_lightgbm_params()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")