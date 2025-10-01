"""
Quick test script for LightGBM integration
"""

import pandas as pd
import numpy as np
from config import *
from model_trainer import ModelTrainer, amex_metric

def test_lightgbm():
    """Test LightGBM integration with synthetic data"""
    print("Testing LightGBM integration...")

    # Create synthetic data
    np.random.seed(SEED)
    n_samples = 1000
    n_features = 20

    # Generate features
    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f'feature_{i}' for i in range(n_features)])

    # Generate target (imbalanced like AmEx)
    y = pd.Series(np.random.binomial(1, 0.269, n_samples))  # 26.9% positive

    print(f"Data shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")

    # Test LightGBM trainer
    print("\nTesting LightGBM trainer...")
    lgb_trainer = ModelTrainer(model_type='lightgbm')

    try:
        models, oof_predictions = lgb_trainer.train_cv(X, y, n_folds=3, num_rounds=100)
        cv_score = amex_metric(y.values, oof_predictions)

        print(f"PASS LightGBM CV Score: {cv_score:.6f}")
        print(f"PASS Number of models trained: {len(models)}")

        # Test predictions
        test_pred = lgb_trainer.predict(X.head(100))
        print(f"PASS Test predictions shape: {test_pred.shape}")
        print(f"PASS Test prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

        # Test model info
        info = lgb_trainer.get_model_info()
        print(f"PASS Model info: {info}")

        return True

    except Exception as e:
        print(f"FAIL LightGBM test failed: {e}")
        return False

def test_xgboost():
    """Test XGBoost integration for comparison"""
    print("\nTesting XGBoost trainer...")

    # Create synthetic data
    np.random.seed(SEED)
    n_samples = 1000
    n_features = 20

    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.binomial(1, 0.269, n_samples))

    # Test XGBoost trainer
    xgb_trainer = ModelTrainer(model_type='xgboost')

    try:
        models, oof_predictions = xgb_trainer.train_cv(X, y, n_folds=3, num_rounds=100)
        cv_score = amex_metric(y.values, oof_predictions)

        print(f"PASS XGBoost CV Score: {cv_score:.6f}")
        print(f"PASS Number of models trained: {len(models)}")

        test_pred = xgb_trainer.predict(X.head(100))
        print(f"PASS Test predictions shape: {test_pred.shape}")
        print(f"PASS Test prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

        return True

    except Exception as e:
        print(f"FAIL XGBoost test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("MULTI-MODEL TRAINER TEST")
    print("="*50)

    # Test both models
    lgb_success = test_lightgbm()
    xgb_success = test_xgboost()

    print("\n" + "="*50)
    print("TEST RESULTS:")
    print(f"LightGBM: {'PASSED' if lgb_success else 'FAILED'}")
    print(f"XGBoost:  {'PASSED' if xgb_success else 'FAILED'}")

    if lgb_success and xgb_success:
        print("\nAll tests passed! Multi-model integration ready.")
    else:
        print("\nSome tests failed. Check error messages above.")