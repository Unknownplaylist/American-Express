"""
Full Training Script - Windows Version
Train on complete dataset with cross-validation
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
import gc
import os
from config import *
from cached_feature_engineering import CachedFeatureEngine

def amex_metric(y_true, y_pred):
    """
    AmEx evaluation metric
    """
    def top_four_percent_captured(y_true, y_pred, y_pred_binary):
        df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'bin': y_pred_binary})
        df = df.sort_values('pred', ascending=False)
        df['weight'] = df['true'].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['true'] == 1).sum() / (df['true'] == 1).sum()

    def weighted_gini(y_true, y_pred):
        df = pd.DataFrame({'true': y_true, 'pred': y_pred})
        df = df.sort_values('pred', ascending=False)
        df['weight'] = df['true'].apply(lambda x: 20 if x == 0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['true'] * df['weight']).sum()
        df['cum_pos_found'] = (df['true'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        n = len(df)
        gini = sum(df['lorentz'] - df['random']) * 2 / n
        return gini

    cutoff = int(0.04 * len(y_true))
    y_pred_binary = (y_pred >= np.sort(y_pred)[-cutoff])

    d = top_four_percent_captured(y_true, y_pred, y_pred_binary)
    g = weighted_gini(y_true, y_pred)

    return 0.5 * (d + g)

def load_and_prepare_data():
    """
    Load and prepare full training data (with caching)
    """
    print("Loading full training data...")

    # Use cached feature engineering
    feature_engine = CachedFeatureEngine()
    model_data = feature_engine.get_train_features(TRAIN_PATH, LABELS_PATH)
    print(f"Final model data: {model_data.shape}")

    return model_data

def clean_data(X):
    """
    Clean data for XGBoost
    """
    print("Cleaning data for XGBoost...")

    # Convert object columns to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Replace inf with nan, then fill with median
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    return X

def train_model():
    """
    Train XGBoost model with cross-validation
    """
    print("="*60)
    print("AMEX XGBOOST - FULL TRAINING (Windows Version)")
    print("="*60)

    # Load data
    model_data = load_and_prepare_data()

    # Prepare features and target
    feature_cols = [col for col in model_data.columns if col not in ['customer_ID', 'target']]
    X = model_data[feature_cols]
    y = model_data['target']

    # Clean data
    X = clean_data(X)

    print(f"Final dataset: {X.shape[0]} customers, {X.shape[1]} features")
    print(f"Target distribution: {y.mean():.4f}")

    # Cross-validation
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    oof_predictions = np.zeros(len(X))
    cv_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{FOLDS}")
        print("-" * 30)

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create DMatrix
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dval = xgb.DMatrix(data=X_val, label=y_val)

        # Train model
        model = xgb.train(
            XGB_PARAMS,
            dtrain,
            num_boost_round=2000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=100
        )

        # Predict
        val_pred = model.predict(dval)
        oof_predictions[val_idx] = val_pred

        # Calculate fold score
        fold_score = amex_metric(y_val.values, val_pred)
        cv_scores.append(fold_score)
        models.append(model)

        print(f"Fold {fold + 1} AmEx Score: {fold_score:.6f}")

        # Clean memory
        del dtrain, dval, X_train, X_val, y_train, y_val
        gc.collect()

    # Overall CV score
    overall_score = amex_metric(y.values, oof_predictions)

    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    for i, score in enumerate(cv_scores):
        print(f"Fold {i+1}: {score:.6f}")
    print(f"Mean CV Score: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
    print(f"Overall Score: {overall_score:.6f}")

    # Save best model (highest CV score)
    best_model = models[np.argmax(cv_scores)]
    best_model.save_model('amex_model_full.xgb')
    print(f"\nBest model saved: amex_model_full.xgb")

    # Save OOF predictions
    oof_df = pd.DataFrame({
        'customer_ID': model_data['customer_ID'],
        'target': y,
        'prediction': oof_predictions
    })
    oof_df.to_csv('oof_predictions.csv', index=False)
    print("OOF predictions saved: oof_predictions.csv")

    return best_model, overall_score

def generate_test_predictions(model):
    """
    Generate predictions for test set (with caching)
    """
    print("\n" + "="*60)
    print("GENERATING TEST PREDICTIONS")
    print("="*60)

    # Use cached feature engineering
    feature_engine = CachedFeatureEngine()
    test_features = feature_engine.get_test_features(TEST_PATH)
    print(f"Test features shape: {test_features.shape}")

    # Prepare for prediction
    feature_cols = [col for col in test_features.columns if col != 'customer_ID']
    X_test = test_features[feature_cols]
    X_test = clean_data(X_test)

    # Predict
    print("Making predictions...")
    dtest = xgb.DMatrix(data=X_test)
    predictions = model.predict(dtest)

    # Create submission with original customer_IDs
    if 'original_customer_ID' in test_features.columns:
        customer_ids = test_features['original_customer_ID']
        print("Using original customer_ID format for submission")
    else:
        # Fallback to converted format
        customer_ids = test_features['customer_ID'].apply(lambda x: hex(x)[2:].upper().zfill(16))
        print("Using converted customer_ID format")

    submission = pd.DataFrame({
        'customer_ID': customer_ids,
        'prediction': predictions
    })

    submission.to_csv('submission_full.csv', index=False)
    print("Submission saved: submission_full.csv")

    return submission

def main():
    """
    Main training pipeline
    """
    try:
        # Train model
        model, cv_score = train_model()

        # Generate test predictions
        submission = generate_test_predictions(model)

        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Final CV Score: {cv_score:.6f}")
        print("Files created:")
        print("- amex_model_full.xgb (trained model)")
        print("- oof_predictions.csv (out-of-fold predictions)")
        print("- submission_full.csv (test predictions)")
        print("\nReady for Kaggle submission!")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()