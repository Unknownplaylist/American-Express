"""
Multi-Model Trainer
Supports both XGBoost and LightGBM with same features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import pickle
import gc
from config import *

class ModelTrainer:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type.lower()
        if self.model_type not in AVAILABLE_MODELS:
            raise ValueError(f"Model type must be one of {AVAILABLE_MODELS}")

        self.models = []
        self.oof_predictions = None

    def _import_model_library(self):
        """Import the appropriate model library"""
        if self.model_type == 'xgboost':
            import xgboost as xgb
            return xgb
        elif self.model_type == 'lightgbm':
            import lightgbm as lgb
            return lgb

    def _get_model_params(self):
        """Get parameters for the selected model"""
        if self.model_type == 'xgboost':
            return XGB_PARAMS.copy()
        elif self.model_type == 'lightgbm':
            return LGBM_PARAMS.copy()

    def _create_dataset(self, X, y, X_val=None, y_val=None):
        """Create model-specific dataset"""
        if self.model_type == 'xgboost':
            xgb = self._import_model_library()
            dtrain = xgb.DMatrix(data=X, label=y)
            if X_val is not None:
                dval = xgb.DMatrix(data=X_val, label=y_val)
                return dtrain, dval
            return dtrain

        elif self.model_type == 'lightgbm':
            lgb = self._import_model_library()
            dtrain = lgb.Dataset(X, label=y)
            if X_val is not None:
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
                return dtrain, dval
            return dtrain

    def _train_model(self, dtrain, dval=None, params=None, num_rounds=1000):
        """Train model with appropriate library"""
        if params is None:
            params = self._get_model_params()

        if self.model_type == 'xgboost':
            xgb = self._import_model_library()

            if dval is not None:
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_rounds,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=50,
                    verbose_eval=False
                )
            else:
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_rounds,
                    verbose_eval=False
                )

        elif self.model_type == 'lightgbm':
            lgb = self._import_model_library()

            if dval is not None:
                model = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_rounds,
                    valid_sets=[dtrain, dval],
                    valid_names=['train', 'val'],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
            else:
                model = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_rounds,
                    callbacks=[lgb.log_evaluation(0)]
                )

        return model

    def _predict(self, model, X):
        """Make predictions with the trained model"""
        if self.model_type == 'xgboost':
            xgb = self._import_model_library()
            dtest = xgb.DMatrix(data=X)
            return model.predict(dtest)

        elif self.model_type == 'lightgbm':
            return model.predict(X, num_iteration=model.best_iteration)

    def _save_model(self, model, filename):
        """Save model to file"""
        if self.model_type == 'xgboost':
            model.save_model(filename)
        elif self.model_type == 'lightgbm':
            model.save_model(filename)

    def _load_model(self, filename):
        """Load model from file"""
        if self.model_type == 'xgboost':
            xgb = self._import_model_library()
            model = xgb.Booster()
            model.load_model(filename)
            return model
        elif self.model_type == 'lightgbm':
            lgb = self._import_model_library()
            return lgb.Booster(model_file=filename)

    def train_cv(self, X, y, n_folds=3, params=None, num_rounds=1000):
        """Train with cross-validation"""
        print(f"Training {self.model_type.upper()} with {n_folds}-fold CV...")

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        self.oof_predictions = np.zeros(len(X))
        self.models = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Fold {fold + 1}/{n_folds}", end=" ")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Create datasets
            dtrain, dval = self._create_dataset(X_train, y_train, X_val, y_val)

            # Train model
            model = self._train_model(dtrain, dval, params, num_rounds)

            # Predict on validation
            val_pred = self._predict(model, X_val)
            self.oof_predictions[val_idx] = val_pred

            # Store model
            self.models.append(model)

            # Cleanup
            del dtrain, dval, X_train, X_val, y_train, y_val
            gc.collect()

        print("CV training complete")
        return self.models, self.oof_predictions

    def predict(self, X, use_best_model=True):
        """Make predictions using trained models"""
        if not self.models:
            raise ValueError("No models trained. Run train_cv first.")

        if use_best_model and len(self.models) > 1:
            # Use the first model (or implement best model selection)
            model = self.models[0]
            return self._predict(model, X)
        else:
            # Average predictions from all models
            predictions = np.zeros(len(X))
            for model in self.models:
                pred = self._predict(model, X)
                predictions += pred
            return predictions / len(self.models)

    def save_models(self, prefix="model"):
        """Save all trained models"""
        saved_files = []
        for i, model in enumerate(self.models):
            filename = f"{prefix}_{self.model_type}_fold_{i}.txt"
            self._save_model(model, filename)
            saved_files.append(filename)
        return saved_files

    def get_model_info(self):
        """Get information about the trained models"""
        return {
            'model_type': self.model_type,
            'n_models': len(self.models),
            'has_oof': self.oof_predictions is not None,
            'oof_shape': self.oof_predictions.shape if self.oof_predictions is not None else None
        }

def amex_metric(y_true, y_pred):
    """AmEx evaluation metric"""
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