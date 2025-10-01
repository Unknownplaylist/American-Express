"""
Windows-Only Configuration for AmEx XGBoost Solution
Simplified for Windows compatibility
"""

import numpy as np

# VERSION AND SETTINGS
SEED = 42
NAN_VALUE = -127
FOLDS = 5

# PATHS - Local data directory
TRAIN_PATH = 'data/train_data.csv'
TEST_PATH = 'data/test_data.csv'
LABELS_PATH = 'data/train_labels.csv'
SAMPLE_SUB_PATH = 'data/sample_submission.csv'

# MODEL SELECTION
AVAILABLE_MODELS = ['xgboost', 'lightgbm']
DEFAULT_MODEL = 'xgboost'

# WINDOWS-OPTIMIZED XGB PARAMETERS
XGB_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.85,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_child_weight': 5,
    'gamma': 0.1,
    'eval_metric': 'logloss',
    'objective': 'binary:logistic',
    'tree_method': 'hist',          # CPU-only for Windows compatibility
    'random_state': SEED,
    'verbosity': 1,
    'n_jobs': -1                    # Use all CPU cores
}

# LIGHTGBM PARAMETERS - Custom Configuration
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'seed': SEED,
    'num_leaves': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.20,
    'bagging_freq': 10,
    'bagging_fraction': 0.50,
    'n_jobs': -1,
    'lambda_l2': 2,
    'min_data_in_leaf': 40,
    'verbosity': -1,
    'force_col_wise': True          # Better for Windows
}

# CATEGORICAL FEATURES
CAT_FEATURES = [
    "B_30", "B_38", "D_114", "D_116", "D_117", "D_120",
    "D_126", "D_63", "D_64", "D_66", "D_68"
]

# SMALL BATCH SETTINGS (for testing)
SMALL_BATCH = {
    'train_rows': 50000,
    'train_customers': 5000,
    'test_rows': 10000,
    'folds': 3,
    'boost_rounds': 500,
    'early_stopping': 50
}