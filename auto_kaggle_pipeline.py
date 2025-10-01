"""
Automated Kaggle Pipeline with Auto-Tuning
Trains, submits, gets score, and tunes parameters automatically
"""

import pandas as pd
import numpy as np
import subprocess
import time
import json
import os
import gc
from datetime import datetime
from config import *
from cached_feature_engineering import CachedFeatureEngine
from model_trainer import ModelTrainer, amex_metric

class AutoKagglePipeline:
    def __init__(self, competition="amex-default-prediction", model_type=DEFAULT_MODEL):
        self.competition = competition
        self.model_type = model_type.lower()
        self.best_score = 0.0
        self.best_params = self._get_default_params()
        self.submission_history = []
        self.iteration = 0
        self.feature_engine = CachedFeatureEngine()
        self.train_data = None  # Cache training data

    def _get_default_params(self):
        """Get default parameters for selected model"""
        if self.model_type == 'xgboost':
            return XGB_PARAMS.copy()
        elif self.model_type == 'lightgbm':
            return LGBM_PARAMS.copy()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def amex_metric(self, y_true, y_pred):
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

    def load_and_prepare_data(self):
        """Load and prepare training data (with caching)"""
        if self.train_data is not None:
            print("Using cached training data...")
            return self.train_data

        print("Loading and engineering training features...")
        self.train_data = self.feature_engine.get_train_features(TRAIN_PATH, LABELS_PATH)
        return self.train_data

    def clean_data(self, X):
        """Clean data for XGBoost"""
        X = X.copy()  # Avoid SettingWithCopyWarning
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        return X

    def train_model(self, params):
        """Train model with given parameters"""
        print(f"\nTraining {self.model_type.upper()} with parameters: {params}")

        model_data = self.load_and_prepare_data()
        feature_cols = [col for col in model_data.columns if col not in ['customer_ID', 'target']]
        X = model_data[feature_cols]
        y = model_data['target']
        X = self.clean_data(X)

        # Use ModelTrainer for flexible model training
        trainer = ModelTrainer(model_type=self.model_type)
        models, oof_predictions = trainer.train_cv(X, y, n_folds=3, params=params, num_rounds=1000)

        cv_score = amex_metric(y.values, oof_predictions)
        print(f"CV Score: {cv_score:.6f}")

        # Use best model for predictions (first model for now)
        best_model = trainer
        return best_model, cv_score

    def generate_predictions(self, model):
        """Generate test predictions (with caching)"""
        print("Generating test predictions...")

        # Use cached feature engineering
        test_features = self.feature_engine.get_test_features(TEST_PATH)

        # Exclude customer_ID columns from features
        feature_cols = [col for col in test_features.columns
                       if col not in ['customer_ID', 'original_customer_ID']]
        X_test = test_features[feature_cols]
        X_test = self.clean_data(X_test)

        # Use ModelTrainer's predict method
        predictions = model.predict(X_test, use_best_model=True)

        # Use original customer_IDs for submission if available
        if 'original_customer_ID' in test_features.columns:
            customer_ids = test_features['original_customer_ID']
            print("   Using original customer_ID format for Kaggle submission")
        else:
            # Fallback to converted format (for backward compatibility)
            customer_ids = test_features['customer_ID'].apply(
                lambda x: hex(x)[2:].upper().zfill(16)
            )
            print("   Using converted customer_ID format")

        submission = pd.DataFrame({
            'customer_ID': customer_ids,
            'prediction': predictions
        })

        return submission

    def submit_to_kaggle(self, submission, message=""):
        """Submit to Kaggle and get score"""
        filename = f"submission_iter_{self.iteration}.csv"
        submission.to_csv(filename, index=False)

        print(f"Submitting {filename} to Kaggle...")

        try:
            cmd = [
                "kaggle", "competitions", "submit",
                "-c", self.competition,
                "-f", filename,
                "-m", message or f"Auto iteration {self.iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Submission failed: {result.stderr}")
                return None

            print("Submission successful! Waiting for score...")

            # Wait for score
            score = self.wait_for_score()
            return score

        except Exception as e:
            print(f"Submission error: {e}")
            return None

    def wait_for_score(self, max_wait_minutes=10):
        """Wait for Kaggle score with improved parsing"""
        max_attempts = max(1, int(max_wait_minutes * 4))  # Check every 15 seconds, minimum 1

        for attempt in range(max_attempts):
            try:
                # Get submissions in CSV format
                cmd = ["kaggle", "competitions", "submissions", "-c", self.competition, "--csv"]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        # Parse header to find column positions
                        header = lines[0].split(',')
                        try:
                            public_score_idx = header.index('publicScore')
                        except ValueError:
                            # Try alternative header names
                            for idx, col in enumerate(header):
                                if 'score' in col.lower() and 'public' in col.lower():
                                    public_score_idx = idx
                                    break
                            else:
                                print(f"Could not find score column in header: {header}")
                                continue

                        # Look for first submission with actual data
                        for line_idx in range(1, len(lines)):
                            submission = lines[line_idx].split(',')

                            if len(submission) > public_score_idx and submission[public_score_idx].strip():
                                try:
                                    score = float(submission[public_score_idx].strip())
                                    print(f"Kaggle Public Score: {score:.6f}")

                                    # Also try to get private score if available
                                    try:
                                        private_score_idx = header.index('privateScore')
                                        if len(submission) > private_score_idx and submission[private_score_idx].strip():
                                            private_score = float(submission[private_score_idx].strip())
                                            print(f"Kaggle Private Score: {private_score:.6f}")
                                    except (ValueError, IndexError):
                                        pass

                                    return score
                                except (ValueError, IndexError) as e:
                                    print(f"Could not parse score from: {submission[public_score_idx]} (error: {e})")
                                    continue

                if attempt < max_attempts - 1:
                    print(f"Waiting for score... ({attempt + 1}/{max_attempts})")
                    time.sleep(15)

            except Exception as e:
                print(f"Error checking score: {e}")
                break

        print("Score not available, continuing...")
        return None

    def tune_parameters(self):
        """Advanced parameter tuning based on iteration and performance"""
        if len(self.submission_history) == 0:
            return self.best_params

        new_params = self.best_params.copy()
        iteration = len(self.submission_history) + 1

        if self.model_type == 'xgboost':
            self._tune_xgboost_params(new_params, iteration)
        elif self.model_type == 'lightgbm':
            self._tune_lightgbm_params(new_params, iteration)

        print(f"   Tuning strategy for iteration {iteration}: {self.get_phase_name(iteration)}")
        return new_params

    def _tune_xgboost_params(self, new_params, iteration):
        """XGBoost-specific parameter tuning"""
        # Phase 1: Learning Rate & Depth (Iterations 1-3)
        if iteration <= 3:
            if self.best_score < 0.795:
                new_params['learning_rate'] = max(0.02, new_params['learning_rate'] * 0.8)
                new_params['max_depth'] = min(8, new_params['max_depth'] + 1)
                new_params['subsample'] = min(0.95, new_params['subsample'] + 0.05)

        # Phase 2: Regularization (Iterations 4-6)
        elif iteration <= 6:
            new_params['reg_alpha'] = max(0.01, new_params['reg_alpha'] * 0.7)
            new_params['reg_lambda'] = max(0.01, new_params['reg_lambda'] * 0.7)
            new_params['min_child_weight'] = max(1, new_params['min_child_weight'] - 1)
            new_params['gamma'] = max(0.01, new_params['gamma'] * 0.5)

        # Phase 3: Feature Sampling (Iterations 7-9)
        elif iteration <= 9:
            new_params['colsample_bytree'] = min(0.9, new_params['colsample_bytree'] + 0.1)
            new_params['colsample_bylevel'] = 0.8
            new_params['colsample_bynode'] = 0.8

        # Phase 4: Advanced Features (Iterations 10-12)
        elif iteration <= 12:
            new_params['max_depth'] = min(10, new_params['max_depth'] + 1)
            new_params['learning_rate'] = max(0.015, new_params['learning_rate'] * 0.9)
            new_params['scale_pos_weight'] = 3.7  # Based on 26.9% positive class
            new_params['max_delta_step'] = 1
            new_params['grow_policy'] = 'lossguide'

        # Phase 5: Fine-tuning (Iterations 13+)
        else:
            # Ensemble approach with different seeds
            seeds = [42, 123, 456, 789, 999]
            new_params['random_state'] = seeds[iteration % len(seeds)]
            new_params['learning_rate'] = max(0.01, new_params['learning_rate'] * 0.95)

    def _tune_lightgbm_params(self, new_params, iteration):
        """LightGBM-specific parameter tuning"""
        # Phase 1: Learning Rate & Depth (Iterations 1-3)
        if iteration <= 3:
            if self.best_score < 0.795:
                new_params['learning_rate'] = max(0.005, new_params['learning_rate'] * 0.8)
                new_params['num_leaves'] = min(127, int(new_params['num_leaves'] * 1.2))
                new_params['bagging_fraction'] = min(0.7, new_params['bagging_fraction'] + 0.1)

        # Phase 2: Regularization (Iterations 4-6)
        elif iteration <= 6:
            new_params['lambda_l2'] = max(0.5, new_params['lambda_l2'] * 0.8)
            new_params['feature_fraction'] = min(0.3, new_params['feature_fraction'] + 0.05)
            new_params['min_data_in_leaf'] = max(20, new_params['min_data_in_leaf'] - 5)

        # Phase 3: Feature Sampling (Iterations 7-9)
        elif iteration <= 9:
            new_params['feature_fraction'] = min(0.4, new_params['feature_fraction'] + 0.1)
            new_params['bagging_freq'] = max(5, new_params['bagging_freq'] - 2)
            new_params['min_data_in_leaf'] = 25

        # Phase 4: Advanced Features (Iterations 10-12)
        elif iteration <= 12:
            new_params['num_leaves'] = min(150, int(new_params['num_leaves'] * 1.1))
            new_params['learning_rate'] = max(0.008, new_params['learning_rate'] * 0.9)
            new_params['bagging_fraction'] = min(0.6, new_params['bagging_fraction'] + 0.05)

        # Phase 5: Fine-tuning (Iterations 13+)
        else:
            # Ensemble approach with different seeds
            seeds = [42, 123, 456, 789, 999]
            new_params['seed'] = seeds[iteration % len(seeds)]
            new_params['learning_rate'] = max(0.005, new_params['learning_rate'] * 0.95)

    def get_phase_name(self, iteration):
        """Get tuning phase name"""
        if iteration <= 3:
            return "Learning Rate & Depth Optimization"
        elif iteration <= 6:
            return "Regularization Tuning"
        elif iteration <= 9:
            return "Feature Sampling Optimization"
        elif iteration <= 12:
            return "Advanced Parameter Tuning"
        else:
            return "Ensemble Fine-tuning"

    def save_progress(self):
        """Save progress to file"""
        progress = {
            'iteration': self.iteration,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'submission_history': self.submission_history
        }

        with open('pipeline_progress.json', 'w') as f:
            json.dump(progress, f, indent=2)

    def load_progress(self):
        """Load previous progress"""
        if os.path.exists('pipeline_progress.json'):
            try:
                with open('pipeline_progress.json', 'r') as f:
                    progress = json.load(f)
                self.iteration = progress.get('iteration', 0)
                self.best_score = progress.get('best_score', 0.0)

                # Use model-appropriate default parameters if not in progress
                saved_params = progress.get('best_params', {})
                if saved_params and self._is_compatible_params(saved_params):
                    self.best_params = saved_params
                else:
                    self.best_params = self._get_default_params()
                    print(f"Using fresh {self.model_type} parameters (incompatible saved params)")

                self.submission_history = progress.get('submission_history', [])
                print(f"Resumed from iteration {self.iteration}, best score: {self.best_score:.6f}")
            except:
                print("Could not load previous progress, starting fresh")

    def _is_compatible_params(self, params):
        """Check if saved parameters are compatible with current model type"""
        if self.model_type == 'xgboost':
            return 'objective' in params and params.get('objective') == 'binary:logistic'
        elif self.model_type == 'lightgbm':
            return 'objective' in params and params.get('objective') == 'binary'
        return False

    def run(self, max_iterations=10, target_score=0.805):
        """Run the automated pipeline"""
        print("="*60)
        print("AUTOMATED KAGGLE PIPELINE")
        print("="*60)
        print(f"Target score: {target_score}")
        print(f"Max iterations: {max_iterations}")

        self.load_progress()

        for i in range(max_iterations):
            self.iteration += 1
            print(f"\n{'='*20} ITERATION {self.iteration} {'='*20}")

            # Get parameters for this iteration
            if self.iteration == 1:
                params = self.best_params
            else:
                params = self.tune_parameters()

            try:
                # Train model
                model, cv_score = self.train_model(params)

                # Generate predictions
                submission = self.generate_predictions(model)

                # Submit to Kaggle
                message = f"Auto-tuning iteration {self.iteration} - CV: {cv_score:.6f}"
                kaggle_score = self.submit_to_kaggle(submission, message)

                # Save model for this iteration (regardless of score)
                model_ext = 'txt' if self.model_type == 'lightgbm' else 'xgb'
                iteration_model_name = f'model_iter_{self.iteration}_cv_{cv_score:.6f}_{self.model_type}.{model_ext}'
                saved_files = model.save_models(f'model_iter_{self.iteration}_cv_{cv_score:.6f}_{self.model_type}')
                print(f"Models saved: {saved_files}")

                # Record results
                result = {
                    'iteration': self.iteration,
                    'params': params,
                    'cv_score': cv_score,
                    'kaggle_score': kaggle_score,
                    'timestamp': datetime.now().isoformat(),
                    'model_files': saved_files,
                    'model_type': self.model_type
                }
                self.submission_history.append(result)

                # Update best if improved
                if kaggle_score and kaggle_score > self.best_score:
                    self.best_score = kaggle_score
                    self.best_params = params
                    print(f"NEW BEST SCORE: {self.best_score:.6f}")

                    # Save best model (additional copy)
                    best_model_files = model.save_models(f'best_model_score_{self.best_score:.6f}_{self.model_type}')
                    print(f"Best models updated: {best_model_files}")

                # Also save if we have a good CV score but no Kaggle score yet
                elif not kaggle_score and cv_score > 0.78:
                    backup_files = model.save_models(f'backup_model_iter_{self.iteration}_cv_{cv_score:.6f}_{self.model_type}')
                    print(f"Backup models saved: {backup_files}")

                # Save progress
                self.save_progress()

                # Check if target reached
                if kaggle_score and kaggle_score >= target_score:
                    print(f"\nTARGET SCORE REACHED! {kaggle_score:.6f} >= {target_score}")
                    break

                print(f"Iteration {self.iteration} complete - Score: {kaggle_score or 'N/A'}")

            except Exception as e:
                print(f"Iteration {self.iteration} failed: {e}")
                continue

        # Final summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Best Score: {self.best_score:.6f}")
        print(f"Best Parameters: {self.best_params}")
        print(f"Total Iterations: {len(self.submission_history)}")

        if self.submission_history:
            scores = [r['kaggle_score'] for r in self.submission_history if r['kaggle_score']]
            if scores:
                print(f"Score Improvement: {min(scores):.6f} -> {max(scores):.6f}")

def main():
    """Main pipeline runner with model selection"""
    print("="*60)
    print("AUTOMATED KAGGLE PIPELINE - MODEL SELECTION")
    print("="*60)

    # Model selection
    print("Available models:")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"{i}. {model.upper()}")

    while True:
        try:
            choice = input(f"\nSelect model (1-{len(AVAILABLE_MODELS)}) or press Enter for XGBoost: ").strip()
            if not choice:
                selected_model = DEFAULT_MODEL
                break
            choice = int(choice)
            if 1 <= choice <= len(AVAILABLE_MODELS):
                selected_model = AVAILABLE_MODELS[choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(AVAILABLE_MODELS)}")
        except ValueError:
            print("Please enter a valid number")

    print(f"\nSelected model: {selected_model.upper()}")

    pipeline = AutoKagglePipeline(model_type=selected_model)

    print("\nStarting automated Kaggle pipeline...")
    print("This will:")
    print("1. Train models with different parameters")
    print("2. Submit predictions to Kaggle")
    print("3. Get scores and tune automatically")
    print("4. Continue until target score or max iterations")

    # Configuration
    MAX_ITERATIONS = 15
    TARGET_SCORE = 0.805

    pipeline.run(max_iterations=MAX_ITERATIONS, target_score=TARGET_SCORE)

if __name__ == "__main__":
    main()