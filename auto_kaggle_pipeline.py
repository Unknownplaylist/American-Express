"""
Automated Kaggle Pipeline with Auto-Tuning
Trains, submits, gets score, and tunes parameters automatically
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
import subprocess
import time
import json
import os
import gc
from datetime import datetime
from config import *
from cached_feature_engineering import CachedFeatureEngine

class AutoKagglePipeline:
    def __init__(self, competition="amex-default-prediction"):
        self.competition = competition
        self.best_score = 0.0
        self.best_params = XGB_PARAMS.copy()
        self.submission_history = []
        self.iteration = 0
        self.feature_engine = CachedFeatureEngine()
        self.train_data = None  # Cache training data

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
        print(f"\nTraining with parameters: {params}")

        model_data = self.load_and_prepare_data()
        feature_cols = [col for col in model_data.columns if col not in ['customer_ID', 'target']]
        X = model_data[feature_cols]
        y = model_data['target']
        X = self.clean_data(X)

        # Quick 3-fold CV for speed
        kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
        oof_predictions = np.zeros(len(X))
        models = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Fold {fold + 1}/3", end=" ")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            dtrain = xgb.DMatrix(data=X_train, label=y_train)
            dval = xgb.DMatrix(data=X_val, label=y_val)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )

            val_pred = model.predict(dval)
            oof_predictions[val_idx] = val_pred
            models.append(model)

            del dtrain, dval, X_train, X_val, y_train, y_val
            gc.collect()

        cv_score = self.amex_metric(y.values, oof_predictions)
        print(f"CV Score: {cv_score:.6f}")

        # Use best model for predictions
        best_model = models[0]  # Could select based on validation score
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

        dtest = xgb.DMatrix(data=X_test)
        predictions = model.predict(dtest)

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
        """Wait for Kaggle score"""
        max_attempts = max_wait_minutes * 4  # Check every 15 seconds

        for attempt in range(max_attempts):
            try:
                cmd = ["kaggle", "competitions", "submissions", "-c", self.competition, "--csv"]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        latest = lines[1].split(',')
                        if len(latest) > 2 and latest[2]:
                            try:
                                score = float(latest[2])
                                print(f"Kaggle Score: {score:.6f}")
                                return score
                            except ValueError:
                                pass

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

        print(f"   Tuning strategy for iteration {iteration}: {self.get_phase_name(iteration)}")
        return new_params

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
                self.best_params = progress.get('best_params', XGB_PARAMS.copy())
                self.submission_history = progress.get('submission_history', [])
                print(f"Resumed from iteration {self.iteration}, best score: {self.best_score:.6f}")
            except:
                print("Could not load previous progress, starting fresh")

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

                # Record results
                result = {
                    'iteration': self.iteration,
                    'params': params,
                    'cv_score': cv_score,
                    'kaggle_score': kaggle_score,
                    'timestamp': datetime.now().isoformat()
                }
                self.submission_history.append(result)

                # Update best if improved
                if kaggle_score and kaggle_score > self.best_score:
                    self.best_score = kaggle_score
                    self.best_params = params
                    print(f"NEW BEST SCORE: {self.best_score:.6f}")

                    # Save best model
                    model.save_model(f'best_model_score_{self.best_score:.6f}.xgb')

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
    """Main pipeline runner"""
    pipeline = AutoKagglePipeline()

    print("Starting automated Kaggle pipeline...")
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