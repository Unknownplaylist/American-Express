"""
Reset pipeline progress for LightGBM training
"""

import os
import json
from config import LGBM_PARAMS

def reset_progress_for_lightgbm():
    """Reset pipeline progress to start fresh with LightGBM"""

    # Remove existing progress file
    if os.path.exists('pipeline_progress.json'):
        os.remove('pipeline_progress.json')
        print("Removed existing pipeline_progress.json")

    # Create fresh progress file with LightGBM parameters
    progress = {
        'iteration': 0,
        'best_score': 0.0,
        'best_params': LGBM_PARAMS.copy(),
        'submission_history': [],
        'model_type': 'lightgbm'
    }

    with open('pipeline_progress.json', 'w') as f:
        json.dump(progress, f, indent=2)

    print("Created fresh pipeline_progress.json for LightGBM")
    print(f"Starting parameters: {LGBM_PARAMS}")

if __name__ == "__main__":
    reset_progress_for_lightgbm()
    print("\nReady to start LightGBM training from iteration 1!")