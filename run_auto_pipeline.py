"""
Auto Pipeline Runner
Easy interface for the automated Kaggle pipeline
"""

import os
import sys
from auto_kaggle_pipeline import AutoKagglePipeline

def check_requirements():
    """Check if all requirements are met"""
    print("Checking requirements...")

    # Check data files
    required_files = ['data/train_data.csv', 'data/test_data.csv', 'data/train_labels.csv']
    missing_files = []

    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("Missing required data files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False

    # Check Kaggle API
    try:
        import subprocess
        result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Kaggle CLI not available. Install with: pip install kaggle")
            return False
        print(f"Kaggle CLI: {result.stdout.strip()}")
    except:
        print("Kaggle CLI not found. Install with: pip install kaggle")
        return False

    print("All requirements met!")
    return True

def main():
    """Main interface"""
    print("="*60)
    print("AUTOMATED KAGGLE PIPELINE")
    print("="*60)

    if not check_requirements():
        print("\nPlease fix the requirements above before running.")
        return

    print("\nPipeline Configuration:")
    print("- Competition: amex-default-prediction")
    print("- Max iterations: 15")
    print("- Target score: 0.805")
    print("- Auto-tuning: Enabled")

    choice = input("\nDo you want to start the pipeline? (y/n): ").strip().lower()

    if choice != 'y':
        print("Pipeline cancelled.")
        return

    print("\nStarting automated pipeline...")
    print("This will run continuously until target score is reached.")
    print("Progress will be saved automatically.")
    print("You can stop anytime with Ctrl+C and resume later.")

    try:
        pipeline = AutoKagglePipeline()
        pipeline.run(max_iterations=15, target_score=0.805)
    except KeyboardInterrupt:
        print("\n\nPipeline stopped by user.")
        print("Progress has been saved. Run again to resume.")
    except Exception as e:
        print(f"\nPipeline error: {e}")
        print("Check the error above and try again.")

if __name__ == "__main__":
    main()