"""
Quick submission script for Kaggle competitions
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime
import time


def submit_to_kaggle(file_path, competition="amex-default-prediction", message="", wait_for_score=True):
    """
    Quick submit function with score retrieval
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    try:
        print(f"🚀 Submitting {file_path} to {competition}...")

        # Construct submission command
        cmd = [
            "kaggle", "competitions", "submit",
            "-c", competition,
            "-f", file_path,
            "-m", message or f"Submission {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]

        # Execute submission
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Submission failed:")
            print(result.stderr)
            return None

        print(f"✅ Submission successful!")
        print(result.stdout)

        if wait_for_score:
            print("⏳ Waiting for score...")
            score = wait_for_kaggle_score(competition)
            if score:
                print(f"📊 Your score: {score}")
                return score
            else:
                print("⚠️  Score not available yet. Check Kaggle manually.")

        return "submitted"

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def wait_for_kaggle_score(competition, max_wait_minutes=5):
    """
    Wait for Kaggle score with timeout
    """
    max_attempts = max_wait_minutes * 4  # Check every 15 seconds

    for attempt in range(max_attempts):
        try:
            # Get latest submissions
            cmd = ["kaggle", "competitions", "submissions", "-c", competition, "--csv"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    # Parse latest submission
                    latest = lines[1].split(',')
                    if len(latest) > 2 and latest[2]:
                        try:
                            score = float(latest[2])
                            return score
                        except ValueError:
                            pass

            # Wait before next check
            if attempt < max_attempts - 1:
                print(f"⏳ Checking score... ({attempt + 1}/{max_attempts})")
                time.sleep(15)

        except Exception as e:
            print(f"Error checking score: {e}")
            break

    return None


def get_latest_submissions(competition="amex-default-prediction", n=5):
    """
    Get latest submission scores
    """
    try:
        cmd = ["kaggle", "competitions", "submissions", "-c", competition]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print(f"📊 Latest {n} submissions:")
            print("="*60)
            for i, line in enumerate(lines[:n+1]):  # +1 for header
                print(line)
                if i >= n:
                    break
        else:
            print(f"❌ Failed to get submissions: {result.stderr}")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Quick Kaggle submission tool")
    parser.add_argument("file", nargs="?", help="Submission file path")
    parser.add_argument("-m", "--message", help="Submission message")
    parser.add_argument("-c", "--competition", default="amex-default-prediction", help="Competition name")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for score")
    parser.add_argument("--status", action="store_true", help="Show recent submissions")

    args = parser.parse_args()

    if args.status:
        get_latest_submissions(args.competition)
        return

    if not args.file:
        print("📁 Available submission files:")
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'submission' in f.lower()]
        if csv_files:
            for i, f in enumerate(csv_files, 1):
                print(f"  {i}. {f}")

            choice = input(f"\nSelect file (1-{len(csv_files)}) or enter custom path: ").strip()

            try:
                if choice.isdigit() and 1 <= int(choice) <= len(csv_files):
                    args.file = csv_files[int(choice) - 1]
                else:
                    args.file = choice
            except:
                print("❌ Invalid selection")
                return
        else:
            print("❌ No submission files found")
            return

    if not args.message:
        args.message = input("Enter submission message (optional): ").strip()

    # Submit file
    score = submit_to_kaggle(
        args.file,
        args.competition,
        args.message,
        wait_for_score=not args.no_wait
    )

    if score:
        print(f"\n🎯 Final Result: {score}")


if __name__ == "__main__":
    main()