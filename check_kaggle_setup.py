"""
Kaggle Setup Checker
Verify that Kaggle API is properly configured
"""

import os
import subprocess
import json

def check_kaggle_installation():
    """Check if Kaggle CLI is installed"""
    print("1. Checking Kaggle CLI installation...")
    try:
        result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✓ Kaggle CLI installed: {result.stdout.strip()}")
            return True
        else:
            print(f"   ✗ Kaggle CLI error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("   ✗ Kaggle CLI not found")
        print("   Install with: pip install kaggle")
        return False

def check_api_credentials():
    """Check if Kaggle API credentials exist"""
    print("\n2. Checking Kaggle API credentials...")

    # Check common credential locations
    possible_paths = [
        os.path.expanduser("~/.kaggle/kaggle.json"),
        os.path.join(os.environ.get('USERPROFILE', ''), '.kaggle', 'kaggle.json'),
        "kaggle.json"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"   ✓ Found credentials at: {path}")

            # Validate JSON format
            try:
                with open(path, 'r') as f:
                    creds = json.load(f)
                if 'username' in creds and 'key' in creds:
                    print(f"   ✓ Valid format - Username: {creds['username']}")
                    return True, path
                else:
                    print("   ✗ Invalid format - missing username or key")
                    return False, path
            except json.JSONDecodeError:
                print("   ✗ Invalid JSON format")
                return False, path

    print("   ✗ No kaggle.json found")
    return False, None

def test_api_connection():
    """Test if API connection works"""
    print("\n3. Testing API connection...")
    try:
        result = subprocess.run(['kaggle', 'competitions', 'list'],
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("   ✓ API connection successful")
            # Show first few competitions
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                print("   Sample competitions:")
                for line in lines[1:4]:  # Show first 3 competitions
                    print(f"     {line}")
            return True
        else:
            print(f"   ✗ API connection failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("   ✗ API connection timeout")
        return False
    except Exception as e:
        print(f"   ✗ API connection error: {e}")
        return False

def check_competition_access():
    """Check access to AmEx competition"""
    print("\n4. Checking AmEx competition access...")
    try:
        result = subprocess.run(['kaggle', 'competitions', 'submissions',
                               '-c', 'amex-default-prediction'],
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("   ✓ Can access AmEx competition")

            # Show submission history if any
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                print(f"   Found {len(lines)-1} previous submissions")
            else:
                print("   No previous submissions")
            return True
        else:
            error = result.stderr.lower()
            if "not found" in error or "404" in error:
                print("   ✗ Competition not found or not joined")
                print("   Please join the competition at:")
                print("   https://www.kaggle.com/competitions/amex-default-prediction")
            elif "forbidden" in error or "401" in error:
                print("   ✗ Access denied - check your credentials")
            else:
                print(f"   ✗ Access error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("   ✗ Competition access timeout")
        return False
    except Exception as e:
        print(f"   ✗ Competition access error: {e}")
        return False

def provide_setup_instructions():
    """Provide setup instructions"""
    print("\n" + "="*60)
    print("KAGGLE SETUP INSTRUCTIONS")
    print("="*60)

    print("\n1. Install Kaggle CLI:")
    print("   pip install kaggle")

    print("\n2. Get your API credentials:")
    print("   a) Go to https://www.kaggle.com/account")
    print("   b) Scroll to 'API' section")
    print("   c) Click 'Create New API Token'")
    print("   d) Download kaggle.json file")

    print("\n3. Place credentials file:")
    kaggle_dir = os.path.expanduser("~/.kaggle")
    print(f"   a) Create directory: {kaggle_dir}")
    print(f"   b) Place kaggle.json in: {kaggle_dir}")
    print("   c) On Windows: C:\\Users\\YourName\\.kaggle\\kaggle.json")

    print("\n4. Join the competition:")
    print("   a) Go to: https://www.kaggle.com/competitions/amex-default-prediction")
    print("   b) Click 'Join Competition'")
    print("   c) Accept rules and terms")

    print("\n5. Test your setup:")
    print("   python check_kaggle_setup.py")

def main():
    """Main setup checker"""
    print("KAGGLE SETUP CHECKER")
    print("="*50)

    all_good = True

    # Run all checks
    if not check_kaggle_installation():
        all_good = False

    creds_ok, creds_path = check_api_credentials()
    if not creds_ok:
        all_good = False

    if creds_ok and not test_api_connection():
        all_good = False

    if creds_ok and not check_competition_access():
        all_good = False

    # Summary
    print("\n" + "="*50)
    if all_good:
        print("✓ KAGGLE SETUP COMPLETE!")
        print("You're ready to run the automated pipeline!")
        print("Run: python run_auto_pipeline.py")
    else:
        print("✗ SETUP INCOMPLETE")
        print("Please fix the issues above.")

        choice = input("\nShow setup instructions? (y/n): ").strip().lower()
        if choice == 'y':
            provide_setup_instructions()

if __name__ == "__main__":
    main()