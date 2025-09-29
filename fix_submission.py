"""
Submission Fixer Tool
Converts submission with wrong customer_ID format to correct format
"""

import pandas as pd
import os
import sys

def load_original_customer_ids():
    """Load original customer IDs from test data"""
    print("Loading original test customer IDs...")

    try:
        # Load test data to get original customer IDs
        test_data = pd.read_csv('data/test_data.csv', usecols=['customer_ID'], engine='c', low_memory=False)
        print(f"Found {len(test_data)} original customer IDs")
        return test_data['customer_ID'].tolist()
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

def create_id_mapping():
    """Create mapping from converted IDs back to original IDs"""
    print("Creating customer_ID mapping...")

    original_ids = load_original_customer_ids()
    if original_ids is None:
        return None

    # Apply same conversion that was used in pipeline
    def safe_hex_convert(x):
        try:
            hex_part = str(x)[-16:]
            return int(hex_part, 16)
        except:
            return hash(str(x)) % (2**63)

    # Create mapping: converted_id -> original_id
    mapping = {}
    for original_id in original_ids:
        converted_id = safe_hex_convert(original_id)
        # Convert back to hex format (like in submission)
        hex_id = hex(converted_id)[2:].upper().zfill(16)
        mapping[hex_id] = original_id

    print(f"Created mapping for {len(mapping)} customer IDs")
    return mapping

def fix_submission_file(submission_file, output_file=None):
    """Fix submission file with correct customer IDs"""
    print(f"Fixing submission file: {submission_file}")

    # Load submission
    if not os.path.exists(submission_file):
        print(f"Error: Submission file not found: {submission_file}")
        return False

    submission = pd.read_csv(submission_file)
    print(f"Loaded submission with {len(submission)} rows")
    print(f"Columns: {list(submission.columns)}")

    # Create ID mapping
    id_mapping = create_id_mapping()
    if id_mapping is None:
        print("Failed to create ID mapping")
        return False

    # Fix customer IDs
    print("Applying customer_ID fixes...")
    fixed_count = 0
    not_found_count = 0

    def fix_customer_id(customer_id):
        nonlocal fixed_count, not_found_count
        if customer_id in id_mapping:
            fixed_count += 1
            return id_mapping[customer_id]
        else:
            not_found_count += 1
            return customer_id  # Keep original if mapping not found

    submission['customer_ID'] = submission['customer_ID'].apply(fix_customer_id)

    print(f"Fixed {fixed_count} customer IDs")
    if not_found_count > 0:
        print(f"Warning: {not_found_count} customer IDs not found in mapping")

    # Save fixed submission
    if output_file is None:
        # Generate output filename
        base_name = os.path.splitext(submission_file)[0]
        output_file = f"{base_name}_fixed.csv"

    submission.to_csv(output_file, index=False)
    print(f"Fixed submission saved: {output_file}")

    # Show sample of fixes
    print("\nSample of original vs fixed customer IDs:")
    sample_mapping = list(id_mapping.items())[:5]
    for hex_id, original_id in sample_mapping:
        print(f"  {hex_id} -> {original_id[:50]}...")

    return output_file

def find_submission_files():
    """Find recent submission files"""
    submission_files = []

    # Look for submission files in current directory
    for file in os.listdir('.'):
        if file.startswith('submission_') and file.endswith('.csv'):
            submission_files.append(file)

    # Sort by modification time (newest first)
    submission_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    return submission_files

def main():
    """Main fixer interface"""
    print("SUBMISSION FIXER TOOL")
    print("="*50)

    # Check if file specified as argument
    if len(sys.argv) > 1:
        submission_file = sys.argv[1]
        if not os.path.exists(submission_file):
            print(f"Error: File not found: {submission_file}")
            return
    else:
        # Find submission files automatically
        submission_files = find_submission_files()

        if not submission_files:
            print("No submission files found in current directory")
            print("Looking for files matching pattern: submission_*.csv")
            return

        print("Found submission files:")
        for i, file in enumerate(submission_files):
            mod_time = os.path.getmtime(file)
            print(f"  {i+1}. {file} (modified: {pd.Timestamp.fromtimestamp(mod_time)})")

        # Use most recent by default, or let user choose
        if len(submission_files) == 1:
            submission_file = submission_files[0]
            print(f"\nUsing: {submission_file}")
        else:
            try:
                choice = input(f"\nSelect file (1-{len(submission_files)}) or press Enter for newest: ").strip()
                if choice == "":
                    submission_file = submission_files[0]
                else:
                    idx = int(choice) - 1
                    submission_file = submission_files[idx]
                print(f"Selected: {submission_file}")
            except:
                print("Invalid selection, using newest file")
                submission_file = submission_files[0]

    # Fix the submission
    print(f"\nFixing submission: {submission_file}")
    fixed_file = fix_submission_file(submission_file)

    if fixed_file:
        print(f"\n✅ SUCCESS!")
        print(f"Fixed submission: {fixed_file}")
        print(f"You can now submit: {fixed_file}")
    else:
        print(f"\n❌ FAILED to fix submission")

if __name__ == "__main__":
    main()