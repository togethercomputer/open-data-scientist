import os
import sys
from huggingface_hub import hf_hub_download

CONTEXT_FILENAMES = [
    "data/context/acquirer_countries.csv",
    "data/context/payments-readme.md",
    "data/context/payments.csv",
    "data/context/merchant_category_codes.csv",
    "data/context/fees.json",
    "data/context/merchant_data.json",
    "data/context/manual.md",
]
DATA_DIR = "/app/downloaded_data"

print(f"Starting data download to {DATA_DIR}")
print(f"Current working directory: {os.getcwd()}")

# Create the data directory if it doesn't exist
# Also create parent 'data' and 'context' directories if they are part of the path
# by iterating through unique directory paths derived from CONTEXT_FILENAMES
unique_dirs = set()
for f_name in CONTEXT_FILENAMES:
    # We want to create the directory structure like /app/downloaded_data/data/context/
    # So, we take the dirname of each filename and prepend DATA_DIR
    # os.path.dirname("data/context/acquirer_countries.csv") -> "data/context"
    dir_path = os.path.join(DATA_DIR, os.path.dirname(f_name))
    unique_dirs.add(dir_path)

print(f"Creating directories: {unique_dirs}")
for d_path in unique_dirs:
    os.makedirs(d_path, exist_ok=True)
    print(f"Created directory: {d_path}")

# Download each file with error handling
failed_downloads = []
successful_downloads = []

for filename in CONTEXT_FILENAMES:
    print(f"\nDownloading {filename}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id="adyen/DABstep",
            repo_type="dataset",
            filename=filename,  # This is the path *within the repository*
            local_dir=DATA_DIR,  # This is the base local directory
            force_download=True,
        )
        print(f"Successfully downloaded to: {downloaded_path}")
        successful_downloads.append(filename)
    except Exception as e:
        print(f"ERROR downloading {filename}: {e}")
        failed_downloads.append((filename, str(e)))

print("\n=== DOWNLOAD SUMMARY ===")
print(f"Successful downloads: {len(successful_downloads)}")
print(f"Failed downloads: {len(failed_downloads)}")

if failed_downloads:
    print("\nFAILED DOWNLOADS:")
    for filename, error in failed_downloads:
        print(f"  - {filename}: {error}")

print("\nChecking if files exist:")
# The files will be at DATA_DIR + original filename structure
downloaded_files_paths = [os.path.join(DATA_DIR, fname) for fname in CONTEXT_FILENAMES]

existing_files = []
missing_files = []

for file_path in downloaded_files_paths:
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f"✓ {file_path} exists ({file_size} bytes)")
        existing_files.append(file_path)
    else:
        print(f"✗ {file_path} does not exist")
        missing_files.append(file_path)

print("\n=== FINAL STATUS ===")
print(f"Files found: {len(existing_files)}")
print(f"Files missing: {len(missing_files)}")

if missing_files:
    print("\nMISSING FILES:")
    for missing in missing_files:
        print(f"  - {missing}")
    print("\nERROR: Not all files were downloaded successfully!")
    sys.exit(1)

print("\n✓ Data download script finished successfully!")
