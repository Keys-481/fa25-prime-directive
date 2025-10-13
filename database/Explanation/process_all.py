#!/usr/bin/env python3
import os
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Base folder where all rosbag folders reside
base_folder = os.path.expanduser("~/rosbag_test")

# Names of folders to exclude
exclude_folders = {"venv", "CSV_JSON_FILES"}

# Path to the main conversion script
conversion_script = os.path.join(base_folder, "rosbag_to_csv_json.py")

if not os.path.exists(conversion_script):
    print(f"Error: Conversion script not found at {conversion_script}")
    exit(1)

# Function to process a single folder
def process_folder(folder_name):
    folder_path = os.path.join(base_folder, folder_name)
    try:
        subprocess.run(["python3", conversion_script, folder_path], check=True)
        return (folder_name, True, "")
    except subprocess.CalledProcessError as e:
        return (folder_name, False, str(e))
    except Exception as e:
        return (folder_name, False, str(e))

# Gather all folders to process
folders_to_process = [
    f for f in os.listdir(base_folder)
    if os.path.isdir(os.path.join(base_folder, f)) and f not in exclude_folders
]

if not folders_to_process:
    print("No rosbag folders found to process.")
    exit(0)

print(f"Automation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
success_folders = []
failed_folders = []

# Process folders in parallel
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = {executor.submit(process_folder, folder): folder for folder in folders_to_process}
    for future in as_completed(futures):
        folder, success, message = future.result()
        if success:
            success_folders.append(folder)
            print(f"Folder '{folder}' processed successfully.")
        else:
            failed_folders.append((folder, message))
            print(f"Error processing folder '{folder}': {message}")

# Summary
print("\nAutomation completed.")
print(f"Total folders processed: {len(success_folders) + len(failed_folders)}")
print(f"Successful folders: {len(success_folders)}")
for f in success_folders:
    print(f"  - {f}")
print(f"Failed folders: {len(failed_folders)}")
for f, msg in failed_folders:
    print(f"  - {f} (Error: {msg})")

print(f"\nFinished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")