import os
import random
import argparse
import numpy as np
from impact import Impact

def check_random_file_in_archive(archive_folder):
    # List all .h5 files in the folder
    h5_files = [file for file in os.listdir(archive_folder) if file.endswith('.h5')]
    
    if not h5_files:
        print("No .h5 files found in the archive folder.")
        return

    # Randomly pick one .h5 file
    random_file = random.choice(h5_files)
    random_file_path = os.path.join(archive_folder, random_file)
    
    print(f"Selected file: {random_file}")
    
    # Load the file using Impact
    try:
        f = Impact.from_archive(random_file_path)
        print(f"Number of particles: {len(f.particles)}")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Check a random .h5 file in the specified archive folder.")
    parser.add_argument("--archive_folder", type=str, help="Path to the archive folder containing .h5 files")
    
    args = parser.parse_args()
    
    # Call the function with the provided path
    check_random_file_in_archive(args.archive_folder)
