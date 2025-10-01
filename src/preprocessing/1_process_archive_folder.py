import glob
import numpy as np
import pandas as pd
import argparse
import os
from impact import Impact  # Assuming this is the correct import for the Impact class
from tqdm import tqdm  # Add tqdm for progress bar

def calc_N(file):
    I = Impact()
    no_exceptions = True
    try:
        I.load_archive(file)
    except Exception as e:
        print(f"Error loading archive for file {file}: {e}")
        no_exceptions = False
        return 0, 0, no_exceptions

    try:
        N = len(I['particles']['initial_particles']['x']) if 'initial_particles' in I['particles'] else 0
        N241 = len(I['particles']['PR10241']['x']) if 'PR10241' in I['particles'] else 0
    except Exception as e:
        print(f"Error processing particle data for file {file}: {e}")
        no_exceptions = False
        N = 0
        N241 = 0

    return N, N241, no_exceptions

def process_archive_folder(data_dir, output_file):
    files = glob.glob(os.path.join(data_dir, '*'))
    files = [f for f in files if os.path.isfile(f)]
    
    valid_files_n241_match = []
    total_files = len(files)

    # Use tqdm for progress bar
    for idx, file in enumerate(tqdm(files, desc="Processing files"), start=1):
        N, N241, no_exceptions = calc_N(file)
        if no_exceptions and N241 == N:
            valid_files_n241_match.append(file)
        # tqdm handles progress display

    print(f"Number of valid files with N241 == N: {len(valid_files_n241_match)}")

    df_valid_n241_match = pd.DataFrame(valid_files_n241_match, columns=['filename'])
    df_valid_n241_match.to_csv(output_file, index=False)
    print(f"Data catalog saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process archive folder and filter by N241.")
    parser.add_argument('--archive_dir', type=str, help="Path to the archive folder")
    parser.add_argument('--output_file', type=str, help="Output file for filtered data catalog")

    args = parser.parse_args()

    process_archive_folder(args.archive_dir, args.output_file)
    
