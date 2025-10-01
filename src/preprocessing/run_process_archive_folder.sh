#!/bin/bash
#SBATCH --job-name=process_archive
#SBATCH --output=logs/process_archive_%j.out
#SBATCH --error=logs/process_archive_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --account=ad:default
#SBATCH --partition=milano
#SBATCH --mail-user=tiffan@slac.stanford.edu
#SBATCH --mail-type=END,FAIL

# Activate your environment if needed
# source /path/to/your/env/bin/activate

# Go to your working directory
cd /sdf/home/t/tiffan/repo/accelerator-surrogate/src/preprocessing_new/

# Run your script
python 1_process_archive_folder.py --archive_dir /sdf/data/ad/ard/u/tiffan/Archive_6 --output_file Archive6_n241_match.csv
