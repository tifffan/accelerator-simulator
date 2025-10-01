import pandas as pd

# Replace 'catalog.csv' with your actual CSV filename
csv_file = '/sdf/home/t/tiffan/repo/accelerator-surrogate/src/preprocessing/electrons_vary_distributions_vary_settings_filtered_total_charge_51_catalog_all_sdf_cleaned.csv'

# Read the CSV file
df = pd.read_csv(csv_file)

# Drop the filepath column to ignore it in the duplicate check
df_no_filepath = df.drop(columns=['filepath'])
# df_no_filepath = df

# Identify duplicate rows (all columns except filepath)
duplicates = df_no_filepath[df_no_filepath.duplicated(keep=False)]

if not duplicates.empty:
    print("Found duplicate rows (ignoring 'filepath'):")
    print(duplicates)
else:
    print("No duplicate rows found (ignoring 'filepath').")
