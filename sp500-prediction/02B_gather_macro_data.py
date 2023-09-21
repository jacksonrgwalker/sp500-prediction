import pandas as pd
from pathlib import Path
import os

# Directory to the data files
directory_path = Path.cwd() / "Macro_Data"

files_and_dirs = os.listdir(directory_path)

# Filter only the files from the list
files = [f for f in files_and_dirs if os.path.isfile(os.path.join(directory_path, f))]

# Load in the data in to Dataframes
df_list = []

for file in files:

    dummy_df = pd.read_csv(directory_path / file)
    dummy_df[dummy_df.columns[0]] = pd.to_datetime(dummy_df[dummy_df.columns[0]])
    dummy_df.set_index(dummy_df.columns[0], inplace=True)
    dummy_df.index.name = 'date'

    df_list.append(dummy_df)

counter = 1
for df in df_list:

    if counter == 1:
        macro_df = df.copy()
        counter += 1
    else:
        macro_df = macro_df.join(df, on='date', how='inner')

# Save the table
macro_save_path = Path("data/macro.parquet")
macro_df.to_parquet(macro_save_path)

