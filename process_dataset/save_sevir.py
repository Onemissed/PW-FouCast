import numpy as np
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm

# Load one array data
data = np.load('/your/path/to/sevir_lr/data/vil_numpy/2019/SEVIR_VIL_STORMEVENTS_2019_0701_1231.npy')
# Choose to extract precipitation events in random or storm
output_folder = '/your/path/to/sevir_lr/data/vil_single/random/'  # Folder to save individual sequences
# output_folder = '/your/path/to/sevir_lr/data/vil_single/storm/'

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load the sevir CSV catalog
catalog = pd.read_csv('CATALOG.csv')

# print("data.shape[0]: ", data.shape[0])
print("len(catalog): ", len(catalog))

# Check the catalog and data dimensions match
assert data.shape[0] == len(catalog), "Mismatch between data sequences and catalog entries."

# Loop over each sequence and save with the appropriate filename
with tqdm(total=data.shape[0]) as pbar:
    for index, row in catalog.iterrows():
        # Get the sequence based on the index
        sequence = data[index]

        # Convert the timestamp to the required format
        timestamp = datetime.strptime(row['time_utc'], '%Y/%m/%d %H:%M')
        formatted_time = timestamp.strftime('%Y%m%d%H%M')

        # Construct the filename
        filename = f"random_{formatted_time}_{index}.npy"
        filepath = os.path.join(output_folder, filename)

        # Save the sequence
        np.save(filepath, sequence)
        pbar.update(1)