import os
import h5py
import numpy as np

# Define the input and output directories
base_folder = '/path/to/sevir_lr/data/vil/'
output_folder = '/path/to/sevir_lr/data/vil_numpy/'  # Replace with the path to save .npy files

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Loop through each .h5 file in the input folder
for i in range(2019, 2020):
    input_folder = base_folder + str(i)
    for filename in os.listdir(input_folder):
        if filename.endswith('.h5'):
            # Open the .h5 file
            with h5py.File(os.path.join(input_folder, filename), 'r') as h5_file:
                # Extract data from the specified key
                for key in h5_file.keys():
                    print(key)
                    data = h5_file[key][:]

                    # Define the .npy file path
                    npy_filename = os.path.splitext(filename)[0] + '.npy'
                    npy_path = os.path.join(output_folder + str(i), npy_filename)

                    # Save the data as a .npy file
                    np.save(npy_path, data)

            print(f"Saved {npy_filename}.")
