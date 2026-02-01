import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

# Used to convert raw radar files into separate npy files and downsampling these data
if __name__ == "__main__":
    examples_root = '/path/to/meteonet/data/nw/reflectivity_old_product/NW_reflectivity_old_product_2018/'
    out_root = '/path/to/meteonet/data/nw/reflectivity_npy/'

    with tqdm(total=len(os.listdir(examples_root))) as pbar:
        for example_name in os.listdir(examples_root):
            reflectivity_data = os.path.join(examples_root, example_name)

            d = np.load(reflectivity_data, allow_pickle=True)
            # Read radar reflectivity data
            data = d['data']
            # Read the datetime corresponding to each radar data
            dates = d['dates']

            # Save the data for each time step into an npy file.
            for i in range(len(data)):
                dt = dates[i]
                file_name = dt.strftime("%Y%m%d_%H%M")

                # Read the corresponding radar data
                file_npy = data[i]
                file_npy = file_npy.astype(np.uint8)
                file_npy[file_npy == 255] = 0

                # Downsampling
                file_tensor = torch.tensor(file_npy)
                video_4d = file_tensor.unsqueeze(0).unsqueeze(0)
                resized_4d = F.interpolate(video_4d, size=(128, 128), mode='bicubic')
                resized = resized_4d.squeeze(0).squeeze(0)  # removes the singular channel dim

                file_npy = resized.numpy()

                os.makedirs(out_root, exist_ok=True)
                base = out_root + file_name
                # save as npy file
                np.save(os.path.join(out_root, f"{base}.npy"), file_npy)

            pbar.update(1)