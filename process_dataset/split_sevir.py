import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
radar_folder = Path("/your/path/to/sevir_lr/data/vil_single/random/")
# radar_folder = Path("/your/path/to/sevir_lr/data/vil_single/storm/")
window_size = 13
stride = 6

def sliding_windows(sequence, window_size, stride):
    """
    Given a NumPy array of shape [H, W, T], returns overlapping windows
    of shape [H, W, window_size] using the specified stride along the T axis.
    """
    H, W, T = sequence.shape
    num_windows = (T - window_size) // stride + 1
    return [
        sequence[:, :, (i * stride):(i * stride + window_size)].copy()
        for i in range(num_windows)
    ]

npy_count = len(list(radar_folder.glob("*.npy")))

with tqdm(total=npy_count) as pbar:
    # Process each .npy file in the directory
    for npy_file in sorted(radar_folder.glob("*.npy")):
        try:
            arr = np.load(npy_file)
        except Exception as e:
            print(f"Failed to load {npy_file.name}: {e}")
            continue

        if arr.ndim != 3 or arr.shape[2] < window_size:
            print(f"Skipping {npy_file.name}: invalid shape {arr.shape}")
            continue

        # Use sliding window to split the original array into 3 npy files
        windows = sliding_windows(arr, window_size, stride)

        base_name = npy_file.stem
        for idx, win in enumerate(windows):
            out_path = f"/your/path/to/sevir_lr/data/vil_split/random/{base_name}_{idx}.npy"
            # out_path = f"/your/path/to/sevir_lr/data/vil_split/storm/{base_name}_{idx}.npy"
            np.save(out_path, win)

        pbar.update(1)