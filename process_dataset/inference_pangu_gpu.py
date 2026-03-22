import os
import numpy as np
import onnx
import onnxruntime as ort
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Modified from: https://github.com/198808xc/Pangu-Weather/blob/main/inference_gpu.py
# -----------------------------------------------------------------------------

# Set GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Directories
input_root = 'pangu_model/input_data'
output_root = 'pangu_model/output_data'
model_path = 'pangu_model/pangu_weather_1.onnx'

# ONNX session options
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
options.intra_op_num_threads = 4
cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested'}

# Initialize ONNX runtime session
ort_session = ort.InferenceSession(model_path, sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

# Iteration settings
step_hours = 1  # each step predicts 3 hours
max_steps = 24   # total prediction steps

# Get all time folders
time_folders = sorted([f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))])

for folder in tqdm(time_folders):
    if folder<'2020-11-09-00-00':
        continue
    input_dir = os.path.join(input_root, folder)
    output_dir = os.path.join(output_root, folder)
    os.makedirs(output_dir, exist_ok=True)

    # Load initial inputs
    prev_upper = np.load(os.path.join(input_dir, 'input_upper.npy')).astype(np.float32)
    prev_surface = np.load(os.path.join(input_dir, 'input_surface.npy')).astype(np.float32)

    # If the model resolution is 3, downsample or slice inputs if needed
    # Example (adjust according to your input shape):
    # prev_upper = prev_upper[:, ::3, ::3, :]
    # prev_surface = prev_surface[:, ::3, ::3, :]

    for step in range(1, max_steps + 1):
        # Run inference
        output_upper, output_surface = ort_session.run(None, {'input': prev_upper, 'input_surface': prev_surface})

        # Save every step
        current_hour = step * step_hours
        #np.save(os.path.join(output_dir, f'output_upper_{current_hour:03d}.npy'), output_upper)   # Inference upper-air variables
        np.save(os.path.join(output_dir, f'output_surface_{current_hour:02d}.npy'), output_surface)   # Inference surface variables

        # Prepare input for next step
        prev_upper = output_upper
        prev_surface = output_surface
