# NOTE: mlx-lm does NOT support models quantized with bitsandbytes. Use only original (non-quantized) models.

# Replace the placeholders below with your actual Hugging Face credentials and repo info.
from huggingface_hub import snapshot_download, HfApi, login
import mlx_lm
import os

# Login to Hugging Face (use your token or huggingface-cli login)
login(token="<YOUR_HF_TOKEN>")

# Download the model from Hugging Face
model_id = "<your-model-id>"  # e.g., "google/gemma-3n-E4B-it"
local_dir = "./model"
if not os.path.exists(local_dir) or not os.listdir(local_dir):
    print(f"Downloading model {model_id} to {local_dir}...")
    snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
else:
    print(f"Model already exists in {local_dir}, skipping download.")

# Quantize the model using MLX
quantized_dir = "./model-quantized"
if os.path.exists(quantized_dir):
    import shutil
    print(f"Removing existing {quantized_dir} directory...")
    shutil.rmtree(quantized_dir)

from mlx_lm import convert
try:
    print("Starting quantization process...")
    convert(
        local_dir,
        mlx_path=quantized_dir,
        quantize=True,
        q_bits=4,
        q_group_size=128,
        upload_repo="<your-username>/<your-quantized-repo>"  # e.g., "username/model-quantized"
    )
    print("Quantization completed successfully!")
except Exception as e:
    print(f"Quantization failed: {e}")
    print("Try running with smaller batch sizes or on a machine with more memory.")

# Push quantized model to Hugging Face Hub
if os.path.exists(quantized_dir) and os.listdir(quantized_dir):
    print(f"Found quantized model in {quantized_dir}, uploading to Hugging Face...")
    api = HfApi()
    api.create_repo(repo_id="<your-username>/<your-quantized-repo>", exist_ok=True, token="<YOUR_HF_TOKEN>")
    api.upload_folder(
        repo_id="<your-username>/<your-quantized-repo>",
        folder_path=quantized_dir,
        token="<YOUR_HF_TOKEN>"
    )
    print("Upload completed!")
else:
    print("No quantized model found to upload.")