# mlx-quant.py

A simple script to download, quantize, and upload Hugging Face models using MLX.

## Features
- Downloads a model from Hugging Face Hub
- Quantizes the model using MLX
- Uploads the quantized model back to the Hugging Face Hub

## Usage
1. Replace the placeholders in the script with your Hugging Face token, model ID, and repo info.
2. Run the script:
   ```bash
   python mlx-quant.py
   ```


## Notes
- mlx-lm does NOT support models quantized with bitsandbytes. Use only original (non-quantized) models.
- **Important:** Be sure the model you select is supported by `mlx-lm`. Not all Hugging Face models are compatible. Check the [mlx-lm documentation](https://github.com/ml-explore/mlx-lm#supported-models) for a list of supported models and architectures.

