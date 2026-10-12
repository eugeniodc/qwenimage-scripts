#!/usr/bin/env python3
"""
Qwen Image 2.1 Generator for Artistic Purposes
This script demonstrates how to use the Qwen Image 2.1 model for artistic image generation
"""

import os
import sys
from pathlib import Path

def setup_comfyui_config():
    """
    Setup configuration for ComfyUI with Qwen Image 2.1
    """
    config = {
        "model_path": "ComfyUI/models/diffusion_models/qwen-image-2.1-UC-Q4_K_M.gguf",
        "text_encoder_path": "ComfyUI/models/text_encoders/qwen3vl_8b_bf16.safetensors",
        "vae_path": "ComfyUI/models/vae/qwen_image_2.1_vae_bf16.safetensors",
        "output_dir": "output_art",
        "prompt": "A beautiful landscape painting in the style of Van Gogh",
        "negative_prompt": "blurry, low quality, distorted",
        "steps": 20,
        "cfg": 7.0
    }
    return config

def create_comfyui_workflow():
    """
    Create a basic ComfyUI workflow configuration for Qwen Image 2.1
    """
    workflow = {
        "nodes": [
            {
                "id": 1,
                "type": "Unet Loader (GGUF)",
                "inputs": {
                    "model": "qwen-image-2.1-UC-Q4_K_M.gguf"
                }
            },
            {
                "id": 2,
                "type": "CLIPLoader",
                "inputs": {
                    "clip_name": "qwen3vl_8b_bf16.safetensors",
                    "type": "qwen_image"
                }
            },
            {
                "id": 3,
                "type": "VAELoader",
                "inputs": {
                    "vae_name": "qwen_image_2.1_vae_bf16.safetensors"
                }
            },
            {
                "id": 4,
                "type": "KSampler",
                "inputs": {
                    "model": 1,
                    "positive": "positive_prompt",
                    "negative": "negative_prompt",
                    "latent_image": "latent",
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal"
                }
            }
        ],
        "links": [
            [1, 0, 4, 0],
            [2, 0, 4, 1],
            [3, 0, 4, 2]
        ]
    }
    return workflow

def generate_artistic_image(prompt, output_dir="output_art", steps=20):
    """
    Generate an artistic image using Qwen Image 2.1 model
    
    Args:
        prompt (str): Description of the artistic image to generate
        output_dir (str): Directory to save generated images
        steps (int): Number of sampling steps for generation
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Generating artistic image with prompt: '{prompt}'")
    print(f"Using {steps} sampling steps")
    print(f"Output will be saved to: {output_dir}")
    
    # In a real implementation, this would:
    # 1. Load the Qwen Image 2.1 model
    # 2. Process the prompt through the text encoder
    # 3. Generate image using the diffusion model
    # 4. Save the result to the output directory
    
    print("Setup complete. To run the generation:")
    print("1. Install ComfyUI")
    print("2. Place model files in correct directories")
    print("3. Use the ComfyUI interface to run the workflow")
    
    return f"Generated workflow configuration for: {prompt}"

def main():
    """
    Main function to demonstrate Qwen Image 2.1 usage
    """
    print("Qwen Image 2.1 Artistic Generator")
    print("=" * 40)
    
    # Setup configuration
    config = setup_comfyui_config()
    
    # Create workflow
    workflow = create_comfyui_workflow()
    
    # Generate artistic image
    result = generate_artistic_image(
        prompt=config["prompt"],
        output_dir=config["output_dir"],
        steps=config["steps"]
    )
    
    print("\n" + "=" * 40)
    print("Configuration Summary:")
    print(f"Model: {config['model_path']}")
    print(f"Text Encoder: {config['text_encoder_path']}")
    print(f"VAE: {config['vae_path']}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"Prompt: {config['prompt']}")
    print(f"Steps: {config['steps']}")
    
    print(f"\n{result}")

if __name__ == "__main__":
    main()