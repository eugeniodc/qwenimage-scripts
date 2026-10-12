#!/usr/bin/env python3
"""
Qwen Image 2.1 Setup and Usage Script
This script helps set up and demonstrates usage of the Qwen Image 2.1 model for artistic purposes
"""

import os
import sys
from pathlib import Path

def print_welcome():
    """Print welcome message and model information"""
    print("=" * 60)
    print("Qwen Image 2.1 Uncensored GGUF Setup")
    print("=" * 60)
    print("Model: Qwen-Image-2.1-Uncensored-GGUF")
    print("Purpose: Artistic image generation with local processing")
    print("License: qwen-research")
    print("=" * 60)

def setup_directories():
    """Setup required directories for ComfyUI"""
    directories = [
        "ComfyUI/models/diffusion_models",
        "ComfyUI/models/text_encoders",
        "ComfyUI/models/vae",
        "output_art"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def show_model_files():
    """Show the available model files"""
    print("\nAvailable Model Files:")
    print("-" * 40)
    files = {
        "Diffusion Models (GGUF)": [
            "qwen-image-2.1-UC-BF16.gguf",
            "qwen-image-2.1-UC-fp8.safetensors", 
            "qwen-image-2.1-UC-int8_convrot.safetensors",
            "qwen-image-2.1-UC-NVFP4.safetensors",
            "qwen-image-2.1-UC-MLX-4bit.safetensors",
            "qwen-image-2.1-UC-MLX-6bit.safetensors",
            "qwen-image-2.1-UC-MLX-8bit.safetensors",
            "qwen-image-2.1-UC-Q8_0.gguf",
            "qwen-image-2.1-UC-Q6_K.gguf",
            "qwen-image-2.1-UC-Q5_K_M.gguf",
            "qwen-image-2.1-UC-Q4_K_M.gguf",
            "qwen-image-2.1-UC-Q4_0.gguf"
        ],
        "Text Encoders": [
            "qwen3vl_8b_bf16.safetensors",
            "qwen3vl_8b_int8_convrot.safetensors"
        ],
        "VAE": [
            "qwen_image_2.1_vae_bf16.safetensors"
        ]
    }
    
    for category, file_list in files.items():
        print(f"\n{category}:")
        for file in file_list:
            print(f"  - {file}")

def show_setup_instructions():
    """Show setup instructions for ComfyUI"""
    print("\nSetup Instructions:")
    print("-" * 40)
    print("1. Install ComfyUI from: https://github.com/comfyanonymous/ComfyUI")
    print("2. Clone ComfyUI-GGUF plugin:")
    print("   cd ComfyUI/custom_nodes")
    print("   git clone https://github.com/leejet/ComfyUI-GGUF")
    print("3. Download model files to respective directories:")
    print("   - Diffusion Model: ComfyUI/models/diffusion_models/")
    print("   - Text Encoder: ComfyUI/models/text_encoders/")
    print("   - VAE: ComfyUI/models/vae/")
    print("4. Recommended model files:")
    print("   - Diffusion: qwen-image-2.1-UC-Q4_K_M.gguf (~4.6 GB)")
    print("   - Text Encoder: qwen3vl_8b_int8_convrot.safetensors (~9.35 GB)")
    print("   - VAE: qwen_image_2.1_vae_bf16.safetensors (676 MB)")
    print("5. Run ComfyUI and use the workflow templates")

def create_sample_prompts():
    """Create sample artistic prompts for legitimate use"""
    prompts = [
        "A beautiful landscape painting in the style of Van Gogh",
        "Abstract geometric art with vibrant colors",
        "Fantasy creature design for a fantasy game",
        "Cyberpunk cityscape at night with neon lights",
        "Portrait of a futuristic robot with detailed textures",
        "Minimalist architecture photography in natural light",
        "Watercolor painting of a serene mountain lake",
        "Steampunk mechanical contraption with brass gears",
        "Astronomical nebula with cosmic colors",
        "Japanese ink painting of cherry blossoms in spring"
    ]
    
    print("\nSample Artistic Prompts:")
    print("-" * 40)
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt}")

def main():
    """Main function to demonstrate setup"""
    print_welcome()
    
    setup_directories()
    show_model_files()
    show_setup_instructions()
    create_sample_prompts()
    
    print("\n" + "=" * 60)
    print("USAGE NOTES:")
    print("- This model is for artistic and creative purposes only")
    print("- Do not use for generating inappropriate content")
    print("- Follow ethical guidelines for AI image generation")
    print("- Ensure proper attribution and licensing compliance")
    print("=" * 60)
    
    print("\nTo run with ComfyUI:")
    print("1. Place the downloaded model files in the correct directories")
    print("2. Run ComfyUI: python main.py")
    print("3. Use the workflow templates provided")
    print("4. Adjust parameters for your artistic vision")

if __name__ == "__main__":
    main()