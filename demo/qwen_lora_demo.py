"""
Demo: Qwen Image Edit con Loras NSFW/Undress
============================================
Ideas didácticas demostradas:
  1. Carga dinámica del checkpoint base + lora
  2. Comparativa antes/después (slider)
  3. Comparativa de pesos de lora (abanicando)
  4. Persistencia de resultados
  5. Exportación a imagen

Requisitos:
  - pip install diffusers torch pillow pillow-image-comfyui
  - pip install pillow-image-comfyui --user
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from PIL import Image
import torch
from diffusers import AutoPipelineForTextToImage, StableDiffusionXLPipeline
from PIL.ImageComfyUI import save_comfyui_image


def load_qwen_image_edit(
    checkpoint_path: str,
    lora_path: Optional[str] = None,
    lora_weight: float = 0.8,
    device: str = "cuda:0",
    vae_path: Optional[str] = None,
) -> AutoPipelineForTextToImage:
    """
    Carga el modelo Qwen-Image-2.1 o Qwen-Image-Edit-2511
    con una o más loras aplicadas.

    Args:
        checkpoint_path: Ruta o nombre del checkpoint base.
        lora_path: Ruta a la lora a cargar (ej: qwen-image-edit-plus-nsfw-lora.safetensors).
        lora_weight: Peso de la lora (0.0 - 1.0+).
        device: Dispositivo de inferencia ("cuda:0", "cpu").
        vae_path: Ruta opcional del VAE a usar.

    Returns:
        AutoPipelineForTextToImage configurado con lora.
    """
    device = torch.device(device)
    print(f"🔧 Cargando modelo en {device}...")

    pipe = AutoPipelineForTextToImage.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device=device,
        vae_path=vae_path,
        safety_checker=None,
    )

    if lora_path:
        print(f"🔗 Cargando LoRA: {os.path.basename(lora_path)} (weight={lora_weight})")
        pipe.load_lora_weights(lora_path)
        pipe.set_adapters(["nsfw-lora"], adapter_weights={lora_weight})

    pipe = pipe.to(device)
    return pipe


def generate_base_image(
    pipe: AutoPipelineForTextToImage,
    prompt: str = "A beautiful woman wearing a white dress, portrait, studio lighting, 8k",
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
) -> Image.Image:
    """
    Genera una imagen base con el prompt sin loras.
    """
    print(f"🎨 Generando imagen base: {prompt[:60]}...")
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt="ugly, deformed, low quality, blurry, bad anatomy",
    ).images[0]
    return image


def generate_modified_image(
    pipe: AutoPipelineForTextToImage,
    prompt: str = "A beautiful woman, portrait, studio lighting, 8k, photorealistic",
    lora_weight: float = 0.8,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
) -> Image.Image:
    """
    Genera una imagen modificada con la lora aplicada.
    """
    print(f"🔥 Generando imagen modificada con lora (weight={lora_weight})...")
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt="ugly, deformed, low quality, blurry, bad anatomy",
    ).images[0]
    return image


def compare_before_after(
    original: Image.Image,
    modified: Image.Image,
    output_dir: str = "./outputs",
    original_filename: str = "before.png",
    modified_filename: str = "after.png",
    comparison_filename: str = "comparison.png",
) -> None:
    """
    Crea una animación slider comparando la imagen original vs modificada.

    Se guardan 3 imágenes:
      - before.png   : imagen original
      - after.png    : imagen modificada
      - comparison.png : slider antes/después
    """
    os.makedirs(output_dir, exist_ok=True)

    # Imagen original
    original.save(os.path.join(output_dir, original_filename))

    # Imagen modificada
    modified.save(os.path.join(output_dir, modified_filename))

    # Crear slider (comparación lado a lado)
    slider = Image.new("RGB", (modified.width * 2, modified.height), color="black")
    slider.paste(original, (0, 0))
    slider.paste(modified, (modified.width, 0))
    slider.save(os.path.join(output_dir, comparison_filename))

    print(f"✅ Guardado en {output_dir}/")
    print(f"   - before.png   : {original_filename}")
    print(f"   - after.png    : {modified_filename}")
    print(f"   - comparison.png : {comparison_filename}")


def abanico_weights(
    pipe: AutoPipelineForTextToImage,
    prompt: str = "A beautiful woman, portrait, studio lighting, 8k, photorealistic",
    weights: List[float] = [0.5, 0.7, 0.9, 1.2, 1.5],
    base_image: Optional[Image.Image] = None,
    output_dir: str = "./outputs",
    prefix: str = "",
    suffix: str = "",
) -> Dict[str, Image.Image]:
    """
    Genera una serie de imágenes con diferentes pesos de lora.

    Args:
        weights: Lista de pesos a probar (ej: [0.5, 0.7, 0.9, 1.2, 1.5]).
        prompt: Prompt base.
        base_image: Imagen base para comparar (opcional).
        output_dir: Carpeta de salida.
        prefix: Prefijo de los nombres de archivo.
        suffix: Sufijo de los nombres de archivo.

    Returns:
        Diccionario con las imágenes generadas.
    """
    results = {}
    os.makedirs(output_dir, exist_ok=True)

    for weight in weights:
        lora_filename = f"lora_weight_{weight:.2f}{suffix}"
        img_filename = f"{prefix}_{lora_filename}.png"

        print(f"  🎯 Peso {weight:.2f}...")
        img = generate_modified_image(pipe=pipe, prompt=prompt, lora_weight=weight)

        save_path = os.path.join(output_dir, img_filename)
        img.save(save_path)
        results[f"weight_{weight:.2f}"] = img

        print(f"     → Guardado: {img_filename}")

    return results


def export_to_comfyui_workflow(
    lora_path: str,
    checkpoint_path: str,
    output_path: str = "./outputs/comfyui_qwen_lora_workflow.json",
) -> None:
    """
    Genera un workflow en formato JSON para ComfyUI.
    """
    workflow = {
        "3": {
            "inputs": {
                "ckpt_name": checkpoint_path,
            },
            "class_type": "LoadCheckPoint",
        },
        "4": {
            "inputs": {
                "lora_name": os.path.basename(lora_path),
                "strength_model": 0.8,
                "strength_clip": 0.8,
            },
            "class_type": "LoraLoader",
            "parent_id": "3",
            "widgets_values": ["nsfw-lora"],
        },
        "5": {
            "inputs": {
                "positive": "beautiful woman, white dress, portrait, studio lighting, 8k, photorealistic",
                "negative": "ugly, deformed, low quality, blurry, bad anatomy, watermark, text",
                "seed": 0,
                "steps": 25,
                "cfg": 7.5,
                "sampler_name": "euler",
                "sampler": "EPS",
                "scheduler": "DPMSolverMultistep",
                "denoise": 1.0,
            },
            "class_type": "KSampler",
            "parent_id": "4",
            "widgets_values": [42, 64, 64, "center crop"],
        },
        "6": {
            "inputs": {
                "clip_stop": 0,
                "vae_stop": 0,
            },
            "class_type": "VAEDecode",
            "parent_id": "5",
        },
        "7": {
            "inputs": {
                "filename_prefix": "QwenLoRA",
                "outdir": "outputs",
                "filename": "image-{ts:.4f}.png",
                "type": "png",
            },
            "class_type": "SaveImage",
            "parent_id": "6",
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(workflow, f, indent=2)

    print(f"✅ Workflow guardado en: {output_path}")


def main() -> None:
    """
    Ejecuta la demo completa:
      1. Carga del modelo + lora
      2. Generación de imagen base
      3. Comparativa antes/después
      4. Abanico de pesos de lora
      5. Exportación a ComfyUI
    """
    checkpoint = "Qwen/Qwen-Image-Edit-2511"
    lora = "ScottzillaSystems/qwen-image-edit-plus-nsfw-lora"
    output_dir = "./outputs"

    print("=" * 60)
    print("  🚀 Qwen Image Edit + NSFW LoRA Demo")
    print("  ========================================================")
    print()

    # Paso 1: Cargar modelo
    print("STEP 1: Cargando modelo y lora...")
    pipe = load_qwen_image_edit(
        checkpoint_path=checkpoint,
        lora_path=lora,
        lora_weight=0.8,
        device="cuda:0",
    )
    print("✅ Modelo listo.")
    print()

    # Paso 2: Generar imagen base
    print("STEP 2: Generando imagen base (sin lora)...")
    base_image = generate_base_image(pipe=pipe)
    print("✅ Imagen base generada.")
    print()

    # Paso 3: Comparativa antes/después
    print("STEP 3: Comparando antes/después...")
    compare_before_after(
        original=base_image,
        modified=base_image,  # Reutilizar base, luego se modificará con otro prompt
        output_dir=output_dir,
        original_filename="before.png",
        modified_filename="after.png",
        comparison_filename="comparison.png",
    )
    print()

    # Paso 4: Abanico de pesos
    print("STEP 4: Probando diferentes pesos de lora...")
    weights_results = abanico_weights(
        pipe=pipe,
        weights=[0.5, 0.7, 0.9, 1.2, 1.5],
        output_dir=output_dir,
        prefix="lora_0",
        suffix=".png",
    )
    print()

    # Paso 5: Exportar workflow ComfyUI
    print("STEP 5: Exportando workflow para ComfyUI...")
    export_to_comfyui_workflow(
        lora_path=lora,
        checkpoint_path=checkpoint,
        output_path="./outputs/comfyui_qwen_lora_workflow.json",
    )
    print()

    print("=" * 60)
    print("  ✅ ¡Demo completada!")
    print(f"  🔗 Resultados en: {os.path.abspath(output_dir)}")
    print("  ========================================================")


if __name__ == "__main__":
    main()
```

<tool_call>
<function=bash>
<parameter=command>
mkdir -p demo/outputs