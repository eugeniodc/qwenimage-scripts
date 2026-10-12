#!/usr/bin/env python3
"""
Script de Expansión de Imagen / Outpainting (Uncrop)
utilizando Qwen-Image-2.1-Turbo y el LoRA especializado:
'ausboss/Qwen-Image-2.1-Outpaint-LoRA' (qwen-image-2.1-outpaint-v2.safetensors)

Ubicación: scripts/demo_outpaint.py
Guarda la imagen por defecto en: outputs/outpaint_output.png

Cómo funciona:
1. Rellena el lienzo con gris plano (#808080) en la dirección deseada (horizontal, vertical o todas).
2. El LoRA detecta las áreas grises y las rellena con una continuación natural y sin cortes del entorno.
"""

import argparse
import os
import sys
from pathlib import Path

# Directorio raíz del proyecto
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Configurar aislamiento local de caché
os.environ["HF_HOME"] = str(PROJECT_DIR / "cache" / "huggingface")
os.environ["TORCH_HOME"] = str(PROJECT_DIR / "cache" / "torch")

import torch
from diffusers import QwenImage21Pipeline
from diffusers.utils import load_image
from PIL import Image

DEFAULT_OUTPAINT_LORA = "ausboss/Qwen-Image-2.1-Outpaint-LoRA"
DEFAULT_WEIGHT_NAME = "qwen-image-2.1-outpaint-v2.safetensors"

def resolve_path(p: str) -> str:
    if not p:
        return p
    path_obj = Path(p)
    if path_obj.exists():
        return str(path_obj.resolve())
    proj_candidate = PROJECT_DIR / path_obj
    if proj_candidate.exists():
        return str(proj_candidate.resolve())
    return p

def get_target_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    else:
        return "cpu", torch.float32

def pad_image_gray(img: Image.Image, target_ratio: str = "16:9") -> Image.Image:
    """Rellena la imagen con gris #808080 para alcanzar la relación de aspecto deseada."""
    w, h = img.size
    
    ratio_dict = {
        "16:9": 16 / 9,
        "9:16": 9 / 16,
        "4:3":  4 / 3,
        "3:4":  3 / 4,
        "1:1":  1.0,
        "expand_all": (w * 1.5) / (h * 1.5),
    }

    target_aspect = ratio_dict.get(target_ratio, 16 / 9)
    current_aspect = w / h

    if target_ratio == "expand_all":
        new_w = int(w * 1.4)
        new_h = int(h * 1.4)
    elif current_aspect < target_aspect:
        # Necesita expandir ancho (horizontalmente)
        new_h = h
        new_w = int(h * target_aspect)
    else:
        # Necesita expandir alto (verticalmente)
        new_w = w
        new_h = int(w / target_aspect)

    # Redondear a múltiplos de 32 para el VAE
    new_w = ((new_w + 31) // 32) * 32
    new_h = ((new_h + 31) // 32) * 32

    # Crear lienzo con gris neutro #808080 (128, 128, 128)
    padded = Image.new("RGB", (new_w, new_h), color=(128, 128, 128))
    # Centrar la imagen original en el nuevo lienzo
    offset_x = (new_w - w) // 2
    offset_y = (new_h - h) // 2
    padded.paste(img, (offset_x, offset_y))

    return padded

def parse_args():
    parser = argparse.ArgumentParser(
        description="Expansión de imagen (Outpainting / Uncrop) con Qwen-Image-2.1-Turbo y Outpaint LoRA"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Ruta de la imagen que deseas expandir (ej: sample_images/mi_foto.jpg)",
    )
    parser.add_argument(
        "--ratio",
        choices=["16:9", "9:16", "4:3", "3:4", "1:1", "expand_all"],
        default="16:9",
        help="Formato o expansión deseada: 16:9 (panorámico), 9:16 (vertical), 4:3, o expand_all (zoom-out general)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Descripción opcional del entorno a expandir (ej: 'in an open sunny field')",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=DEFAULT_OUTPAINT_LORA,
        help=f"Repositorio de LoRA (por defecto: '{DEFAULT_OUTPAINT_LORA}')",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Fuerza del LoRA (recomendado por el autor: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outpaint_output.png",
        help="Nombre del archivo de salida (guardado en outputs/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad (por defecto: 42)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device, dtype = get_target_device_and_dtype()
    cache_dir = PROJECT_DIR / "cache" / "huggingface"
    outputs_dir = PROJECT_DIR / "outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output)
    if out_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        out_path = out_path.with_suffix(".png")
    if not out_path.is_absolute():
        out_path = outputs_dir / out_path

    print("=" * 65)
    print("🖼️  Qwen-Image-2.1-Turbo | Outpainting (Extensión de Lienzo)")
    print("=" * 65)
    print(f"[*] Imagen origen:          {args.image}")
    print(f"[*] Relación objetivo:      {args.ratio}")
    print(f"[*] LoRA de Outpainting:    {args.lora} (fuerza: {args.lora_scale})")
    print(f"[*] Dispositivo:            {device.upper()} (dtype: {dtype})")
    print(f"[*] Archivo de salida:      {out_path}")

    # Cargar y preparar lienzo con padding gris #808080
    print("\n[+] Cargando y preparando lienzo con padding neutro #808080...")
    try:
        raw_image = load_image(resolve_path(args.image)).convert("RGB")
        padded_canvas = pad_image_gray(raw_image, target_ratio=args.ratio)
    except Exception as e:
        print(f"[!] Error al procesar imagen: {e}")
        sys.exit(1)

    print(f"[+] Tamaño original: {raw_image.size} -> Tamaño expandido: {padded_canvas.size}")

    # Prompt oficial del autor del LoRA
    prompt = (
        "Outpaint the image: replace the solid gray areas with a seamless continuation "
        "of the scene, keeping the existing picture unchanged."
    )
    if args.description:
        prompt += f" Scene: {args.description}"

    print(f"\n[+] Prompt de Outpaint:")
    print(f"    \"{prompt}\"")

    # Cargar pipeline
    print("\n[+] Cargando modelo base Qwen-Image-2.1-Turbo...")
    pipe = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1-Turbo",
        dtype=dtype,
        cache_dir=str(cache_dir),
    ).to(device)

    # Cargar LoRA Outpaint
    print(f"[+] Cargando LoRA: {args.lora} ({DEFAULT_WEIGHT_NAME})...")
    try:
        pipe.load_lora_weights(
            args.lora,
            weight_name=DEFAULT_WEIGHT_NAME,
            cache_dir=str(cache_dir),
        )
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters(list(pipe.get_active_adapters()), adapter_weights=[args.lora_scale])
        print("[✓] LoRA Outpaint cargado exitosamente.")
    except Exception as e:
        print(f"[!] Advertencia al cargar LoRA: {e}")
        print("[!] Continuando en modo nativo...")

    # Generación
    print("\n[+] Procesando expansión de imagen (8 pasos Turbo)...")
    generator = torch.Generator("cpu").manual_seed(args.seed)

    result = pipe(
        prompt=prompt,
        image=padded_canvas,
        width=padded_canvas.size[0],
        height=padded_canvas.size[1],
        num_inference_steps=8,
        use_kv_cache=True,
        generator=generator,
    )

    output_image = result.images[0]
    try:
        output_image.save(out_path)
    except Exception:
        output_image.save(out_path, format="PNG")
    print(f"\n[✓] ¡Lienzo expandido exitosamente!")
    print(f"[✓] Guardado en: {out_path}")

if __name__ == "__main__":
    main()
