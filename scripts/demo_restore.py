#!/usr/bin/env python3
"""
Script de Restauración y Mejora de Fotos (Photo Restoration & Upscale)
utilizando Qwen-Image-2.1-Turbo y el LoRA especializado:
'ausboss/Qwen-Image-2.1-Restore-LoRA' (qwen-image-2.1-restore.safetensors)

Ubicación: scripts/demo_restore.py
Guarda la imagen por defecto en: outputs/restore_output.png

Qué hace:
- Elimina ruido, artefactos de compresión y pixelado sin envejecer la piel.
- Modo foto antigua: remueve arañazos, polvo y desvanecimiento de color.
- Conserva al 100% las facciones de la persona y la iluminación original.
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

DEFAULT_RESTORE_LORA = "ausboss/Qwen-Image-2.1-Restore-LoRA"
DEFAULT_WEIGHT_NAME = "qwen-image-2.1-restore.safetensors"

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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Restauración de fotos con Qwen-Image-2.1-Turbo y LoRA Restore"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Ruta o URL de la imagen a restaurar (ej: sample_images/mi_foto.jpg)",
    )
    parser.add_argument(
        "--mode",
        choices=["modern", "old_photo"],
        default="modern",
        help="Modo: 'modern' (desenfoque/compresión) o 'old_photo' (arañazos/polvo/fotos antiguas)",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=DEFAULT_RESTORE_LORA,
        help=f"Repositorio de LoRA (por defecto: '{DEFAULT_RESTORE_LORA}')",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Fuerza del LoRA: 1.0 para fotos nítidas/comprimidas, 0.5-0.75 para fotos muy borrosas",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="restore_output.png",
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
    print("✨ Qwen-Image-2.1-Turbo | Restauración de Fotos (Restore LoRA)")
    print("=" * 65)
    print(f"[*] Imagen origen:          {args.image}")
    print(f"[*] Modo de restauración:   {args.mode.upper()}")
    print(f"[*] LoRA de restauración:   {args.lora} (fuerza: {args.lora_scale})")
    print(f"[*] Dispositivo:            {device.upper()} (dtype: {dtype})")
    print(f"[*] Archivo de salida:      {out_path}")

    # Cargar imagen de entrada
    print("\n[+] Cargando imagen...")
    try:
        input_image = load_image(resolve_path(args.image)).convert("RGB")
    except Exception as e:
        print(f"[!] Error al cargar la imagen: {e}")
        sys.exit(1)

    # Redimensionar suavemente si es muy pequeña para mejor muestreo a 2K
    w, h = input_image.size
    # Asegurar múltiplos de 32
    target_w = ((w + 31) // 32) * 32
    target_h = ((h + 31) // 32) * 32
    if target_w < 1024 or target_h < 1024:
        scale = max(1024 / target_w, 1024 / target_h)
        target_w = int(((target_w * scale) + 31) // 32) * 32
        target_h = int(((target_h * scale) + 31) // 32) * 32

    # Prompt oficial del autor según modo
    if args.mode == "old_photo":
        prompt = (
            "Restore the old photo: remove scratches, dust, fading and grain and recover "
            "sharp, natural detail and color, keeping the same composition and people."
        )
    else:
        prompt = (
            "Restore the photo: remove blur, noise and compression artifacts and recover "
            "sharp, natural detail, keeping the same composition, people and colors."
        )

    print(f"\n[+] Prompt de restauración:")
    print(f"    \"{prompt}\"")

    # Cargar pipeline
    print("\n[+] Cargando modelo base Qwen-Image-2.1-Turbo...")
    pipe = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1-Turbo",
        dtype=dtype,
        cache_dir=str(cache_dir),
    ).to(device)

    # Cargar LoRA Restore
    print(f"[+] Cargando LoRA: {args.lora} ({DEFAULT_WEIGHT_NAME})...")
    try:
        pipe.load_lora_weights(
            args.lora,
            weight_name=DEFAULT_WEIGHT_NAME,
            cache_dir=str(cache_dir),
        )
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters(list(pipe.get_active_adapters()), adapter_weights=[args.lora_scale])
        print("[✓] LoRA Restore cargado exitosamente.")
    except Exception as e:
        print(f"[!] Advertencia al cargar LoRA: {e}")
        print("[!] Continuando en modo nativo...")

    # Generación
    print("\n[+] Procesando restauración (8 pasos Turbo)...")
    generator = torch.Generator("cpu").manual_seed(args.seed)

    result = pipe(
        prompt=prompt,
        image=input_image,
        width=target_w,
        height=target_h,
        num_inference_steps=8,
        use_kv_cache=True,
        generator=generator,
    )

    output_image = result.images[0]
    try:
        output_image.save(out_path)
    except Exception:
        output_image.save(out_path, format="PNG")
    print(f"\n[✓] ¡Foto restaurada exitosamente!")
    print(f"[✓] Guardado en: {out_path}")

if __name__ == "__main__":
    main()
