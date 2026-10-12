#!/usr/bin/env python3
"""
Script de Cambio de Ropa / Outfit Swap (Virtual Try-On)
utilizando Qwen-Image-2.1-Turbo y el LoRA especializado:
'ausboss/Qwen-Image-2.1-Outfit-Swap-Consistency-LoRA'

Ubicación: scripts/demo_outfit_swap.py
Guarda la imagen generada por defecto en: outputs/outfit_swap_output.png

Cómo funciona:
 - <image1>: La persona base (se mantiene su cara, pelo, cuerpo, pose y fondo).
 - <image2>: La prenda o vestimenta de referencia que se desea poner a la persona.
"""

import argparse
import os
import sys
from pathlib import Path

# Directorio raíz del proyecto (un nivel arriba de scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Configurar aislamiento local de caché en cache/
os.environ["HF_HOME"] = str(PROJECT_DIR / "cache" / "huggingface")
os.environ["TORCH_HOME"] = str(PROJECT_DIR / "cache" / "torch")

import torch
from diffusers import QwenImage21Pipeline
from diffusers.utils import load_image

# Preajustes oficiales de resolución
ASPECT_RATIOS = {
    "1:1":  (2048, 2048),
    "4:3":  (2400, 1792),
    "3:4":  (1792, 2400),
    "3:2":  (2528, 1696),
    "2:3":  (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

DEFAULT_OUTFIT_LORA = "ausboss/Qwen-Image-2.1-Outfit-Swap-Consistency-LoRA"
DEFAULT_WEIGHT_NAME = "qwen-image-2.1-outfit-swap.safetensors"

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
        description="Cambio de ropa / Outfit Swap con Qwen-Image-2.1-Turbo y LoRA de consistencia"
    )
    parser.add_argument(
        "--person-image",
        type=str,
        required=True,
        help="Ruta o URL de la foto de la persona a la que cambiar la ropa (<image1>)",
    )
    parser.add_argument(
        "--outfit-image",
        type=str,
        required=True,
        help="Ruta o URL de la prenda o conjunto de ropa de referencia (<image2>)",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=DEFAULT_OUTFIT_LORA,
        help=f"Repositorio de LoRA para outfit swap (por defecto: '{DEFAULT_OUTFIT_LORA}')",
    )
    parser.add_argument(
        "--lora-weight-name",
        type=str,
        default=DEFAULT_WEIGHT_NAME,
        help=f"Archivo safetensors del LoRA (por defecto: '{DEFAULT_WEIGHT_NAME}')",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Desactivar el LoRA y usar únicamente el modelo base nativo",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Fuerza del LoRA (recomendado por el autor: 1.0)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt personalizado de cambio de ropa si deseas especificar detalles",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        choices=list(ASPECT_RATIOS.keys()),
        default="1:1",
        help="Relación de aspecto de salida (1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16). Por defecto: 1:1",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outfit_swap_output.png",
        help="Nombre del archivo de salida (se guardará en la carpeta outputs/)",
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
    width, height = ASPECT_RATIOS[args.aspect_ratio]
    cache_dir = PROJECT_DIR / "cache" / "huggingface"
    outputs_dir = PROJECT_DIR / "outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Determinar ruta de salida dentro de outputs/ (asegurar extensión válida)
    out_path = Path(args.output)
    if out_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        out_path = out_path.with_suffix(".png")
    if not out_path.is_absolute():
        out_path = outputs_dir / out_path

    use_lora = not args.no_lora and bool(args.lora)

    print("=" * 65)
    print("👔 Qwen-Image-2.1-Turbo | Outfit Swap / Cambio de Ropa")
    print("=" * 65)
    print(f"[*] Directorio raíz:           {PROJECT_DIR}")
    print(f"[*] Dispositivo:              {device.upper()} (dtype: {dtype})")
    print(f"[*] Persona base (<image1>):  {args.person_image}")
    print(f"[*] Ropa referencia (<image2>): {args.outfit_image}")
    print(f"[*] LoRA de consistencia:     {args.lora if use_lora else 'Desactivado (Nativo)'}")
    print(f"[*] Resolución:               {args.aspect_ratio} ({width}x{height})")
    print(f"[*] Semilla:                  {args.seed}")
    print(f"[*] Salida:                   {out_path}")

    # Cargar imágenes de referencia
    print("\n[+] Cargando imágenes...")
    try:
        person_img = load_image(resolve_path(args.person_image)).convert("RGB")
        outfit_img = load_image(resolve_path(args.outfit_image)).convert("RGB")
    except Exception as e:
        print(f"[!] Error al cargar las imágenes: {e}")
        sys.exit(1)

    # Prompt estructurado según las especificaciones del autor del LoRA
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = (
            "Dress the person in image 1 in the outfit shown in image 2. "
            "Keep their face, hair, hands, pose and the background exactly the same. "
            "Replace everything they wear. Photorealistic, high fashion quality."
        )

    print(f"\n[+] Prompt de transformación:")
    print(f"    \"{prompt}\"")

    # Cargar el pipeline
    print("\n[+] Cargando modelo base Qwen-Image-2.1-Turbo...")
    pipe = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1-Turbo",
        dtype=dtype,
        cache_dir=str(cache_dir),
    ).to(device)

    # Cargar LoRA si está activado
    if use_lora:
        print(f"\n[+] Cargando LoRA de Outfit Swap: {args.lora}...")
        print(f"[+] Archivo de pesos: {args.lora_weight_name} (fuerza: {args.lora_scale})")
        try:
            kwargs_lora = {"cache_dir": str(cache_dir)}
            if args.lora_weight_name:
                kwargs_lora["weight_name"] = args.lora_weight_name
            pipe.load_lora_weights(args.lora, **kwargs_lora)
            if hasattr(pipe, "set_adapters"):
                pipe.set_adapters(list(pipe.get_active_adapters()), adapter_weights=[args.lora_scale])
            print("[✓] LoRA de Outfit Swap cargado exitosamente.")
        except Exception as e:
            print(f"[!] Aviso al cargar LoRA: {e}")
            print("[!] Continuando la generación en modo nativo...")

    # Inferencia (8 pasos Turbo)
    print("\n[+] Procesando cambio de ropa (8 pasos Turbo)...")
    generator = torch.Generator("cpu").manual_seed(args.seed)

    result = pipe(
        prompt=prompt,
        image=[person_img, outfit_img],
        width=width,
        height=height,
        num_inference_steps=8,
        use_kv_cache=True,
        generator=generator,
    )

    output_image = result.images[0]
    try:
        output_image.save(out_path)
    except Exception:
        output_image.save(out_path, format="PNG")
    print(f"\n[✓] ¡Cambio de ropa completado exitosamente!")
    print(f"[✓] Imagen guardada en: {out_path}")

if __name__ == "__main__":
    main()
