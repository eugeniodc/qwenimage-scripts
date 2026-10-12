#!/usr/bin/env python3
"""
CLI interactivo y configurable para Qwen-Image-2.1-Turbo
Ubicación: scripts/demo_cli.py
Guarda las imágenes generadas por defecto en: outputs/output.png
"""

import argparse
import os
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

# Relaciones de aspecto y dimensiones recomendadas según la documentación de Qwen
ASPECT_RATIOS = {
    "1:1":  (2048, 2048),
    "4:3":  (2400, 1792),
    "3:4":  (1792, 2400),
    "3:2":  (2528, 1696),
    "2:3":  (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

ASPECT_RATIOS_FAST = {
    "1:1":  (1024, 1024),
    "4:3":  (1152, 896),
    "3:4":  (896, 1152),
    "3:2":  (1216, 832),
    "2:3":  (832, 1216),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
}

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
        description="Demo CLI para Qwen/Qwen-Image-2.1-Turbo (Diffusers 8-step)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Texto descriptivo para generar o editar la imagen",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta local o URL de la imagen de entrada si se desea modo edición (ej. sample_images/mi_imagen.png)",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        choices=list(ASPECT_RATIOS.keys()),
        default="1:1",
        help="Relación de aspecto (1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16). Por defecto: 1:1",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Modo rápido: genera a resolución 1K (~1024x1024), acelerando la generación hasta 4x",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Número de pasos de inferencia (por defecto: 8 para el modelo Turbo)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Nombre o ruta del archivo de salida (por defecto: outputs/output.png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad (por defecto: 42)",
    )
    parser.add_argument(
        "--disable-kv-cache",
        action="store_true",
        help="Desactivar use_kv_cache (activado por defecto para mayor eficiencia)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device, dtype = get_target_device_and_dtype()
    ratio_dict = ASPECT_RATIOS_FAST if args.fast else ASPECT_RATIOS
    width, height = ratio_dict[args.aspect_ratio]
    cache_dir = PROJECT_DIR / "cache" / "huggingface"
    outputs_dir = PROJECT_DIR / "outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Determinar ruta de salida dentro de outputs/ si es relativa
    out_path = Path(args.output)
    if out_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        out_path = out_path.with_suffix(".png")
    if not out_path.is_absolute():
        out_path = outputs_dir / out_path

    print("=" * 65)
    print("🚀 Qwen-Image-2.1-Turbo | CLI")
    print("=" * 65)
    print(f"[*] Directorio raíz:        {PROJECT_DIR}")
    print(f"[*] Caché local de modelos: {cache_dir}")
    print(f"[*] Dispositivo:            {device.upper()} | Tipo de dato: {dtype}")
    print(f"[*] Modo:                   {'Image Editing' if args.image else 'Text-to-Image'}")
    print(f"[*] Perfil:                 {'Rápido (~1K)' if args.fast else 'Alta Calidad (~2K)'}")
    print(f"[*] Relación de aspecto:    {args.aspect_ratio} ({width}x{height})")
    print(f"[*] Pasos de inferencia:    {args.steps} pasos Turbo")
    print(f"[*] Semilla:                {args.seed}")
    print(f"[*] Prompt:                 {args.prompt}")
    print(f"[*] Archivo de salida:      {out_path}")

    print("\n[*] Cargando modelo...")
    pipe = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1-Turbo",
        dtype=dtype,
        cache_dir=str(cache_dir),
    ).to(device)

    generator = torch.Generator("cpu").manual_seed(args.seed)
    kwargs = {
        "prompt": args.prompt,
        "width": width,
        "height": height,
        "num_inference_steps": args.steps,
        "use_kv_cache": not args.disable_kv_cache,
        "generator": generator,
    }

    if args.image:
        resolved_img_path = resolve_path(args.image)
        print(f"[*] Cargando imagen de referencia desde: {resolved_img_path}")
        input_img = load_image(resolved_img_path).convert("RGBA")
        kwargs["image"] = input_img

    print("[*] Ejecutando generación con schedule Turbo de 8 pasos...")
    output = pipe(**kwargs)
    image = output.images[0]
    try:
        image.save(out_path)
    except Exception:
        image.save(out_path, format="PNG")
    print(f"\n[✓] Guardado exitosamente en: {out_path}")

if __name__ == "__main__":
    main()
