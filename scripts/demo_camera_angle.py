#!/usr/bin/env python3
"""
Script de Cambio de Ángulo de Cámara (Camera Angle / Multi-View / LoRA)
para Qwen-Image-2.1-Turbo.

Ubicación: scripts/demo_camera_angle.py
Salida por defecto: outputs/angle_<tipo>.png

Especificaciones del autor (akhaliq/Qwen-Image-2.1-Multiple-Angles-LoRA):
 - Trigger: <mva>
 - Gramática estricta: <mva> {azimuth}, {elevation}[ close-up]
 - 12 azimuts (pasos de 30°) y 4 elevaciones (0°, 30°, 60°, 90°)
 - Fuerza recomendada: 0.8 - 1.0
 - Archivo recomendado: multiple_angles_step1000.safetensors
"""

import argparse
import os
import sys
from pathlib import Path

# Directorio raíz del proyecto
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Configurar aislamiento local de caché
os.environ["HF_HOME"] = str(PROJECT_DIR / "cache" / "huggingface")
os.environ["TORCH_HOME"] = str(PROJECT_DIR / "cache" / "torch")

import torch
from diffusers import QwenImage21Pipeline
from diffusers.utils import load_image
from core.prompt_utils import translate_and_enrich_prompt

DEFAULT_CAMERA_LORA = "akhaliq/Qwen-Image-2.1-Multiple-Angles-LoRA"
DEFAULT_CAMERA_WEIGHT = "multiple_angles_step1000.safetensors"

# Gramática oficial del LoRA Multiple-Angles
ANGLE_PRESETS = {
    "profile_right": {
        "desc": "full right side profile view (90 degrees right)",
        "mva": "<mva> right side view, eye-level shot",
    },
    "profile_left": {
        "desc": "full left side profile view (90 degrees left)",
        "mva": "<mva> left side view, eye-level shot",
    },
    "three_quarter_right": {
        "desc": "three-quarters view turned to the right",
        "mva": "<mva> front-right quarter view, eye-level shot",
    },
    "three_quarter_left": {
        "desc": "three-quarters view turned to the left",
        "mva": "<mva> front-left quarter view, eye-level shot",
    },
    "low_angle": {
        "desc": "dramatic low angle shot looking upwards at the subject",
        "mva": "<mva> front view, eye-level shot",  # El ángulo base a nivel
    },
    "high_angle": {
        "desc": "high angle shot looking down from above (60 degrees elevation)",
        "mva": "<mva> front view, high-angle shot",
    },
    "top_down": {
        "desc": "zenith bird's-eye view from directly above (90 degrees elevation)",
        "mva": "<mva> front view, top-down shot",
    },
    "back_view": {
        "desc": "shot from behind showing the back of the subject (180 degrees)",
        "mva": "<mva> back view, eye-level shot",
    },
    "back_right": {
        "desc": "rear three-quarters view from the right (120 degrees)",
        "mva": "<mva> back-right quarter view, eye-level shot",
    },
    "back_left": {
        "desc": "rear three-quarters view from the left (240 degrees)",
        "mva": "<mva> back-left quarter view, eye-level shot",
    },
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
        description="Cambio de ángulo de cámara con Qwen-Image-2.1-Turbo (Nativo o LoRA)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Ruta local de la imagen de entrada (ej: sample_images/eugenio.jpg)",
    )
    parser.add_argument(
        "--angle",
        type=str,
        choices=list(ANGLE_PRESETS.keys()),
        default="three_quarter_right",
        help=f"Ángulo deseado: {list(ANGLE_PRESETS.keys())}. Por defecto: three_quarter_right",
    )
    parser.add_argument(
        "--close-up",
        action="store_true",
        help="Encuadre en primer plano (close-up)",
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        default=None,
        help="Instrucción de ángulo personalizada si no deseas usar los preajustes",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help=f"Repositorio de LoRA (ej: '{DEFAULT_CAMERA_LORA}')",
    )
    parser.add_argument(
        "--lora-weight-name",
        type=str,
        default=DEFAULT_CAMERA_WEIGHT,
        help=f"Archivo safetensors del LoRA (por defecto: '{DEFAULT_CAMERA_WEIGHT}')",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Fuerza del LoRA (recomendado: 0.8 a 1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Nombre del archivo de salida (por defecto: outputs/angle_<tipo>.png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad (por defecto: 42)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Modo rápido (~1024px, 4x más veloz, por defecto)",
    )
    parser.add_argument(
        "--2k",
        dest="high_quality",
        action="store_true",
        help="Modo alta calidad 2K (~2048px)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device, dtype = get_target_device_and_dtype()
    cache_dir = PROJECT_DIR / "cache" / "huggingface"
    outputs_dir = PROJECT_DIR / "outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Definir nombre de salida
    if args.output:
        out_name = args.output
    else:
        out_name = f"angle_{args.angle}.png"

    out_path = Path(out_name)
    if out_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        out_path = out_path.with_suffix(".png")
    if not out_path.is_absolute():
        out_path = outputs_dir / out_path

    print("=" * 65)
    print("🎥 Qwen-Image-2.1-Turbo | Cambio de Ángulo de Cámara")
    print("=" * 65)
    print(f"[*] Imagen origen:          {args.image}")
    print(f"[*] Ángulo seleccionado:     {args.angle}")
    print(f"[*] Modo LoRA:              {'Sí (' + args.lora + ')' if args.lora else 'Modo Nativo 3D-Aware'}")
    print(f"[*] Dispositivo:            {device.upper()} (dtype: {dtype})")
    print(f"[*] Salida:                 {out_path}")

    # Cargar imagen de entrada
    print("\n[+] Cargando imagen de referencia...")
    try:
        resolved_img = resolve_path(args.image)
        input_image = load_image(resolved_img).convert("RGBA")
    except Exception as e:
        print(f"[!] Error al cargar la imagen: {e}")
        sys.exit(1)

    # Configurar prompt según uso de LoRA o modo nativo
    preset_info = ANGLE_PRESETS[args.angle]
    if args.custom_prompt:
        prompt = translate_and_enrich_prompt(args.custom_prompt, context_type="edit")
    elif args.lora:
        # Gramática oficial requerida por el LoRA Multiple-Angles
        prompt = preset_info["mva"]
        if args.close_up:
            prompt += " close-up"
    else:
        # Prompt descriptivo para el modo nativo
        prompt = (
            f"Using <image1> as reference, render the exact same person and scene from a {preset_info['desc']}. "
            f"Maintain the subject's exact facial identity, clothing, hair texture, and ambient lighting, "
            f"seamlessly adapting the perspective and 3D camera viewpoint. Photorealistic portrait."
        )

    print(f"\n[+] Prompt generado:")
    print(f"    \"{prompt}\"")

    # Cargar pipeline
    print("\n[+] Cargando modelo base Qwen-Image-2.1-Turbo...")
    pipe = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1-Turbo",
        dtype=dtype,
        cache_dir=str(cache_dir),
    ).to(device)

    # Cargar LoRA si está especificado
    if args.lora:
        print(f"\n[+] Cargando LoRA de control de cámara: {args.lora}...")
        print(f"[+] Archivo de pesos: {args.lora_weight_name} (fuerza: {args.lora_scale})")
        try:
            kwargs_lora = {"cache_dir": str(cache_dir)}
            if args.lora_weight_name:
                kwargs_lora["weight_name"] = args.lora_weight_name
            pipe.load_lora_weights(args.lora, **kwargs_lora)
            if hasattr(pipe, "set_adapters"):
                pipe.set_adapters(list(pipe.get_active_adapters()), adapter_weights=[args.lora_scale])
            print("[✓] LoRA de cámara cargado exitosamente.")
        except Exception as e:
            print(f"[!] Aviso al cargar LoRA: {e}")
            print("[!] Continuando la generación en modo nativo...")

    # Generación
    width, height = (2048, 2048) if args.high_quality else (1024, 1024)
    print(f"\n[+] Renderizando nuevo ángulo {width}x{height} (8 pasos Turbo)...")
    generator = torch.Generator("cpu").manual_seed(args.seed)

    result = pipe(
        prompt=prompt,
        image=input_image,
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
    print(f"\n[✓] ¡Nuevo ángulo generado con éxito!")
    print(f"[✓] Guardado en: {out_path}")

if __name__ == "__main__":
    main()
