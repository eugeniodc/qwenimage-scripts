#!/usr/bin/env python3
"""
Demostración oficial de Text-to-Image para Qwen-Image-2.1-Turbo
Ubicación: scripts/demo_text_to_image.py
Guarda la imagen generada en: outputs/t2i.png
"""

import argparse
import os
import sys
from pathlib import Path

# Directorio raíz del proyecto (un nivel arriba de scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Configurar aislamiento local de caché en cache/
os.environ["HF_HOME"] = str(PROJECT_DIR / "cache" / "huggingface")
os.environ["TORCH_HOME"] = str(PROJECT_DIR / "cache" / "torch")

import torch
from diffusers import QwenImage21Pipeline
from core.prompt_utils import translate_and_enrich_prompt

def get_target_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    else:
        return "cpu", torch.float32

def main():
    parser = argparse.ArgumentParser(description="Demo Text-to-Image para Qwen-Image-2.1-Turbo")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt de generación (si se omite, usa la muestra oficial)")
    parser.add_argument("--output", type=str, default="t2i.png", help="Archivo de salida")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    args = parser.parse_args()

    device, dtype = get_target_device_and_dtype()
    cache_dir = PROJECT_DIR / "cache" / "huggingface"
    outputs_dir = PROJECT_DIR / "outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output)
    if out_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        out_path = out_path.with_suffix(".png")
    if not out_path.is_absolute():
        out_path = outputs_dir / out_path.name

    default_prompt = (
        "A vertically oriented educational chemistry study poster fills the frame, "
        "designed like a hand-drawn classroom notes page with a clean white background, "
        "thick dark navy-blue outer border, and thin inner navy rectangular border lines. "
        "The overall aesthetic is colorful, high-contrast, and hand-lettered in a signature Topper style "
        "using marker and watercolor effects. Centered across the top is the main lesson heading in large "
        "stylized black-and-yellow bold lettering: \"Science – Chapter 1:\" followed below by the much larger "
        "title \"Chemical Reactions and Equations\". Inside an equation panel, the reaction is written very large "
        "as \"Fe + CuSO₄ → FeSO₄ + Cu\". Realistic glassware illustrations, test tubes, and molecular diagrams."
    )

    if args.prompt:
        prompt = translate_and_enrich_prompt(args.prompt, context_type="t2i")
    else:
        prompt = default_prompt

    print("=" * 65)
    print("🎨 Qwen-Image-2.1-Turbo | Text-to-Image")
    print("=" * 65)
    print(f"[*] Directorio raíz:        {PROJECT_DIR}")
    print(f"[*] Caché local de modelos: {cache_dir}")
    print(f"[*] Carpeta de salida:      {outputs_dir}")
    print(f"[*] Dispositivo detectado:  {device.upper()} (dtype: {dtype})")
    print(f"[*] Prompt aplicado:        {prompt[:120]}...")
    print(f"[*] Archivo de salida:      {out_path}")

    print("\n[+] Cargando pipeline QwenImage21Pipeline desde Qwen/Qwen-Image-2.1-Turbo...")
    pipe = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1-Turbo",
        dtype=dtype,
        cache_dir=str(cache_dir),
    ).to(device)

    print("\n[+] Generando imagen (resolución 1680x2512, 8 pasos Turbo)...")
    image = pipe(
        prompt=prompt,
        width=1680,
        height=2512,
        num_inference_steps=8,
        use_kv_cache=True,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]

    image.save(out_path)
    print(f"\n[✓] Imagen guardada exitosamente en la carpeta de construidas:")
    print(f"    {out_path}")

if __name__ == "__main__":
    main()
