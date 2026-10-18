#!/usr/bin/env python3
"""
Demostración oficial de Image Editing para Qwen-Image-2.1-Turbo
Ubicación: scripts/demo_image_editing.py
Guarda la imagen generada en: outputs/edit.png
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
from diffusers.utils import load_image
from core.prompt_utils import translate_and_enrich_prompt

def get_target_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    else:
        return "cpu", torch.float32

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

def main():
    parser = argparse.ArgumentParser(description="Demo Image Editing para Qwen-Image-2.1-Turbo")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt de edición (si se omite, usa la muestra oficial)")
    parser.add_argument("--image", type=str, default=None, help="Ruta o URL de imagen de referencia")
    parser.add_argument("--output", type=str, default="edit.png", help="Archivo de salida")
    parser.add_argument("--seed", type=int, default=20261009, help="Semilla para reproducibilidad")
    args = parser.parse_args()

    device, dtype = get_target_device_and_dtype()
    cache_dir = PROJECT_DIR / "cache" / "huggingface"
    outputs_dir = PROJECT_DIR / "outputs"
    sample_dir = PROJECT_DIR / "sample_images"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output)
    if out_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        out_path = out_path.with_suffix(".png")
    if not out_path.is_absolute():
        out_path = outputs_dir / out_path.name

    default_prompt = (
        "Create a photorealistic photograph of the motor yacht from the input sketch, sailing on open sea, "
        "captured in full side profile with the camera at water level. Build the vessel with the structure shown "
        "in the input sketch: a long low hull; one tall mast stepped forward with a small cross-tree platform near "
        "its top; a wheelhouse superstructure with an open bridge wing just behind the mast; a midships deckhouse "
        "carrying a row of four rectangular windows with a lifeboat stowed on its roof; a rectangular funnel behind "
        "that deckhouse; an aft deckhouse with a row of three rectangular windows and a second lifeboat above it; "
        "thin railings along every deck edge. Paint the hull black, the superstructures and the mast white, and the funnel yellow. "
        "Set the yacht broadside on a calm deep-blue sea with a low gentle swell and a thin white wake trailing from "
        "the bow, under a clear sky with soft scattered clouds; sunlight from the front left lights the white "
        "superstructures and leaves a bright reflection path on the water beside the hull."
    )

    if args.prompt:
        edit_prompt = translate_and_enrich_prompt(args.prompt, context_type="edit")
    else:
        edit_prompt = default_prompt

    print("=" * 65)
    print("🖌️  Qwen-Image-2.1-Turbo | Image Editing")
    print("=" * 65)
    print(f"[*] Directorio raíz:        {PROJECT_DIR}")
    print(f"[*] Caché local de modelos: {cache_dir}")
    print(f"[*] Carpeta de salida:      {outputs_dir}")
    print(f"[*] Dispositivo detectado:  {device.upper()} (dtype: {dtype})")
    print(f"[*] Prompt aplicado:        {edit_prompt[:120]}...")
    print(f"[*] Archivo de salida:      {out_path}")

    # Imagen de entrada
    if args.image:
        resolved_img = resolve_path(args.image)
        print(f"[+] Cargando imagen de referencia: {resolved_img}")
        input_image = load_image(resolved_img).convert("RGBA")
    else:
        input_image_url = 'https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/resolve/main/assets/turbo-boat-input.png'
        print(f"[+] Descargando imagen de entrada de referencia: {input_image_url}")
        input_image = load_image(input_image_url).convert("RGBA")
        sample_copy = sample_dir / "turbo-boat-input.png"
        if not sample_copy.exists():
            input_image.save(sample_copy)
            print(f"[+] Muestra guardada localmente en: {sample_copy}")

    print("\n[+] Cargando pipeline QwenImage21Pipeline desde Qwen/Qwen-Image-2.1-Turbo...")
    pipe = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1-Turbo",
        dtype=dtype,
        cache_dir=str(cache_dir),
    ).to(device)

    print("\n[+] Procesando edición (resolución 2048x2048, 8 pasos Turbo)...")
    image = pipe(
        prompt=edit_prompt,
        image=input_image,
        width=2048,
        height=2048,
        num_inference_steps=8,
        use_kv_cache=True,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]

    image.save(out_path)
    print(f"\n[✓] Imagen guardada exitosamente en la carpeta de construidas:")
    print(f"    {out_path}")

if __name__ == "__main__":
    main()
