#!/usr/bin/env python3
"""
Demostración oficial de Image Editing para Qwen-Image-2.1-Turbo
Ubicación: scripts/demo_image_editing.py
Guarda la imagen generada en: outputs/edit.png
"""

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

def get_target_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    else:
        return "cpu", torch.float32

def main():
    device, dtype = get_target_device_and_dtype()
    cache_dir = PROJECT_DIR / "cache" / "huggingface"
    outputs_dir = PROJECT_DIR / "outputs"
    sample_dir = PROJECT_DIR / "sample_images"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("🖌️  Qwen-Image-2.1-Turbo | Image Editing")
    print("=" * 65)
    print(f"[*] Directorio raíz:        {PROJECT_DIR}")
    print(f"[*] Caché local de modelos: {cache_dir}")
    print(f"[*] Carpeta de salida:      {outputs_dir}")
    print(f"[*] Dispositivo detectado:  {device.upper()} (dtype: {dtype})")

    print("\n[+] Cargando pipeline QwenImage21Pipeline desde Qwen/Qwen-Image-2.1-Turbo...")
    pipe = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1-Turbo",
        dtype=dtype,
        cache_dir=str(cache_dir),
    ).to(device)

    # Imagen de entrada oficial
    input_image_url = 'https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/resolve/main/assets/turbo-boat-input.png'
    print(f"[+] Descargando imagen de entrada de referencia: {input_image_url}")
    input_image = load_image(input_image_url).convert("RGBA")

    # Guardar copia de muestra en sample_images si no existe
    sample_copy = sample_dir / "turbo-boat-input.png"
    if not sample_copy.exists():
        input_image.save(sample_copy)
        print(f"[+] Muestra guardada localmente en: {sample_copy}")

    edit_prompt = (
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

    print("\n[+] Prompt de edición:")
    print(f"    \"{edit_prompt[:120]}...\"")
    print("\n[+] Procesando edición (resolución 2048x2048, 8 pasos Turbo)...")

    image = pipe(
        prompt=edit_prompt,
        image=input_image,
        width=2048,
        height=2048,
        num_inference_steps=8,
        use_kv_cache=True,
        generator=torch.Generator("cpu").manual_seed(20261009),
    ).images[0]

    output_file = outputs_dir / "edit.png"
    image.save(output_file)
    print(f"\n[✓] Imagen guardada exitosamente en la carpeta de construidas:")
    print(f"    {output_file}")

if __name__ == "__main__":
    main()
