#!/usr/bin/env python3
"""
Script de Intercambio de Caras (Face Swap / Identity Transfer)
utilizando Qwen-Image-2.1-Turbo (Multi-Reference Editing con 8 pasos Turbo).

Soporta dos modalidades:
 1. Modo Nativo: Utiliza el modelado multi-referencia base de Qwen-Image-2.1.
 2. Modo LoRA BFS (Best Face Swap): Carga el adaptador de la comunidad
    'Alissonerdx/BFS-Best-Face-Swap' (bfs_head_v1_qwen_2.1.safetensors)
    para máxima fidelidad facial, nitidez de piel y conservación de rasgos.

Ubicación: scripts/demo_face_swap.py
Guarda la imagen generada por defecto en: outputs/face_swap_output.png
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

# Preajustes rápidos (~1K, hasta 4x más veloz, recomendado por defecto)
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
        description="Intercambio de caras con Qwen-Image-2.1-Turbo (Multi-Reference + LoRA BFS)"
    )
    parser.add_argument(
        "--source-face",
        type=str,
        required=True,
        help="Ruta o URL de la imagen con la cara que deseas transferir (ej: sample_images/eugenio.jpg)",
    )
    parser.add_argument(
        "--target-image",
        type=str,
        required=True,
        help="Ruta o URL de la imagen de destino donde se colocará la cara (ej: sample_images/iron-man.avif)",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="Ruta o repositorio de LoRA para face swap (ej: 'Alissonerdx/BFS-Best-Face-Swap')",
    )
    parser.add_argument(
        "--lora-weight-name",
        type=str,
        default="bfs_head_v1_qwen_2.1.safetensors",
        help="Nombre del archivo de pesos LoRA en el repositorio (por defecto: bfs_head_v1_qwen_2.1.safetensors)",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Fuerza del LoRA (típicamente 0.8 a 1.0)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt personalizado. Si se omite, se usa el prompt estándar optimizado para face-swap",
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
        default="face_swap_output.png",
        help="Nombre del archivo de salida (se guardará en la carpeta outputs/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad (por defecto: 42)",
    )
    parser.add_argument(
        "--invert-roles",
        action="store_true",
        help="Invertir el orden de las imágenes (<image1> como cara y <image2> como destino)",
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
    ratios = ASPECT_RATIOS if args.high_quality else ASPECT_RATIOS_FAST
    width, height = ratios[args.aspect_ratio]
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

    print("=" * 65)
    print("🎭 Qwen-Image-2.1-Turbo | Face Swap / Identity Transfer")
    print("=" * 65)
    print(f"[*] Directorio raíz:           {PROJECT_DIR}")
    print(f"[*] Dispositivo:              {device.upper()} (dtype: {dtype})")
    print(f"[*] Rostro a transferir:      {args.source_face}")
    print(f"[*] Escena de destino base:   {args.target_image}")
    print(f"[*] Modo LoRA:                {'Sí (' + str(args.lora) + ')' if args.lora else 'Modo Nativo'}")
    print(f"[*] Resolución:               {args.aspect_ratio} ({width}x{height})")
    print(f"[*] Semilla:                  {args.seed}")
    print(f"[*] Archivo de salida:        {out_path}")

    # Cargar imágenes de referencia
    print("\n[+] Cargando imágenes de referencia...")
    try:
        source_img = load_image(resolve_path(args.source_face)).convert("RGB")
        target_img = load_image(resolve_path(args.target_image)).convert("RGB")
    except Exception as e:
        print(f"[!] Error al cargar las imágenes: {e}")
        sys.exit(1)

    # <image1> = Lienzo base (target_img)
    # <image2> = Identidad y rostro que se transfiere (source_img)
    if not args.invert_roles:
        images_order = [target_img, source_img]
        image1_desc = f"Escena base (<image1>): {args.target_image}"
        image2_desc = f"Rostro a transferir (<image2>): {args.source_face}"
        
        if args.lora:
            # Prompt con trigger 'head_swap:' para el LoRA BFS
            default_prompt = (
                "head_swap: In <image1>, keep the exact body posture, clothing, lighting, "
                "and entire background unchanged. Replace the head and face of the character in <image1> "
                "with the exact facial features, eyes, nose, mouth, and likeness from <image2>. "
                "Seamless skin tone blending, sharp facial pores and details, photorealistic 8k."
            )
        else:
            default_prompt = (
                "In the target scene <image1>, keep the exact body posture, clothing, hairstyle, lighting, "
                "and entire background from <image1> unchanged. Replace only the face of the character in <image1> "
                "with the facial features, eyes, nose, mouth, and identity of the person from <image2>. "
                "The face from <image2> must blend seamlessly and naturally into the head in <image1>, "
                "matching the perspective, skin tone and lighting of <image1>. Photorealistic, high quality."
            )
    else:
        images_order = [source_img, target_img]
        image1_desc = f"Rostro (<image1>): {args.source_face}"
        image2_desc = f"Escena base (<image2>): {args.target_image}"
        default_prompt = (
            "Using <image1> as the facial identity reference and <image2> as the target portrait reference, "
            "replace the face of the subject in <image2> with the facial features and identity of the person in <image1>. "
            "Keep the pose, clothes and background of <image2>. Seamless and photorealistic."
        )

    print(f"[+] {image1_desc}")
    print(f"[+] {image2_desc}")

    prompt = args.prompt if args.prompt else default_prompt
    print(f"\n[+] Prompt de fusión:")
    print(f"    \"{prompt}\"")

    # Cargar el pipeline
    print("\n[+] Cargando modelo base Qwen-Image-2.1-Turbo...")
    pipe = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1-Turbo",
        dtype=dtype,
        cache_dir=str(cache_dir),
    ).to(device)

    # Cargar adaptador LoRA si está activado
    if args.lora:
        print(f"\n[+] Cargando LoRA especializado de Face Swap: {args.lora}...")
        try:
            kwargs_lora = {"cache_dir": str(cache_dir)}
            if args.lora_weight_name:
                kwargs_lora["weight_name"] = args.lora_weight_name
            pipe.load_lora_weights(args.lora, **kwargs_lora)
            if hasattr(pipe, "set_adapters"):
                pipe.set_adapters(list(pipe.get_active_adapters()), adapter_weights=[args.lora_scale])
            print("[✓] LoRA de Face Swap cargado exitosamente.")
        except Exception as e:
            print(f"[!] Advertencia al cargar el archivo de LoRA: {e}")
            print("[!] Continuando la generación en modo multi-referencia nativo...")

    # Inferencia
    print("\n[+] Procesando intercambio de caras (8 pasos Turbo)...")
    generator = torch.Generator("cpu").manual_seed(args.seed)

    result = pipe(
        prompt=prompt,
        image=images_order,
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
    print(f"\n[✓] ¡Intercambio completado exitosamente!")
    print(f"[✓] Imagen guardada en: {out_path}")

if __name__ == "__main__":
    main()
