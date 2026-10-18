#!/usr/bin/env python3
"""
Script para Unir Múltiples Personas en una Misma Foto (Group Photo / Multi-Person Merge)
utilizando Qwen-Image-2.1-Turbo (Multi-Reference Editing con 8 pasos Turbo).

Permite combinar 2 o más fotos de personas para que parezcan que estuvieron juntos:
  1. Modo Nueva Escena (new_scene):
     Toma la identidad de <image1> y <image2> (y opcionalmente más) y los sitúa juntos
     en un entorno nuevo (cafetería, playa, parque, cena, etc.).
  2. Modo Conservar Fondo (<image1> como base):
     Conserva la persona y el fondo de <image1> y añade a la persona de <image2> a su lado,
     ajustando la iluminación, sombras y perspectiva para máxima coherencia.

Ubicación: scripts/demo_group_photo.py
Guarda la imagen generada por defecto en: outputs/group_photo_output.png
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
        description="Unir varias fotos de personas en una sola con Qwen-Image-2.1-Turbo"
    )
    parser.add_argument(
        "--person1",
        type=str,
        default=None,
        help="Ruta a la foto de la primera persona (<image1>)",
    )
    parser.add_argument(
        "--person2",
        type=str,
        default=None,
        help="Ruta a la foto de la segunda persona (<image2>)",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Lista de rutas de imágenes de entrada (alternativa a --person1 y --person2)",
    )
    parser.add_argument(
        "--mode",
        choices=["new_scene", "keep_scene"],
        default="new_scene",
        help="Modo de composición: 'new_scene' (los sitúa en un nuevo entorno) o 'keep_scene' (mantiene el fondo y persona de <image1>)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="at a stylish cozy cafe, sitting together enjoying coffee, warm ambient lighting",
        help="Descripción del entorno o escenario donde estarán juntos (para modo 'new_scene')",
    )
    parser.add_argument(
        "--interaction",
        type=str,
        default="standing shoulder to shoulder, smiling warmly, looking at the camera",
        help="Descripción de la pose o interacción entre ellos",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt personalizado completo. Si se especifica, anula la generación automática de prompt.",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        choices=list(ASPECT_RATIOS.keys()),
        default="4:3",
        help="Relación de aspecto (1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16). Por defecto: 4:3",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Modo rápido: genera a ~1K de resolución para acelerar hasta 4x",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Número de pasos de inferencia (por defecto: 8 pasos Turbo)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="group_photo_output.png",
        help="Nombre del archivo de salida (se guardará en outputs/)",
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

    # Recoger lista de imágenes
    input_paths = []
    if args.images:
        input_paths = args.images
    else:
        if args.person1:
            input_paths.append(args.person1)
        if args.person2:
            input_paths.append(args.person2)

    if len(input_paths) < 2:
        print("[!] Error: Debes proporcionar al menos 2 imágenes para unir.")
        print("    Ejemplo: --person1 sample_images/eugenio.jpg --person2 sample_images/mujer.avif")
        print("    O bien:  --images foto1.jpg foto2.jpg [foto3.jpg ...]")
        sys.exit(1)

    ratio_dict = ASPECT_RATIOS_FAST if args.fast else ASPECT_RATIOS
    width, height = ratio_dict[args.aspect_ratio]
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
    print("👥 Qwen-Image-2.1-Turbo | Unir Personas (Group Photo Merge)")
    print("=" * 65)
    print(f"[*] Directorio raíz:        {PROJECT_DIR}")
    print(f"[*] Dispositivo:            {device.upper()} | Tipo de dato: {dtype}")
    print(f"[*] Modo de unión:          {args.mode}")
    print(f"[*] Perfil:                 {'Rápido (~1K)' if args.fast else 'Alta Calidad (~2K)'}")
    print(f"[*] Relación de aspecto:    {args.aspect_ratio} ({width}x{height})")
    print(f"[*] Semilla:                {args.seed}")
    print(f"[*] Archivo de salida:      {out_path}")
    print(f"[*] Número de personas:     {len(input_paths)}")

    # Cargar imágenes de referencia
    loaded_images = []
    print("\n[+] Cargando imágenes de referencia...")
    for idx, path_str in enumerate(input_paths, 1):
        resolved = resolve_path(path_str)
        print(f"    <image{idx}>: {resolved}")
        try:
            img = load_image(resolved).convert("RGB")
            loaded_images.append(img)
        except Exception as e:
            print(f"[!] Error al cargar la imagen {path_str}: {e}")
            sys.exit(1)

    # Construir prompt optimizado
    if args.prompt:
        final_prompt = translate_and_enrich_prompt(args.prompt, context_type="edit")
    else:
        # Traducir y enriquecer la escena e interacción si son provistas
        scene_desc = translate_and_enrich_prompt(args.scene, context_type="scene") if args.scene else ""
        interaction_desc = translate_and_enrich_prompt(args.interaction, context_type="interaction") if args.interaction else ""
        if args.mode == "keep_scene":
            final_prompt = (
                f"In <image1>, keep the person and the entire background scenery. "
                f"Seamlessly add the person from <image2> {interaction_desc}. "
                f"Match the exact lighting, shadows, colors, perspective, and camera depth of field of <image1>. "
                f"Natural skin tones, sharp facial features matching <image2>, realistic photorealistic integration."
            )
        else:  # new_scene
            if len(loaded_images) == 2:
                final_prompt = (
                    f"A authentic photograph of the person from <image1> and the person from <image2> "
                    f"together in the same scene. They are {interaction_desc}, {scene_desc}. "
                    f"Preserve the exact facial features, likeness, and identities of both individuals from <image1> and <image2>. "
                    f"Harmonious natural lighting, realistic cast shadows, matching camera depth of field, sharp focus."
                )
            else:
                refs = ", ".join([f"<image{i}>" for i in range(1, len(loaded_images) + 1)])
                final_prompt = (
                    f"A authentic photograph of the people from {refs} together in the same scene. "
                    f"They are {interaction_desc}, {scene_desc}. "
                    f"Preserve the exact facial features, likeness, and identities of each person from their reference images. "
                    f"Harmonious ambient lighting, realistic cast shadows, sharp focus."
                )

    print(f"\n[+] Prompt generado:")
    print(f"    \"{final_prompt}\"")

    print("\n[+] Cargando modelo base Qwen-Image-2.1-Turbo...")
    pipe = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1-Turbo",
        dtype=dtype,
        cache_dir=str(cache_dir),
    ).to(device)

    print("\n[+] Procesando fusión de personas (8 pasos Turbo)...")
    generator = torch.Generator("cpu").manual_seed(args.seed)

    result = pipe(
        prompt=final_prompt,
        image=loaded_images,
        width=width,
        height=height,
        num_inference_steps=args.steps,
        use_kv_cache=True,
        generator=generator,
    )

    output_image = result.images[0]
    try:
        output_image.save(out_path)
    except Exception:
        output_image.save(out_path, format="PNG")

    print(f"\n[✓] ¡Composición creada exitosamente!")
    print(f"[✓] Guardada en: {out_path}")

if __name__ == "__main__":
    main()
