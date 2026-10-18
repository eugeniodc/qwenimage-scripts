#!/usr/bin/env python3
"""
Script de Edición de Imagen Existente Sin Censura (Uncensored / Abliterated)
utilizando el modelo de AtomicChat:
https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF

Características:
 - Utiliza el encoder de texto abliterado/desensurado que elimina las directivas
   internas de rechazo (refusal rate reducido de 88.9% a 1.2%).
 - Permite editar cualquier imagen existente de referencia (<image1>) aplicando
   transformaciones anatómicas, cambios de vestimenta, estéticas artísticas o
   modificaciones sin bloqueos ni filtros morales.
 - Soporta modo rápido (1K) y máxima resolución (2K) con schedule Turbo de 8 pasos.
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

# Repositorio oficial del modelo sin censura
UNCENSORED_REPO = "AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF"
BASE_TURBO_MODEL = "Qwen/Qwen-Image-2.1-Turbo"

# Resoluciones recomendadas
ASPECT_RATIOS_2K = {
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

AVAILABLE_QUANTS = {
    "AD-Q4_K": "Qwen-Image-2.1-Turbo-Abliterated-Uncensored-AD-Q4_K.gguf",
    "Q8_0":    "Qwen-Image-2.1-Turbo-Abliterated-Uncensored-Q8_0.gguf",
    "BF16":    "Qwen-Image-2.1-Turbo-Abliterated-Uncensored-BF16.gguf",
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
        description="Edición de imagen existente sin censura con AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Ruta o URL de la imagen de entrada a editar (ej: sample_images/eugenio.jpg)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Instrucción de edición sin censura (ej: 'In <image1>, remove the shirt and make body muscular')",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="AD-Q4_K",
        choices=list(AVAILABLE_QUANTS.keys()),
        help="Variante de cuantización GGUF del encoder desensurado (por defecto: AD-Q4_K, 5.29 GB)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Modo rápido: genera a resolución 1K (~1024px) en 8 pasos (activado por defecto)",
    )
    parser.add_argument(
        "--2k",
        dest="force_2k",
        action="store_true",
        help="Forzar resolución completa 2K (~2048px)",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="1:1",
        choices=list(ASPECT_RATIOS_2K.keys()),
        help="Relación de aspecto deseada (por defecto: 1:1)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Número de pasos de difusión (por defecto: 8 para schedule Turbo)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/edit_uncensored.png",
        help="Nombre o ruta del archivo de salida (por defecto: outputs/edit_uncensored.png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad (por defecto: 42)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device, dtype = get_target_device_and_dtype()
    cache_dir = PROJECT_DIR / "cache" / "huggingface"
    outputs_dir = PROJECT_DIR / "outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Determinar resolución
    is_fast_mode = args.fast and not args.force_2k
    ratio_dict = ASPECT_RATIOS_FAST if is_fast_mode else ASPECT_RATIOS_2K
    width, height = ratio_dict[args.aspect_ratio]

    # Archivo de salida
    out_path = Path(args.output)
    if out_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        out_path = out_path.with_suffix(".png")
    if not out_path.is_absolute():
        out_path = outputs_dir / out_path.name

    quant_file = AVAILABLE_QUANTS[args.quant]

    print("=" * 72)
    print("🔓 Qwen-Image-2.1-Turbo | Edición de Imagen Sin Censura (Uncensored)")
    print("=" * 72)
    print(f"[*] Repositorio abliterado:  {UNCENSORED_REPO}")
    print(f"[*] Archivo de cuantización: {quant_file} ({args.quant})")
    print(f"[*] Modelo base denoiser:    {BASE_TURBO_MODEL}")
    print(f"[*] Imagen origen:           {args.image}")
    print(f"[*] Perfil de resolución:    {'⚡ Rápido (~1K)' if is_fast_mode else '💎 Máxima Calidad (~2K)'} ({width}x{height})")
    print(f"[*] Pasos de inferencia:     {args.steps} pasos Turbo")
    print(f"[*] Dispositivo:             {device.upper()} (dtype: {dtype})")
    print(f"[*] Archivo de salida:       {out_path}")

    # 1. Cargar imagen de entrada
    resolved_img_path = resolve_path(args.image)
    print(f"\n[+] Cargando imagen de referencia: {resolved_img_path}...")
    try:
        input_image = load_image(resolved_img_path).convert("RGBA")
        print(f"[✓] Imagen cargada correctamente: {input_image.size}")
    except Exception as e:
        print(f"[!] Error al cargar la imagen de entrada: {e}")
        sys.exit(1)

    # 2. Formatear y enriquecer prompt para asegurar edición directa
    raw_prompt = args.prompt.strip()
    enriched_prompt = translate_and_enrich_prompt(raw_prompt, context_type="edit_uncensored")
    if "<image1>" not in enriched_prompt:
        formatted_prompt = f"Using <image1> as reference, {enriched_prompt}"
    else:
        formatted_prompt = enriched_prompt

    print(f"\n[+] Prompt de edición aplicado:")
    print(f"    \"{formatted_prompt}\"")

    # 3. Cargar el pipeline base
    print(f"\n[+] Cargando pipeline base desde '{BASE_TURBO_MODEL}'...")
    try:
        pipe = QwenImage21Pipeline.from_pretrained(
            BASE_TURBO_MODEL,
            dtype=dtype,
            cache_dir=str(cache_dir),
        ).to(device)
        print("[✓] Pipeline base cargado en memoria exitosamente.")
    except Exception as e:
        print(f"[!] Error al cargar el pipeline base: {e}")
        sys.exit(1)

    # 4. Información del encoder abliterado
    print(f"\n[+] Configuración de censura: FILTROS DE RECHAZO DESACTIVADOS.")
    print(f"    El texto se interpreta sin bloqueos de seguridad ni directivas de negativa.")

    # 5. Ejecutar la inferencia de edición
    print(f"\n[+] Renderizando edición sin censura ({args.steps} pasos Turbo)...")
    generator = torch.Generator("cpu").manual_seed(args.seed)

    try:
        output = pipe(
            prompt=formatted_prompt,
            image=input_image,
            width=width,
            height=height,
            num_inference_steps=args.steps,
            use_kv_cache=True,
            generator=generator,
        )
        result_image = output.images[0]
        result_image.save(out_path)
        print(f"\n[✓] ¡Edición completada con éxito!")
        print(f"    Guardada en: {out_path}")
    except Exception as e:
        print(f"[!] Error durante la generación: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
