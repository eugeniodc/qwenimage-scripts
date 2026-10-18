#!/usr/bin/env python3
"""
Script estándar para Cargar, Unir y Fusionar (Merge / Fuse) LoRAs en Qwen-Image-2.1-Turbo
Ubicación: scripts/merge_lora.py

Permite:
 1. Cargar cualquier LoRA (archivo local .safetensors de Civitai/disco o repositorio Hugging Face).
 2. Unir y fusionar sus pesos permanentemente en el modelo base con `pipe.fuse_lora(lora_scale=...)`.
 3. Opcionalmente guardar el modelo completo fusionado a disco (`--save-dir`) para reutilizarlo sin recargar el LoRA.
 4. Opcionalmente generar una imagen directa de prueba con el modelo fusionado (`--prompt`).
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

def get_target_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    else:
        return "cpu", torch.float32

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script estándar para fusionar (merge/fuse) LoRAs en Qwen-Image-2.1-Turbo"
    )
    parser.add_argument(
        "--lora",
        type=str,
        required=True,
        help="Ruta a archivo local (.safetensors) o identificador en Hugging Face (ej: 'Alissonerdx/BFS-Best-Face-Swap')",
    )
    parser.add_argument(
        "--lora-weight-name",
        type=str,
        default=None,
        help="Nombre del archivo de pesos en el repositorio HF (ej: 'bfs_head_v1_qwen_2.1.safetensors'). Opcional.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Fuerza o escala con la que se fusiona el LoRA en los pesos del modelo base (por defecto: 1.0)",
    )
    parser.add_argument(
        "--adapter-name",
        type=str,
        default="custom_adapter",
        help="Nombre interno temporal para registrar el adaptador antes de la fusión (por defecto: 'custom_adapter')",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen-Image-2.1-Turbo",
        help="Identificador o ruta del modelo base (por defecto: 'Qwen/Qwen-Image-2.1-Turbo')",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directorio local donde exportar y guardar permanentemente el modelo con los pesos ya fusionados (save_pretrained).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt de texto para probar la generación inmediatamente tras la fusión de pesos.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta local o URL a imagen de referencia para LoRAs de edición / multi-referencia.",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="1:1",
        choices=list(ASPECT_RATIOS.keys()),
        help="Relación de aspecto para la imagen generada (por defecto: 1:1)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Ancho personalizado en píxeles (anula el aspect ratio)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Alto personalizado en píxeles (anula el aspect ratio)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/fused_lora_output.png",
        help="Ruta de guardado de la imagen generada (por defecto: outputs/fused_lora_output.png)",
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

    print("=" * 70)
    print("🔀 Qwen-Image-2.1-Turbo | Fusión y Unión de LoRA (Merge & Fuse)")
    print("=" * 70)
    print(f"[*] Modelo base:             {args.base_model}")
    print(f"[*] LoRA a fusionar:         {args.lora}")
    print(f"[*] Escala de fusión:        {args.lora_scale}")
    print(f"[*] Dispositivo detectado:   {device.upper()} (dtype: {dtype})")
    print(f"[*] Caché local:             {cache_dir}")

    # 1. Cargar pipeline del modelo base
    print(f"\n[+] Cargando pipeline desde '{args.base_model}'...")
    try:
        pipe = QwenImage21Pipeline.from_pretrained(
            args.base_model,
            dtype=dtype,
            cache_dir=str(cache_dir),
        ).to(device)
        print("[✓] Modelo base cargado en memoria exitosamente.")
    except Exception as e:
        print(f"[!] Error al cargar el modelo base: {e}")
        sys.exit(1)

    # 2. Determinar origen del LoRA (archivo local vs Hugging Face)
    lora_input = Path(args.lora)
    if not lora_input.exists() and (PROJECT_DIR / lora_input).exists():
        lora_input = PROJECT_DIR / lora_input
    kwargs_lora = {"cache_dir": str(cache_dir)}

    if lora_input.is_file():
        # Archivo local directo (ej: /ruta/mi_lora.safetensors)
        print(f"\n[+] Detectado archivo LoRA local: {lora_input.resolve()}")
        load_target = str(lora_input.parent)
        kwargs_lora["weight_name"] = lora_input.name
    elif lora_input.is_dir():
        # Directorio local existente
        print(f"\n[+] Detectado directorio LoRA local: {lora_input.resolve()}")
        load_target = str(lora_input)
        if args.lora_weight_name:
            kwargs_lora["weight_name"] = args.lora_weight_name
    else:
        # Repositorio de Hugging Face
        print(f"\n[+] Detectado identificador de Hugging Face: {args.lora}")
        load_target = args.lora
        if args.lora_weight_name:
            kwargs_lora["weight_name"] = args.lora_weight_name

    # 3. Cargar pesos del LoRA
    print(f"[+] Inyectando pesos del adaptador '{args.adapter_name}'...")
    try:
        pipe.load_lora_weights(load_target, adapter_name=args.adapter_name, **kwargs_lora)
        print(f"[✓] Adaptador '{args.adapter_name}' cargado en la estructura del pipeline.")
    except Exception as e:
        print(f"[!] Error al cargar los pesos del LoRA: {e}")
        sys.exit(1)

    # 4. Fusionar (Fuse / Merge) los pesos de LoRA en el modelo base
    print(f"\n[+] Fusionando matemáticamente (fuse_lora) con factor de escala: {args.lora_scale}...")
    try:
        pipe.fuse_lora(lora_scale=args.lora_scale, adapter_names=[args.adapter_name])
        print("[✓] ¡FUSIÓN COMPLETADA!")
        print("    Los pesos del LoRA han sido incorporados permanentemente a los tensores base.")
        print("    Inferencia acelerada sin sobrecarga de capas adaptadoras.")
    except Exception as e:
        print(f"[!] Error durante la fusión (fuse_lora): {e}")
        sys.exit(1)

    # 5. Guardar modelo fusionado si se solicitó (--save-dir)
    if args.save_dir:
        save_target = Path(args.save_dir)
        if not save_target.is_absolute():
            save_target = PROJECT_DIR / save_target
        save_target.mkdir(parents=True, exist_ok=True)
        print(f"\n[+] Exportando modelo fusionado completo a: {save_target}...")
        try:
            pipe.save_pretrained(str(save_target))
            print(f"[✓] ¡Modelo exportado y guardado con éxito en: {save_target}!")
            print(f"    Ahora puedes cargarlo directamente sin LoRA usando:")
            print(f"    QwenImage21Pipeline.from_pretrained('{save_target}')")
        except Exception as e:
            print(f"[!] Error al exportar el modelo fusionado: {e}")

    # 6. Generación de prueba si se proporcionó un prompt
    if args.prompt:
        print("\n" + "=" * 70)
        print("🎨 Generando imagen de prueba con el modelo fusionado...")
        print("=" * 70)
        print(f"[*] Prompt: \"{args.prompt}\"")

        # Dimensiones
        if args.width and args.height:
            w, h = args.width, args.height
        else:
            w, h = ASPECT_RATIOS[args.aspect_ratio]
        print(f"[*] Resolución: {w}x{h} ({args.aspect_ratio})")

        # Cargar imagen de referencia si existe
        if args.image:
            image_p = Path(args.image)
            if not image_p.exists() and (PROJECT_DIR / image_p).exists():
                image_p = PROJECT_DIR / image_p
            print(f"[*] Cargando imagen de referencia: {image_p}...")
            try:
                ref_image = load_image(str(image_p)).convert("RGBA")
            except Exception as e:
                print(f"[!] Error al cargar la imagen de referencia: {e}")
                sys.exit(1)

        # Destino de salida
        out_path = Path(args.output)
        if out_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            out_path = out_path.with_suffix(".png")
        if not out_path.is_absolute():
            out_path = outputs_dir / out_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        generator = torch.Generator("cpu").manual_seed(args.seed)

        context_type = "edit" if ref_image is not None else "t2i"
        processed_prompt = translate_and_enrich_prompt(args.prompt, context_type=context_type)

        call_kwargs = {
            "prompt": processed_prompt,
            "width": w,
            "height": h,
            "num_inference_steps": 8,
            "use_kv_cache": True,
            "generator": generator,
        }
        if ref_image is not None:
            call_kwargs["image"] = ref_image

        try:
            result = pipe(**call_kwargs)
            final_img = result.images[0]
            final_img.save(out_path)
            print(f"\n[✓] Imagen generada con éxito:")
            print(f"    {out_path}")
        except Exception as e:
            print(f"[!] Error durante la generación: {e}")
            sys.exit(1)
    else:
        if not args.save_dir:
            print("\n💡 Sugerencias de uso:")
            print("  • Para generar una imagen de prueba con este LoRA fusionado:")
            print(f"    python scripts/merge_lora.py --lora \"{args.lora}\" --prompt \"tu descripción aquí\"")
            print("  • Para exportar y guardar permanentemente el modelo fusionado en disco:")
            print(f"    python scripts/merge_lora.py --lora \"{args.lora}\" --save-dir \"models/qwen_fused\"")

    print("\n[✓] Proceso finalizado.")

if __name__ == "__main__":
    main()
