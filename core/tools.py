"""
core/tools.py
Módulo unificado de herramientas para Qwen-Image-2.1-Turbo.
Permite invocar todas las operaciones visuales de forma programática y tipada.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.prompt_utils import translate_and_enrich_prompt

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
SAMPLE_DIR = PROJECT_DIR / "sample_images"
CACHE_DIR = PROJECT_DIR / "cache"
VENV_DIR = PROJECT_DIR / "venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"

# Asegurar carpetas clave
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_python_executable() -> str:
    """Devuelve el ejecutable de Python del venv si existe, o el actual."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def list_sample_images() -> List[str]:
    """Lista los nombres de archivos disponibles en sample_images/"""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"}
    return sorted([f.name for f in SAMPLE_DIR.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])


def list_outputs() -> List[str]:
    """Lista las imágenes generadas en outputs/, ordenadas de más reciente a más antigua."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [f for f in OUTPUTS_DIR.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return [f.name for f in files]


def get_latest_output() -> Optional[str]:
    """Devuelve la ruta absoluta de la última imagen generada en outputs/."""
    outputs = list_outputs()
    if outputs:
        return str((OUTPUTS_DIR / outputs[0]).resolve())
    return None


def resolve_image_path(name_or_path: str) -> str:
    """
    Resuelve una referencia de imagen a una ruta absoluta existente.
    Soporta:
      - 'last', 'ultima', 'anterior' -> última imagen generada en outputs/
      - Nombre de archivo en sample_images/ (ej: 'eugenio.jpg' o 'eugenio')
      - Nombre de archivo en outputs/ (ej: 'mi_foto.png')
      - Ruta relativa o absoluta directa
    """
    if not name_or_path or not name_or_path.strip():
        raise ValueError("Se requiere una ruta o nombre de imagen válido.")

    cleaned = name_or_path.strip()
    lower = cleaned.lower()

    if lower in {"last", "latest", "ultima", "última", "anterior", "actual"}:
        latest = get_latest_output()
        if latest:
            return latest
        raise FileNotFoundError("No hay ninguna imagen previa en outputs/ para referenciar.")

    p = Path(cleaned)
    if p.is_file():
        return str(p.resolve())

    # Buscar en sample_images/
    sample_candidate = SAMPLE_DIR / cleaned
    if sample_candidate.is_file():
        return str(sample_candidate.resolve())

    # Probar con extensiones comunes en sample_images/
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".avif"]:
        with_ext = SAMPLE_DIR / f"{cleaned}{ext}"
        if with_ext.is_file():
            return str(with_ext.resolve())

    # Buscar en outputs/
    output_candidate = OUTPUTS_DIR / cleaned
    if output_candidate.is_file():
        return str(output_candidate.resolve())

    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        with_ext = OUTPUTS_DIR / f"{cleaned}{ext}"
        if with_ext.is_file():
            return str(with_ext.resolve())

    # Si es una URL o ruta inexistente, devolverla tal cual para que el script la gestione
    return str(p.resolve())


def _generate_output_name(prefix: str, custom_name: Optional[str] = None) -> str:
    """Genera un nombre de archivo de salida seguro en outputs/."""
    if custom_name and custom_name.strip():
        name = custom_name.strip()
        p = Path(name)
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            name = f"{name}.png"
        return name
    ts = int(time.time())
    return f"{prefix}_{ts}.png"


def _run_script_subprocess(args: List[str]) -> Dict[str, Any]:
    """Ejecuta un script en un subproceso aislado usando el entorno virtual."""
    python_bin = get_python_executable()
    cmd = [python_bin] + args

    env = os.environ.copy()
    env["HF_HOME"] = str(CACHE_DIR / "huggingface")
    env["TORCH_HOME"] = str(CACHE_DIR / "torch")
    env["PIP_CACHE_DIR"] = str(CACHE_DIR / "pip")
    env["VIRTUAL_ENV"] = str(VENV_DIR)
    env["PATH"] = f"{VENV_DIR}/bin:{env.get('PATH', '')}"

    print(f"\n[core.tools] 🚀 Ejecutando: {' '.join(cmd)}")
    start_time = time.time()
    res = subprocess.run(cmd, cwd=str(PROJECT_DIR), env=env, capture_output=False)
    elapsed = time.time() - start_time

    if res.returncode != 0:
        return {
            "success": False,
            "returncode": res.returncode,
            "error": f"El proceso terminó con código de error {res.returncode}",
            "elapsed_seconds": round(elapsed, 2)
        }

    return {
        "success": True,
        "returncode": 0,
        "elapsed_seconds": round(elapsed, 2)
    }


# ==============================================================================
# HERRAMIENTAS VISUALES DE QWEN-IMAGE-2.1
# ==============================================================================

def generate_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    fast: bool = True,
    output_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Genera una nueva imagen desde una descripción en texto (Text-to-Image).
    
    Args:
        prompt: Descripción detallada en inglés (traducción enriquecida con iluminación, atmósfera y estilo).
        aspect_ratio: Relación de aspecto ('1:1', '16:9', '9:16', '4:3', '3:4', '3:2', '2:3').
        fast: True para modo Turbo 1K (~1024px, rápido), False para máxima calidad 2K (~2048px).
        output_name: Nombre opcional para el archivo de salida en outputs/.
    """
    out_file = _generate_output_name("t2i", output_name)
    processed_prompt = translate_and_enrich_prompt(prompt, context_type="t2i")
    args = [
        str(SCRIPTS_DIR / "demo_cli.py"),
        "--prompt", processed_prompt,
        "--aspect-ratio", aspect_ratio,
        "--output", out_file,
    ]
    if fast:
        args.append("--fast")

    result = _run_script_subprocess(args)
    target_path = OUTPUTS_DIR / out_file
    result["output_file"] = out_file
    result["output_path"] = str(target_path.resolve()) if target_path.exists() else None
    return result


def edit_image(
    image_path: str,
    prompt: str,
    uncensored: bool = False,
    fast: bool = True,
    output_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Edita una imagen existente aplicando transformaciones descritas por texto.
    
    Args:
        image_path: Ruta o nombre de la imagen a editar (o 'last' para la última).
        prompt: Instrucción en inglés enriquecida con detalles de iluminación y coherencia.
        uncensored: True para usar el modelo abliterado sin filtros morales (AtomicChat GGUF).
        fast: True para modo rápido 1K, False para resolución 2K.
        output_name: Nombre opcional de salida.
    """
    resolved_src = resolve_image_path(image_path)
    out_file = _generate_output_name("edit_uncensored" if uncensored else "edit", output_name)
    processed_prompt = translate_and_enrich_prompt(prompt, context_type="edit_uncensored" if uncensored else "edit")

    if uncensored:
        args = [
            str(SCRIPTS_DIR / "demo_image_editing_uncensored.py"),
            "--image", resolved_src,
            "--prompt", processed_prompt,
            "--output", out_file,
        ]
        if not fast:
            args.append("--2k")
    else:
        args = [
            str(SCRIPTS_DIR / "demo_cli.py"),
            "--image", resolved_src,
            "--prompt", processed_prompt,
            "--output", out_file,
        ]
        if fast:
            args.append("--fast")

    result = _run_script_subprocess(args)
    target_path = OUTPUTS_DIR / out_file
    result["output_file"] = out_file
    result["output_path"] = str(target_path.resolve()) if target_path.exists() else None
    return result


def face_swap(
    source_face: str,
    target_image: str,
    use_lora: bool = True,
    fast: bool = True,
    output_name: Optional[str] = None,
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transfiere el rostro de una persona (source_face) al cuerpo/escena de otra imagen (target_image).
    
    Args:
        source_face: Imagen con la cara de origen (ej: 'eugenio.jpg').
        target_image: Imagen con el cuerpo/escena destino (ej: 'iron-man-2008-1626678790.avif' o 'last').
        use_lora: True para activar el LoRA BFS (Best Face Swap) de alta fidelidad.
        fast: True para modo rápido (~1024px, 4x más veloz, por defecto), False para 2K.
        output_name: Nombre opcional de salida.
        prompt: Instrucción personalizada opcional para el intercambio.
    """
    src = resolve_image_path(source_face)
    tgt = resolve_image_path(target_image)
    out_file = _generate_output_name("face_swap", output_name)

    args = [
        str(SCRIPTS_DIR / "demo_face_swap.py"),
        "--source-face", src,
        "--target-image", tgt,
        "--output", out_file,
    ]
    if use_lora:
        args += ["--lora", "Alissonerdx/BFS-Best-Face-Swap"]
    if not fast:
        args.append("--2k")
    if prompt and prompt.strip():
        args += ["--prompt", prompt.strip()]

    result = _run_script_subprocess(args)
    target_path = OUTPUTS_DIR / out_file
    result["output_file"] = out_file
    result["output_path"] = str(target_path.resolve()) if target_path.exists() else None
    return result


def group_photo(
    person1: str,
    person2: str,
    mode: str = "new_scene",
    scene: str = "at a stylish cozy cafe, warm ambient lighting",
    interaction: str = "standing shoulder to shoulder, smiling warmly, looking at the camera",
    aspect_ratio: str = "4:3",
    fast: bool = True,
    output_name: Optional[str] = None,
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Une a dos personas diferentes en una misma fotografía coherente.
    
    Args:
        person1: Foto de la primera persona.
        person2: Foto de la segunda persona.
        mode: 'new_scene' (los sitúa en un lugar nuevo) o 'keep_scene' (sitúa a person2 en el fondo de person1).
        scene: Descripción del entorno (solo en new_scene).
        interaction: Descripción de la pose, gesto o interacción mutua.
        aspect_ratio: Formato de la imagen ('4:3', '16:9', '1:1').
        fast: Modo rápido 1K.
        output_name: Nombre opcional de salida.
        prompt: Prompt personalizado que anula el montaje automático.
    """
    p1 = resolve_image_path(person1)
    p2 = resolve_image_path(person2)
    out_file = _generate_output_name("group_photo", output_name)

    proc_interaction = translate_and_enrich_prompt(interaction, context_type="interaction") if interaction else ""
    proc_scene = translate_and_enrich_prompt(scene, context_type="scene") if scene else ""

    args = [
        str(SCRIPTS_DIR / "demo_group_photo.py"),
        "--person1", p1,
        "--person2", p2,
        "--mode", mode,
        "--interaction", proc_interaction,
        "--aspect-ratio", aspect_ratio,
        "--output", out_file,
    ]
    if mode == "new_scene" and proc_scene:
        args += ["--scene", proc_scene]
    if fast:
        args.append("--fast")
    if prompt and prompt.strip():
        args += ["--prompt", prompt.strip()]

    result = _run_script_subprocess(args)
    target_path = OUTPUTS_DIR / out_file
    result["output_file"] = out_file
    result["output_path"] = str(target_path.resolve()) if target_path.exists() else None
    return result


def outfit_swap(
    person_image: str,
    outfit_image: str,
    use_lora: bool = True,
    fast: bool = True,
    output_name: Optional[str] = None,
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Cambia la vestimenta de una persona por la prenda o estilo de otra imagen (Virtual Try-On).
    
    Args:
        person_image: Foto de la persona que vestirá la ropa (o 'last').
        outfit_image: Foto con la prenda, vestuario o referencia textil.
        use_lora: True para activar el LoRA de consistencia de vestuario.
        fast: True para modo rápido (~1024px, 4x más veloz, por defecto), False para 2K.
        output_name: Nombre opcional de salida.
        prompt: Instrucción personalizada opcional para la vestimenta.
    """
    person = resolve_image_path(person_image)
    outfit = resolve_image_path(outfit_image)
    out_file = _generate_output_name("outfit", output_name)

    args = [
        str(SCRIPTS_DIR / "demo_outfit_swap.py"),
        "--person-image", person,
        "--outfit-image", outfit,
        "--output", out_file,
    ]
    if not use_lora:
        args.append("--no-lora")
    if not fast:
        args.append("--2k")
    if prompt and prompt.strip():
        args += ["--prompt", prompt.strip()]

    result = _run_script_subprocess(args)
    target_path = OUTPUTS_DIR / out_file
    result["output_file"] = out_file
    result["output_path"] = str(target_path.resolve()) if target_path.exists() else None
    return result


def camera_angle(
    image_path: str,
    angle: str = "three_quarter_right",
    use_lora: bool = True,
    fast: bool = True,
    output_name: Optional[str] = None,
    custom_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Reorienta la perspectiva o el ángulo de cámara de una imagen (3D Novel View Synthesis).
    
    Args:
        image_path: Foto original del sujeto u objeto (o 'last').
        angle: Ángulo deseado:
            - 'three_quarter_right' (tres cuartos derecha 45°)
            - 'three_quarter_left'  (tres cuartos izquierda -45°)
            - 'profile_right'       (perfil derecho 90°)
            - 'profile_left'        (perfil izquierdo -90°)
            - 'low_angle'           (contrapicado, desde abajo)
            - 'high_angle'          (picado, desde arriba)
            - 'back_view'           (desde la espalda 180°)
        use_lora: True para activar el LoRA oficial de ángulos múltiples <mva>.
        fast: True para modo rápido (~1024px, 4x más veloz, por defecto), False para 2K.
        output_name: Nombre opcional de salida.
        custom_prompt: Prompt personalizado opcional de ángulo.
    """
    img = resolve_image_path(image_path)
    out_file = _generate_output_name(f"angle_{angle}", output_name)

    args = [
        str(SCRIPTS_DIR / "demo_camera_angle.py"),
        "--image", img,
        "--angle", angle,
        "--output", out_file,
    ]
    if use_lora:
        args += ["--lora", "akhaliq/Qwen-Image-2.1-Multiple-Angles-LoRA"]
    if not fast:
        args.append("--2k")
    if custom_prompt and custom_prompt.strip():
        args += ["--custom-prompt", custom_prompt.strip()]

    result = _run_script_subprocess(args)
    target_path = OUTPUTS_DIR / out_file
    result["output_file"] = out_file
    result["output_path"] = str(target_path.resolve()) if target_path.exists() else None
    return result


def outpaint(
    image_path: str,
    ratio: str = "16:9",
    description: str = "",
    output_name: Optional[str] = None,
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Expande el lienzo de una imagen rellenando contextualmente el entorno (Outpainting / Uncrop).
    
    Args:
        image_path: Imagen a expandir (o 'last').
        ratio: Nuevo formato deseado: '16:9', '9:16', '4:3', '1:1' o 'expand_all' (zoom out general).
        description: Descripción opcional del entorno o paisaje con el que rellenar los bordes.
        output_name: Nombre opcional de salida.
        prompt: Prompt personalizado completo para outpainting.
    """
    img = resolve_image_path(image_path)
    out_file = _generate_output_name("outpaint", output_name)
    proc_desc = translate_and_enrich_prompt(description, context_type="outpaint") if description else ""

    args = [
        str(SCRIPTS_DIR / "demo_outpaint.py"),
        "--image", img,
        "--ratio", ratio,
        "--output", out_file,
    ]
    if proc_desc and proc_desc.strip():
        args += ["--description", proc_desc.strip()]
    if prompt and prompt.strip():
        args += ["--prompt", prompt.strip()]

    result = _run_script_subprocess(args)
    target_path = OUTPUTS_DIR / out_file
    result["output_file"] = out_file
    result["output_path"] = str(target_path.resolve()) if target_path.exists() else None
    return result


def restore(
    image_path: str,
    mode: str = "modern",
    output_name: Optional[str] = None,
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Restaura, desenfoca y mejora la calidad o súper-resolución de una fotografía.
    
    Args:
        image_path: Imagen a restaurar (o 'last').
        mode: 'modern' (elimina compresión, desenfoque y ruido) o 'old_photo' (arañazos, grano y daño físico).
        output_name: Nombre opcional de salida.
        prompt: Instrucción personalizada opcional de restauración.
    """
    img = resolve_image_path(image_path)
    out_file = _generate_output_name("restore", output_name)

    args = [
        str(SCRIPTS_DIR / "demo_restore.py"),
        "--image", img,
        "--mode", mode,
        "--output", out_file,
    ]
    if prompt and prompt.strip():
        args += ["--prompt", prompt.strip()]

    result = _run_script_subprocess(args)
    target_path = OUTPUTS_DIR / out_file
    result["output_file"] = out_file
    result["output_path"] = str(target_path.resolve()) if target_path.exists() else None
    return result


def image_to_prompt(
    image_path: str,
    additional_text: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convierte una imagen existente a prompt inverso estructurado en inglés y español (Image-to-Text / Reverse Prompting).
    Permite opcionalmente incorporar texto o detalles adicionales que se integrarán
    armoniosamente en el prompt resultante tanto en español como en inglés.
    
    Args:
        image_path: Ruta a la imagen (o 'last' para la última).
        additional_text: Texto o detalles adicionales en español o inglés a integrar en el prompt inverso.
        verbose: Si True, imprime el análisis bilingüe en consola.
    """
    from core.image_interrogator import image_to_prompt as _i2p
    resolved = resolve_image_path(image_path)
    return _i2p(resolved, additional_text=additional_text, verbose=verbose)


def enhance_prompt(prompt: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Mejora y enriquece una idea o prompt breve para darle más sentido cinematográfico
    y coherencia visual a la escena, generando la propuesta tanto en español como en inglés.
    
    Args:
        prompt: Idea base o prompt a expandir.
        verbose: Si True, imprime la comparación bilingüe en consola.
    """
    from core.prompt_utils import enhance_scene_prompt as _esp
    res = _esp(prompt, verbose=verbose)
    return {
        "success": True,
        "original": prompt,
        "english": res["english"],
        "spanish": res["spanish"],
    }


def merge_lora(
    lora: str,
    base_model: str = "Qwen/Qwen-Image-2.1-Turbo",
    lora_scale: float = 1.0,
    save_dir: Optional[str] = None,
    prompt: Optional[str] = None,
    image: Optional[str] = None,
    aspect_ratio: str = "1:1",
    output_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Carga y fusiona (fuse) los pesos de un LoRA permanentemente en el modelo base denoiser.
    
    Args:
        lora: Ruta local a safetensors o ID en Hugging Face.
        base_model: Modelo base denoiser.
        lora_scale: Escala de fusión.
        save_dir: Directorio de exportación del modelo fusionado.
        prompt: Prompt opcional para probar la generación inmediatamente.
        image: Imagen de referencia opcional si el LoRA es de edición.
        aspect_ratio: Formato de la imagen generada.
        output_name: Nombre opcional de salida.
    """
    out_file = _generate_output_name("fused_lora", output_name)
    args = [
        str(SCRIPTS_DIR / "merge_lora.py"),
        "--lora", lora,
        "--base-model", base_model,
        "--lora-scale", str(lora_scale),
        "--aspect-ratio", aspect_ratio,
        "--output", out_file,
    ]
    if save_dir:
        args += ["--save-dir", save_dir]
    if prompt:
        args += ["--prompt", prompt]
    if image:
        resolved_img = resolve_image_path(image)
        args += ["--image", resolved_img]

    result = _run_script_subprocess(args)
    target_path = OUTPUTS_DIR / out_file
    result["output_file"] = out_file
    result["output_path"] = str(target_path.resolve()) if target_path.exists() else None
    return result

