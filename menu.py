#!/usr/bin/env python3
"""
Menú interactivo y cómodo para Qwen-Image-2.1-Turbo
Ejecución directa: python menu.py o ./menu.py

Incluye AUTO-ACTIVACIÓN del entorno virtual:
Si lo ejecutas con el python del sistema sin hacer 'source venv/bin/activate',
se auto-detecta y se re-ejecuta automáticamente dentro del entorno virtual.
"""

import os
import subprocess
import sys
from pathlib import Path

# Directorio raíz del proyecto
PROJECT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
SAMPLE_DIR = PROJECT_DIR / "sample_images"
CACHE_DIR = PROJECT_DIR / "cache"
VENV_DIR = PROJECT_DIR / "venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"

# ==============================================================================
# AUTO-ACTIVACIÓN AUTOMÁTICA DEL ENTORNO VIRTUAL (VENV)
# ==============================================================================
if not VENV_DIR.exists():
    print("=" * 62)
    print("⚠️  El entorno virtual 'venv' aún no ha sido instalado.")
    print("=" * 62)
    resp = input("👉 ¿Deseas crearlo e instalar dependencias ahora? (S/n): ").strip().lower()
    if resp in {"", "s", "si", "y", "yes"}:
        print("\n[+] Instalando entorno virtual automáticamente...")
        subprocess.run(["bash", str(PROJECT_DIR / "setup_env.sh")], check=True)
    else:
        print("[!] No se puede ejecutar la demo sin el entorno virtual.")
        sys.exit(1)

# Si se ejecuta con un python que no es el del venv, auto-reemplazar el proceso
if VENV_PYTHON.exists():
    current_python = Path(sys.executable).resolve()
    target_python = VENV_PYTHON.resolve()
    if current_python != target_python:
        os.environ["VIRTUAL_ENV"] = str(VENV_DIR)
        os.environ["PATH"] = f"{VENV_DIR}/bin:{os.environ.get('PATH', '')}"
        os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
        os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
        os.environ["PIP_CACHE_DIR"] = str(CACHE_DIR / "pip")
        # os.execv reemplaza el proceso actual por el python del entorno virtual
        os.execv(str(target_python), [str(target_python)] + sys.argv)

# Asegurar carpetas del proyecto
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def clear_screen():
    print("\n" * 2)

def list_sample_images():
    """Lista las imágenes disponibles en sample_images/"""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"}
    files = [f for f in SAMPLE_DIR.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
    return sorted(files)

def choose_file(prompt_text):
    """Permite al usuario elegir un archivo de sample_images mediante un número o escribir una ruta."""
    images = list_sample_images()
    print(f"\n{prompt_text}:")
    if images:
        print("  Imágenes encontradas en sample_images/:")
        for i, img in enumerate(images, 1):
            print(f"   [{i}] {img.name}")
        print("   [0] Escribir otra ruta o URL manualmente")
        choice = input("\n👉 Selecciona una opción (ej: 1): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(images):
                return str(images[idx - 1])
    
    manual = input("👉 Ingresa la ruta del archivo o URL: ").strip()
    return manual

def sanitize_output_filename(name: str, default: str) -> str:
    """Asegura que el nombre de archivo tenga una extensión válida de imagen (por defecto .png)"""
    cleaned = (name or "").strip()
    if not cleaned:
        cleaned = default
    p = Path(cleaned)
    if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        p = p.with_suffix(".png")
    return p.name

def run_script(args):
    """Ejecuta un script usando el intérprete de Python del entorno virtual"""
    python_bin = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    cmd = [python_bin] + args
    print("\n" + "=" * 60)
    print(f"🚀 Ejecutando con entorno virtual: {' '.join(cmd)}")
    print("=" * 60 + "\n")
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_DIR))
    except KeyboardInterrupt:
        print("\n[!] Operación cancelada por el usuario.")
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Error durante la ejecución (código {e.returncode}).")
    input("\nPresiona [Enter] para volver al menú principal...")

def menu_text_to_image():
    print("\n--- 🎨 1. Generación Texto a Imagen (Text-to-Image) ---")
    print("Deja en blanco para usar el prompt oficial del póster de química.")
    prompt = input("👉 Escribe tu prompt: ").strip()
    
    print("\nRelaciones de aspecto: [1] 1:1 (cuadrado)  [2] 16:9 (horizontal)  [3] 9:16 (vertical)")
    ar_choice = input("👉 Elige formato (1, 2, 3) [por defecto 1]: ").strip()
    ar_map = {"1": "1:1", "2": "16:9", "3": "9:16"}
    aspect = ar_map.get(ar_choice, "1:1")

    print("\nModo de velocidad y resolución:")
    print("  [1] ⚡ Rápido (~1024px, recomendado, hasta 4x más veloz)")
    print("  [2] 💎 Alta Calidad 2K (~2048px)")
    speed_choice = input("👉 Elige modo (1 o 2) [por defecto 1]: ").strip()
    is_fast = speed_choice != "2"

    out_name = sanitize_output_filename(
        input("\n👉 Nombre de archivo de salida [t2i_custom.png]: ").strip(),
        "t2i_custom.png"
    )

    if prompt:
        args = [
            str(SCRIPTS_DIR / "demo_cli.py"),
            "--prompt", prompt,
            "--aspect-ratio", aspect,
            "--output", out_name,
        ]
        if is_fast:
            args.append("--fast")
    else:
        args = [str(SCRIPTS_DIR / "demo_text_to_image.py")]
    
    run_script(args)

def menu_image_editing():
    print("\n--- 🖌️  2. Edición de Imagen (Image Editing) ---")
    img_path = choose_file("Elige la imagen a editar")
    if not img_path:
        return
    
    prompt = input("👉 Instrucción de edición (ej: 'Convert into oil painting'): ").strip()
    if not prompt:
        print("[!] El prompt de edición no puede estar vacío.")
        input("Presiona Enter...")
        return

    print("\nModo de velocidad y resolución:")
    print("  [1] ⚡ Rápido (~1024px, recomendado, hasta 4x más veloz)")
    print("  [2] 💎 Alta Calidad 2K (~2048px)")
    speed_choice = input("👉 Elige modo (1 o 2) [por defecto 1]: ").strip()
    is_fast = speed_choice != "2"

    out_name = sanitize_output_filename(
        input("\n👉 Nombre de archivo de salida [edit_custom.png]: ").strip(),
        "edit_custom.png"
    )
    args = [
        str(SCRIPTS_DIR / "demo_cli.py"),
        "--image", img_path,
        "--prompt", prompt,
        "--output", out_name,
    ]
    if is_fast:
        args.append("--fast")
    run_script(args)

def menu_image_editing_uncensored():
    print("\n--- 🔓 3. Editar Imagen Existente SIN CENSURA (Uncensored / AtomicChat) ---")
    print("Modelo: AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF")
    print("Filtros de negativa y directivas de rechazo desactivadas (abliterated).")
    img_path = choose_file("Elige la imagen a editar")
    if not img_path:
        return

    prompt = input("\n👉 Instrucción de edición sin censura: ").strip()
    if not prompt:
        print("[!] El prompt de edición no puede estar vacío.")
        input("Presiona Enter...")
        return

    print("\nModo de velocidad y resolución:")
    print("  [1] ⚡ Rápido (~1024px, recomendado, 8 pasos Turbo)")
    print("  [2] 💎 Máxima Calidad 2K (~2048px)")
    speed_choice = input("👉 Elige modo (1 o 2) [por defecto 1]: ").strip()
    is_fast = speed_choice != "2"

    out_name = sanitize_output_filename(
        input("\n👉 Nombre de archivo de salida [edit_uncensored.png]: ").strip(),
        "edit_uncensored.png"
    )

    args = [
        str(SCRIPTS_DIR / "demo_image_editing_uncensored.py"),
        "--image", img_path,
        "--prompt", prompt,
        "--output", out_name,
    ]
    if not is_fast:
        args.append("--2k")

    run_script(args)

def menu_face_swap():
    print("\n--- 🎭 4. Intercambio de Caras (Face Swap) ---")
    print("Paso 1: Selecciona la imagen con el ROSTRO que deseas transferir (ej: tu cara)")
    src = choose_file("Imagen del rostro a transferir")
    if not src:
        return

    print("\nPaso 2: Selecciona la imagen de la ESCENA / CUERPO de destino (ej: Iron Man)")
    tgt = choose_file("Imagen de destino (cuerpo y fondo que se conservarán)")
    if not tgt:
        return

    out_name = sanitize_output_filename(
        input("\n👉 Nombre de salida [face_swap_result.png]: ").strip(),
        "face_swap_result.png"
    )
    use_lora = input("\n👉 ¿Deseas activar el LoRA especializado BFS (Best Face Swap) para máxima fidelidad? (s/N) [por defecto No]: ").strip().lower()

    args = [
        str(SCRIPTS_DIR / "demo_face_swap.py"),
        "--source-face", src,
        "--target-image", tgt,
        "--output", out_name,
    ]
    if use_lora in {"s", "si", "y", "yes"}:
        args += ["--lora", "Alissonerdx/BFS-Best-Face-Swap"]

    run_script(args)

def menu_group_photo():
    print("\n--- 👥 5. Unir Personas en una Misma Foto (Group Photo Merge) ---")
    print("Paso 1: Selecciona la foto de la PRIMERA persona (<image1>)")
    p1 = choose_file("Foto de la primera persona")
    if not p1:
        return

    print("\nPaso 2: Selecciona la foto de la SEGUNDA persona (<image2>)")
    p2 = choose_file("Foto de la segunda persona")
    if not p2:
        return

    print("\n¿Cómo deseas unir a las personas?")
    print("  [1] 🏖️  Nueva escena / entorno (sitúa a ambos en un lugar que elijas)")
    print("  [2] 🖼️  Mantener el fondo de la primera foto (añade a la segunda persona al lado)")
    mode_choice = input("👉 Elige modo (1 o 2) [por defecto 1]: ").strip()
    is_keep_scene = mode_choice == "2"
    mode = "keep_scene" if is_keep_scene else "new_scene"

    scene = ""
    if not is_keep_scene:
        print("\nEjemplos de escenario:")
        print("  • 'en una cafetería moderna y acogedora'")
        print("  • 'en una playa al atardecer frente al mar'")
        print("  • 'en una terraza de restaurante de noche con luces cálidas'")
        scene = input("👉 Describe el entorno o escenario [cafetería acogedora]: ").strip()
        if not scene:
            scene = "at a stylish cozy cafe, sitting together enjoying coffee, warm ambient lighting"

    print("\nEjemplos de pose o interacción:")
    print("  • 'abrazados amistosamente sonriendo a la cámara'")
    print("  • 'hombro con hombro sonriendo naturalmente'")
    print("  • 'sentados juntos charlando relajadamente'")
    interaction = input("👉 Describe la pose o interacción [hombro con hombro sonriendo]: ").strip()
    if not interaction:
        interaction = "standing shoulder to shoulder, smiling warmly, looking at the camera"

    print("\nRelaciones de aspecto: [1] 4:3 (recomendado)  [2] 16:9 (horizontal)  [3] 1:1 (cuadrado)")
    ar_choice = input("👉 Elige formato (1, 2, 3) [por defecto 1]: ").strip()
    ar_map = {"1": "4:3", "2": "16:9", "3": "1:1"}
    aspect = ar_map.get(ar_choice, "4:3")

    print("\nModo de velocidad y resolución:")
    print("  [1] ⚡ Rápido (~1024px, recomendado)")
    print("  [2] 💎 Alta Calidad 2K (~2048px)")
    speed_choice = input("👉 Elige modo (1 o 2) [por defecto 1]: ").strip()
    is_fast = speed_choice != "2"

    out_name = sanitize_output_filename(
        input("\n👉 Nombre de archivo de salida [group_photo.png]: ").strip(),
        "group_photo.png"
    )

    args = [
        str(SCRIPTS_DIR / "demo_group_photo.py"),
        "--person1", p1,
        "--person2", p2,
        "--mode", mode,
        "--interaction", interaction,
        "--aspect-ratio", aspect,
        "--output", out_name,
    ]
    if scene:
        args += ["--scene", scene]
    if is_fast:
        args.append("--fast")

    run_script(args)


def menu_camera_angle():
    print("\n--- 🎥 7. Cambio de Ángulo de Cámara (Camera Angle / Perspective) ---")
    img = choose_file("Elige la imagen a reorientar")
    if not img:
        return

    print("\nÁngulos disponibles:")
    angles = [
        ("three_quarter_right", "Tres cuartos hacia la derecha (45°)"),
        ("three_quarter_left", "Tres cuartos hacia la izquierda (-45°)"),
        ("profile_right", "Perfil lateral derecho completo (90°)"),
        ("profile_left", "Perfil lateral izquierdo completo (-90°)"),
        ("low_angle", "Contrapicado (desde abajo hacia arriba)"),
        ("high_angle", "Picado (desde arriba hacia abajo)"),
        ("back_view", "Vista desde la espalda (180°)"),
    ]
    for i, (k, desc) in enumerate(angles, 1):
        print(f"  [{i}] {desc}")
    
    choice = input("\n👉 Selecciona ángulo (1-7) [por defecto 1]: ").strip()
    idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(angles) else 0
    selected_angle = angles[idx][0]

    out_name = sanitize_output_filename(
        input(f"\n👉 Nombre de salida [angle_{selected_angle}.png]: ").strip(),
        f"angle_{selected_angle}.png"
    )

    use_lora = input("\n👉 ¿Deseas usar LoRA oficial de ángulos <mva>? (s/N) [por defecto No]: ").strip().lower()
    
    args = [
        str(SCRIPTS_DIR / "demo_camera_angle.py"),
        "--image", img,
        "--angle", selected_angle,
        "--output", out_name,
    ]
    if use_lora in {"s", "si", "y", "yes"}:
        args += ["--lora", "akhaliq/Qwen-Image-2.1-Multiple-Angles-LoRA"]

    run_script(args)

def menu_outfit_swap():
    print("\n--- 👔 6. Cambio de Ropa (Outfit Swap / Virtual Try-On) ---")
    print("Paso 1: Selecciona la foto de la PERSONA (se mantendrán su rostro, cuerpo y fondo)")
    person = choose_file("Foto de la persona a vestir")
    if not person:
        return

    print("\nPaso 2: Selecciona la foto con la PRENDA / ROPA de referencia")
    outfit = choose_file("Foto de la prenda de referencia")
    if not outfit:
        return

    out_name = sanitize_output_filename(
        input("\n👉 Nombre de salida [outfit_result.png]: ").strip(),
        "outfit_result.png"
    )
    use_lora = input("👉 ¿Usar LoRA de consistencia 'ausboss/Qwen-Image-2.1-Outfit-Swap-Consistency-LoRA'? (S/n) [por defecto Sí]: ").strip().lower()

    args = [
        str(SCRIPTS_DIR / "demo_outfit_swap.py"),
        "--person-image", person,
        "--outfit-image", outfit,
        "--output", out_name,
    ]
    if use_lora in {"n", "no"}:
        args += ["--no-lora"]

    run_script(args)

def menu_outpaint():
    print("\n--- 🖼️  8. Expansión de Lienzo (Outpainting / Uncrop) ---")
    img = choose_file("Elige la imagen a expandir")
    if not img:
        return

    print("\nFormato o expansión deseada:")
    print("  [1] 16:9 (Panorámico / Horizontal)")
    print("  [2] 9:16 (Vertical / Formato historia o reel)")
    print("  [3] 4:3  (Fotografía estándar)")
    print("  [4] 1:1  (Cuadrado)")
    print("  [5] expand_all (Zoom-out general en todas las direcciones)")
    choice = input("\n👉 Selecciona formato (1-5) [por defecto 1]: ").strip()
    ratios = {"1": "16:9", "2": "9:16", "3": "4:3", "4": "1:1", "5": "expand_all"}
    ratio = ratios.get(choice, "16:9")

    desc = input("\n👉 Descripción opcional del entorno a rellenar (ej: 'en un bosque soleado'): ").strip()
    out_name = sanitize_output_filename(
        input("👉 Nombre de archivo de salida [outpaint_result.png]: ").strip(),
        "outpaint_result.png"
    )

    args = [
        str(SCRIPTS_DIR / "demo_outpaint.py"),
        "--image", img,
        "--ratio", ratio,
        "--output", out_name,
    ]
    if desc:
        args += ["--description", desc]

    run_script(args)

def menu_restore():
    print("\n--- ✨ 9. Restauración y Mejora de Fotos (Restore & Upscale) ---")
    img = choose_file("Elige la foto a restaurar")
    if not img:
        return

    print("\nModo de restauración:")
    print("  [1] Foto moderna (eliminar desenfoque, compresión y pixelado)")
    print("  [2] Foto antigua (eliminar arañazos, polvo, grano y restaurar color)")
    m_choice = input("👉 Elige modo (1 o 2) [por defecto 1]: ").strip()
    mode = "old_photo" if m_choice == "2" else "modern"

    out_name = sanitize_output_filename(
        input("\n👉 Nombre de archivo de salida [restore_result.png]: ").strip(),
        "restore_result.png"
    )

    args = [
        str(SCRIPTS_DIR / "demo_restore.py"),
        "--image", img,
        "--mode", mode,
        "--output", out_name,
    ]
    run_script(args)

def menu_merge_lora():
    print("\n--- 🔀 10. Unir / Fusionar un LoRA (Merge & Fuse LoRA) ---")
    print("Introduce el identificador de Hugging Face o la ruta a un archivo .safetensors local.")
    print("Ejemplos:")
    print("  • Hugging Face: Alissonerdx/BFS-Best-Face-Swap")
    print("  • Hugging Face: akhaliq/Qwen-Image-2.1-Multiple-Angles-LoRA")
    print("  • Local: /ruta/a/mi_lora.safetensors")
    lora = input("\n👉 Repositorio o ruta del LoRA: ").strip()
    if not lora:
        return

    scale_str = input("👉 Escala / fuerza de fusión [por defecto 1.0]: ").strip()
    scale = scale_str if scale_str else "1.0"

    print("\n¿Qué deseas hacer con el modelo fusionado?")
    print("  [1] Generar imagen de prueba ahora")
    print("  [2] Guardar / Exportar el modelo fusionado a disco (save_pretrained)")
    print("  [3] Ambas cosas (guardar modelo y generar imagen)")
    print("  [4] Solo validar la fusión en memoria")
    action = input("👉 Elige una opción (1-4) [por defecto 1]: ").strip()

    args = [
        str(SCRIPTS_DIR / "merge_lora.py"),
        "--lora", lora,
        "--lora-scale", scale,
    ]

    if action in {"2", "3"}:
        save_dir = input("\n👉 Directorio donde guardar el modelo [cache/qwen_fused]: ").strip()
        args += ["--save-dir", save_dir if save_dir else "cache/qwen_fused"]

    if action in {"1", "3", ""}:
        prompt = input("\n👉 Escribe el prompt para la imagen de prueba: ").strip()
        if prompt:
            args += ["--prompt", prompt]
            out_name = sanitize_output_filename(
                input("👉 Nombre de archivo de salida [fused_output.png]: ").strip(),
                "fused_output.png"
            )
            args += ["--output", out_name]

    run_script(args)

def menu_check_outputs():
    print("\n--- 📂 11. Galería de Imágenes Construidas (outputs/) ---")
    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = [f for f in OUTPUTS_DIR.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
    if not files:
        print("  (Aún no hay imágenes generadas en la carpeta outputs/)")
    else:
        for i, f in enumerate(sorted(files), 1):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  [{i}] {f.name:<28} ({size_mb:.2f} MB)")
        
        print("\n  [A] Abrir la carpeta outputs en macOS Finder")
        op = input("👉 Selecciona una opción o presiona Enter para volver: ").strip()
        if op.lower() == "a":
            subprocess.run(["open", str(OUTPUTS_DIR)])

def menu_check_cache():
    print("\n--- 🔍 12. Estado del Almacenamiento y Caché ---")
    def get_dir_size(path):
        total = 0
        if path.exists():
            for p in path.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
        return total / (1024 * 1024 * 1024)

    hf_gb = get_dir_size(CACHE_DIR / "huggingface")
    pip_gb = get_dir_size(CACHE_DIR / "pip")
    torch_gb = get_dir_size(CACHE_DIR / "torch")
    total_gb = hf_gb + pip_gb + torch_gb

    print(f"  • Modelos Hugging Face: {hf_gb:.2f} GB")
    print(f"  • Caché de Pip:          {pip_gb:.2f} GB")
    print(f"  • Caché de PyTorch:      {torch_gb:.2f} GB")
    print(f"  ─────────────────────────────────────")
    print(f"  • Total en carpeta cache: {total_gb:.2f} GB")

    print("\nOpciones de limpieza:")
    print("  [1] Vaciar caché de Pip (libera espacio de descargadores temporales)")
    print("  [2] Escanear modelos con huggingface-cli")
    print("  [0] Volver")
    
    sub = input("\n👉 Elige una opción: ").strip()
    if sub == "1":
        pip_path = CACHE_DIR / "pip"
        if pip_path.exists():
            for f in pip_path.glob("**/*"):
                if f.is_file():
                    f.unlink()
            print("[✓] Caché de Pip vaciada.")
        input("Presiona Enter...")
    elif sub == "2":
        subprocess.run(["huggingface-cli", "cache", "scan"])
        input("Presiona Enter...")

def main():
    while True:
        clear_screen()
        venv_active = sys.prefix == str(VENV_DIR) or Path(sys.executable).resolve() == VENV_PYTHON.resolve()
        status_tag = "🟢 VENV ACTIVO" if venv_active else "🟡 SISTEMA"
        print("=" * 62)
        print(f"      🤖 PANEL DE CONTROL — QWEN-IMAGE-2.1-TURBO [{status_tag}]")
        print("=" * 62)
        print("  [A] 🧠 MODO AGENTE LOCAL (vía oMLX • Qwen2.5-7B Uncensored)")
        print("  ─────────────────────────────────────────────────────────────")
        print("  [1] 🎨 Generar imagen desde texto (Text-to-Image)")
        print("  [2] 🖌️  Editar imagen existente (Image Editing)")
        print("  [3] 🔓 Editar imagen existente SIN CENSURA (Uncensored / AtomicChat)")
        print("  [4] 🎭 Intercambio de caras (Face Swap / LoRA BFS)")
        print("  [5] 👥 Unir personas en una foto (Group Photo Merge)")
        print("  [6] 👔 Cambio de ropa (Outfit Swap / LoRA Consistency)")
        print("  [7] 🎥 Cambio de ángulo de cámara (Camera Angle / LoRA)")
        print("  [8] 🖼️  Expansión de lienzo (Outpainting / Uncrop)")
        print("  [9] ✨ Restauración de fotos (Photo Restore & Upscale)")
        print("  [10] 🔀 Unir / Fusionar un LoRA (Merge & Fuse LoRA)")
        print("  [11] 📂 Ver imágenes generadas (en outputs/)")
        print("  [12] 🔍 Ver uso de disco y caché")
        print("  [0] 🚪 Salir")
        print("=" * 62)
        
        choice = input("👉 Selecciona una opción (A, 0-12): ").strip()
        choice_lower = choice.lower()

        if choice_lower in {"a", "agente", "agent"}:
            run_script([str(PROJECT_DIR / "agent.py")])
        elif choice == "1":
            menu_text_to_image()
        elif choice == "2":
            menu_image_editing()
        elif choice == "3":
            menu_image_editing_uncensored()
        elif choice == "4":
            menu_face_swap()
        elif choice == "5":
            menu_group_photo()
        elif choice == "6":
            menu_outfit_swap()
        elif choice == "7":
            menu_camera_angle()
        elif choice == "8":
            menu_outpaint()
        elif choice == "9":
            menu_restore()
        elif choice == "10":
            menu_merge_lora()
        elif choice == "11":
            menu_check_outputs()
        elif choice == "12":
            menu_check_cache()
        elif choice in {"0", "q", "exit", "salir"}:
            print("\n¡Hasta luego!")
            break
        else:
            print("[!] Opción inválida.")

if __name__ == "__main__":
    main()
