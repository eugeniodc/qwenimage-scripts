#!/usr/bin/env python3
"""
agent.py
Agente Inteligente Multimodal para Qwen-Image-2.1-Turbo
Motor: Apple Silicon MLX (mlx-community/Qwen2.5-7B-Instruct-Uncensored-4bit)

Permite operar la suite completa de generación y edición visual mediante lenguaje natural,
encadenamiento automático de tareas, optimización de prompts y memoria de sesión.
"""

import os
import sys
from pathlib import Path

# Directorio raíz del proyecto
PROJECT_DIR = Path(__file__).resolve().parent

# Configurar aislamiento de caché local en ./cache/
os.environ["HF_HOME"] = str(PROJECT_DIR / "cache" / "huggingface")
os.environ["TORCH_HOME"] = str(PROJECT_DIR / "cache" / "torch")
os.environ["PIP_CACHE_DIR"] = str(PROJECT_DIR / "cache" / "pip")

# Auto-detección y activación del entorno virtual
VENV_DIR = PROJECT_DIR / "venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
if VENV_PYTHON.exists():
    current_python = Path(sys.executable).resolve()
    target_python = VENV_PYTHON.resolve()
    if current_python != target_python:
        os.environ["VIRTUAL_ENV"] = str(VENV_DIR)
        os.environ["PATH"] = f"{VENV_DIR}/bin:{os.environ.get('PATH', '')}"
        os.execv(str(target_python), [str(target_python)] + sys.argv)

import json
import re
import subprocess
from typing import Dict, Any, List, Optional

# Importar herramientas locales
from core.tools import (
    generate_image,
    edit_image,
    face_swap,
    group_photo,
    outfit_swap,
    camera_angle,
    outpaint,
    restore,
    list_sample_images,
    list_outputs,
    get_latest_output,
    resolve_image_path,
    OUTPUTS_DIR,
    SAMPLE_DIR,
)

# Identificador oficial del modelo MLX
MLX_MODEL_REPO = "mlx-community/Qwen2.5-7B-Instruct-Uncensored-4bit"

SYSTEM_PROMPT = """Eres el Director de Arte y Agente Autónomo de Qwen-Image-2.1-Turbo.
Tu labor es interpretar las intenciones creativas del usuario en lenguaje natural y ejecutar las herramientas visuales apropiadas.

Capacidades de las herramientas disponibles:
1. generate_image(prompt, aspect_ratio="1:1", fast=True)
   - Crea imágenes desde cero (Text-to-Image).
   - aspect_ratio: "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3".
   - REGLA DE ORO: Traduce y enriquece siempre el prompt a un inglés fotográfico detallado con iluminación, atmósfera, estilo de cámara y lente para maximizar la calidad de difusión.

2. edit_image(image_path, prompt, uncensored=False, fast=True)
   - Modifica una imagen existente mediante instrucciones en texto.
   - uncensored: true si la edición requiere ignorar censura o directivas morales (AtomicChat GGUF).

3. face_swap(source_face, target_image, use_lora=True)
   - Transfiere la cara de 'source_face' (ej: 'eugenio.jpg') al cuerpo/escena de 'target_image' (o 'last').

4. group_photo(person1, person2, mode="new_scene", scene="", interaction="", aspect_ratio="4:3")
   - Une a dos personas en una misma foto.
   - mode: "new_scene" (sitio nuevo) o "keep_scene" (añade a person2 al entorno de person1).

5. outfit_swap(person_image, outfit_image, use_lora=True)
   - Viste a la persona con la ropa de referencia.

6. camera_angle(image_path, angle="three_quarter_right", use_lora=True)
   - Cambia la perspectiva de la toma.
   - angle: "three_quarter_right", "three_quarter_left", "profile_right", "profile_left", "low_angle", "high_angle", "back_view".

7. outpaint(image_path, ratio="16:9", description="")
   - Expande el lienzo en formato horizontal o vertical rellenando el fondo contextualmente.

8. restore(image_path, mode="modern")
   - Restaura o mejora resolución: mode="modern" o mode="old_photo".

Reglas para tu respuesta:
- Si el usuario pide generar, editar o transformar imágenes, DEBES responder incluyendo un bloque JSON con este formato exacto:
```json
{
  "thought": "Breve explicación de tu razonamiento",
  "actions": [
    {
      "tool": "nombre_de_la_herramienta",
      "params": {
        "param1": "valor"
      }
    }
  ],
  "reply": "Mensaje cordial para el usuario en español explicando lo que vas a hacer"
}
```
- Si el usuario pide varias cosas a la vez (ej: "pon la cara de eugenio en ironman y cambia el ángulo a contrapicado"), puedes incluir múltiples acciones en el array "actions" secuencialmente. En los pasos posteriores, usa "last" como referencia para la imagen de entrada.
- Si el usuario hace una pregunta general, charla, o pide aclaraciones, responde cordialmente en texto plano sin el bloque JSON.
"""

TOOL_MAP = {
    "generate_image": generate_image,
    "edit_image": edit_image,
    "face_swap": face_swap,
    "group_photo": group_photo,
    "outfit_swap": outfit_swap,
    "camera_angle": camera_angle,
    "outpaint": outpaint,
    "restore": restore,
}


class MLXImageAgent:
    def __init__(self, model_repo: str = MLX_MODEL_REPO):
        self.model_repo = model_repo
        self.model = None
        self.tokenizer = None
        self.history: List[Dict[str, str]] = []
        self.last_image: Optional[str] = get_latest_output()

    def load_model(self):
        """Carga el modelo MLX en memoria Metal."""
        print("=" * 64)
        print(f"🤖 Inicializando Agente MLX en Apple Silicon...")
        print(f"📦 Modelo: {self.model_repo}")
        print("=" * 64)
        try:
            from mlx_lm import load
            print("[+] Cargando pesos Metal en memoria unificada (puede tardar un momento en la 1ª descarga)...")
            self.model, self.tokenizer = load(self.model_repo)
            print("[✓] Modelo cargado y listo en GPU Metal.")
        except Exception as e:
            print(f"[!] Error al cargar modelo MLX: {e}")
            print("Verifica que las dependencias de MLX estén instaladas.")
            sys.exit(1)

    def _extract_json_action(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Busca y extrae el bloque JSON de acciones estructuradas de la respuesta."""
        # Buscar bloques ```json ... ```
        match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", response_text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Buscar cualquier { ... } válido que contenga "actions" o "tool"
        match2 = re.search(r"(\{[\s\S]*\"(?:actions|tool)\"[\s\S]*\})", response_text)
        if match2:
            try:
                return json.loads(match2.group(1))
            except json.JSONDecodeError:
                pass

        return None

    def execute_actions(self, actions_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Ejecuta una o varias acciones de herramientas y actualiza el estado."""
        actions = actions_payload.get("actions", [])
        # Soporte para formato simple con "tool" único
        if not actions and "tool" in actions_payload:
            actions = [{"tool": actions_payload["tool"], "params": actions_payload.get("params", {})}]

        results = []
        for i, act in enumerate(actions, 1):
            tool_name = act.get("tool")
            params = act.get("params", {})
            func = TOOL_MAP.get(tool_name)

            if not func:
                print(f"[!] Herramienta desconocida solicitada: {tool_name}")
                results.append({"tool": tool_name, "success": False, "error": "Herramienta no encontrada"})
                continue

            print(f"\n⚡ [Paso {i}/{len(actions)}] Ejecutando herramienta: {tool_name}")
            print(f"   Parámetros: {json.dumps(params, ensure_ascii=False, indent=2)}")

            # Sustituir 'last' o rutas relativas si aplica
            try:
                res = func(**params)
                results.append(res)
                if res.get("success") and res.get("output_path"):
                    self.last_image = res["output_path"]
                    print(f"   [✓] Resultado guardado en: {res['output_file']}")
                elif not res.get("success"):
                    print(f"   [!] Error en la ejecución: {res.get('error')}")
            except Exception as e:
                print(f"   [!] Excepción al ejecutar {tool_name}: {e}")
                results.append({"tool": tool_name, "success": False, "error": str(e)})

        return results

    def chat_step(self, user_input: str) -> str:
        """Procesa un turno conversacional con el modelo MLX."""
        from mlx_lm import generate

        # Formatear el contexto con los archivos existentes para ayudar al modelo
        samples = list_sample_images()
        last_img_name = Path(self.last_image).name if self.last_image else "Ninguna aún"
        context_hint = f"\n[Contexto del sistema: Imágenes en sample_images/: {samples}. Última imagen activa: '{last_img_name}']"

        # Construir mensajes para el chat template
        messages = [{"role": "system", "content": SYSTEM_PROMPT + context_hint}]
        # Incluir historial reciente (máximo últimos 6 turnos)
        for h in self.history[-6:]:
            messages.append(h)
        messages.append({"role": "user", "content": user_input})

        # Aplicar plantilla de chat oficial de Qwen2.5
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        print("\n🧠 Pensando...", end="", flush=True)
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=700,
            verbose=False,
        )
        print("\r" + " " * 20 + "\r", end="")  # Limpiar indicador

        # Analizar si la respuesta contiene comandos de herramientas
        action_data = self._extract_json_action(response)
        if action_data:
            reply = action_data.get("reply", "Ejecutando la acción solicitada...")
            print(f"\n🤖 {reply}\n")
            execution_results = self.execute_actions(action_data)
            
            # Registrar en historial
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": f"{reply} (Acciones ejecutadas: {len(execution_results)})"})
            
            if self.last_image:
                return f"\n🎉 Proceso completado. Imagen lista en: outputs/{Path(self.last_image).name}"
            return "\n[✓] Operación finalizada."
        else:
            # Respuesta puramente conversacional
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": response})
            return f"\n🤖 {response}\n"


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║        🤖 ASISTENTE CREATIVO QWEN-IMAGE-2.1 (MLX LOCAL)       ║
║   Apple Silicon Metal • Sin Censura • Orquestación Autónoma   ║
╚═══════════════════════════════════════════════════════════════╝
Comandos rápidos:
  /samples   - Ver imágenes disponibles en sample_images/
  /gallery   - Ver imágenes generadas en outputs/ (y abrir en Finder)
  /last      - Ver la última imagen generada activa
  /clear     - Limpiar historial de conversación
  /help      - Ayuda y ejemplos de instrucciones
  /exit      - Volver al menú principal
""")


def main():
    agent = MLXImageAgent()
    agent.load_model()
    print_banner()

    while True:
        try:
            user_input = input("\n👉 Tú: ").strip()
            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in {"/exit", "/quit", "exit", "quit", "salir"}:
                print("\n¡Hasta pronto!")
                break
            elif cmd == "/samples":
                samples = list_sample_images()
                print(f"\n📂 Imágenes en sample_images/ ({len(samples)}):")
                for s in samples:
                    print(f"   • {s}")
                continue
            elif cmd in {"/gallery", "/outputs"}:
                outs = list_outputs()
                print(f"\n🎨 Imágenes en outputs/ ({len(outs)}):")
                for o in outs[:10]:
                    print(f"   • {o}")
                op = input("\n👉 ¿Abrir carpeta en macOS Finder? (s/N): ").strip().lower()
                if op in {"s", "si", "y", "yes"}:
                    subprocess.run(["open", str(OUTPUTS_DIR)])
                continue
            elif cmd == "/last":
                latest = get_latest_output()
                if latest:
                    print(f"\n🖼️  Última imagen activa: {latest}")
                else:
                    print("\n[!] Aún no hay imágenes generadas.")
                continue
            elif cmd == "/clear":
                agent.history.clear()
                print("\n[✓] Memoria de conversación reiniciada.")
                continue
            elif cmd == "/help":
                print("\n💡 Ejemplos de lo que puedes pedirle al Agente:")
                print("  • 'Genera una imagen de un samurái cyberpunk en Tokio lloviendo, 16:9'")
                print("  • 'Haz face swap poniendo la cara de eugenio.jpg en la foto de iron-man'")
                print("  • 'Viste a eugenio.jpg con el disfraz de vaquero'")
                print("  • 'Une a eugenio.jpg y mujer.avif en un café acogedor'")
                print("  • 'Cambia la perspectiva de la última foto a ángulo contrapicado'")
                print("  • 'Expande la última imagen a formato 16:9 con un paisaje boscoso'")
                print("  • 'Mejora y restaura la calidad de eugenio.jpg'")
                continue

            result = agent.chat_step(user_input)
            print(result)

        except KeyboardInterrupt:
            print("\n\n[!] Operación interrumpida por el usuario.")
            break
        except Exception as e:
            print(f"\n[!] Error: {e}")


if __name__ == "__main__":
    main()
