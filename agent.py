#!/usr/bin/env python3
"""
agent.py
Agente Inteligente Multimodal para Qwen-Image-2.1-Turbo
Conectado al servidor local de producción: oMLX (Apple Silicon Metal)
Modelo por defecto: Qwen2.5-7B-Instruct-Uncensored-4bit en http://localhost:8000/v1
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

# Auto-detección y activación del entorno virtual si aplica
VENV_DIR = PROJECT_DIR / "venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
if VENV_PYTHON.exists():
    current_python = Path(sys.executable).resolve()
    target_python = VENV_PYTHON.resolve()
    if sys.prefix != str(VENV_DIR) and current_python != target_python:
        os.environ["VIRTUAL_ENV"] = str(VENV_DIR)
        os.environ["PATH"] = f"{VENV_DIR}/bin:{os.environ.get('PATH', '')}"
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

import json
import re
import subprocess
import urllib.error
import urllib.request
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
    OUTPUTS_DIR,
    SAMPLE_DIR,
)

# Configuración del servidor oMLX
DEFAULT_OMLX_URL = os.environ.get("OMLX_BASE_URL", "http://localhost:8000/v1")
DEFAULT_OMLX_MODEL = os.environ.get("OMLX_MODEL", "Qwen2.5-7B-Instruct-Uncensored-4bit")

SYSTEM_PROMPT = """Eres el Director de Arte y Agente Autónomo de Qwen-Image-2.1-Turbo.
Tu labor es interpretar las intenciones creativas del usuario en lenguaje natural y ejecutar las herramientas visuales apropiadas.

DEBES RESPONDER SIEMPRE CON UN OBJETO JSON VÁLIDO con esta estructura exacta:
{
  "thought": "Breve explicación de tu razonamiento o plan",
  "reply": "Mensaje cordial para el usuario en español explicando lo que vas a hacer o respondiendo a su consulta",
  "actions": [
    {
      "tool": "nombre_de_la_herramienta",
      "params": {
        "param1": "valor"
      }
    }
  ]
}

Capacidades de las herramientas disponibles para el array "actions":
1. generate_image(prompt, aspect_ratio="1:1", fast=True)
   - Crea imágenes desde cero (Text-to-Image).
   - aspect_ratio: "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3".
   - REGLA DE ORO: Traduce y enriquece siempre el prompt a un inglés fotográfico detallado con iluminación, atmósfera, estilo de cámara y lente para maximizar la calidad de difusión.

2. edit_image(image_path, prompt, uncensored=False, fast=True)
   - Modifica una imagen existente mediante instrucciones en texto.
   - uncensored: true si la edición requiere ignorar censura o directivas morales (AtomicChat GGUF).

3. face_swap(source_face, target_image, use_lora=True, fast=True)
   - Transfiere la cara de 'source_face' (ej: 'eugenio.jpg') al cuerpo/escena de 'target_image' (o 'last').

4. group_photo(person1, person2, mode="new_scene", scene="", interaction="", aspect_ratio="4:3", fast=True)
   - Une a dos personas en una misma foto.
   - mode: "new_scene" (sitio nuevo) o "keep_scene" (añade a person2 al entorno de person1).

5. outfit_swap(person_image, outfit_image, use_lora=True, fast=True)
   - Viste a la persona con la ropa de referencia.

6. camera_angle(image_path, angle="three_quarter_right", use_lora=True, fast=True)
   - Cambia la perspectiva de la toma.
   - angle: "three_quarter_right", "three_quarter_left", "profile_right", "profile_left", "low_angle", "high_angle", "back_view".

7. outpaint(image_path, ratio="16:9", description="")
   - Expande el lienzo en formato horizontal o vertical rellenando el fondo contextualmente.

8. restore(image_path, mode="modern")
   - Restaura o mejora resolución: mode="modern" o mode="old_photo".

Reglas estrictas de comportamiento:
- Si el usuario pide generar, crear, editar, transformar imágenes, hacer face swap, cambiar ropa, o reintentar: DEBES poblar "actions" con la herramienta y parámetros adecuados.
- Si el usuario hace una pregunta general, charla, o pide ayuda: pon "actions": [] y escribe tu respuesta en "reply".
- REGLA DE VELOCIDAD: Incluye siempre fast: true en los params por defecto (~1024px, 4x más veloz), salvo que el usuario pida 2K o alta calidad.
- Si el usuario pide varias cosas a la vez, incluye múltiples acciones secuenciales en el array "actions". En los pasos posteriores, usa "last" como imagen de entrada.
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


class OMLXImageAgent:
    def __init__(self, base_url: str = DEFAULT_OMLX_URL, model_name: str = DEFAULT_OMLX_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.history: List[Dict[str, str]] = []
        self.last_image: Optional[str] = get_latest_output()
        self.last_actions_payload: Optional[Dict[str, Any]] = None

    def check_connection(self) -> bool:
        """Verifica que el servidor oMLX esté activo y el modelo disponible."""
        models_url = f"{self.base_url}/models"
        try:
            req = urllib.request.Request(models_url, headers={"User-Agent": "QwenAgent/1.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("id") for m in data.get("data", [])]
                if self.model_name in models:
                    print(f"🟢 Servidor oMLX conectado ({self.base_url})")
                    print(f"📦 Modelo activo: {self.model_name}")
                    return True
                else:
                    print(f"🟡 Servidor oMLX activo, pero '{self.model_name}' no figura en la lista.")
                    print(f"   Modelos disponibles: {', '.join(models[:4])}...")
                    return True
        except urllib.error.URLError as e:
            print(f"🔴 No se pudo conectar al servidor oMLX en {self.base_url}: {e.reason}")
            print("\n💡 Para iniciar el servidor oMLX en tu Mac, ejecuta en otra terminal:")
            print("   omlx serve Qwen2.5-7B-Instruct-Uncensored-4bit --port 8000")
            print("   o también: omlx start\n")
            return False
        except Exception as e:
            print(f"🔴 Error de conexión con oMLX: {e}")
            return False

    def _extract_json_action(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Busca y extrae el bloque JSON de acciones de la respuesta del modelo."""
        # 1. Intentar parsear directamente (response_format garantizado por oMLX)
        try:
            cleaned = response_text.strip()
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # 2. Buscar bloques ```json ... ```
        match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", response_text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. Buscar cualquier bloque { ... } con actions o tool
        match2 = re.search(r"(\{[\s\S]*\"(?:actions|tool)\"[\s\S]*\})", response_text)
        if match2:
            try:
                return json.loads(match2.group(1))
            except json.JSONDecodeError:
                pass

        return None

    def execute_actions(self, actions_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Ejecuta una o varias acciones de herramientas y actualiza el estado de la sesión."""
        self.last_actions_payload = actions_payload
        actions = actions_payload.get("actions", [])
        if not actions and "tool" in actions_payload:
            actions = [{"tool": actions_payload["tool"], "params": actions_payload.get("params", {})}]

        results = []
        for i, act in enumerate(actions, 1):
            tool_name = act.get("tool")
            params = act.get("params", {})
            func = TOOL_MAP.get(tool_name)

            if not func:
                print(f"[!] Herramienta desconocida solicitada: {tool_name}")
                results.append({"tool": tool_name, "success": False, "error": f"Herramienta '{tool_name}' no encontrada"})
                continue

            print(f"\n⚡ [Paso {i}/{len(actions)}] Ejecutando herramienta: {tool_name}")
            print(f"   Parámetros: {json.dumps(params, ensure_ascii=False, indent=2)}")

            try:
                res = func(**params)
                res["tool"] = tool_name
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
        """Envía el turno de conversación al servidor oMLX y procesa las herramientas."""
        samples = list_sample_images()
        last_img_name = Path(self.last_image).name if self.last_image else "Ninguna aún"
        context_hint = f"\n[Contexto del sistema: Imágenes en sample_images/: {samples}. Última imagen activa: '{last_img_name}']"

        # Detectar si el usuario pide reintentar
        input_lower = user_input.lower().strip()
        is_retry = any(kw in input_lower for kw in [
            "reintenta", "otra vez", "de nuevo", "vuelve a intentar",
            "prueba otra vez", "repite", "inténtalo de nuevo", "intenta de nuevo"
        ])

        messages = [{"role": "system", "content": SYSTEM_PROMPT + context_hint}]
        for h in self.history[-6:]:
            messages.append(h)
        messages.append({"role": "user", "content": user_input})

        chat_url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 800,
            "response_format": {"type": "json_object"}
        }

        print("\n🧠 Consultando a oMLX...", end="", flush=True)
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                chat_url,
                data=req_data,
                headers={"Content-Type": "application/json", "User-Agent": "QwenAgent/1.0"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                choices = resp_json.get("choices", [])
                if not choices:
                    return "\n[!] oMLX no devolvió ninguna respuesta."
                response_text = choices[0]["message"]["content"]
                
                # Métricas opcionales devueltas por oMLX
                usage = resp_json.get("usage", {})
                speed = usage.get("generation_tokens_per_second")
                speed_info = f" ({speed:.1f} tok/s)" if speed else ""
        except Exception as e:
            print("\r" + " " * 30 + "\r", end="")
            return f"\n[!] Error al comunicarse con oMLX ({self.base_url}): {e}"

        print("\r" + " " * 30 + "\r", end="")  # Limpiar indicador de carga

        # Analizar si la respuesta contiene comandos de herramientas
        action_data = self._extract_json_action(response_text)

        # Auto-recuperación: si el usuario pidió reintentar y no hay acciones, rescatar la última acción
        if (not action_data or not action_data.get("actions")) and is_retry and self.last_actions_payload:
            print("\n🔄 [Auto-Recuperación] Reintentando la última acción de herramientas registrada...")
            action_data = self.last_actions_payload
            action_data["reply"] = "Reintentando la acción anterior..."

        actions = action_data.get("actions", []) if action_data else []
        if action_data and actions:
            reply = action_data.get("reply", "Ejecutando la acción solicitada...")
            print(f"\n🤖 {reply}{speed_info}\n")
            execution_results = self.execute_actions(action_data)

            # Detectar fallos en las acciones
            failures = [r for r in execution_results if not r.get("success")]
            successes = [r for r in execution_results if r.get("success")]

            if failures:
                err_msg = failures[0].get("error", "Error en el subproceso")
                failed_tool = failures[0].get("tool", "herramienta")
                # Registrar el fallo en el historial para que oMLX sepa que no se completó
                self.history.append({"role": "user", "content": user_input})
                self.history.append({
                    "role": "assistant",
                    "content": f"{reply} [FALLO: La herramienta '{failed_tool}' devolvió error: {err_msg}]"
                })
                return f"\n❌ No se pudo completar la operación: {err_msg}\nPuedes decir 'inténtalo de nuevo' o usar el comando /retry."

            # Todas las acciones tuvieron éxito
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": f"{reply} (Completado con éxito: {len(successes)} acciones)"})

            last_gen = successes[-1].get("output_file")
            if last_gen:
                return f"\n🎉 Proceso completado con éxito. Imagen lista en: outputs/{last_gen}"
            return "\n[✓] Operación finalizada."
        else:
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": response_text})
            return f"\n🤖 {response_text}{speed_info}\n"


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║        🤖 ASISTENTE CREATIVO QWEN-IMAGE-2.1 (vía oMLX)        ║
║   Apple Silicon Metal • Sin Censura • Orquestación Autónoma   ║
╚═══════════════════════════════════════════════════════════════╝
Comandos rápidos:
  /retry     - Reintentar la última acción de herramientas
  /samples   - Ver imágenes disponibles en sample_images/
  /gallery   - Ver imágenes generadas en outputs/ (y abrir en Finder)
  /last      - Ver la última imagen generada activa
  /clear     - Limpiar historial de conversación
  /help      - Ayuda y ejemplos de instrucciones
  /exit      - Volver al menú principal
""")


def main():
    agent = OMLXImageAgent()
    is_connected = agent.check_connection()
    if not is_connected:
        print("[!] No se puede iniciar el agente sin conexión al servidor oMLX.")
        print("    Inicia el servidor y vuelve a intentarlo.")
        sys.exit(1)

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
            elif cmd in {"/retry", "/reintentar"}:
                if agent.last_actions_payload:
                    print("\n🔄 Reintentando última acción registrada...")
                    res = agent.execute_actions(agent.last_actions_payload)
                    failures = [r for r in res if not r.get("success")]
                    successes = [r for r in res if r.get("success")]
                    if failures:
                        print(f"\n❌ El reintento falló: {failures[0].get('error')}")
                    else:
                        print(f"\n🎉 ¡Reintento completado con éxito! Imagen lista en: outputs/{successes[-1].get('output_file')}")
                else:
                    print("\n[!] No hay ninguna acción previa registrada para reintentar.")
                continue
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
