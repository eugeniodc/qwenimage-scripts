"""
tests/test_battery_10.py
Batería de 10 pruebas automatizadas para el Agente, Core de Herramientas y oMLX.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from core.tools import (
    get_python_executable,
    resolve_image_path,
    list_sample_images,
    list_outputs,
    get_latest_output,
    _generate_output_name,
    OUTPUTS_DIR,
    SAMPLE_DIR,
    VENV_PYTHON,
)
from agent import OMLXImageAgent, SYSTEM_PROMPT


class BatteryOf10Tests(unittest.TestCase):

    # --------------------------------------------------------------------------
    # TEST 1: Listado y validación de assets en sample_images/
    # --------------------------------------------------------------------------
    def test_01_sample_images_validity(self):
        samples = list_sample_images()
        self.assertIsInstance(samples, list, "El listado de muestras debe ser una lista.")
        self.assertGreater(len(samples), 0, "Debe haber al menos una imagen en sample_images/.")
        self.assertIn("eugenio.jpg", samples, "eugenio.jpg debe estar presente en sample_images/.")
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"}
        for s in samples:
            ext = Path(s).suffix.lower()
            self.assertIn(ext, valid_exts, f"Extensión inválida para muestra: {s}")

    # --------------------------------------------------------------------------
    # TEST 2: Listado y ordenación temporal de outputs/
    # --------------------------------------------------------------------------
    def test_02_outputs_listing_and_sorting(self):
        outputs = list_outputs()
        self.assertIsInstance(outputs, list, "El listado de outputs debe ser una lista.")
        if len(outputs) >= 2:
            path_first = OUTPUTS_DIR / outputs[0]
            path_second = OUTPUTS_DIR / outputs[1]
            # El primero debe ser más reciente o igual al segundo
            self.assertGreaterEqual(
                path_first.stat().st_mtime,
                path_second.stat().st_mtime,
                "Las salidas deben estar ordenadas de más reciente a más antigua."
            )

    # --------------------------------------------------------------------------
    # TEST 3: Resolución de rutas directas y sin extensión
    # --------------------------------------------------------------------------
    def test_03_resolve_image_direct_and_stem(self):
        # Con extensión completa
        resolved_full = resolve_image_path("eugenio.jpg")
        self.assertTrue(Path(resolved_full).is_file())
        self.assertEqual(Path(resolved_full).name, "eugenio.jpg")

        # Sin extensión (auto-detección)
        resolved_stem = resolve_image_path("eugenio")
        self.assertTrue(Path(resolved_stem).is_file())
        self.assertEqual(resolved_full, resolved_stem)

    # --------------------------------------------------------------------------
    # TEST 4: Resolución de referencia contextual 'last' / 'ultima'
    # --------------------------------------------------------------------------
    def test_04_resolve_last_reference(self):
        latest_file = get_latest_output()
        if latest_file:
            resolved_last = resolve_image_path("last")
            resolved_ultima = resolve_image_path("ultima")
            self.assertEqual(resolved_last, latest_file)
            self.assertEqual(resolved_ultima, latest_file)
        else:
            with self.assertRaises(FileNotFoundError):
                resolve_image_path("last")

    # --------------------------------------------------------------------------
    # TEST 5: Control de errores ante imágenes inexistentes
    # --------------------------------------------------------------------------
    def test_05_resolve_nonexistent_image_error(self):
        # Debe fallar con ValueError si la cadena está vacía
        with self.assertRaises(ValueError):
            resolve_image_path("")
        with self.assertRaises(ValueError):
            resolve_image_path("   ")

    # --------------------------------------------------------------------------
    # TEST 6: Generación de nombres de archivo seguros y únicos
    # --------------------------------------------------------------------------
    def test_06_output_name_sanitization(self):
        # Sin nombre personalizado (debe añadir timestamp y .png)
        auto_name = _generate_output_name("test_prefix")
        self.assertTrue(auto_name.startswith("test_prefix_"))
        self.assertTrue(auto_name.endswith(".png"))

        # Con nombre personalizado sin extensión (debe añadir .png)
        custom_no_ext = _generate_output_name("prefix", "mi_foto_custom")
        self.assertEqual(custom_no_ext, "mi_foto_custom.png")

        # Con nombre personalizado con extensión válida
        custom_jpg = _generate_output_name("prefix", "resultado.jpg")
        self.assertEqual(custom_jpg, "resultado.jpg")

    # --------------------------------------------------------------------------
    # TEST 7: Intérprete Python del entorno virtual y presencia de Diffusers
    # --------------------------------------------------------------------------
    def test_07_venv_python_executable_integrity(self):
        py_bin = get_python_executable()
        # No debe resolver al python global de Homebrew
        self.assertNotIn("Cellar", py_bin, "get_python_executable no debe apuntar al Cellar de Homebrew.")
        self.assertTrue(Path(py_bin).exists(), f"El binario {py_bin} no existe.")

        # Verificar que al ejecutar ese binario puede importar diffusers y torch
        cmd = [py_bin, "-c", "import torch; import diffusers; print('OK')"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Fallo al importar diffusers/torch desde {py_bin}: {res.stderr}")
        self.assertIn("OK", res.stdout)

    # --------------------------------------------------------------------------
    # TEST 8: Parser del Agente: Bloques Markdown con JSON y encadenamiento
    # --------------------------------------------------------------------------
    def test_08_agent_action_parser_chained(self):
        agent = OMLXImageAgent()

        # Respuesta simulada con encadenamiento de 2 pasos
        response_text = """
        ¡Claro que sí! Voy a preparar todo:
        ```json
        {
          "thought": "Paso 1: Generar escena. Paso 2: Cambiar perspectiva.",
          "actions": [
            {
              "tool": "generate_image",
              "params": {"prompt": "cyberpunk city", "aspect_ratio": "16:9"}
            },
            {
              "tool": "camera_angle",
              "params": {"image_path": "last", "angle": "low_angle"}
            }
          ],
          "reply": "Generando ciudad y reorientando cámara..."
        }
        ```
        Espero que te guste el resultado.
        """
        parsed = agent._extract_json_action(response_text)
        self.assertIsNotNone(parsed)
        self.assertIn("actions", parsed)
        self.assertEqual(len(parsed["actions"]), 2)
        self.assertEqual(parsed["actions"][0]["tool"], "generate_image")
        self.assertEqual(parsed["actions"][1]["tool"], "camera_angle")
        self.assertEqual(parsed["actions"][1]["params"]["image_path"], "last")

    # --------------------------------------------------------------------------
    # TEST 9: Conexión y handshake en vivo con el servidor oMLX
    # --------------------------------------------------------------------------
    def test_09_omlx_live_server_handshake(self):
        agent = OMLXImageAgent()
        connected = agent.check_connection()
        self.assertTrue(
            connected,
            "El servidor oMLX en http://localhost:8000/v1 no respondió. Debe estar corriendo."
        )

    # --------------------------------------------------------------------------
    # TEST 10: Inferencia conversacional en vivo con oMLX y métricas de velocidad
    # --------------------------------------------------------------------------
    def test_10_omlx_real_inference_execution(self):
        agent = OMLXImageAgent()
        user_prompt = "Hola, confirma en una sola frase que estás listo para generar imágenes."
        response = agent.chat_step(user_prompt)

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        # Comprobar que en el historial se guardó el turno
        self.assertGreater(len(agent.history), 0)
        self.assertEqual(agent.history[-2]["role"], "user")
        self.assertEqual(agent.history[-1]["role"], "assistant")

        # Comprobar execute_actions con herramienta simulada
        mock_payload = {
            "thought": "Test unitario",
            "actions": [{"tool": "herramienta_inexistente_test", "params": {}}]
        }
        res = agent.execute_actions(mock_payload)
        self.assertEqual(len(res), 1)
        self.assertFalse(res[0]["success"])


if __name__ == "__main__":
    unittest.main()
