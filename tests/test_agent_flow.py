"""
tests/test_agent_flow.py
Pruebas de validación para el Agente y el Core de Herramientas.
"""

import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from core.tools import (
    resolve_image_path,
    list_sample_images,
    list_outputs,
    get_latest_output,
    generate_image,
    face_swap,
    camera_angle,
    outpaint,
    restore,
    group_photo,
    outfit_swap,
    edit_image,
)
from agent import OMLXImageAgent, SYSTEM_PROMPT


class TestAgentAndCoreTools(unittest.TestCase):

    def test_samples_and_outputs_listing(self):
        samples = list_sample_images()
        self.assertIsInstance(samples, list)
        self.assertIn("eugenio.jpg", samples)

        outputs = list_outputs()
        self.assertIsInstance(outputs, list)

    def test_resolve_image_path(self):
        # Resolver imagen existente en sample_images
        resolved = resolve_image_path("eugenio.jpg")
        self.assertTrue(Path(resolved).is_file())
        self.assertTrue(resolved.endswith("eugenio.jpg"))

        # Resolver sin extensión
        resolved_no_ext = resolve_image_path("eugenio")
        self.assertTrue(Path(resolved_no_ext).is_file())

        # Resolver 'last' o 'ultima'
        if list_outputs():
            last_img = resolve_image_path("last")
            self.assertTrue(Path(last_img).is_file())

    def test_json_action_parsing(self):
        agent = OMLXImageAgent()

        # Caso 1: JSON con bloque markdown ```json
        sample_response = """
        Aquí tienes el resultado:
        ```json
        {
          "thought": "El usuario quiere generar un gato astronauta",
          "actions": [
            {
              "tool": "generate_image",
              "params": {
                "prompt": "A cute cat astronaut on Mars",
                "aspect_ratio": "16:9"
              }
            }
          ],
          "reply": "Generando gato astronauta en formato panorámico..."
        }
        ```
        """
        parsed = agent._extract_json_action(sample_response)
        self.assertIsNotNone(parsed)
        self.assertEqual(len(parsed["actions"]), 1)
        self.assertEqual(parsed["actions"][0]["tool"], "generate_image")
        self.assertEqual(parsed["actions"][0]["params"]["aspect_ratio"], "16:9")

        # Caso 2: JSON plano
        sample_response2 = '{"thought": "Cambio de ropa", "tool": "outfit_swap", "params": {"person_image": "eugenio.jpg", "outfit_image": "disfraz_vaquero.jpg"}}'
        parsed2 = agent._extract_json_action(sample_response2)
        self.assertIsNotNone(parsed2)
        self.assertEqual(parsed2["tool"], "outfit_swap")

        # Caso 3: Charla conversacional sin JSON
        sample_response3 = "¡Hola! Soy tu asistente creativo. ¿Qué imagen deseas generar hoy?"
        parsed3 = agent._extract_json_action(sample_response3)
        self.assertIsNone(parsed3)

    def test_system_prompt_english_translation_and_enrichment_rule(self):
        """Verifica que el SYSTEM_PROMPT exija la traducción al inglés y enriquecimiento de prompts."""
        self.assertIn("REGLA FUNDAMENTAL DE TRADUCCIÓN Y ENRIQUECIMIENTO", SYSTEM_PROMPT)
        self.assertIn("inglés", SYSTEM_PROMPT.lower())
        self.assertIn("esencia", SYSTEM_PROMPT.lower())
        self.assertIn("elementos necesarios", SYSTEM_PROMPT.lower())

    def test_tool_map_completeness(self):
        """Verifica que todas las herramientas clave están mapeadas en el Agente."""
        from agent import TOOL_MAP
        expected_tools = [
            "generate_image",
            "edit_image",
            "face_swap",
            "group_photo",
            "outfit_swap",
            "camera_angle",
            "outpaint",
            "restore",
            "image_to_prompt",
            "enhance_prompt",
            "merge_lora",
        ]
        for tool in expected_tools:
            self.assertIn(tool, TOOL_MAP, f"La herramienta '{tool}' debe estar en TOOL_MAP")

    def test_execute_actions_param_normalization(self):
        """Verifica que execute_actions normaliza alias de parámetros."""
        agent = OMLXImageAgent()
        # Mock func en TOOL_MAP
        from unittest.mock import MagicMock
        mock_func = MagicMock(return_value={"success": True, "output_path": "/tmp/out.png", "output_file": "out.png"})
        from agent import TOOL_MAP
        orig_edit = TOOL_MAP["edit_image"]
        TOOL_MAP["edit_image"] = mock_func

        try:
            payload = {
                "actions": [
                    {
                        "tool": "edit_image",
                        "params": {
                            "image": "sample_images/eugenio.jpg",
                            "prompt": "make it cinematic"
                        }
                    }
                ]
            }
            results = agent.execute_actions(payload)
            mock_func.assert_called_once()
            called_kwargs = mock_func.call_args.kwargs
            self.assertIn("image_path", called_kwargs)
            self.assertEqual(called_kwargs["image_path"], "sample_images/eugenio.jpg")
            self.assertTrue(results[0]["success"])
        finally:
            TOOL_MAP["edit_image"] = orig_edit

    def test_omlx_live_connection(self):
        agent = OMLXImageAgent()
        # Verificar handshake con el servidor local oMLX si está en ejecución
        connected = agent.check_connection()
        if not connected:
            self.skipTest("El servidor local oMLX no está activo en http://localhost:8000/v1")
        self.assertTrue(connected)


if __name__ == "__main__":
    unittest.main()

