"""
tests/test_image_interrogator.py
Pruebas automatizadas para el módulo de conversión de Imagen a Texto (Image-to-Prompt / Reverse Prompting).
"""

import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from core.image_interrogator import image_to_prompt
from core.tools import image_to_prompt as tools_image_to_prompt, resolve_image_path


class TestImageInterrogator(unittest.TestCase):

    def test_image_to_prompt_nonexistent_file(self):
        res = image_to_prompt("archivo_que_no_existe_12345.jpg")
        self.assertFalse(res["success"])
        self.assertIn("error", res)

    def test_image_to_prompt_valid_sample(self):
        sample_path = resolve_image_path("eugenio.jpg")
        res = image_to_prompt(sample_path, verbose=False)

        self.assertTrue(res["success"], f"Error en interrogación: {res.get('error')}")
        self.assertIn("caption_base", res)
        self.assertIn("english", res)
        self.assertIn("spanish", res)

        # Verificar que la descripción base en inglés existe
        self.assertTrue(len(res["caption_base"]) > 0)

        # Verificar que el prompt en inglés está enriquecido para difusión
        en_prompt = res["english"].lower()
        self.assertTrue(len(en_prompt) >= len(res["caption_base"]))
        self.assertTrue(any(w in en_prompt for w in ["lighting", "lens", "photorealistic", "detailed", "depth"]))

        # Verificar que el prompt en español existe y está traducido
        es_prompt = res["spanish"].lower()
        self.assertTrue(len(es_prompt) > 0)
        self.assertTrue(any(w in es_prompt for w in ["fotografía", "fotografia", "hombre", "muelle", "luz", "iluminación", "lente"]))

    def test_core_tools_image_to_prompt_wrapper(self):
        res = tools_image_to_prompt("eugenio.jpg", verbose=False)
        self.assertTrue(res["success"])
        self.assertIn("english", res)
        self.assertIn("spanish", res)

    def test_person_details_interrogation(self):
        sample_path = resolve_image_path("jasonstatham.jpg")
        res = image_to_prompt(sample_path, verbose=False)
        self.assertTrue(res["success"])
        en = res["english"].lower()
        es = res["spanish"].lower()

        # Comprobar que capta rasgos específicos de la persona (barba, traje, corbata, lentes/gafas)
        self.assertTrue(any(w in en for w in ["beard", "suit", "tie", "sunglasses", "bald"]))
        # Comprobar que enriquece detalles de piel, expresión y ropa
        self.assertTrue(any(w in en for w in ["skin", "facial", "expression", "garment", "fabric"]))
        # Comprobar que la traducción al español refleja esos detalles
        self.assertTrue(any(w in es for w in ["barba", "traje", "corbata", "gafas", "piel", "expresión", "expresion"]))

    def test_person_limbs_and_posture_interrogation(self):
        sample_path = resolve_image_path("eugenio_cuerpo_entero.jpg")
        res = image_to_prompt(sample_path, verbose=False)
        self.assertTrue(res["success"])
        en = res["english"].lower()
        es = res["spanish"].lower()

        # Comprobar que capta posición de manos (pockets) y piernas (legs)
        self.assertTrue("pocket" in en or "hands" in en)
        self.assertTrue("leg" in en or "legs" in en or "crossed" in en or "standing" in en)

        # Comprobar traducción al español de extremidades y postura
        self.assertTrue("bolsillo" in es or "manos" in es)
        self.assertTrue("pierna" in es or "piernas" in es or "cruzadas" in es or "parado" in es)

    def test_image_to_prompt_with_additional_text(self):
        sample_path = resolve_image_path("eugenio.jpg")
        extra_detail = "con un sombrero vaquero y una guitarra acústica"
        res = image_to_prompt(sample_path, additional_text=extra_detail, verbose=False)
        self.assertTrue(res["success"])
        self.assertEqual(res.get("additional_text"), extra_detail)

        en = res["english"].lower()
        es = res["spanish"].lower()

        # Comprobar que los detalles adicionales están integrados en inglés
        self.assertTrue("cowboy hat" in en or "hat" in en)
        self.assertTrue("guitar" in en)

        # Comprobar que los detalles adicionales están integrados de la mejor forma en español
        self.assertTrue("sombrero" in es)
        self.assertTrue("guitarra" in es)


if __name__ == "__main__":
    unittest.main()
