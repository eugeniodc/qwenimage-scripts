"""
tests/test_prompt_utils.py
Pruebas de validación del módulo core/prompt_utils.py:
Traducción al inglés y enriquecimiento manteniendo la esencia original.
"""

import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from core.prompt_utils import (
    translate_and_enrich_prompt,
    enhance_scene_prompt,
    _enrich_english_prompt,
    _translate_to_english_api,
)



class TestPromptUtils(unittest.TestCase):

    def test_empty_or_whitespace_prompt(self):
        self.assertEqual(translate_and_enrich_prompt("", verbose=False), "")
        self.assertEqual(translate_and_enrich_prompt("   ", verbose=False), "")

    def test_spanish_t2i_translation_and_enrichment(self):
        spanish_input = "un gato astronauta explorando marte"
        result = translate_and_enrich_prompt(spanish_input, context_type="t2i", verbose=False)

        # Debe estar en inglés y contener los conceptos clave
        lower = result.lower()
        self.assertTrue("cat" in lower or "astronaut" in lower or "mars" in lower)
        # Debe contener elementos necesarios enriquecidos (iluminación, composición, calidad)
        self.assertTrue(any(term in lower for term in ["lighting", "detailed", "photorealistic", "focus"]))

    def test_spanish_edit_translation_and_enrichment(self):
        spanish_input = "cambia el fondo por un bosque lluvioso de noche"
        result = translate_and_enrich_prompt(spanish_input, context_type="edit", verbose=False)

        lower = result.lower()
        self.assertTrue("forest" in lower or "night" in lower or "rain" in lower or "background" in lower)
        self.assertTrue(any(term in lower for term in ["seamless", "lighting", "shadow", "quality"]))

    def test_group_photo_scene_and_interaction(self):
        scene_input = "en una playa tropical al atardecer"
        result_scene = translate_and_enrich_prompt(scene_input, context_type="scene", verbose=False)
        lower_scene = result_scene.lower()
        self.assertTrue("beach" in lower_scene or "sunset" in lower_scene or "tropical" in lower_scene)
        self.assertTrue("lighting" in lower_scene or "ambient" in lower_scene or "atmosphere" in lower_scene)

        interaction_input = "abrazados sonriendo a la cámara"
        result_inter = translate_and_enrich_prompt(interaction_input, context_type="interaction", verbose=False)
        lower_inter = result_inter.lower()
        self.assertTrue("smiling" in lower_inter or "camera" in lower_inter or "hugging" in lower_inter or "posing" in lower_inter)

    def test_outpaint_translation_and_enrichment(self):
        desc_input = "ampliar con montañas nevadas y pinos"
        result_outpaint = translate_and_enrich_prompt(desc_input, context_type="outpaint", verbose=False)
        lower_outpaint = result_outpaint.lower()
        self.assertTrue("mountain" in lower_outpaint or "pine" in lower_outpaint or "snow" in lower_outpaint)
        self.assertTrue("extension" in lower_outpaint or "continuity" in lower_outpaint or "seamless" in lower_outpaint)

    def test_english_input_preserved_and_enriched(self):
        english_input = "a cybernetic tiger in a neon Tokyo alley"
        result = translate_and_enrich_prompt(english_input, context_type="t2i", verbose=False)
        lower = result.lower()
        self.assertIn("cybernetic tiger", lower)
        self.assertIn("tokyo", lower)
        self.assertTrue(any(term in lower for term in ["lighting", "detailed", "photorealistic"]))

    def test_enhance_scene_prompt_bilingual(self):
        idea = "un samurai bajo la lluvia en un bosque de bambú"
        res = enhance_scene_prompt(idea, verbose=False)
        self.assertIsInstance(res, dict)
        self.assertIn("spanish", res)
        self.assertIn("english", res)

        es_text = res["spanish"].lower()
        en_text = res["english"].lower()

        # Comprobar contenido en español
        self.assertTrue(len(es_text) > len(idea))
        self.assertTrue("samurái" in es_text or "samurai" in es_text)

        # Comprobar contenido en inglés
        self.assertTrue(len(en_text) > len(idea))
        self.assertTrue("samurai" in en_text)
        self.assertTrue("bamboo" in en_text or "forest" in en_text or "rain" in en_text)
        self.assertTrue(any(w in en_text for w in ["lighting", "lens", "atmospheric", "detailed"]))


    def test_qwen_official_rules_compliance(self):
        # 1. Prohibición de meta-palabras como 4k, 8k, 16:9 en el cuerpo del prompt
        input_with_buzzwords = "un dragón volando sobre un volcán 4k 8k 16:9 resolution"
        enhanced = translate_and_enrich_prompt(input_with_buzzwords, context_type="t2i", verbose=False)
        lower_en = enhanced.lower()
        self.assertNotIn("4k", lower_en)
        self.assertNotIn("8k", lower_en)
        self.assertNotIn("16:9", lower_en)
        self.assertNotIn("resolution", lower_en)

        # 2. Apertura con estilo/medio en T2I
        self.assertTrue(any(lower_en.startswith(p) for p in ["the image is", "a realistic photograph", "a detailed photograph", "a photograph of"]))

        # 3. Cláusula de preservación y acción directa en Edición
        edit_input = "ponle un sombrero de copa al hombre"
        enhanced_edit = translate_and_enrich_prompt(edit_input, context_type="edit", verbose=False)
        lower_edit = enhanced_edit.lower()
        self.assertIn("preserve", lower_edit)
        self.assertIn("identity", lower_edit)


if __name__ == "__main__":
    unittest.main()


