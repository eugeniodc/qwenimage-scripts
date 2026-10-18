"""
tests/test_scripts_prompt_integration.py
Valida que todos los scripts en scripts/ importen y utilicen el módulo unificado
core/prompt_utils.py (translate_and_enrich_prompt) para la revisión de prompts.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core.prompt_utils import translate_and_enrich_prompt


class TestScriptsPromptIntegration(unittest.TestCase):

    def test_scripts_import_prompt_utils(self):
        """Verifica que todos los scripts importan translate_and_enrich_prompt."""
        scripts_to_check = [
            "demo_cli",
            "demo_image_editing_uncensored",
            "demo_face_swap",
            "demo_group_photo",
            "demo_outfit_swap",
            "demo_camera_angle",
            "demo_outpaint",
            "demo_restore",
            "merge_lora",
            "demo_text_to_image",
            "demo_image_editing",
        ]
        for script_name in scripts_to_check:
            script_path = PROJECT_DIR / "scripts" / f"{script_name}.py"
            self.assertTrue(script_path.exists(), f"{script_name}.py no existe")
            content = script_path.read_text(encoding="utf-8")
            self.assertIn(
                "from core.prompt_utils import translate_and_enrich_prompt",
                content,
                f"{script_name}.py no importa translate_and_enrich_prompt"
            )
            self.assertIn(
                "translate_and_enrich_prompt(",
                content,
                f"{script_name}.py no invoca translate_and_enrich_prompt"
            )

    @patch("diffusers.QwenImage21Pipeline.from_pretrained")
    @patch("core.prompt_utils.translate_and_enrich_prompt")
    def test_demo_cli_invokes_prompt_utils(self, mock_translate, mock_pipe_cls):
        mock_pipe = MagicMock()
        mock_pipe.return_value = MagicMock(images=[MagicMock()])
        mock_pipe_cls.return_value.to.return_value = mock_pipe
        mock_translate.return_value = "Enriched English Prompt"

        from scripts import demo_cli
        test_args = ["demo_cli.py", "--prompt", "un perro corriendo en la playa"]
        with patch.object(sys, "argv", test_args):
            demo_cli.main()

        mock_translate.assert_called_once()
        call_args, call_kwargs = mock_translate.call_args
        self.assertEqual(call_args[0], "un perro corriendo en la playa")
        self.assertEqual(call_kwargs.get("context_type"), "t2i")

    @patch("diffusers.QwenImage21Pipeline.from_pretrained")
    @patch("diffusers.utils.load_image")
    @patch("core.prompt_utils.translate_and_enrich_prompt")
    def test_demo_image_editing_uncensored_invokes_prompt_utils(self, mock_translate, mock_load_img, mock_pipe_cls):
        mock_img = MagicMock()
        mock_img.convert.return_value = MagicMock(size=(1024, 1024))
        mock_load_img.return_value = mock_img

        mock_pipe = MagicMock()
        mock_pipe.return_value = MagicMock(images=[MagicMock()])
        mock_pipe_cls.return_value.to.return_value = mock_pipe
        mock_translate.return_value = "Remove the shirt and make body muscular"

        from scripts import demo_image_editing_uncensored
        test_args = [
            "demo_image_editing_uncensored.py",
            "--image", "sample_images/eugenio.jpg",
            "--prompt", "quita la camisa"
        ]
        with patch.object(sys, "argv", test_args):
            demo_image_editing_uncensored.main()

        mock_translate.assert_called_once()
        call_args, call_kwargs = mock_translate.call_args
        self.assertEqual(call_args[0], "quita la camisa")
        self.assertEqual(call_kwargs.get("context_type"), "edit_uncensored")

    @patch("diffusers.QwenImage21Pipeline.from_pretrained")
    @patch("diffusers.utils.load_image")
    @patch("core.prompt_utils.translate_and_enrich_prompt")
    def test_demo_outfit_swap_invokes_prompt_utils(self, mock_translate, mock_load_img, mock_pipe_cls):
        mock_img = MagicMock()
        mock_img.convert.return_value = MagicMock(size=(1024, 1024))
        mock_load_img.return_value = mock_img

        mock_pipe = MagicMock()
        mock_pipe.return_value = MagicMock(images=[MagicMock()])
        mock_pipe_cls.return_value.to.return_value = mock_pipe
        mock_translate.return_value = "Dress the person in image 1 in the leather jacket from image 2"

        from scripts import demo_outfit_swap
        test_args = [
            "demo_outfit_swap.py",
            "--person-image", "sample_images/eugenio.jpg",
            "--outfit-image", "sample_images/iron-man.avif",
            "--prompt", "ponle una chaqueta de cuero elegante",
            "--no-lora"
        ]
        with patch.object(sys, "argv", test_args):
            demo_outfit_swap.main()

        mock_translate.assert_called_once()
        call_args, call_kwargs = mock_translate.call_args
        self.assertEqual(call_args[0], "ponle una chaqueta de cuero elegante")
        self.assertEqual(call_kwargs.get("context_type"), "edit")


if __name__ == "__main__":
    unittest.main()
