"""
core/image_interrogator.py
Módulo de Conversión de Imagen a Texto y Generación de Prompts Inversos (Image-to-Prompt)
para Qwen-Image-2.1-Turbo.

Analiza cualquier fotografía existente y genera el prompt inverso óptimo tanto
en inglés (listo para modelos de difusión como Qwen-Image) como en español (con sentido de escena,
narrativa y detalles fotográficos).
"""

import base64
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(PROJECT_DIR / "cache" / "huggingface")
os.environ["TORCH_HOME"] = str(PROJECT_DIR / "cache" / "torch")

from core.prompt_utils import (
    enhance_scene_prompt,
    _translate_api,
    _translate_to_english_api,
    _is_omlx_available,
    DEFAULT_OMLX_URL,
    DEFAULT_OMLX_MODEL
)

# Variable singleton para cachear el procesador y modelo en memoria
_BLIP_PROCESSOR = None
_BLIP_MODEL = None


def _clean_caption_text(text: str) -> str:
    """Elimina repeticiones consecutivas de palabras y caracteres residuales."""
    # Desduplica palabras repetidas consecutivamente (e.g. 'bald bald' -> 'bald')
    cleaned = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(" .,;")
    return cleaned


def _has_high_overlap(phrase: str, reference: str, threshold: float = 0.6) -> bool:
    """Determina si las palabras significativas de phrase ya están contenidas en reference."""
    phrase_words = set(re.findall(r'\b\w+\b', phrase.lower()))
    ref_words = set(re.findall(r'\b\w+\b', reference.lower()))
    stopwords = {"a", "an", "the", "and", "or", "of", "in", "on", "at", "with", "her", "his", "their", "is", "wearing", "has", "like"}
    key_phrase_words = phrase_words - stopwords
    if not key_phrase_words:
        return True
    overlap = key_phrase_words.intersection(ref_words)
    return (len(overlap) / len(key_phrase_words)) >= threshold


def _clean_clauses(phrase: str, reference: str) -> str:
    """Extrae únicamente cláusulas o atributos que no estén ya descritos en reference."""
    clauses = [p.strip() for p in re.split(r'\band\b|,', phrase) if p.strip()]
    kept = []
    ref_lower = reference.lower()
    for c in clauses:
        c_words = set(re.findall(r'\b\w+\b', c.lower())) - {"a", "an", "the", "on", "in", "with", "is", "has", "her", "his", "wearing"}
        if not c_words:
            continue
        overlap = [w for w in c_words if w in ref_lower or (w == "glasses" and "sunglasses" in ref_lower)]
        if len(overlap) < len(c_words):
            kept.append(c)
    if not kept:
        return ""
    if len(kept) == 1:
        return kept[0]
    return ", ".join(kept[:-1]) + " and " + kept[-1]


def _filter_valid_hands_phrase(text: str) -> Optional[str]:
    """Valida si el texto extraído describe una posición real de manos o brazos."""
    valid_positions = [
        "pocket", "pockets", "hip", "hips", "side", "sides", "lap", "waist",
        "crossed", "folded", "resting", "holding", "clasped", "knees", "counter",
        "table", "chest", "behind", "raised", "air"
    ]
    words = [w.strip(".,;") for w in text.lower().split()]
    if any(pos in words for pos in valid_positions):
        clean = re.sub(r'^(the hands are|the person has hands|hands are)\s+', '', text, flags=re.IGNORECASE).strip()
        return clean
    return None


def _filter_valid_legs_phrase(text: str) -> Optional[str]:
    """Valida si el texto extraído describe una postura real de piernas o apoyo."""
    valid_stances = [
        "crossed", "apart", "spread", "wide", "together", "open", "bent",
        "planted", "stretched", "extended", "resting", "shoulder-width"
    ]
    words = [w.strip(".,;") for w in text.lower().split()]
    if any(stance in words for stance in valid_stances):
        clean = re.sub(r'^(the person is standing with legs|the legs are|legs are)\s+', '', text, flags=re.IGNORECASE).strip()
        return clean
    return None


def _probe_blip_attribute(processor, model, device, raw_image, prompt_prefix: str) -> str:
    """Sondea el modelo BLIP con un prefijo específico para extraer un atributo concreto."""
    import torch
    inputs = processor(raw_image, prompt_prefix, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=35,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2
        )
        decoded = processor.decode(out[0], skip_special_tokens=True).strip()
        if decoded.lower().startswith(prompt_prefix.lower()):
            decoded = decoded[len(prompt_prefix):].strip()
        return _clean_caption_text(decoded)


def _get_blip_pipeline():
    """Carga y cachea en memoria el modelo de visión BLIP en el dispositivo óptimo (MPS/CPU)."""
    global _BLIP_PROCESSOR, _BLIP_MODEL
    if _BLIP_PROCESSOR is not None and _BLIP_MODEL is not None:
        return _BLIP_PROCESSOR, _BLIP_MODEL

    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float16

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=dtype
    ).to(device)

    _BLIP_PROCESSOR = processor
    _BLIP_MODEL = model
    return _BLIP_PROCESSOR, _BLIP_MODEL


def _analyze_with_omlx_vision(image_path: str, additional_text: Optional[str] = None, base_url: str = DEFAULT_OMLX_URL) -> Optional[Dict[str, str]]:
    """Intenta consultar al servidor local oMLX si tiene soporte multimodal / VLM."""
    vision_model = os.environ.get("OMLX_VISION_MODEL", "")
    target_model = vision_model or DEFAULT_OMLX_MODEL
    # Solo invocar vision oMLX si el modelo tiene capacidades multimodales/visión comprobadas
    is_vision = any(k in target_model.lower() for k in ["vision", "-vl", "cpm-v", "llava", "omni", "multimodal"])
    if not is_vision:
        return None

    try:
        with open(image_path, "rb") as img_file:
            b64_img = base64.b64encode(img_file.read()).decode("utf-8")

        ext = Path(image_path).suffix.lower().replace(".", "")
        mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"

        system_prompt = (
            "You are a master reverse prompt engineer and photographer for Qwen-Image diffusion models.\n"
            "Analyze this image and generate the perfect reverse diffusion prompt to recreate it with maximum fidelity.\n\n"
            "CRITICAL DIRECTIVE FOR PERSONS, LIMBS, AND BODY POSTURE:\n"
            "If any person is present in the image, you MUST focus with meticulous precision on their anatomy, limbs, and posture:\n"
            "- Face & Identity: estimated age, facial bone structure, skin tone and natural texture, eye color, gaze direction, eyebrow shape, nose, lips, facial hair (beard, mustache, stubble), and exact facial expression.\n"
            "- Hair: exact hairstyle, hair color, length, texture (curly, straight, wavy, bald), parting, volume.\n"
            "- Hands and Arms (MANDATORY): describe the exact position and placement of both hands and arms (e.g., hands deep in pockets, hands placed on hips, arms crossed over chest, hands resting at sides, hands placed on a counter/table, hands gesturing, holding an object).\n"
            "- Legs, Feet, and Stance (MANDATORY): describe the exact placement and stance of the legs and feet (e.g., standing with legs spread shoulder-width apart / open legs, legs crossed at ankles or knees, feet firmly planted on the ground, sitting with legs open / apart, sitting with legs crossed or tucked, kneeling, one leg bent).\n"
            "- Torso and Pose: orientation (facing forward, three-quarter view, profile), posture (straight, leaning, slouching), and framing (close-up portrait, medium shot, cowboy shot, full-body).\n"
            "- Clothing & Accessories: detailed description of all garments (color, fabric texture, cut, collar, sleeves, layering) and accessories (glasses, jewelry, watch, hat, belt, footwear).\n"
            "- Lighting on Subject: how the light sculpts the subject's face and body (key light, catchlights in eyes, cheek shadows, rim light on shoulders and hair).\n\n"
            "Return a JSON object with this exact format:\n"
            "{\n"
            '  "spanish": "Descripción cinematográfica completa en español con máximo detalle de la persona (rostro, expresión, peinado, posición exacta de manos y brazos, postura de piernas y pies [ej: de pie con piernas abiertas/cruzadas o manos en los bolsillos], ropa, accesorios), encuadre, iluminación y entorno.",\n'
            '  "english": "The image is a realistic portrait photograph of [meticulous person details: face, hair, expression, exact placement of hands and arms, exact stance of legs and feet, posture, clothing, accessories], set against [background], [sculpting lighting], captured on 50mm portrait cinema lens, f/1.8 shallow depth of field, sharp focus on the subject."\n'
            "}\n"
            "STRICT RULES:\n"
            "1. Output ONLY the JSON object.\n"
            "2. NEVER include banned meta-words such as '4K', '8K', 'hyperrealistic', or 'resolution'."
        )

        user_text = "Describe this image for reverse diffusion prompting:"
        if additional_text and additional_text.strip():
            user_text += (
                f"\n\nADDITIONAL USER INSTRUCTION: Seamlessly integrate the following user-requested details or modifications into BOTH the Spanish and English descriptions in the best and most natural photographic way possible:\n"
                f"\"{additional_text.strip()}\""
            )

        payload = {
            "model": DEFAULT_OMLX_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}}
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 500,
            "response_format": {"type": "json_object"}
        }

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=req_data,
            headers={"Content-Type": "application/json", "User-Agent": "QwenAgent/1.0"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if choices:
                content = choices[0]["message"]["content"].strip()
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "english" in parsed and "spanish" in parsed:
                    return {
                        "english": parsed["english"].strip(),
                        "spanish": parsed["spanish"].strip(),
                    }
    except Exception:
        pass
    return None


def image_to_prompt(image_path: str, additional_text: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Convierte una imagen existente a prompt inverso estructurado en inglés y español.
    Permite opcionalmente incorporar texto o detalles adicionales que se integrarán
    armoniosamente en el prompt resultante tanto en español como en inglés.
    
    1. Si oMLX Vision está disponible, analiza visualmente la imagen con el LLM multimodal.
    2. Si no, utiliza el modelo local de visión BLIP en Apple Silicon / MPS para extraer
       la descripción visual base, integra los detalles adicionales solicitados y luego
       la expande cinematográficamente con enhance_scene_prompt.
       
    Args:
        image_path: Ruta a la imagen (absoluta o relativa).
        additional_text: Texto o detalles adicionales a incluir/añadir en el prompt inverso.
        verbose: Si True, imprime los resultados bilingües en consola.
        
    Returns:
        Dict con:
            - 'success': bool
            - 'caption_base': str (descripción directa en inglés con añadidos)
            - 'additional_text': Optional[str]
            - 'english': str (prompt inverso optimizado para difusión)
            - 'spanish': str (prompt inverso en español con sentido de escena)
            - 'image_path': str
    """
    path_obj = Path(image_path).resolve()
    if not path_obj.is_file():
        return {
            "success": False,
            "error": f"La imagen no existe: {image_path}",
            "additional_text": additional_text,
            "english": "",
            "spanish": ""
        }

    # 1. Intentar con oMLX VLM si está activo
    if _is_omlx_available():
        omlx_res = _analyze_with_omlx_vision(str(path_obj), additional_text=additional_text)
        if omlx_res:
            res = {
                "success": True,
                "caption_base": omlx_res["english"],
                "additional_text": additional_text.strip() if additional_text else None,
                "english": omlx_res["english"],
                "spanish": omlx_res["spanish"],
                "image_path": str(path_obj)
            }
            if verbose:
                _print_bilingual_result(res)
            return res

    # 2. Análisis local con el modelo de visión BLIP
    try:
        import torch
        processor, model = _get_blip_pipeline()
        device = next(model.parameters()).device

        raw_image = Image.open(str(path_obj)).convert("RGB")

        # 2.1 Generar descripción base con control estricto de repetición
        inputs = processor(raw_image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=45,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2
            )
            base_caption = _clean_caption_text(processor.decode(out[0], skip_special_tokens=True))

        # 2.2 Detectar si la imagen contiene una persona o sujeto humano
        person_words = [
            "man", "woman", "person", "boy", "girl", "model", "lady", "guy",
            "child", "people", "portrait", "face", "human", "gentleman", "male", "female"
        ]
        is_person = any(re.search(r'\b' + re.escape(w) + r'\b', base_caption.lower()) for w in person_words)

        if is_person:
            # 2.3 Sondeo específico de rasgos físicos, vestimenta, extremidades y expresión
            hair_face = _probe_blip_attribute(processor, model, device, raw_image, "the person has")
            clothing = _probe_blip_attribute(processor, model, device, raw_image, "the person is wearing")
            hands = _probe_blip_attribute(processor, model, device, raw_image, "the hands are")
            legs = _probe_blip_attribute(processor, model, device, raw_image, "the person is standing with legs")
            gaze_pose = _probe_blip_attribute(processor, model, device, raw_image, "the person looks")

            current = base_caption

            # 1. Integrar o expandir vestimenta
            if clothing:
                clean_cloth = re.sub(r'^(the person is wearing|wearing|is wearing)\s+', '', clothing, flags=re.IGNORECASE).strip()
                expanded = False
                for word in re.findall(r'\b\w+\b', clean_cloth.lower()):
                    if word not in {"a", "an", "the", "and", "or"} and re.search(r'\ba\s+' + re.escape(word) + r'\b', current, flags=re.IGNORECASE):
                        if clean_cloth.lower() != f"a {word}".lower() and word in clean_cloth.lower():
                            current = re.sub(r'\ba\s+' + re.escape(word) + r'\b', clean_cloth, current, flags=re.IGNORECASE, count=1)
                            expanded = True
                            break
                if not expanded:
                    new_cloth = _clean_clauses(clean_cloth, current)
                    if new_cloth:
                        current = f"{current}, wearing {new_cloth}"

            # 2. Integrar rasgos faciales y cabello
            if hair_face:
                clean_hf = re.sub(r'^(the person has|has)\s+', '', hair_face, flags=re.IGNORECASE).strip()
                new_hf = _clean_clauses(clean_hf, current)
                if new_hf:
                    current = f"{current}, with {new_hf}"

            # 3. Integrar posición de manos y brazos (ej: en bolsillos, en caderas)
            valid_hands = _filter_valid_hands_phrase(hands)
            if valid_hands and not _has_high_overlap(valid_hands, current):
                if valid_hands.startswith("in") or valid_hands.startswith("on") or valid_hands.startswith("at"):
                    current = f"{current}, with hands {valid_hands}"
                else:
                    current = f"{current}, hands {valid_hands}"

            # 4. Integrar postura de piernas y apoyo (ej: piernas abiertas, cruzadas)
            valid_legs = _filter_valid_legs_phrase(legs)
            if valid_legs and not _has_high_overlap(valid_legs, current):
                if "standing" in current.lower():
                    current = f"{current}, with legs {valid_legs}"
                else:
                    current = f"{current}, standing with legs {valid_legs}"

            # 5. Integrar mirada, expresión o postura
            if gaze_pose:
                clean_gaze = re.sub(r'^(the person looks|looks|is looking)\s+', '', gaze_pose, flags=re.IGNORECASE).strip()
                new_gaze = _clean_clauses(clean_gaze, current)
                if new_gaze and not any(new_gaze.lower().startswith(b) for b in ["like a bald", "like a man", "like a woman", "like a person"]):
                    if new_gaze.lower().startswith("at") or new_gaze.lower() in {"down", "up", "away", "forward", "ahead", "sideways", "directly"}:
                        current = f"{current}, looking {new_gaze}"
                    else:
                        current = f"{current}, {new_gaze}"

            caption = current
        else:
            caption = base_caption

        # 2.4 Integrar texto o detalles adicionales solicitados por el usuario si existen
        if additional_text and additional_text.strip():
            user_extra = additional_text.strip()
            en_extra = _translate_to_english_api(user_extra)
            clean_extra = _clean_caption_text(en_extra)
            if clean_extra and not _has_high_overlap(clean_extra, caption):
                caption = f"{caption}, {clean_extra}"

        # Enriquecer la descripción para darle sentido cinematográfico y fotográfico a la escena
        enhanced_bundle = enhance_scene_prompt(caption, verbose=False)

        res = {
            "success": True,
            "caption_base": caption,
            "additional_text": additional_text.strip() if additional_text else None,
            "english": enhanced_bundle["english"],
            "spanish": enhanced_bundle["spanish"],
            "image_path": str(path_obj)
        }
        if verbose:
            _print_bilingual_result(res)
        return res

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "additional_text": additional_text,
            "english": "",
            "spanish": "",
            "image_path": str(path_obj)
        }


def _print_bilingual_result(result: Dict[str, Any]):
    """Imprime el resultado del análisis inverso con formato claro."""
    print("\n" + "=" * 62)
    print("📸 ANÁLISIS DE IMAGEN A TEXTO (REVERSE PROMPTING)")
    print("=" * 62)
    print(f"📁 Imagen analizada: {Path(result.get('image_path', '')).name}")
    if result.get("additional_text"):
        print(f"✍️  Detalles adicionales solicitados: \"{result['additional_text']}\"")
    if result.get("caption_base"):
        print(f"🔍 Detección visual integrada: \"{result['caption_base']}\"")
    print("-" * 62)
    print("🇪🇸 PROMPT INVERSO EN ESPAÑOL (Sentido de Escena y Composición):")
    print(result.get("spanish", ""))
    print("-" * 62)
    print("🇬🇧 REVERSE PROMPT EN INGLÉS (Optimizado para Difusión Qwen-Image):")
    print(result.get("english", ""))
    print("=" * 62 + "\n")
