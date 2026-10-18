"""
core/prompt_utils.py
Utilidades de traducción y enriquecimiento automático de prompts para Qwen-Image-2.1-Turbo.

Asegura que los prompts introducidos por el usuario (en cualquier idioma) sean traducidos
al inglés manteniendo la esencia de la petición pero añadiendo todos los elementos
necesarios (iluminación, composición, detalles fotográficos, consistencia y atmósfera)
para maximizar la calidad de difusión visual.
"""

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Optional, Dict

# Configuración por defecto del servidor local oMLX
DEFAULT_OMLX_URL = os.environ.get("OMLX_BASE_URL", "http://localhost:8000/v1")
DEFAULT_OMLX_MODEL = os.environ.get("OMLX_MODEL", "Qwen2.5-7B-Instruct-Uncensored-4bit")


def _is_omlx_available(base_url: str = DEFAULT_OMLX_URL) -> bool:
    """Verifica rápidamente si el servidor oMLX local está respondiendo."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers={"User-Agent": "QwenAgent/1.0"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def _translate_with_omlx(text: str, context_type: str, base_url: str = DEFAULT_OMLX_URL, model: str = DEFAULT_OMLX_MODEL) -> Optional[str]:
    """Usa el LLM local oMLX aplicando las directrices oficiales de prompt rewriting de Qwen-Image-2.1."""
    if context_type in {"edit", "edit_uncensored"}:
        system_prompt = (
            "You are the official Qwen-Image-2.1 image editing prompt expert.\n"
            "Your task is to rewrite the user's edit instruction following the official Attribute Disentanglement principle:\n"
            "1. Language: Write the descriptive instruction in English. Any text to be rendered into the image must remain exactly in its original language inside double quotes (\"...\").\n"
            "2. Direct Action: Lead with the specific action verb (e.g., 'Replace...', 'Add...', 'Transform...', 'Modify...').\n"
            "3. Full Strength: Push the requested edit to an unmistakable, high-impact degree (avoid faint under-editing).\n"
            "4. Strict Preservation: Hold unmentioned attributes at input fidelity. Conclude with a concise preservation clause: 'Preserve the original subject identity, pose, composition, and background lighting exactly as shown in the input image.'\n"
            "5. Conciseness: Do NOT describe unedited content in detail, as repainting descriptions causes drift.\n"
            "STRICT: Output ONLY the final English edit instruction. No quotes or conversational filler."
        )
    elif context_type == "scene":
        system_prompt = (
            "You are the official Qwen-Image-2.1 scene description expert.\n"
            "Your task is to translate and describe the background setting and environment in English:\n"
            "1. Describe the environment with ambient lighting, atmosphere, and depth.\n"
            "2. Do NOT mention people or subjects unless specified.\n"
            "3. NO Meta-Words (4K, 8K, resolution).\n"
            "STRICT: Output ONLY the English scene description phrase or sentence."
        )
    elif context_type == "outpaint":
        system_prompt = (
            "You are the official Qwen-Image-2.1 outpainting and canvas uncrop expert.\n"
            "Your task is to translate and describe the outpainting extension in English:\n"
            "1. Describe seamless background extension outward with scene lighting and perspective continuity.\n"
            "2. NO Meta-Words (4K, 8K, resolution).\n"
            "STRICT: Output ONLY the English outpainting description phrase or sentence."
        )
    elif context_type == "interaction":
        system_prompt = (
            "You are the official Qwen-Image-2.1 character interaction and posing expert.\n"
            "Your task is to translate and describe the interaction, posing, and connection between subjects in English:\n"
            "1. Describe natural candid posing and authentic body language.\n"
            "STRICT: Output ONLY the English interaction description phrase or sentence."
        )
    else:
        system_prompt = (
            "You are the official Qwen-Image-2.1 image prompt rewriting expert.\n"
            "Your task is to turn the user's brief into a cohesive English description as an observer reporting what is in the frame:\n"
            "1. Language: Write the descriptive prose in English. Any text to be rendered into the image must remain exactly in its original script and language inside double quotes (\"...\").\n"
            "2. Opening Sentence: Formulate an opening sentence stating medium, style, subject, and background/palette: 'The image is a [style] [photograph/illustration/portrait/scene] of [subject], set against [background/palette].'\n"
            "3. Spatial Mapping ('Walk the Frame'): Distribute the scene elements across the frame using concrete positional phrases (foreground, in the upper-left, across the background, in the center, lower-third).\n"
            "4. Atmosphere & Lighting: Detail natural lighting, atmosphere, lens depth of field, and textures.\n"
            "5. NO Meta-Words: NEVER write aspect ratios, pixel dimensions, '4K', '8K', or 'resolution' into the prompt.\n"
            "STRICT: Output ONLY the final English prompt paragraph. No quotes or introductory text."
        )

    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.3,
            "max_tokens": 350,
        }
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=req_data,
            headers={"Content-Type": "application/json", "User-Agent": "QwenAgent/1.0"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if choices:
                result = choices[0]["message"]["content"].strip()
                result = re.sub(r'^["\']|["\']$', '', result).strip()
                if result:
                    return result
    except Exception:
        pass
    return None



def _translate_api(text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
    """Traduce texto entre idiomas usando el endpoint libre de traducción."""
    if not text or not text.strip():
        return ""
    try:
        url = (
            f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q="
            + urllib.parse.quote(text.strip())
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            translated = "".join([part[0] for part in data[0] if part[0]]).strip()
            if translated:
                return translated
    except Exception:
        pass
    return text


def _translate_to_english_api(text: str) -> str:
    """Wrapper hacia _translate_api para traducción al inglés."""
    return _translate_api(text, source_lang="auto", target_lang="en")



def _enrich_english_prompt(english_text: str, context_type: str = "t2i") -> str:
    """
    Enriquece un texto en inglés según los criterios oficiales de Qwen-Image-2.1:
    - Apertura con estilo + medio + sujeto + fondo.
    - Distribución espacial ("Walk the Frame").
    - Principio de Attribute Disentanglement con cláusula de preservación para edición.
    - Eliminación estricta de meta-palabras de resolución ('4k', '8k', 'resolution', etc.).
    """
    cleaned = english_text.strip().rstrip(".,; ")
    if not cleaned:
        return ""

    # Limpiar meta-palabras técnicas prohibidas por la documentación oficial de Qwen
    cleaned = re.sub(r'\b(4k|8k|16:9|9:16|1:1|resolution|uhd)\b', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip().rstrip(".,; ")
    lower_text = cleaned.lower()

    person_words = [
        "man", "woman", "person", "boy", "girl", "model", "lady", "guy",
        "child", "people", "portrait", "face", "human", "gentleman", "male", "female"
    ]
    is_person = any(re.search(r'\b' + re.escape(w) + r'\b', lower_text) for w in person_words)

    if context_type == "t2i":
        has_opening = any(lower_text.startswith(p) for p in [
            "the image is", "a photograph of", "a photo of", "an illustration of",
            "a portrait of", "a cinematic", "a realistic photograph", "a detailed photograph"
        ])
        if is_person:
            prefix = "" if has_opening else "A realistic portrait photograph of "
            additions = []
            if not any(w in lower_text for w in ["skin", "pores", "facial", "catchlights"]):
                additions.append("natural skin texture and authentic facial expression")
            if not any(w in lower_text for w in ["lighting", "light", "sunlight", "cinematic"]):
                additions.append("sculpted portrait lighting with soft catchlights in the eyes and subtle rim light")
            if not any(w in lower_text for w in ["detailed", "hyperdetailed", "texture", "textures", "fabric"]):
                additions.append("detailed clothing textures and natural folds")
            if not any(w in lower_text for w in ["lens", "focus", "shot", "composition"]):
                additions.append("sharp focus on 50mm portrait cinema lens, f/1.8")
        else:
            prefix = "" if has_opening else "A realistic photograph of "
            additions = []
            if not any(w in lower_text for w in ["lighting", "light", "sunlight", "cinematic"]):
                additions.append("cinematic volumetric lighting")
            if not any(w in lower_text for w in ["detailed", "hyperdetailed", "texture", "textures"]):
                additions.append("detailed textures and natural atmospheric depth")
            if not any(w in lower_text for w in ["lens", "focus", "shot", "composition"]):
                additions.append("sharp focus on 35mm lens")

        result = f"{prefix}{cleaned}"
        if additions:
            result = f"{result}, {', '.join(additions)}"
        return result

    elif context_type in {"edit", "edit_uncensored"}:
        # Principio oficial Qwen: Attribute Disentanglement at Full Strength
        # Acción directa + modificación fuerte + cláusula de preservación concisa
        additions = []
        if not any(w in lower_text for w in ["seamless", "coherent", "integration"]):
            additions.append("seamless realistic integration")
        if not any(w in lower_text for w in ["lighting", "shadow", "shadows"]):
            additions.append("matching scene lighting and natural shadows")

        preservation = "Preserve the original subject identity, pose, and background lighting exactly as shown in the input image."
        if not any(w in lower_text for w in ["preserve", "keep"]):
            additions.append(preservation)

        if additions:
            return f"{cleaned}, {', '.join(additions)}"
        return cleaned

    elif context_type == "scene":
        additions = []
        if not any(w in lower_text for w in ["lighting", "ambient"]):
            additions.append("warm ambient cinematic lighting")
        if not any(w in lower_text for w in ["atmosphere", "atmospheric", "depth"]):
            additions.append("detailed environment with cohesive atmospheric depth across the frame")

        if additions:
            return f"{cleaned}, {', '.join(additions)}"
        return cleaned

    elif context_type == "interaction":
        additions = []
        if not any(w in lower_text for w in ["natural", "candid", "authentic"]):
            additions.append("natural candid posing")
        if not any(w in lower_text for w in ["expression", "smile", "looking"]):
            additions.append("authentic expressive connection and body language")

        if additions:
            return f"{cleaned}, {', '.join(additions)}"
        return cleaned

    elif context_type == "outpaint":
        additions = []
        if not any(w in lower_text for w in ["seamless", "extension", "continuation"]):
            additions.append("seamless background extension outward")
        if not any(w in lower_text for w in ["lighting", "perspective"]):
            additions.append("matching original scene lighting and perspective continuity")

        if additions:
            return f"{cleaned}, {', '.join(additions)}"
        return cleaned

    # Contexto general
    return f"A realistic photograph of {cleaned}, cinematic lighting, detailed textures, sharp focus"



def translate_and_enrich_prompt(
    text: str,
    context_type: str = "t2i",
    verbose: bool = True
) -> str:
    """
    Punto de entrada principal para procesar prompts del usuario en todas las opciones.
    
    1. Traduce al inglés manteniendo fielmente la esencia.
    2. Enriquece con los elementos visuales, lumínicos y técnicos necesarios.
    3. Muestra feedback en consola si verbose=True.
    
    Args:
        text: Prompt o instrucción introducida por el usuario.
        context_type: Tipo de contexto ('t2i', 'edit', 'edit_uncensored', 'scene', 'interaction', 'outpaint', 'general').
        verbose: Si True, imprime en consola el prompt original y el procesado.
        
    Returns:
        str: Prompt optimizado en inglés listo para el modelo de difusión.
    """
    if not text or not text.strip():
        return ""

    original = text.strip()

    # 1. Intentar con oMLX local si está activo (traducción + enriquecimiento en 1 paso inteligente)
    if _is_omlx_available():
        omlx_result = _translate_with_omlx(original, context_type)
        if omlx_result:
            enriched_omlx = _enrich_english_prompt(omlx_result, context_type)
            if verbose:
                print(f"\n🌐 [Traductor/oMLX]: \"{original}\"")
                print(f"✨ [Prompt en inglés optimizado]: \"{enriched_omlx}\"\n")
            return enriched_omlx

    # 2. Vía fallback API libre + enriquecimiento heurístico
    translated = _translate_to_english_api(original)
    enriched = _enrich_english_prompt(translated, context_type)

    if verbose and enriched:
        print(f"\n🌐 [Traducción]: \"{original}\" -> \"{translated}\"")
        print(f"✨ [Prompt en inglés optimizado]: \"{enriched}\"\n")

    return enriched


def _build_cinematic_scene(en_text: str) -> str:
    """Construye una descripción de escena enriquecida siguiendo las directrices oficiales de Qwen-Image-2.1."""
    cleaned = en_text.strip().rstrip(".,; ")
    if not cleaned:
        return ""

    # Limpiar posibles meta-palabras técnicas prohibidas por la guía oficial de Qwen
    cleaned = re.sub(r'\b(4k|8k|16:9|9:16|1:1|resolution|uhd)\b', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip().rstrip(".,; ")
    lower = cleaned.lower()

    person_words = [
        "man", "woman", "person", "boy", "girl", "model", "lady", "guy",
        "child", "people", "portrait", "face", "human", "gentleman", "male", "female"
    ]
    is_person = any(re.search(r'\b' + re.escape(w) + r'\b', lower) for w in person_words)

    # Si no tiene oración de apertura con medio/estilo, estructurarla según la fórmula oficial de Qwen:
    # "The image is a [style] [medium] of [subject], set against [background/palette]."
    has_opening = any(lower.startswith(p) for p in [
        "the image is", "a photograph of", "a photo of", "an illustration of",
        "a portrait of", "a cinematic", "a realistic photograph", "a detailed photograph"
    ])

    if is_person:
        prefix = "" if has_opening else "The image is a realistic cinematic portrait photograph of "
        components = []
        # 1. Rasgos de la persona: piel, expresión, reflejos en los ojos
        if not any(w in lower for w in ["skin", "pores", "complexion", "facial", "catchlights"]):
            components.append("featuring natural skin texture, defined facial contours, authentic facial expression, and lifelike catchlights in the eyes")

        # 2. Ropa y textura textil
        if not any(w in lower for w in ["fabric", "garment", "outfit", "folds", "drape", "textile"]):
            components.append("detailed clothing texture with natural garment folds")

        # 3. Posición de manos, brazos, piernas y postura corporal (abiertas, cruzadas, en bolsillos)
        has_hands = any(w in lower for w in ["hands", "hand", "pocket", "pockets", "arms", "arm", "holding", "gesturing", "hips"])
        has_legs = any(w in lower for w in ["legs", "leg", "stance", "feet", "foot", "apart", "crossed", "seated", "sitting", "kneeling", "spread", "wide"])
        if not has_hands and not has_legs:
            components.append("standing in a grounded posture with legs positioned naturally slightly apart, hands and arms in a relaxed pose")
        elif not has_hands:
            components.append("hands and arms positioned in a relaxed natural posture")
        elif not has_legs:
            components.append("grounded stance with legs positioned firmly shoulder-width apart")

        # 4. Composición y encuadre
        if not any(w in lower for w in ["foreground", "background", "center", "centre", "position", "placed"]):
            components.append("positioned centrally in the frame with confident posture")

        # 5. Iluminación esculpida en rostro y silueta
        if not any(w in lower for w in ["lighting", "light", "sunlight", "volumetric", "neon", "shadow", "shadows"]):
            components.append("sculpted by soft volumetric portrait lighting, gentle key light on the face, and subtle rim light defining the silhouette")

        # 6. Óptica de retrato
        if not any(w in lower for w in ["lens", "shot", "angle", "view", "composition", "frame", "depth of field"]):
            components.append("captured on 50mm portrait cinema lens, f/1.8 shallow depth of field, sharp focus on the subject")
    else:
        prefix = "" if has_opening else "The image is a realistic cinematic photograph of "
        components = []
        # 1. Sujeto y composición espacial (Walk the Frame)
        if not any(w in lower for w in ["foreground", "background", "center", "centre", "position", "placed"]):
            components.append("positioned prominently in the center of the frame")

        # 2. Atmósfera y entorno circundante
        if not any(w in lower for w in ["atmosphere", "ambient", "weather", "fog", "mist", "rain", "sun", "dusk", "dawn", "night", "golden hour"]):
            components.append("with rich atmospheric depth and cohesive environmental mood across the background")

        # 3. Iluminación volumétrica y luz de borde
        if not any(w in lower for w in ["lighting", "light", "sunlight", "volumetric", "neon", "shadow", "shadows"]):
            components.append("illuminated by soft volumetric cinematic lighting with gentle rim light")

        # 4. Parámetros ópticos de cámara (lente, profundidad de campo, enfoque)
        if not any(w in lower for w in ["lens", "shot", "angle", "view", "composition", "frame", "depth of field"]):
            components.append("captured on 35mm cinema lens, shallow depth of field, f/1.8, crisp focus and tactile textures")

    core_desc = f"{prefix}{cleaned}"
    if components:
        return f"{core_desc}, {', '.join(components)}"
    return core_desc


def _enhance_with_omlx(text: str, base_url: str = DEFAULT_OMLX_URL, model: str = DEFAULT_OMLX_MODEL) -> Optional[Dict[str, str]]:
    """Usa oMLX local aplicando los criterios oficiales de Prompt Rewrite de Qwen-Image-2.1."""
    system_prompt = (
        "You are the official Qwen-Image-2.1 Master Prompt Rewriting Expert and Art Director.\n"
        "Your task is to take the user's scene brief and enhance it into a professional observer-perspective description following Qwen's official guidelines:\n"
        "1. Two Languages Decision: The descriptive prose must be in English. Any text meant to be rendered into the image must be preserved verbatim in double quotes (\"...\").\n"
        "2. Opening Sentence: 'The image is a [style] [medium] of [subject], set against [background/palette].'\n"
        "3. Walk the Frame (Spatial Mapping): Settle positions across foreground, center, upper, and background with concrete positional phrases.\n"
        "4. Human Anatomy, Limbs, and Posture (CRITICAL): When a person or portrait is described, describe in meticulous detail their facial features, natural skin texture, eye catchlights, authentic expression, hair styling, exact placement of hands and arms (e.g., hands deep in pockets, hands on hips, arms crossed, hanging at sides), exact position and stance of legs and feet (e.g., standing with legs spread shoulder-width apart / open legs, legs crossed, feet firmly planted, sitting with legs apart or crossed), and tactile garment fabric folds.\n"
        "5. Cinematic Atmosphere: Describe volumetric lighting, shadows, 35mm/50mm cinema lens, f/1.8 depth of field, and tactile textures.\n"
        "6. NO Meta-Words: Never include '4k', '8k', pixel counts, or aspect ratios in the prompt body.\n"
        "7. Provide BOTH Spanish and English versions:\n"
        "You MUST respond with a valid JSON object with this EXACT structure:\n"
        "{\n"
        '  "spanish": "Descripción cinematográfica completa en español: estilo, medio, sujeto (con detalles meticulosos de rostro, manos, postura de piernas [abiertas/cruzadas], ropa y accesorios), distribución espacial, iluminación y atmósfera.",\n'
        '  "english": "The image is a realistic photograph of... [Full English description following Qwen-Image-2.1 official structure]"\n'
        "}\n"
        "STRICT RULE: Output ONLY the raw JSON object. Do not include markdown codeblocks or extra text."
    )

    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.4,
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
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if choices:
                content = choices[0]["message"]["content"].strip()
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "spanish" in parsed and "english" in parsed:
                    es_res = parsed["spanish"].strip()
                    en_res = parsed["english"].strip()
                    en_lower = en_res.lower()
                    additions = []
                    
                    person_words = [
                        "man", "woman", "person", "boy", "girl", "model", "lady", "guy",
                        "child", "people", "portrait", "face", "human", "gentleman", "male", "female", "bald"
                    ]
                    is_person = any(re.search(r'\b' + re.escape(w) + r'\b', en_lower) for w in person_words)
                    if is_person:
                        if not any(w in en_lower for w in ["skin", "facial", "expression"]):
                            additions.append("natural skin texture and authentic facial expression")
                        if not any(w in en_lower for w in ["garment", "fabric", "folds"]):
                            additions.append("detailed garment fabric folds")

                    if not any(w in en_lower for w in ["lighting", "light", "sunlight", "volumetric"]):
                        additions.append("volumetric cinematic lighting")
                    if not any(w in en_lower for w in ["lens", "focus", "shot"]):
                        additions.append("sharp focus on cinema lens, f/1.8")
                    if not any(w in en_lower for w in ["atmospheric", "atmosphere", "depth"]):
                        additions.append("natural atmospheric depth")
                    if not any(w in en_lower for w in ["detailed", "texture", "textures"]):
                        additions.append("detailed textures")
                    if additions:
                        en_res = f"{en_res}, {', '.join(additions)}"
                    return {
                        "spanish": es_res,
                        "english": en_res
                    }
    except Exception:
        pass
    return None


def enhance_scene_prompt(prompt_text: str, verbose: bool = False) -> Dict[str, str]:
    """
    Mejora un prompt para darle mayor sentido de escena, profundidad narrativa y cinematográfica.
    
    Genera y devuelve el texto enriquecido tanto en ESPAÑOL como en INGLÉS:
    {
        "spanish": "...",
        "english": "..."
    }
    
    Args:
        prompt_text: Idea o borrador de escena introducido por el usuario.
        verbose: Si True, imprime ambas versiones en consola.
        
    Returns:
        Dict con claves 'spanish' y 'english'.
    """
    if not prompt_text or not prompt_text.strip():
        return {"spanish": "", "english": ""}

    raw_text = prompt_text.strip()

    # 1. Intentar con oMLX local (máxima calidad contextual)
    if _is_omlx_available():
        omlx_enhanced = _enhance_with_omlx(raw_text)
        if omlx_enhanced and omlx_enhanced.get("english") and omlx_enhanced.get("spanish"):
            if verbose:
                print("\n========================================================")
                print("🇪🇸 VERSIÓN EN ESPAÑOL (Sentido Narrativo y Escena):")
                print(omlx_enhanced["spanish"])
                print("--------------------------------------------------------")
                print("🇬🇧 VERSIÓN EN INGLÉS (Optimizado para Difusión Qwen-Image):")
                print(omlx_enhanced["english"])
                print("========================================================\n")
            return omlx_enhanced

    # 2. Fallback inteligente: Traducción bidireccional + Enriquecimiento cinematográfico
    en_base = _translate_to_english_api(raw_text)
    en_scene = _build_cinematic_scene(en_base)
    es_scene = _translate_api(en_scene, source_lang="en", target_lang="es")

    result = {
        "spanish": es_scene,
        "english": en_scene
    }

    if verbose:
        print("\n========================================================")
        print("🇪🇸 VERSIÓN EN ESPAÑOL (Sentido Narrativo y Escena):")
        print(result["spanish"])
        print("--------------------------------------------------------")
        print("🇬🇧 VERSIÓN EN INGLÉS (Optimizado para Difusión Qwen-Image):")
        print(result["english"])
        print("========================================================\n")

    return result

