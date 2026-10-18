# AGENTS.md — Qwen-Image-2.1-Turbo & Agente oMLX

## 1. Contexto del Proyecto
- **Propósito:** Generación, edición y transformación multimodal de imágenes mediante Qwen-Image-2.1-Turbo (Diffusers 8 pasos en MPS) y un Agente Autónomo local conectado al servidor oMLX (`http://localhost:8000/v1`).
- **Stack Tecnológico:** Python 3.14, PyTorch MPS (Apple Silicon Metal), Hugging Face Diffusers, oMLX API.
- **Arquitectura:**
  - `agent.py`: Orquestador conversacional autónomo con auto-recuperación y normalización de parámetros.
  - `core/tools.py`: Registro tipado de herramientas visuales (`generate_image`, `edit_image`, `face_swap`, `group_photo`, `outfit_swap`, `camera_angle`, `outpaint`, `restore`, `image_to_prompt`, `enhance_prompt`, `merge_lora`).
  - `core/prompt_utils.py`: Módulo unificado de traducción al inglés y enriquecimiento escénico según directrices oficiales de Qwen.
  - `core/image_interrogator.py`: Inferencia de prompt inverso estructurado (BLIP local + oMLX Vision).
  - `scripts/*.py`: Scripts deterministas de difusión por tarea.
  - `cache/`: Aislamiento obligatorio de modelos (`HF_HOME`, `TORCH_HOME`).
  - `outputs/`: Carpeta de almacenamiento de imágenes generadas.

## 2. Guardrails Inmutables
1. **Regla de Oro de Prompts:** Todo prompt hacia difusión DEBE traducirse al inglés y enriquecerse siguiendo las directrices oficiales de Qwen (apertura compositiva, iluminación volumétrica, profundidad de lente, cláusula de preservación y eliminación de meta-palabras tipo 4K/8K/resolución).
2. **Aislamiento de Entorno:** Trabajar siempre dentro de `./venv` y con almacenamiento de modelos exclusivamente en `./cache/`.
3. **Autonomía y Ejecución Directa:** Ejecutar comprobaciones de sintaxis y tests unitarios de forma proactiva sin pausas innecesarias.

## 3. Comandos de Verificación
- **Tests Unitarios e Integración:**
  ```bash
  ./venv/bin/python -m unittest tests/test_agent_flow.py tests/test_scripts_prompt_integration.py tests/test_prompt_utils.py tests/test_image_interrogator.py
  ```
- **Comprobación de Sintaxis:**
  ```bash
  ./venv/bin/python -m py_compile scripts/*.py core/*.py agent.py menu.py
  ```
- **Lanzadores del Sistema:**
  - Menú interactivo: `./start.sh` o `python menu.py`
  - Asistente autónomo oMLX: `./start.sh --agent` o `python agent.py`
