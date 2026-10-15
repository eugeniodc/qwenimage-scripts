# Demo Qwen-Image-2.1-Turbo

Proyecto organizado según los estándares de la comunidad de IA para el modelo **[Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)** de Alibaba Cloud / Qwen Team.

---

## ⚡ Inicio Rápido (Menú Interactivo)

Para no tener que recordar comandos ni rutas, hemos añadido un **menú interactivo**:

```bash
cd qwen_image_2_1_turbo
./start.sh
```

*(O si lo prefieres: `python3 menu.py`)*

### ¿Qué hace el menú?
* Te permite elegir mediante números las imágenes de `sample_images/` (como `eugenio.jpg`).
* Genera texto a imagen, edición, intercambio de caras (*face swap*) y cambios de ángulo de cámara con un par de pulsaciones.
* Muestra el tamaño que ocupan los modelos en la carpeta `cache/` y te permite limpiarlos.
* Lista las imágenes generadas en `outputs/` y te ofrece abrirlas en el Finder de macOS.

---

## 📂 Estructura del Proyecto

```text
qwen_image_2_1_turbo/
├── start.sh                  # 👈 LANZADOR RÁPIDO (activa venv y abre el menú)
├── menu.py                   # Panel interactivo principal (auto-activa venv)
├── scripts/                  # Scripts ejecutables individuales
│   ├── demo_text_to_image.py # Generación texto a imagen
│   ├── demo_image_editing.py # Edición a partir de bocetos
│   ├── demo_image_editing_uncensored.py # 👈 Edición SIN CENSURA (AtomicChat GGUF)
│   ├── demo_face_swap.py     # Intercambio de caras (LoRA BFS)
│   ├── demo_group_photo.py   # 👈 Unir múltiples personas en una misma foto
│   ├── demo_outfit_swap.py   # Cambio de ropa (LoRA Outfit Consistency)
│   ├── demo_camera_angle.py  # Cambio de ángulo de cámara (LoRA Multiple-Angles <mva>)
│   ├── demo_outpaint.py      # Expansión de lienzo / Uncrop (LoRA Outpaint)
│   ├── demo_restore.py       # Restauración y súper-resolución (LoRA Restore)
│   ├── merge_lora.py         # Fusión y exportación de LoRAs (Merge & Fuse)
│   └── demo_cli.py           # Herramienta CLI configurable con aspect ratios
├── sample_images/            # Imágenes de muestra / entradas de referencia
├── outputs/                  # Imágenes construidas / resultados generados
├── cache/                    # Almacenamiento local aislado (pesos 7B, LoRAs y pip)
├── requirements.txt
├── setup_env.sh
└── README.md
```

---

## 💻 Ejecución Manual por Comandos

Si prefieres ejecutar los scripts directamente sin el menú:

### 1. Intercambio de Caras (Face Swap)
* **Modo Nativo:**
  ```bash
  python scripts/demo_face_swap.py \
    --source-face sample_images/eugenio.jpg \
    --target-image sample_images/iron-man-2008-1626678790.avif \
    --output eugenio_ironman.png
  ```

* **Con LoRA especializado BFS (Best Face Swap):**
  ```bash
  python scripts/demo_face_swap.py \
    --source-face sample_images/eugenio.jpg \
    --target-image sample_images/iron-man-2008-1626678790.avif \
    --lora "Alissonerdx/BFS-Best-Face-Swap" \
    --output eugenio_ironman_bfs.png
  ```

### 2. Unir Personas en una Misma Foto (Group Photo Merge)
* **Modo Nueva Escena (los sitúa en un lugar nuevo juntos):**
  ```bash
  python scripts/demo_group_photo.py \
    --person1 sample_images/eugenio.jpg \
    --person2 sample_images/mujer.avif \
    --mode new_scene \
    --scene "en una cafetería moderna y acogedora con iluminación cálida" \
    --interaction "hombro con hombro sonriendo a la cámara" \
    --aspect-ratio 4:3 \
    --output eugenio_amiga_cafe.png
  ```

* **Modo Mantener Fondo (añade a la segunda persona al entorno de la primera):**
  ```bash
  python scripts/demo_group_photo.py \
    --person1 sample_images/eugenio.jpg \
    --person2 sample_images/mujer.avif \
    --mode keep_scene \
    --interaction "de pie al lado sonriendo juntos amistosamente" \
    --output eugenio_amiga_juntos.png
  ```

### 3. Cambio de Ropa (Outfit Swap / Virtual Try-On)
```bash
python scripts/demo_outfit_swap.py \
  --person-image sample_images/eugenio.jpg \
  --outfit-image sample_images/ropa_referencia.jpg \
  --output eugenio_nuevo_outfit.png
```

### 3. Cambio de Ángulo de Cámara (Camera Angle / LoRA)
* **Modo Nativo 3D:**
  ```bash
  python scripts/demo_camera_angle.py \
    --image sample_images/eugenio.jpg \
    --angle profile_right
  ```

* **Con LoRA oficial Multiple-Angles (<mva>):**
  ```bash
  python scripts/demo_camera_angle.py \
    --image sample_images/eugenio.jpg \
    --angle profile_right \
    --lora "akhaliq/Qwen-Image-2.1-Multiple-Angles-LoRA"
  ```

### 4. Expansión de Lienzo (Outpainting / Uncrop)
```bash
python scripts/demo_outpaint.py \
  --image sample_images/eugenio.jpg \
  --ratio 16:9 \
  --output eugenio_panoramico.png
```

### 5. Restauración y Mejora de Fotos (Restore & Upscale)
```bash
python scripts/demo_restore.py \
  --image sample_images/eugenio.jpg \
  --mode modern \
  --output eugenio_restaurado.png
```

### 6. Texto a Imagen
```bash
python scripts/demo_text_to_image.py
```

### 7. CLI General
```bash
python scripts/demo_cli.py \
  --prompt "Un astronauta en Marte explorando ruinas alienígenas, iluminación cinematográfica" \
  --aspect-ratio 16:9 \
  --output marte_ruinas.png
```

### 8. Fusión y Unión de LoRAs (Merge & Fuse LoRA)
Permite fusionar matemáticamente los pesos de cualquier LoRA (de Hugging Face o archivo `.safetensors` local) en el modelo base para acelerar la inferencia o exportar un checkpoint unificado:

* **Fusión + Generación de prueba inmediata:**
  ```bash
  python scripts/merge_lora.py \
    --lora "Alissonerdx/BFS-Best-Face-Swap" \
    --lora-scale 1.0 \
    --prompt "A photorealistic portrait of an astronaut on Mars" \
    --output outputs/prueba_fused.png
  ```

* **Fusión + Exportación permanente del modelo completo (save_pretrained):**
  ```bash
  python scripts/merge_lora.py \
    --lora "/ruta/a/tu_lora_local.safetensors" \
    --lora-scale 1.0 \
    --save-dir "models/qwen_custom_fused"
  ```

### 9. Edición de Imagen Sin Censura (Uncensored / AtomicChat)
Utiliza la versión desensurada y abliterada de [AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF) que elimina las directivas internas de rechazo moral/ético para permitir cualquier modificación de la imagen:

```bash
python scripts/demo_image_editing_uncensored.py \
  --image sample_images/eugenio.jpg \
  --prompt "In <image1>, transform the outfit into a futuristic cyberpunk armor, retaining face and pose" \
  --fast \
  --output outputs/edit_uncensored.png
```


