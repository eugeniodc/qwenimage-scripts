#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configurar todas las cachés locales para que nada se descargue fuera de este directorio
export PIP_CACHE_DIR="$SCRIPT_DIR/cache/pip"
export HF_HOME="$SCRIPT_DIR/cache/huggingface"
export TORCH_HOME="$SCRIPT_DIR/cache/torch"

# Crear estructura de carpetas organizadas
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$SCRIPT_DIR/outputs" "$SCRIPT_DIR/sample_images"

echo "=========================================================="
echo "Instalando entorno virtual para Qwen-Image-2.1-Turbo"
echo "Directorio de trabajo:    $SCRIPT_DIR"
echo "Scripts:                  $SCRIPT_DIR/scripts"
echo "Muestras de entrada:      $SCRIPT_DIR/sample_images"
echo "Imágenes construidas:     $SCRIPT_DIR/outputs"
echo "Caché de modelos aislada: $SCRIPT_DIR/cache"
echo "=========================================================="

if [ ! -d "venv" ]; then
    echo "[+] Creando entorno virtual en $SCRIPT_DIR/venv..."
    python3 -m venv venv
else
    echo "[!] El entorno virtual 'venv' ya existe."
fi

echo "[+] Activando entorno virtual..."
source venv/bin/activate

# Inyectar variables de entorno en el script de activación para que persistan al usar 'source venv/bin/activate'
if ! grep -q "HF_HOME" venv/bin/activate; then
    cat << 'EOF' >> venv/bin/activate

# Cachés locales aisladas para este proyecto (rutas relativas al entorno)
export HF_HOME="$VIRTUAL_ENV/../cache/huggingface"
export TORCH_HOME="$VIRTUAL_ENV/../cache/torch"
export PIP_CACHE_DIR="$VIRTUAL_ENV/../cache/pip"
EOF
fi

echo "[+] Actualizando pip y herramientas básicas..."
pip install --upgrade pip setuptools wheel

echo "[+] Instalando dependencias requeridas (Diffusers git repo, Transformers>=5.17, Accelerate, Pillow, PyTorch)..."
pip install -r requirements.txt

echo "=========================================================="
echo "[✓] Instalación completada con éxito."
echo ""
echo "Para activar el entorno:"
echo "    cd \"$SCRIPT_DIR\""
echo "    source venv/bin/activate"
echo ""
echo "Para ejecutar las demos:"
echo "    python scripts/demo_text_to_image.py"
echo "    python scripts/demo_image_editing.py"
echo "    python scripts/demo_face_swap.py --source-face sample_images/cara.jpg --target-image sample_images/escena.jpg"
echo "=========================================================="
