#!/usr/bin/env bash
# Lanzador rápido que activa el entorno virtual y abre el menú
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "[!] No se encontró el entorno virtual 'venv'. Ejecutando instalación inicial..."
    bash setup_env.sh
fi

source venv/bin/activate
python menu.py
