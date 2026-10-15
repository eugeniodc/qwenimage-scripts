#!/usr/bin/env bash
# Lanzador rápido que activa el entorno virtual y abre el menú
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "[!] No se encontró el entorno virtual 'venv'. Ejecutando instalación inicial..."
    bash setup_env.sh
fi

source venv/bin/activate
if [ "$1" = "--agent" ] || [ "$1" = "-a" ]; then
    python agent.py
else
    python menu.py
fi
