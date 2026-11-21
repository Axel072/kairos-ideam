#!/usr/bin/env bash
# Comando único del proyecto: archivos de entrada de CPT con el pronóstico más reciente.
#
#   bash kairos.sh                 corrida normal (todo desde config/params.yaml)
#   bash kairos.sh --help          todas las opciones
#   bash kairos.sh qc              control de calidad de la Fase 2
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONIOENCODING=utf-8  # tildes legibles en la consola de Windows

if [[ -x venv/Scripts/python.exe ]]; then
    PY=venv/Scripts/python.exe
elif [[ -x venv/bin/python ]]; then
    PY=venv/bin/python
else
    echo "No se encontró el venv. Crea uno con: python -m venv venv && pip install -e ." >&2
    exit 1
fi

exec "$PY" -m ideam_seasonal.kairos "$@"
