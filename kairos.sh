#!/usr/bin/env bash
# bash kairos.sh [opciones]   (bash kairos.sh --help)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONIOENCODING=utf-8  # tildes en la consola de Windows

if [[ -x venv/Scripts/python.exe ]]; then
    PY=venv/Scripts/python.exe
elif [[ -x venv/bin/python ]]; then
    PY=venv/bin/python
else
    echo "No se encontró el venv. Crea uno con: python -m venv venv && pip install -e ." >&2
    exit 1
fi

exec "$PY" -m ideam_seasonal.kairos "$@"
