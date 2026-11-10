#!/usr/bin/env bash
#   bash cpt.sh [--modelo ncep|dwd|bom] [--variable precipitation|temperature] [--force]
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

"$PY" -m ideam_seasonal.cpt "$@"
echo "Archivos en data/cpt/"
