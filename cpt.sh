#!/usr/bin/env bash
# compatibilidad: equivale a kairos.sh
exec bash "$(dirname "${BASH_SOURCE[0]}")/kairos.sh" "$@"
