"""Comando único: archivos de entrada de CPT (X, Z) con el pronóstico más reciente.

Sin opciones usa config/params.yaml: pronóstico más reciente publicado en el CDS, hindcast
del mismo mes de inicio, los tres modelos y su promedio, lluvia y temperatura, región y
temporada de `cpt`. Cada opción reemplaza ese valor solo en esta corrida.

Ejemplos:
  bash kairos.sh
  bash kairos.sh --region 5,-75,-2,-70            cuadro propio N,W,S,E
  bash kairos.sh -m ncep -v precipitation          un modelo, una variable
  bash kairos.sh --modelo ncep,dwd                 dos modelos y su promedio
  bash kairos.sh --inicio 2026-10 --temporada 2-4  un inicio y una temporada concretos
  bash kairos.sh --anios 1995-2015 --salida otra/carpeta
  bash kairos.sh qc [--force]                      control de calidad de la Fase 2
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from ideam_seasonal.common import CENTRES, load_config, resolve_region

VARIABLES_CLI = ("precipitation", "temperature")


# ---------------------------------------------------------------------------------------------
# Lectura de valores (errores claros para argparse)
# ---------------------------------------------------------------------------------------------

def parse_models(values: list[str] | None) -> list[str]:
    """`-m ncep -m dwd` o `-m ncep,dwd` -> ['ncep', 'dwd'] (orden de CENTRES, sin repetir)."""
    if not values:
        return list(CENTRES)
    asked = {v.strip().lower() for item in values for v in item.split(",") if v.strip()}
    unknown = asked - set(CENTRES)
    if unknown:
        raise argparse.ArgumentTypeError(f"modelo(s) desconocido(s): {', '.join(sorted(unknown))}; "
                                         f"usa {', '.join(CENTRES)}")
    return [c for c in CENTRES if c in asked]


def parse_start(text: str) -> tuple[int, int]:
    """'2026-10' -> (2026, 10)."""
    try:
        y, m = (int(p) for p in text.split("-"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"inicio '{text}': usa AAAA-MM, p. ej. 2026-10") from None
    if not (1 <= m <= 12 and 1981 <= y <= 2100):
        raise argparse.ArgumentTypeError(f"inicio '{text}': mes 1-12 y año razonable")
    return y, m


def parse_leads(text: str) -> list[int]:
    """'2-4' -> [2, 3, 4]; '3' -> [3]. Lead 1 = mes de inicio."""
    try:
        parts = [int(p) for p in text.split("-")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"temporada '{text}': usa leads como 2-4 o 3") from None
    if len(parts) == 1:
        parts *= 2
    if len(parts) != 2 or not (1 <= parts[0] <= parts[1] <= 6):
        raise argparse.ArgumentTypeError(f"temporada '{text}': leads entre 1 y 6, de menor a mayor")
    return list(range(parts[0], parts[1] + 1))


def parse_years(text: str) -> tuple[int, int]:
    """'1993-2016' -> (1993, 2016)."""
    try:
        y0, y1 = (int(p) for p in text.split("-"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"años '{text}': usa AAAA-AAAA, p. ej. 1993-2016") from None
    if y0 > y1:
        raise argparse.ArgumentTypeError(f"años '{text}': el primero debe ser menor o igual")
    return y0, y1


# ---------------------------------------------------------------------------------------------
# Opciones
# ---------------------------------------------------------------------------------------------

def build_parser(cfg: dict) -> argparse.ArgumentParser:
    c = cfg["cpt"]
    y0, y1 = cfg["centres"]["ncep"]["years"]
    lead_default = f"{c['season_leads'][0]}-{c['season_leads'][-1]}"
    p = argparse.ArgumentParser(prog="kairos.sh", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-r", "--region", metavar="REGION",
                   help="prueba, global, " + ", ".join(c["regions"]) + " o un cuadro N,W,S,E "
                        f"(longitudes con signo). Por defecto: {c['region']}")
    p.add_argument("-m", "--modelo", action="append", metavar="MODELO",
                   help=f"{', '.join(CENTRES)}; repetible o separado por comas. Con más de uno "
                        "se escribe también el promedio. Por defecto: los tres")
    p.add_argument("-v", "--variable", choices=[*VARIABLES_CLI, "ambas"], default="ambas",
                   help="Por defecto: ambas")
    p.add_argument("-i", "--inicio", type=parse_start, metavar="AAAA-MM",
                   help="mes de inicio del pronóstico. Por defecto: el más reciente publicado")
    p.add_argument("-t", "--temporada", type=parse_leads, metavar="LEADS",
                   help=f"leads que forman la temporada (1 = mes de inicio). Por defecto: {lead_default}")
    p.add_argument("-a", "--anios", type=parse_years, metavar="AAAA-AAAA",
                   help=f"periodo del hindcast. Por defecto: {y0}-{y1}")
    p.add_argument("-o", "--salida", type=Path, default=Path("data") / "cpt", metavar="CARPETA",
                   help="carpeta de los archivos CPT. Por defecto: data/cpt")
    p.add_argument("-f", "--force", action="store_true",
                   help="vuelve a descargar aunque el archivo esté en data/raw/")
    return p


def build_qc_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kairos.sh qc",
        description="Control de calidad de la Fase 2 (caja de prueba, mes de params.yaml): "
                    "descarga, verificación, homogeneización e informe.")
    p.add_argument("-f", "--force", action="store_true", help="vuelve a descargar")
    return p


# ---------------------------------------------------------------------------------------------
# Corridas
# ---------------------------------------------------------------------------------------------

def run_cpt(args: argparse.Namespace, command: str) -> list[Path]:
    from ideam_seasonal import cpt

    models = parse_models(args.modelo)
    variables = list(VARIABLES_CLI) if args.variable == "ambas" else [args.variable]
    written = []
    for v in variables:
        print(f"\n== {v} ==")
        written += cpt.build(models, v, force=args.force, out_dir=args.salida,
                             region=args.region, start=args.inicio, leads=args.temporada,
                             years=args.anios, command=command)
    return written


def run_qc(args: argparse.Namespace) -> None:
    from ideam_seasonal import homogenize, ingest, verify

    cfg = load_config()
    region = resolve_region(cfg, "prueba")
    print("== 1/4 Descarga ==")
    paths = [ingest.download(cfg, c, region=region, force=args.force) for c in CENTRES]
    y0, y1 = cfg["chirps"]["years"]
    for y in range(y0, y1 + 1):
        ingest.download_chirps_year(cfg, y, force=args.force)
    print("== 2/4 Verificación de cada GRIB ==")
    for p in paths:
        if not verify.verify(p):
            raise ValueError(f"{p.name} no pasó la verificación; se detiene el control de calidad")
    print("== 3/4 Homogeneización ==")
    print(homogenize.build())
    print("== 4/4 Informe de control de calidad ==")
    print(verify.build_report())


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cfg = load_config()
    command = "bash kairos.sh" + ("" if not argv else " " + shlex.join(argv))
    if argv[:1] == ["qc"]:
        args = build_qc_parser().parse_args(argv[1:])
        runner = lambda: run_qc(args)  # noqa: E731
    else:
        parser = build_parser(cfg)
        args = parser.parse_args(argv)
        try:  # errores de opciones antes de descargar nada
            resolve_region(cfg, args.region, default=cfg["cpt"]["region"])
            args.modelo = parse_models(args.modelo)
        except (ValueError, argparse.ArgumentTypeError) as e:
            parser.error(str(e))
        runner = lambda: [print(f"[ok] {p}") for p in run_cpt(args, command)]  # noqa: E731
    try:
        runner()
    except (ValueError, RuntimeError, FileNotFoundError, NotImplementedError) as e:
        sys.stdout.flush()
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
