"""Descarga de NCEP/DWD/BOM (CDS) y CHIRPS (UCSB)."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from ideam_seasonal.common import (Region, chirps_path, load_config, raw_path,  # noqa: F401
                                   resolve_region)


def build_request(cfg: dict, centre: str, variable: str = "precipitation",
                  region: Region | None = None, month: int | None = None,
                  years: list[int] | None = None) -> dict:
    """Sin `region`, la caja de prueba (`area`)."""
    c = cfg["centres"][centre]
    if None in (c["originating_centre"], c["system"], c["years"]):
        raise NotImplementedError(
            f"'{centre}': originating_centre/system/años sin confirmar en config/params.yaml. "
            "Verificar en el formulario del CDS."
        )
    if variable not in cfg["variables"]:
        raise NotImplementedError(f"Variable '{variable}' no confirmada (ver hallazgos.md).")
    region = region or resolve_region(cfg)
    start, end = c["years"]
    month = month or cfg["month"]
    years = years or list(range(start, end + 1))
    request = {
        "originating_centre": c["originating_centre"],
        "system": c["system"],
        "variable": [cfg["variables"][variable]],
        "product_type": [cfg["product_type"]],
        "year": [str(y) for y in years],
        "month": [f"{month:02d}"],
        "leadtime_month": [str(m) for m in cfg["leadtime_months"]],
        "data_format": cfg["data_format"],
    }
    if region.area is not None:
        a = region.area
        request["area"] = [a["north"], a["west"], a["south"], a["east"]]
    return request


def download(cfg: dict, centre: str, variable: str = "precipitation",
             out_dir: Path = Path("data") / "raw", region: Region | None = None,
             force: bool = False, month: int | None = None,
             forecast_year: int | None = None) -> Path:
    """Reutiliza el GRIB si el request guardado es igual."""
    region = region or resolve_region(cfg)
    month = month or cfg["month"]
    years = [forecast_year] if forecast_year else None
    request = build_request(cfg, centre, variable, region, month, years)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = raw_path(out_dir, centre, variable, month, region, forecast_year)
    sidecar = target.with_suffix(".request.json")
    cached = (target.exists() and sidecar.exists()
              and json.loads(sidecar.read_text(encoding="utf-8")) == request)
    if cached and not force:
        print(f"[cache] {target}")
        return target

    import cdsapi

    cdsapi.Client().retrieve(cfg["dataset"], request).download(str(target))
    sidecar.write_text(json.dumps(request, indent=2), encoding="utf-8")
    print(f"[ok] {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def forecast_unavailable(exc: BaseException) -> bool:
    """Pronóstico aún no publicado: HTTPError 400 con error de MARS (comprobado 2026-11-21)."""
    from requests import HTTPError

    if not isinstance(exc, HTTPError):
        return False
    status = getattr(exc.response, "status_code", None)
    return status == 400 and "mars" in str(exc).lower()


def _mars_reason(exc: BaseException) -> str:
    for line in str(exc).splitlines():
        msg = line.split(" - ")[-1].strip()
        if line.lower().startswith("mars") and "Error:" in msg:
            return msg[:160]
    return str(exc).splitlines()[0][:160]


def latest_forecast(cfg: dict, centres, variable: str = "precipitation",
                    region: Region | None = None, force: bool = False,
                    today: dt.date | None = None,
                    months_back: int = 3) -> tuple[int, int, dict[str, Path]]:
    """Último mes publicado para todos los centros; solo retrocede si no está publicado."""
    today = today or dt.date.today()
    y, m = today.year, today.month
    for _ in range(months_back):
        try:
            paths = {c: download(cfg, c, variable, region=region, force=force,
                                 month=m, forecast_year=y)
                     for c in centres}
            return y, m, paths
        except Exception as exc:
            if not forecast_unavailable(exc):
                raise
            print(f"[--] pronóstico {y}-{m:02d} aún no publicado para todos los centros "
                  f"({_mars_reason(exc)}); se prueba el mes anterior")
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    raise RuntimeError(f"ningún pronóstico publicado en los últimos {months_back} meses")


def download_chirps_year(cfg: dict, year: int, out_dir: Path = Path("data") / "raw",
                         force: bool = False) -> Path:
    """Un año de CHIRPS recortado a la caja de prueba."""
    import requests
    import xarray as xr

    out_dir.mkdir(parents=True, exist_ok=True)
    region = resolve_region(cfg, "prueba")
    a = region.area
    target = chirps_path(out_dir, year, region)
    if target.exists() and not force:
        print(f"[cache] {target}")
        return target
    url = cfg["chirps"]["url_template"].format(year=year)
    full = out_dir / f"chirps_v3_{year}_global.nc.part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(full, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    with xr.open_dataset(full, mask_and_scale=False, decode_times=False) as ds:
        lat, lon = ds["latitude"], ds["longitude"]
        box = ds.where((lat >= a["south"]) & (lat <= a["north"])
                       & (lon >= a["west"]) & (lon <= a["east"]), drop=True)
        box.load().to_netcdf(target)
    full.unlink()
    print(f"[ok] {target} ({target.stat().st_size / 1e6:.2f} MB)")
    return target


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("centre", choices=[*cfg["centres"], "chirps"])
    p.add_argument("--variable", default="precipitation", choices=cfg["variables"])
    p.add_argument("--year", type=int, help="solo CHIRPS: un año (por defecto todo el rango)")
    p.add_argument("--global", dest="is_global", action="store_true",
                   help="descarga global en vez de la caja de prueba")
    p.add_argument("--force", action="store_true", help="ignora la caché")
    a = p.parse_args()
    if a.centre == "chirps":
        y0, y1 = cfg["chirps"]["years"]
        for y in ([a.year] if a.year else range(y0, y1 + 1)):
            download_chirps_year(cfg, y, force=a.force)
        return
    region = resolve_region(cfg, "global" if a.is_global else "prueba")
    download(cfg, a.centre, a.variable, region=region, force=a.force)


if __name__ == "__main__":
    main()
