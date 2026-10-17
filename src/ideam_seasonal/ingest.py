"""Descarga de NCEP/DWD/BOM (CDS) y CHIRPS (UCSB)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

CONFIG_PATH = Path("config") / "params.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_request(cfg: dict, centre: str, variable: str = "precipitation",
                  use_area: bool = True) -> dict:
    """Request del CDS desde la config; `use_area=False` es global."""
    c = cfg["centres"][centre]
    if None in (c["originating_centre"], c["system"], c["years"]):
        raise NotImplementedError(
            f"'{centre}': originating_centre/system/años sin confirmar en config/params.yaml. "
            "Verificar en el formulario del CDS."
        )
    if variable not in cfg["variables"]:
        raise NotImplementedError(f"Variable '{variable}' no confirmada (ver hallazgos.md).")
    start, end = c["years"]
    request = {
        "originating_centre": c["originating_centre"],
        "system": c["system"],
        "variable": [cfg["variables"][variable]],
        "product_type": [cfg["product_type"]],
        "year": [str(y) for y in range(start, end + 1)],
        "month": [f"{cfg['month']:02d}"],
        "leadtime_month": [str(m) for m in cfg["leadtime_months"]],
        "data_format": cfg["data_format"],
    }
    if use_area:
        a = cfg["area"]
        request["area"] = [a["north"], a["west"], a["south"], a["east"]]
    return request


def download(cfg: dict, centre: str, variable: str = "precipitation",
             out_dir: Path = Path("data") / "raw", use_area: bool = True,
             force: bool = False) -> Path:
    """Descarga con caché por request."""
    request = build_request(cfg, centre, variable, use_area)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{centre}_{variable}_m{cfg['month']:02d}_{'box' if use_area else 'global'}"
    target, sidecar = out_dir / f"{stem}.grib", out_dir / f"{stem}.request.json"
    cached = (target.exists() and sidecar.exists()
              and json.loads(sidecar.read_text(encoding="utf-8")) == request)
    if cached and not force:
        print(f"[cache] {target}")
        return target

    import cdsapi  # tardío: los tests no necesitan credenciales

    cdsapi.Client().retrieve(cfg["dataset"], request).download(str(target))
    sidecar.write_text(json.dumps(request, indent=2), encoding="utf-8")
    print(f"[ok] {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def download_chirps_year(cfg: dict, year: int, out_dir: Path = Path("data") / "raw",
                         force: bool = False) -> Path:
    """Baja un año de CHIRPS, lo recorta a `area` y borra el global."""
    import requests
    import xarray as xr

    out_dir.mkdir(parents=True, exist_ok=True)
    a = cfg["area"]
    target = out_dir / f"chirps_v3_{year}_box.nc"
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
    download(cfg, a.centre, a.variable, use_area=not a.is_global, force=a.force)


if __name__ == "__main__":
    main()
