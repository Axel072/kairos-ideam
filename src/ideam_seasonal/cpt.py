"""
Uso:  python -m ideam_seasonal.cpt [--modelo ncep|dwd|bom] [--variable precipitation|temperature]
                                   [--region colombia|prueba|global]
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
from pathlib import Path

import numpy as np
import xarray as xr

from ideam_seasonal.common import CENTRES, open_grib
from ideam_seasonal.homogenize import reduce_lagged
from ideam_seasonal.ingest import download, latest_forecast, load_config
from ideam_seasonal.verify import verify

CPT_HEADER = "xmlns:cpt=http://iri.columbia.edu/CPT/v10/"
MISSING = -999.0
ENSEMBLE = "promedio"


def season_mean(ds: xr.Dataset, leads: list[int]) -> xr.DataArray:
    name = "tas" if "tas" in ds else "pr"
    return ds[name].sel(lead=leads).mean(("member", "lead"), keep_attrs=True)


def season_label(year: int, start_month: int, leads: list[int]) -> str:
    first = start_month - 1 + leads[0] - 1
    last = start_month - 1 + leads[-1] - 1
    return f"{year + first // 12}-{first % 12 + 1:02d}/{year + last // 12}-{last % 12 + 1:02d}"


def write_cpt(da: xr.DataArray, path: Path, field: str, start_month: int,
              leads: list[int]) -> Path:
    da = da.transpose("year", "lat", "lon").sortby("lat", ascending=False)
    lat, lon = da["lat"].values, da["lon"].values
    lead = leads[0] - 0.5
    buf = io.StringIO()
    buf.write(CPT_HEADER + "\n")
    buf.write("cpt:nfields=1\n")
    for i, year in enumerate(da["year"].values):
        tags = [f"cpt:field={field}",
                f"cpt:L={lead:g} months",
                f"cpt:S={int(year)}-{start_month:02d}-01T00:00",
                f"cpt:T={season_label(int(year), start_month, leads)}",
                f"cpt:nrow={lat.size}", f"cpt:ncol={lon.size}",
                "cpt:row=Y", "cpt:col=X",
                f"cpt:units={da.attrs.get('units', 'mm/day')}",
                f"cpt:missing={MISSING:g}"]
        buf.write(", ".join(tags) + "\n")
        buf.write("\t" + "\t".join(f"{v:g}" for v in lon) + "\n")
        values = np.where(np.isfinite(da.values[i]), da.values[i], MISSING)
        np.savetxt(buf, np.column_stack([lat, values]), fmt="%.6g", delimiter="\t")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue(), encoding="utf-8")
    return path


def ensemble(fields: dict[str, xr.DataArray]) -> xr.DataArray:
    ref = next(iter(fields.values()))
    for name, da in fields.items():
        for coord in ("year", "lat", "lon"):
            if not np.allclose(da[coord], ref[coord]):
                raise ValueError(f"{name}: '{coord}' distinto; no se promedia")
    out = xr.concat(list(fields.values()), dim="model").mean("model", keep_attrs=True)
    out.attrs = ref.attrs
    return out


def region_args(cfg: dict, region: str) -> dict:
    if region == "global":
        return {"use_area": False, "region": "global"}
    if region == "prueba":
        return {"use_area": True, "region": "box"}
    return {"use_area": True, "area": cfg["cpt"]["regions"][region], "region": region}


def write_log(path: Path, cfg: dict, variable: str, year: int, month: int, leads: list[int],
              region: str, centres: list[str], members: dict, files: dict,
              grid: xr.DataArray) -> Path:
    if region == "global":
        bounds = "global"
    else:
        a = cfg["cpt"]["regions"][region] if region != "prueba" else cfg["area"]
        bounds = f"N {a['north']}, W {a['west']}, S {a['south']}, E {a['east']}"
    y0, y1 = (int(v) for v in grid["year"].values[[0, -1]])
    corrida = "A" if len(centres) == 1 else "A y B"
    lines = [f"# Registro de corrida: {variable}, inicio {year}-{month:02d}", "",
             "| Campo | Valor |", "|---|---|",
             f"| Fecha | {dt.datetime.now():%Y-%m-%d %H:%M} |",
             f"| Corrida | {corrida} |",
             f"| Variable | {variable} ({grid.attrs.get('units', '')}) |",
             f"| Inicio del pronóstico | {year}-{month:02d} |",
             f"| Temporada | {season_label(year, month, leads)} (leads {', '.join(map(str, leads))}) |",
             f"| Región | {region}: {bounds} |",
             f"| Rejilla | {grid.sizes['lat']} × {grid.sizes['lon']} celdas |",
             f"| Años del hindcast | {y0}–{y1} |",
             "",
             "| Modelo | Sistema | Miembros X | Miembros Z | Archivo X | Archivo Z |",
             "|---|---|---|---|---|---|"]
    for name in files:
        if name == ENSEMBLE:
            system = "media de " + " + ".join(c.upper() for c in centres)
            mx = mz = "—"
        else:
            system = cfg["centres"][name]["system"]
            mx, mz = members[name]
        lines.append(f"| {name} | {system} | {mx} | {mz} | {files[name]['X'].name} | "
                     f"{files[name]['Z'].name} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build(centres: list[str], variable: str = "precipitation", force: bool = False,
          out_dir: Path = Path("data") / "cpt", region: str | None = None) -> list[Path]:
    cfg = load_config()
    leads = cfg["cpt"]["season_leads"]
    n_leads = len(cfg["leadtime_months"])
    region = region or cfg["cpt"]["region"]
    reg = region_args(cfg, region)

    year, month, z_raw = latest_forecast(cfg, centres, variable, force=force, **reg)
    print(f"Pronóstico más reciente: inicio {year}-{month:02d}")
    x_raw = {c: download(cfg, c, variable, force=force, month=month, **reg) for c in centres}

    for p in [*x_raw.values(), *z_raw.values()]:
        if not verify(p):
            raise ValueError(f"{p.name} no pasó la verificación; no se escriben archivos CPT")

    x, z, members = {}, {}, {}
    for c in centres:
        hx = reduce_lagged(open_grib(x_raw[c]), month, n_leads)
        hz = reduce_lagged(open_grib(z_raw[c]), month, n_leads)
        members[c] = (hx.sizes["member"], hz.sizes["member"])
        x[c], z[c] = season_mean(hx, leads), season_mean(hz, leads)
    if len(centres) > 1:
        x[ENSEMBLE], z[ENSEMBLE] = ensemble(x), ensemble(z)

    tag = f"{variable}_i{year}{month:02d}_{region}"
    files = {}
    for name in x:
        files[name] = {kind: write_cpt(data[name], out_dir / f"{kind}_{name}_{tag}.txt",
                                       f"{variable}_{name}", month, leads)
                       for kind, data in (("X", x), ("Z", z))}
    log = write_log(out_dir / f"registro_{tag}.md", cfg, variable, year, month, leads, region,
                    centres, members, files, x[centres[0]])
    return [p for f in files.values() for p in f.values()] + [log]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", choices=CENTRES,
                   help="solo ese modelo (X y Z); sin esta opción, los tres y el promedio")
    p.add_argument("--force", action="store_true", help="ignora la caché de data/raw/")
    p.add_argument("--variable", choices=["precipitation", "temperature", "ambas"], default="ambas")
    p.add_argument("--region", choices=["colombia", "prueba", "global"],
                   help="por defecto la de config/params.yaml (cpt.region)")
    a = p.parse_args()
    variables = ["precipitation", "temperature"] if a.variable == "ambas" else [a.variable]
    for v in variables:
        for path in build([a.modelo] if a.modelo else list(CENTRES), v, force=a.force,
                          region=a.region):
            print(f"[ok] {path}")


if __name__ == "__main__":
    main()
