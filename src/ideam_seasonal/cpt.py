"""Escritura de los archivos de entrada de CPT (X, Z) y verificación del Y.

Se usa desde `bash kairos.sh` (ver `ideam_seasonal.kairos`); `python -m ideam_seasonal.cpt`
acepta las mismas opciones.
"""
from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import numpy as np
import xarray as xr

from ideam_seasonal.common import (Region, load_config, open_grib, predictand_path,
                                   resolve_region)
from ideam_seasonal.homogenize import monthly_mean, reduce_lagged
from ideam_seasonal.ingest import download, forecast_unavailable, latest_forecast
from ideam_seasonal.verify import verify, verify_predictand

CPT_HEADER = "xmlns:cpt=http://iri.columbia.edu/CPT/v10/"
MISSING = -999.0
ENSEMBLE = "promedio"


def season_mean(ds: xr.Dataset, leads: list[int]) -> xr.DataArray:
    """Temporada: promedio de los valores mensuales (`monthly_mean`) de los leads dados."""
    return monthly_mean(ds).sel(lead=leads).mean("lead", keep_attrs=True)


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


def write_log(path: Path, cfg: dict, variable: str, year: int, month: int, leads: list[int],
              region: Region, centres: list[str], members: dict, files: dict,
              grid: xr.DataArray, predictand: Path | None = None,
              command: str | None = None, start_mode: str = "más reciente") -> Path:
    y0, y1 = (int(v) for v in grid["year"].values[[0, -1]])
    corrida = "A" if len(centres) == 1 else "A y B"
    y_text = (f"{predictand.name} (verificado)" if predictand
              else "sin definir para esta variable: solo se escriben X y Z")
    lines = [f"# Registro de corrida: {variable}, inicio {year}-{month:02d}", "",
             "| Campo | Valor |", "|---|---|",
             f"| Fecha | {dt.datetime.now():%Y-%m-%d %H:%M} |",
             *([f"| Comando | `{command}` |"] if command else []),
             f"| Corrida | {corrida} |",
             f"| Modelos | {', '.join(centres)} |",
             f"| Elección del inicio | {start_mode} |",
             f"| Variable | {variable} ({grid.attrs.get('units', '')}) |",
             f"| Inicio del pronóstico | {year}-{month:02d} |",
             f"| Temporada | {season_label(year, month, leads)} (leads {', '.join(map(str, leads))}) |",
             f"| Región | {region.name}: {region.bounds()} |",
             f"| Rejilla | {grid.sizes['lat']} × {grid.sizes['lon']} celdas |",
             f"| Años del hindcast | {y0}–{y1} |",
             f"| Predictando (Y) | {y_text} |",
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


def forecast_files(cfg: dict, centres: list[str], variable: str, region: Region, force: bool,
                   start: tuple[int, int] | None) -> tuple[int, int, dict[str, Path], str]:
    """Pronóstico (Z): el más reciente publicado o el del inicio pedido (año, mes)."""
    if start is None:
        year, month, paths = latest_forecast(cfg, centres, variable, region=region, force=force)
        print(f"Pronóstico más reciente: inicio {year}-{month:02d}")
        return year, month, paths, "más reciente publicado"
    year, month = start
    try:
        paths = {c: download(cfg, c, variable, region=region, force=force, month=month,
                             forecast_year=year) for c in centres}
    except Exception as exc:
        if forecast_unavailable(exc):
            raise RuntimeError(f"el pronóstico con inicio {year}-{month:02d} no está publicado "
                               "en el CDS (todavía o para algún centro)") from exc
        raise
    print(f"Pronóstico pedido: inicio {year}-{month:02d}")
    return year, month, paths, "pedido con --inicio"


def hindcast_years(cfg: dict, centres: list[str], years: tuple[int, int] | None) -> tuple[int, int]:
    """Periodo del hindcast: el pedido, que debe caber en el de todos los centros elegidos."""
    lo = max(cfg["centres"][c]["years"][0] for c in centres)
    hi = min(cfg["centres"][c]["years"][1] for c in centres)
    if years is None:
        return lo, hi
    y0, y1 = years
    if not (lo <= y0 <= y1 <= hi):
        raise ValueError(f"años del hindcast {y0}–{y1} fuera del periodo disponible {lo}–{hi} "
                         f"para {', '.join(centres)}")
    return y0, y1


def build(centres: list[str], variable: str = "precipitation", force: bool = False,
          out_dir: Path = Path("data") / "cpt", region: str | None = None,
          start: tuple[int, int] | None = None, leads: list[int] | None = None,
          years: tuple[int, int] | None = None, command: str | None = None,
          run_cpt: bool = False) -> list[Path]:
    """X, Z (y verificación del Y) en formato CPT. Lo que no se pase sale de params.yaml.

    Con `run_cpt`, además corre las corridas de prueba en CPT batch (A: cada modelo,
    B: promedio) para las variables que tienen Y; ValueError si alguna no pasa.
    """
    cfg = load_config()
    leads = leads or cfg["cpt"]["season_leads"]
    n_leads = len(cfg["leadtime_months"])
    if not set(leads) <= set(cfg["leadtime_months"]):
        raise ValueError(f"temporada (leads {leads}) fuera de los leads descargados "
                         f"{cfg['leadtime_months']}")
    reg = resolve_region(cfg, region, default=cfg["cpt"]["region"])
    y0, y1 = hindcast_years(cfg, centres, years)

    year, month, z_raw, start_mode = forecast_files(cfg, centres, variable, reg, force, start)
    x_raw = {c: download(cfg, c, variable, region=reg, force=force, month=month)
             for c in centres}

    for p in [*x_raw.values(), *z_raw.values()]:
        if not verify(p):
            raise ValueError(f"{p.name} no pasó la verificación; no se escriben archivos CPT")

    x, z, members = {}, {}, {}
    for c in centres:
        hx = reduce_lagged(open_grib(x_raw[c]), month, n_leads).sel(year=slice(y0, y1))
        if [int(hx["year"][0]), int(hx["year"][-1])] != [y0, y1]:
            raise ValueError(f"{c}: el hindcast descargado no cubre {y0}–{y1}")
        hz = reduce_lagged(open_grib(z_raw[c]), month, n_leads)
        members[c] = (hx.sizes["member"], hz.sizes["member"])
        x[c], z[c] = season_mean(hx, leads), season_mean(hz, leads)
    if len(centres) > 1:
        x[ENSEMBLE], z[ENSEMBLE] = ensemble(x), ensemble(z)

    y_path = predictand_path(cfg, variable)
    if y_path is None:
        print(f"[--] sin predictando (Y) configurado para {variable}: se escriben solo X y Z")
    elif not verify_predictand(y_path, variable, x[centres[0]]["year"].values, month, leads, reg):
        raise ValueError(f"{y_path.name} no pasó la verificación; no se escriben archivos CPT")

    tag = f"{variable}_i{year}{month:02d}_{reg.name}"
    files = {}
    for name in x:
        files[name] = {kind: write_cpt(data[name], out_dir / f"{kind}_{name}_{tag}.txt",
                                       f"{variable}_{name}", month, leads)
                       for kind, data in (("X", x), ("Z", z))}
    log = write_log(out_dir / f"registro_{tag}.md", cfg, variable, year, month, leads, reg,
                    centres, members, files, x[centres[0]], predictand=y_path,
                    command=command, start_mode=start_mode)
    written = [p for f in files.values() for p in f.values()] + [log]
    if run_cpt:
        if y_path is None:
            print(f"[--] {variable}: sin Y no se puede correr CPT; se omiten las corridas")
            return written
        from ideam_seasonal.cpt_run import run_all

        results, cpt_log = run_all(files, y_path, out_dir, variable, tag, reg, cfg["cpt_batch"],
                                   [int(v) for v in x[centres[0]]["year"].values], command)
        written.append(cpt_log)
        failed = [r.name for r in results if not r.ok]
        if failed:
            raise ValueError(f"corridas de CPT con fallas: {', '.join(failed)} (ver {cpt_log})")
    return written


if __name__ == "__main__":
    # Compatibilidad: `python -m ideam_seasonal.cpt ...` = `bash kairos.sh ...`
    import sys

    from ideam_seasonal.kairos import main

    sys.exit(main())
