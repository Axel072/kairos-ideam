"""NMME de la IRI Data Library para la corrida C (B + NMME)."""
from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr

from ideam_seasonal.common import VARIABLES, Region

NAME = "nmme"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
UNITS = {"mm/day": (1.0, 0.0), "Kelvin_scale": (1.0, -273.15), "Celsius_scale": (1.0, 0.0)}


class Unavailable(Exception):
    pass


def url(cfg: dict, model: str, variable: str, month: int, years: tuple[int, int],
        leads: list[int], region: Region, forecast: bool) -> str:
    n = cfg["nmme"]
    m = n["models"][model]
    y0, y1 = years
    when = f"0000 1 {MONTHS[month - 1]} {y0}" + ("" if y0 == y1 else f"-{y1}")
    parts = [n["url"].rstrip("/"), m["forecast" if forecast else "hindcast"],
             "." + n["variables"][variable], f"S/({when})/VALUES",
             f"L/{leads[0] - 0.5:g}/{leads[-1] - 0.5:g}/RANGEEDGES", "[L]//keepgrids/average",
             "[M]/average"]
    if region.area is not None:
        a = region.area
        parts += [f"Y/{a['south'] - 1:g}/{a['north'] + 1:g}/RANGEEDGES",
                  f"X/{a['west'] - 1:g}/{a['east'] + 1:g}/RANGEEDGES"]
    parts.append("data.nc")
    return urllib.parse.quote("/".join(parts), safe=":/().-")


def raw_path(out_dir: Path, model: str, variable: str, month: int, region: Region,
             forecast_year: int | None = None) -> Path:
    tag = f"f{forecast_year}{month:02d}" if forecast_year else f"m{month:02d}"
    return out_dir / f"nmme_{model}_{variable}_{tag}_{region.name}.nc"


def download(cfg: dict, model: str, variable: str, month: int, years: tuple[int, int],
             leads: list[int], region: Region, forecast: bool, force: bool = False,
             out_dir: Path = Path("data") / "raw") -> Path:
    path = raw_path(out_dir, model, variable, month, region, years[0] if forecast else None)
    src = url(cfg, model, variable, month, years, leads, region, forecast)
    meta = path.with_suffix(".url")
    if path.exists() and not force and meta.exists() and meta.read_text() == src:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(src, timeout=cfg["nmme"].get("timeout_s", 600)) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise Unavailable("no publicado en la IRI Data Library") from e
        raise RuntimeError(f"NMME {model}: la IRI Data Library respondió {e.code}") from e
    path.write_bytes(data)
    meta.write_text(src)
    return path


def open_season(path: Path, variable: str) -> xr.DataArray:
    """(year, lat, lon) en las unidades de VARIABLES y longitudes -180..180."""
    ds = xr.open_dataset(path, decode_times=False)
    name = next(iter(ds.data_vars))
    da = ds[name].squeeze("L", drop=True)
    units = da.attrs.get("units", "")
    if units not in UNITS:
        raise ValueError(f"{path.name}: unidad NMME '{units}' no reconocida")
    scale, offset = UNITS[units]
    years = (1960 + ds["S"].values // 12).astype(int)
    lon = ((ds["X"].values + 180) % 360) - 180
    out = xr.DataArray(da.values * scale + offset, dims=("year", "lat", "lon"),
                       coords={"year": years, "lat": ds["Y"].values, "lon": lon})
    out = out.sortby("lon").sortby("lat")
    out.attrs["units"] = VARIABLES[variable].units
    return out


def to_grid(da: xr.DataArray, lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Bilineal a la rejilla de los modelos del CDS (centros en ,5)."""
    if da["lon"].size > 2 and float(da["lon"][-1] - da["lon"][0]) >= 359 - 1e-6:
        da = xr.concat([da, da.isel(lon=0).assign_coords(lon=da["lon"][0] + 360)], "lon")
        da = xr.concat([da.isel(lon=-2).assign_coords(lon=da["lon"][-2] - 360), da], "lon")
    src_lat, src_lon = da["lat"].values, da["lon"].values
    if lat.min() < src_lat.min() or lat.max() > src_lat.max() or \
            lon.min() < src_lon.min() or lon.max() > src_lon.max():
        raise ValueError("NMME: la rejilla descargada no cubre la región")
    v = np.apply_along_axis(lambda c: np.interp(lat, src_lat, c), 1, da.values)
    v = np.apply_along_axis(lambda r: np.interp(lon, src_lon, r), 2, v)
    return xr.DataArray(v, dims=("year", "lat", "lon"),
                        coords={"year": da["year"].values, "lat": lat, "lon": lon},
                        attrs=da.attrs)


def build(cfg: dict, variable: str, month: int, forecast_year: int, years: tuple[int, int],
          leads: list[int], region: Region, lat: np.ndarray, lon: np.ndarray,
          force: bool = False) -> tuple[xr.DataArray, xr.DataArray, dict[str, str]]:
    """Promedio simple de los modelos NMME que cubren el hindcast y el pronóstico."""
    lo, hi = VARIABLES[variable].plausible
    x, z, status = [], [], {}
    for model in cfg["nmme"]["models"]:
        try:
            px = download(cfg, model, variable, month, years, leads, region, False, force)
            pz = download(cfg, model, variable, month, (forecast_year,) * 2, leads, region,
                          True, force)
        except Unavailable as e:
            status[model] = f"no entra: {e}"
            continue
        hx, hz = open_season(px, variable), open_season(pz, variable)
        hx = hx.sel(year=slice(*years))
        expected = list(range(years[0], years[1] + 1))
        if hx["year"].values.tolist() != expected or not np.isfinite(hx.values).all():
            status[model] = f"no entra: hindcast incompleto ({hx.sizes['year']} de {len(expected)} años)"
            continue
        if hz["year"].values.tolist() != [forecast_year]:
            status[model] = "no entra: sin pronóstico para el inicio"
            continue
        mean = float(hx.mean())
        if not lo <= mean <= hi:
            raise ValueError(f"NMME {model}: media {mean:.2f} fuera del rango plausible "
                             f"{lo}..{hi} {VARIABLES[variable].units}")
        x.append(to_grid(hx, lat, lon))
        z.append(to_grid(hz, lat, lon))
        status[model] = "entra"
    if not x:
        raise RuntimeError(f"ningún modelo NMME cubre {years[0]}–{years[1]} y el inicio "
                           f"{forecast_year}-{month:02d}: " +
                           "; ".join(f"{m} {s}" for m, s in status.items()))
    ex = xr.concat(x, "model").mean("model")
    ez = xr.concat(z, "model").mean("model")
    ex.attrs = ez.attrs = {"units": VARIABLES[variable].units}
    return ex, ez, status
