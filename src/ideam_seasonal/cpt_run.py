"""Corridas de prueba en CPT batch (A: cada modelo, B: promedio) y validación de lo leído."""
from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ideam_seasonal.common import Region, open_cpt_stations, read_cpt_grid

# CPT agrega ".txt"
OUTPUTS = {"101": "X_leido_por_cpt", "102": "Y_leido_por_cpt", "542": "Z_leido_por_cpt",
           "511": "pronostico", "501": "probabilidades"}
MAX_OUTPUT_BYTES = 5_000_000  # sin entrada, CPT repite la última pregunta sin fin
GLOBAL_DOMAIN = {"north": 90, "south": -90, "west": -180, "east": 180}


class CPTError(RuntimeError):
    pass


@dataclass
class RunResult:
    name: str
    run_dir: Path
    goodness: float | None = None
    n_modes: int | None = None
    checks: dict[str, tuple[bool, str]] = field(default_factory=dict)
    forecast: pd.DataFrame | None = None

    @property
    def ok(self) -> bool:
        return all(ok for ok, _ in self.checks.values())


def batch_input(area: dict | None, x_modes: tuple[int, int]) -> str:
    """Opciones de menú de CPT 18 (PCR)."""
    a = area or GLOBAL_DOMAIN
    dom = [a["north"], a["south"], a["west"], a["east"]]
    steps = ["612",                       # PCR
             "1", "X.txt", *dom, *x_modes,  # X, dominio, modos mín./máx.
             "2", "Y.txt", *dom,
             "3", "Z.txt",
             "512",                        # imprime la configuración en el log
             "311",                        # validación cruzada
             "451",                        # pronósticos
             "111"]                        # guardar resultados
    for code, name in OUTPUTS.items():
        steps += [code, name]
    steps += ["0", "0"]                    # salir
    return "\n".join(str(s) for s in steps) + "\n"


def clean_output(raw: bytes) -> str:
    """Quita colores ANSI y el texto que CPT borra con retrocesos."""
    txt = re.sub(r"\x1b\[[0-9;]*m", "", raw.decode("latin-1"))
    txt = re.sub(r".\x08", "", txt)
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[\x00-\x08\x0b-\x1f]", "", txt)


def run_cpt_batch(exe: Path, bin_dir: Path, run_dir: Path, stdin_text: str,
                  timeout_s: float, argv: list[str] | None = None) -> str:
    """`argv` reemplaza el comando (tests)."""
    if argv is None and not exe.exists():
        raise CPTError(f"no se encontró CPT batch en {exe} (config/params.yaml: cpt_batch.exe)")
    env = dict(os.environ, CPT_BIN_DIR=str(bin_dir))
    p = subprocess.Popen(argv or [str(exe)], cwd=run_dir, env=env, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.stdin.write(stdin_text.encode())
    p.stdin.close()
    buf, state = bytearray(), {"overflow": False}

    def reader():
        while chunk := p.stdout.read(65536):
            buf.extend(chunk)
            if len(buf) > MAX_OUTPUT_BYTES:
                state["overflow"] = True
                p.kill()
                return

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    start = time.monotonic()
    while p.poll() is None and time.monotonic() - start < timeout_s:
        time.sleep(0.2)
    timed_out = p.poll() is None
    if timed_out:
        p.kill()
    t.join(timeout=10)
    out = clean_output(bytes(buf))
    (run_dir / "cpt_salida.log").write_text(out, encoding="utf-8")
    tail = re.sub(r"\s+", " ", out[-400:]).strip()
    if state["overflow"]:
        raise CPTError(f"CPT pidió una entrada no prevista y se detuvo; última pregunta: …{tail[-200:]} "
                       f"(log completo en {run_dir / 'cpt_salida.log'})")
    if timed_out:
        raise CPTError(f"CPT no terminó en {timeout_s:g} s (log en {run_dir / 'cpt_salida.log'})")
    if p.returncode != 0:
        raise CPTError(f"CPT terminó con código {p.returncode} (log en {run_dir / 'cpt_salida.log'})")
    if re.search(r"\bERROR\b", out):
        line = next(ln for ln in out.splitlines() if re.search(r"\bERROR\b", ln))
        raise CPTError(f"CPT informó un error: {line.strip()} (log en {run_dir / 'cpt_salida.log'})")
    return out


def compare_grids(src: Path, cpt: Path) -> tuple[bool, str]:
    a, b = read_cpt_grid(src), read_cpt_grid(cpt)
    if list(a) != list(b):
        return False, f"temporadas distintas: {list(a)[:2]}… vs {list(b)[:2]}…"
    worst = 0.0
    for k in a:
        if a[k].shape != b[k].shape or not (np.allclose(a[k].lat, b[k].lat)
                                            and np.allclose(a[k].lon, b[k].lon)):
            return False, f"{k}: rejilla distinta"
        worst = max(worst, float(np.nanmax(np.abs(a[k].values - b[k].values))))
    return worst < 1e-6, f"{len(a)} campos, misma rejilla; dif. máx {worst:.1e}"


def _station_rows(path: Path) -> list[tuple[str, list[str], np.ndarray]]:
    ids, rows = [], []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        cells = ln.split("\t")
        if ln.startswith("\t"):
            ids = [c.strip() for c in cells[1:] if c.strip()]
        elif ln[:4].isdigit():
            rows.append((cells[0].strip(), ids, np.array(cells[1:len(ids) + 1], dtype=float)))
    return rows


def compare_predictand(src: Path, cpt: Path, years: list[int]) -> tuple[bool, str]:
    """CPT arma la temporada sumando los meses del Y (comprobado 2026-11-21)."""
    y = open_cpt_stations(src)
    rows = _station_rows(cpt)
    first_years = [int(t[:4]) for t, _, _ in rows]
    if first_years != list(years):
        return False, f"años de la temporada {first_years[:2]}… no coinciden con el hindcast"
    worst = 0.0
    for t, ids, vals in rows:
        t0, t1 = t.split("/")
        months = pd.date_range(t0, pd.Period(t1).to_timestamp(), freq="MS")
        total = y.sel(time=months, station=ids).sum("time").values
        worst = max(worst, float(np.max(np.abs(total - vals))))
    return worst < 1e-6, (f"{len(rows)} temporadas = suma de los meses del Y mensual; "
                          f"dif. máx {worst:.1e}")


def forecast_table(run_dir: Path) -> pd.DataFrame:
    (t, ids, value), = _station_rows(run_dir / "pronostico.txt")
    probs = _station_rows(run_dir / "probabilidades.txt")
    df = pd.DataFrame({"estacion": ids, "pronostico": value})
    for (_, _, p), cat in zip(probs, ("bajo", "normal", "sobre")):
        df[f"p_{cat}"] = p
    df.attrs["temporada"] = t
    return df


def run_one(name: str, x: Path, y: Path, z: Path, run_dir: Path, batch_cfg: dict,
            region: Region, years: list[int]) -> RunResult:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    for src, dst in ((x, "X.txt"), (y, "Y.txt"), (z, "Z.txt")):
        shutil.copyfile(src, run_dir / dst)
    stdin_text = batch_input(region.area, tuple(batch_cfg["x_modes"]))
    (run_dir / "cpt_entrada.txt").write_text(stdin_text, encoding="utf-8")
    out = run_cpt_batch(Path(batch_cfg["exe"]), Path(batch_cfg["bin_dir"]), run_dir, stdin_text,
                        float(batch_cfg.get("timeout_s", 900)))

    res = RunResult(name, run_dir)
    m = re.search(r"Goodness Index:\s+(-?[\d.]+)\s+\(using\s+(\d+)\s+X-modes\)", out)
    if m:
        res.goodness, res.n_modes = float(m.group(1)), int(m.group(2))
    missing = [f"{n}.txt" for n in OUTPUTS.values() if not (run_dir / f"{n}.txt").exists()]
    res.checks["salidas"] = (not missing and m is not None and "Done!" in out,
                             "CPT terminó el cálculo y escribió sus salidas" if not missing
                             else f"faltan salidas de CPT: {', '.join(missing)}")
    if missing:
        return res
    res.checks["X leído = origen"] = compare_grids(run_dir / "X.txt", run_dir / "X_leido_por_cpt.txt")
    res.checks["Z leído = origen"] = compare_grids(run_dir / "Z.txt", run_dir / "Z_leido_por_cpt.txt")
    res.checks["Y leído = origen"] = compare_predictand(run_dir / "Y.txt",
                                                       run_dir / "Y_leido_por_cpt.txt", years)
    res.forecast = forecast_table(run_dir)
    p = res.forecast[["p_bajo", "p_normal", "p_sobre"]].sum(axis=1)
    res.checks["pronóstico coherente"] = (
        bool(np.allclose(p, 100, atol=0.5) and np.isfinite(res.forecast["pronostico"]).all()),
        f"{len(res.forecast)} estaciones con pronóstico; probabilidades suman 100 %")
    return res


def write_log(path: Path, results: list[RunResult], variable: str, tag: str, region: Region,
              batch_cfg: dict, command: str | None) -> Path:
    lines = [f"# Corridas de prueba en CPT: {variable} ({tag})", "",
             "| Campo | Valor |", "|---|---|",
             f"| Fecha | {dt.datetime.now():%Y-%m-%d %H:%M} |",
             *([f"| Comando | `{command}` |"] if command else []),
             f"| CPT | `{batch_cfg['exe']}` (batch) |",
             f"| Método | PCR, validación cruzada; modos X {batch_cfg['x_modes'][0]}–"
             f"{batch_cfg['x_modes'][1]} (los demás ajustes: cpt.ini de la instalación, "
             "impresos en cada cpt_salida.log) |",
             f"| Dominio X e Y | {region.name}: {region.bounds()} |", "",
             "| Corrida | Modelo | Resultado | Goodness Index | Modos X | Carpeta |",
             "|---|---|---|---|---|---|"]
    for r in results:
        kind = "B" if r.name == "promedio" else "A"
        good = f"{r.goodness:.3f}" if r.goodness is not None else "—"
        lines.append(f"| {kind} | {r.name} | {'OK' if r.ok else 'FALLA'} | {good} | "
                     f"{r.n_modes or '—'} | `{r.run_dir.name}` |")
    for r in results:
        lines += ["", f"## {r.name}", ""]
        lines += [f"- [{'OK' if ok else 'FALLA'}] {k}: {msg}" for k, (ok, msg) in r.checks.items()]
        if r.forecast is not None:
            f = r.forecast
            lines += ["", f"Pronóstico {f.attrs['temporada']} (total de la temporada en las "
                          "unidades del Y, y probabilidad por tercil en %):", "",
                      "| Estación | Pronóstico | Bajo lo normal | Normal | Sobre lo normal |",
                      "|---|---|---|---|---|"]
            lines += [f"| {row.estacion} | {row.pronostico:.1f} | {row.p_bajo:.0f} | "
                      f"{row.p_normal:.0f} | {row.p_sobre:.0f} |" for row in f.itertuples()]
    lines += ["", "Habilidad: el Goodness Index es solo informativo; la verificación de "
                  "destreza queda fuera del alcance (corridas de prueba de ingreso)."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_all(files: dict[str, dict[str, Path]], y: Path, out_dir: Path, variable: str, tag: str,
            region: Region, batch_cfg: dict, years: list[int],
            command: str | None = None) -> tuple[list[RunResult], Path]:
    results = []
    base = out_dir / "corridas_cpt"
    for name, xz in files.items():
        print(f"[cpt] corrida {'B' if name == 'promedio' else 'A'}: {name}")
        r = run_one(name, xz["X"], y, xz["Z"], base / f"{name}_{tag}", batch_cfg, region, years)
        for k, (ok, msg) in r.checks.items():
            print(f"  [{'OK' if ok else 'FALLA'}] {k}: {msg}")
        results.append(r)
    log = write_log(out_dir / f"registro_cpt_{tag}.md", results, variable, tag, region,
                    batch_cfg, command)
    return results, log
