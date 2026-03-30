"""
plot_lipm_benchmark.py — Figures for LIPM walking benchmark.

Generates two figures (PDF + PNG):

  fig01_lipm_performance.pdf    Avg solve time (ms) per solver (caching OFF/ON)
  fig02_lipm_iterations.pdf    Iteration counts per solver (where available)

Usage:
    python plot_lipm_benchmark.py <benchmark_output.txt> [output_dir]

This parses the "Results (caching = OFF)" and "Results (caching = ON)" tables
printed by `benchmark_lipm_walking` and makes simple comparison plots.
"""

import os
import sys
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "GeoQP": "#2166ac",
    "qpOASES": "#4daf4a",
    "OSQP": "#ff7f00",
}

# match colors used in plot_benchmarks.py
C_GEO = COLORS["GeoQP"]
C_QPO = COLORS["qpOASES"]
C_OSQP = COLORS["OSQP"]

import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.framealpha": 0.92,
    "lines.linewidth": 1.5,
})

W2 = 7.16


def _extract_sub_block(lines, header_re):
    out, found = [], False
    for line in lines:
        if not found:
            if re.search(header_re, line):
                found = True
            continue
        if re.search(r"^={10,}", line):
            break
        out.append(line.rstrip())
    return out


def _parse_results_table(block):
    # Parse CASE lines emitted by the modified benchmark (key=value tokens)
    # Example:
    # CASE n=4 QP_n=4 QP_m=4 Geo_us=0.123 qpO_us=0.456 OSQP_us=0.789 Geo_it=5 qpO_it=100 OSQP_it=-1 Geo_hops=2 AvgObjGeo=... Diff_vs_qpO=... OK_geo=1 OK_qpO=1 OK_osqp=1
    rows = []
    for line in block:
        s = line.strip()
        if not s or not s.startswith("CASE "):
            continue
        toks = s.split()
        d = {}
        for tok in toks[1:]:
            if '=' not in tok:
                continue
            k, v = tok.split('=', 1)
            # try numeric conversion
            try:
                if '.' in v or 'e' in v or 'E' in v:
                    d[k] = float(v)
                else:
                    d[k] = int(v)
            except Exception:
                d[k] = v
        # require numeric n (skip header-like CASE lines with placeholders)
        if 'n' in d and isinstance(d['n'], (int, float)):
            rows.append(d)
    return rows


def fig01_performance(off_rows, on_rows, outdir):
    if not off_rows:
        print("  [skip] fig01_lipm_performance (no data)"); return
    rows = sorted(off_rows, key=lambda r: int(r.get('QP_n', r.get('n', 0))))
    n = np.array([int(r.get('QP_n', r.get('n'))) for r in rows])
    geo = np.array([float(r.get('Geo_us', np.nan)) for r in rows])
    qpo = np.array([float(r.get('qpO_us', np.nan)) for r in rows])
    osqp = np.array([float(r.get('OSQP_us', np.nan)) for r in rows])

    fig, ax = plt.subplots(figsize=(W2, 3.0))
    ax.plot(n, geo, "o-", color=C_GEO, label="GeoQP", linewidth=1.6)
    ax.plot(n, qpo, "s--", color=C_QPO, label="qpOASES", linewidth=1.4)
    if not np.all(np.isnan(osqp)):
        ax.plot(n, osqp, "^:", color=C_OSQP, label="OSQP", linewidth=1.2)

    ax.set_xscale("log")
    ax.set_xlabel(r"QP dimension $n$")
    ax.set_ylabel(r"Solve time (ms)")
    ax.set_title("LIPM Benchmark — Avg solve time vs QP size")
    ax.legend()
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig01_lipm_performance.{ext}"), dpi=300)
    plt.close(fig)
    print("  fig01_lipm_performance")


def fig02_iterations(off_rows, on_rows, outdir):
    if not off_rows:
        print("  [skip] fig02_lipm_iterations (no data)"); return
    rows = sorted(off_rows, key=lambda r: int(r.get('QP_n', r.get('n', 0))))
    n = np.array([int(r.get('QP_n', r.get('n'))) for r in rows])
    geo_it = np.array([np.nan if r.get('Geo_it') is None else float(r.get('Geo_it')) for r in rows])
    qpo_it = np.array([np.nan if r.get('qpO_it') is None else float(r.get('qpO_it')) for r in rows])
    geo_hops = np.array([np.nan if r.get('Geo_hops') is None else float(r.get('Geo_hops')) for r in rows])

    # Choose Geo metric: prefer Geo iterations if any nonzero, otherwise use facet hops
    use_geo_iters = not np.all(np.isnan(geo_it) | (geo_it == 0))
    if use_geo_iters:
        geo_metric = geo_it
        geo_label = "GeoQP (phase2 iters)"
    else:
        geo_metric = geo_hops
        geo_label = "GeoQP (facet hops)"

    fig, ax = plt.subplots(figsize=(W2, 3.0))
    ax.plot(n, geo_metric, "o-", color=C_GEO, label=geo_label, linewidth=1.6)
    ax.plot(n, qpo_it, "s--", color=C_QPO, label="qpOASES (iters)", linewidth=1.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"QP dimension $n$")
    ax.set_ylabel(r"Iterations (avg)")
    ax.set_title("LIPM Benchmark — Iterations vs QP size")
    ax.legend()

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig02_lipm_iterations.{ext}"), dpi=300)
    plt.close(fig)
    print("  fig02_lipm_iterations")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    filepath = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(filepath)), "figures_lipm")
    os.makedirs(outdir, exist_ok=True)

    with open(filepath) as fh:
        raw = fh.readlines()

    # New CASE-based output: parse the "Results (horizons)" section
    horizon_blk = _extract_sub_block(raw, r"Results \(horizons\)")
    rows = _parse_results_table(horizon_blk) if horizon_blk else []

    off_rows = rows
    on_rows = []

    print(f"Parsing:  {filepath}")
    print(f"Output:   {outdir}")
    print(f"  horizon rows: {len(rows)}")

    fig01_performance(off_rows, on_rows, outdir)
    fig02_iterations(off_rows, on_rows, outdir)

    print(f"Done. 2 figures (PDF + PNG) saved to: {outdir}")


if __name__ == "__main__":
    main()
