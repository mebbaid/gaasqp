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
import csv
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
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
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

    # Group rows by QP size (QP_n or n). Allow multiple runs per size.
    rows = sorted(off_rows, key=lambda r: int(r.get('QP_n', r.get('n', 0))))
    sizes = []
    groups = {}
    for r in rows:
        nn = int(r.get('QP_n', r.get('n')))
        if nn not in groups:
            groups[nn] = []
            sizes.append(nn)
        groups[nn].append(r)

    sizes = np.array(sorted(sizes))

    def solver_series(key):
        series = []
        for nn in sizes:
            vals = []
            for r in groups[nn]:
                v = r.get(key)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            series.append(np.array(vals) if vals else np.array([np.nan]))
        return series

    geo_series = solver_series('Geo_us')
    qpo_series = solver_series('qpO_us')
    osqp_series = solver_series('OSQP_us')

    fig, ax = plt.subplots(figsize=(W2, 3.0))

    # helper to plot median + IQR + 5-95
    def plot_with_bands(series, color, label, marker):
        med = np.array([np.nanmedian(s) for s in series])
        p25 = np.array([np.nanpercentile(s, 25) for s in series])
        p75 = np.array([np.nanpercentile(s, 75) for s in series])
        p05 = np.array([np.nanpercentile(s, 5) for s in series])
        p95 = np.array([np.nanpercentile(s, 95) for s in series])
        ax.plot(sizes, med, marker + '-', color=color, label=label, linewidth=1.6)
        ax.fill_between(sizes, p25, p75, color=color, alpha=0.18)
        ax.fill_between(sizes, p05, p95, color=color, alpha=0.08)

    plot_with_bands(geo_series, C_GEO, 'GeoQP', 'o')
    plot_with_bands(qpo_series, C_QPO, 'qpOASES', 's')
    # only plot OSQP if there is numeric data
    if not all(np.isnan(np.array([np.nanmedian(s) for s in osqp_series]))):
        plot_with_bands(osqp_series, C_OSQP, 'OSQP', '^')

    ax.set_xscale("log")
    # choose readable x-ticks: a small set of nicely spaced ticks (log-spaced for wide ranges)
    min_n = int(sizes.min())
    max_n = int(sizes.max())
    if max_n <= 30:
        step = max(2, int(round(max_n / 6.0)))
        ticks = list(range(min_n, max_n + 1, step))
    else:
        ticks = np.unique(np.round(np.logspace(np.log10(min_n), np.log10(max_n), num=6))).astype(int).tolist()
        if ticks[-1] != max_n:
            ticks.append(max_n)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(t)) for t in ticks])
    ax.set_xlabel(r"QP dimension $n$")
    ax.set_ylabel(r"Solve time (ms)")
    ax.set_title("LIPM Benchmark — Avg solve time vs QP size")
    ax.legend()
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig01_lipm_performance.{ext}"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  fig01_lipm_performance")


def _write_csv(rows, outpath):
    if not rows:
        return
    # collect all keys
    keys = set()
    for r in rows:
        keys.update(r.keys())
    keys = sorted(keys)
    with open(outpath, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(keys)
        for r in rows:
            writer.writerow([r.get(k, '') for k in keys])


def fig03_violin(off_rows, on_rows, outdir):
    if not off_rows:
        print("  [skip] fig03_violin (no data)"); return
    rows = sorted(off_rows, key=lambda r: int(r.get('QP_n', r.get('n', 0))))
    sizes = sorted({int(r.get('QP_n', r.get('n'))) for r in rows})
    if not sizes:
        return
    # pick small, medium, large
    idxs = [0, len(sizes)//2, -1]
    picks = [sizes[i] for i in idxs]

    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.6), sharey=False)
    for ax, nn in zip(axes, picks):
        # collect candidate metrics for Geo
        geo_it_vals = [r.get('Geo_it') for r in rows if int(r.get('QP_n', r.get('n'))) == nn]
        avg_pivot_vals = [r.get('AvgPivot') for r in rows if int(r.get('QP_n', r.get('n'))) == nn]
        geo_hops_vals = [r.get('Geo_hops') for r in rows if int(r.get('QP_n', r.get('n'))) == nn]

        # choose Geo metric: prefer phase2 iters (Geo_it) if present, otherwise AvgPivot, otherwise facet hops
        has_geo_iters = any((v is not None and v != 0) for v in geo_it_vals)
        if has_geo_iters:
            geo_vals = [float(v) for v in geo_it_vals if v is not None and v != 0]
            ylabel = 'Iterations'
        else:
            if any(v is not None and v >= 0 for v in avg_pivot_vals):
                geo_vals = [float(v) for v in avg_pivot_vals if v is not None]
                ylabel = 'Phase2 iters'
            else:
                geo_vals = [float(v) for v in geo_hops_vals if v is not None]
                ylabel = 'Facet hops'

        # qpOASES iterations (filter sentinel negatives)
        qpo_vals = [float(r['qpO_it']) for r in rows if int(r.get('QP_n', r.get('n'))) == nn and r.get('qpO_it') is not None and r.get('qpO_it') >= 0]

        geo = np.array(geo_vals) if geo_vals else np.array([])
        qpo = np.array(qpo_vals) if qpo_vals else np.array([])
        data = [geo, qpo]

        # If no iteration-like data at all, fallback to solve-time distributions
        total_samples = sum(int(d.size) for d in data)
        if total_samples == 0:
            geo = np.array([float(r['Geo_us']) for r in rows if int(r.get('QP_n', r.get('n'))) == nn and r.get('Geo_us') is not None])
            qpo = np.array([float(r['qpO_us']) for r in rows if int(r.get('QP_n', r.get('n'))) == nn and r.get('qpO_us') is not None])
            data = [geo, qpo]
            total_samples = sum(int(d.size) for d in data)
            ylabel = 'Solve time (ms)'

        if total_samples == 0:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
            continue

        # If every dataset has at least 2 samples, use violinplot; otherwise scatter+box
        enough_for_violin = all(d.size >= 2 for d in data)
        colors = [C_GEO, C_QPO]
        if enough_for_violin:
            parts = ax.violinplot([d for d in data], showmeans=False, showextrema=False)
            # color bodies per series
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.5)
        else:
            for i, d in enumerate(data, start=1):
                if d.size:
                    xs = np.random.normal(i, 0.05, size=d.size)
                    ax.scatter(xs, d, alpha=0.8, color=colors[i-1], s=18)
                    ax.boxplot(d, positions=[i], widths=0.3)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['GeoQP', 'qpOASES'])
        ax.set_title(f'n={nn}')
        ax.set_ylabel(ylabel)

    # minimal layout for journal; rely on panel titles and axis labels
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'fig03_violin.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  fig03_violin')


def fig04_cdf(off_rows, on_rows, outdir):
    if not off_rows:
        print("  [skip] fig04_cdf (no data)"); return
    rows = sorted(off_rows, key=lambda r: int(r.get('QP_n', r.get('n', 0))))
    sizes = sorted({int(r.get('QP_n', r.get('n'))) for r in rows})
    if not sizes:
        return
    # pick size with most samples; if none has enough samples, aggregate across sizes
    counts_per_size = {}
    for nn in sizes:
        counts_per_size[nn] = sum(1 for r in rows if int(r.get('QP_n', r.get('n'))) == nn and r.get('Geo_us') is not None)
    nn = max(counts_per_size, key=lambda k: counts_per_size[k])
    geo = sorted([float(r['Geo_us']) for r in rows if int(r.get('QP_n', r.get('n'))) == nn and r.get('Geo_us') is not None])
    qpo = sorted([float(r['qpO_us']) for r in rows if int(r.get('QP_n', r.get('n'))) == nn and r.get('qpO_us') is not None])
    osqp = sorted([float(r['OSQP_us']) for r in rows if int(r.get('QP_n', r.get('n'))) == nn and r.get('OSQP_us') is not None])
    total_samples = len(geo) + len(qpo) + len(osqp)
    if total_samples < 4:
        # aggregate across all sizes
        geo = sorted([float(r['Geo_us']) for r in rows if r.get('Geo_us') is not None])
        qpo = sorted([float(r['qpO_us']) for r in rows if r.get('qpO_us') is not None])
        osqp = sorted([float(r['OSQP_us']) for r in rows if r.get('OSQP_us') is not None])
        nn = 'all'

    def ecdf(data):
        data = np.array(sorted(data))
        if data.size == 0:
            return np.array([]), np.array([])
        y = np.arange(1, len(data)+1) / float(len(data))
        return data, y

    fig, ax = plt.subplots(figsize=(W2, 3.0))
    for data, color, label in ((geo, C_GEO, 'GeoQP'), (qpo, C_QPO, 'qpOASES'), (osqp, C_OSQP, 'OSQP')):
        x, y = ecdf(data)
        if x.size:
            ax.step(x, y, where='post', color=color, label=label)
    ax.set_xscale('log')
    ax.set_xlabel('Solve time (ms)')
    ax.set_ylabel('Empirical CDF')
    ax.set_title(f'Empirical CDF of solve time (n={nn})')
    ax.legend()
    fig.tight_layout()
    for ext in ('pdf','png'):
        fig.savefig(os.path.join(outdir, f'fig04_cdf_n{nn}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  fig04_cdf')


def fig02_iterations(off_rows, on_rows, outdir):
    if not off_rows:
        print("  [skip] fig02_lipm_iterations (no data)"); return
    rows = sorted(off_rows, key=lambda r: int(r.get('QP_n', r.get('n', 0))))
    n = np.array([int(r.get('QP_n', r.get('n'))) for r in rows])
    geo_it = np.array([np.nan if r.get('Geo_it') is None else float(r.get('Geo_it')) for r in rows])
    qpo_it = np.array([np.nan if r.get('qpO_it') is None else float(r.get('qpO_it')) for r in rows])
    geo_hops = np.array([np.nan if r.get('Geo_hops') is None else float(r.get('Geo_hops')) for r in rows])
    avg_pivot = np.array([np.nan if r.get('AvgPivot') is None else float(r.get('AvgPivot')) for r in rows])
    # Choose Geo metric: prefer Geo phase2 iters if present, otherwise prefer phase2 pivot counts, then facet hops
    use_geo_iters = not np.all(np.isnan(geo_it) | (geo_it == 0))
    if use_geo_iters:
        geo_metric = geo_it
        geo_label = "GeoQP (phase2 iters)"
    else:
        has_avg_pivot = not np.all(np.isnan(avg_pivot))
        if has_avg_pivot:
            geo_metric = avg_pivot
            geo_label = "GeoQP (phase2 iters - avg pivot)"
        else:
            geo_metric = geo_hops
            geo_label = "GeoQP (facet hops)"

    fig, ax = plt.subplots(figsize=(W2, 3.0))
    ax.plot(n, geo_metric, "o-", color=C_GEO, label=geo_label, linewidth=1.6)
    ax.plot(n, qpo_it, "s--", color=C_QPO, label="qpOASES (iters)", linewidth=1.4)
    ax.set_xscale("log")
    # readable x-ticks as in performance plot
    min_n = int(n.min())
    max_n = int(n.max())
    if max_n <= 30:
        step = max(2, int(round(max_n / 6.0)))
        ticks = list(range(min_n, max_n + 1, step))
    else:
        ticks = np.unique(np.round(np.logspace(np.log10(min_n), np.log10(max_n), num=6))).astype(int).tolist()
        if ticks[-1] != max_n:
            ticks.append(max_n)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(t)) for t in ticks])
    ax.set_xlabel(r"QP dimension $n$")
    ax.set_ylabel(r"Iterations (avg)")
    ax.set_title("LIPM Benchmark — Iterations vs QP size")
    ax.legend()

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig02_lipm_iterations.{ext}"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  fig02_lipm_iterations")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    filepath = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else os.path.join(os.path.dirname(os.path.abspath(filepath)), "figures_lipm")
    os.makedirs(outdir, exist_ok=True)

    save_csv = any(a == '--save-csv' for a in sys.argv)
    extended = any(a == '--extended-plots' for a in sys.argv)

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
    if extended:
        fig03_violin(off_rows, on_rows, outdir)
        fig04_cdf(off_rows, on_rows, outdir)

    if save_csv:
        _write_csv(off_rows, os.path.join(outdir, 'results_horizons.csv'))

    print(f"Done. 2 figures (PDF + PNG) saved to: {outdir}")


if __name__ == "__main__":
    main()
