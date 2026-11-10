"""
plot_lipm_benchmark.py — Figures for LIPM walking benchmark.

Generates two figures (PDF + PNG):

  fig01_lipm_performance.pdf    Median solve time (ms) per solver over the repeats
                                (GaasQP and every third-party solver on the CASE lines)
  fig02_lipm_iterations.pdf     Iterations of the active-set solvers (GaasQP pivots,
                                qpOASES / qpOASES-B working-set changes, DAQP)

"""

import os
import sys
import re
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps  # noqa: E402

COLORS = ps.COLOR
C_GEO = ps.COLOR["GaasQP"]
C_QPO = ps.COLOR["qpOASES"]
C_OSQP = ps.COLOR["OSQP"]
ps.apply()

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
            try:
                if '.' in v or 'e' in v or 'E' in v:
                    d[k] = float(v)
                else:
                    d[k] = int(v)
            except Exception:
                d[k] = v
        if 'n' in d and isinstance(d['n'], (int, float)):
            rows.append(d)
    return rows


def _by_size(rows):
    groups = {}
    for r in rows:
        groups.setdefault(int(r.get('QP_n', r.get('n'))), []).append(r)
    return np.array(sorted(groups)), groups


def _solver_keys(rows):
    """CASE-line key for each solver's time / iterations / success flag."""
    keys = {"GaasQP": ("Geo_ms", "AvgPivot", "OK_geo")}
    present = set()
    for r in rows:
        present.update(k for k in r if k.startswith("S_") and k.endswith("_ms"))
    for s in ps.SOLVERS[1:]:
        if f"S_{s}_ms" in present:
            keys[s] = (f"S_{s}_ms", f"S_{s}_it", f"S_{s}_ok")
    if "qpOASES" not in keys and any("qpO_ms" in r for r in rows):  # older output
        keys["qpOASES"] = ("qpO_ms", "qpO_it", "OK_qpO")
    if "OSQP" not in keys and any("OSQP_ms" in r for r in rows):
        keys["OSQP"] = ("OSQP_ms", "OSQP_it", "OK_osqp")
    return keys


def _median_series(sizes, groups, key, ok_key):
    out = []
    for nn in sizes:
        vals = [float(r[key]) for r in groups[nn]
                if key in r and int(r.get(ok_key, 1)) == 1 and float(r[key]) >= 0]
        out.append(np.median(vals) if vals else np.nan)
    return np.array(out)


def _x_ticks(ax, sizes):
    ax.set_xscale("log")
    ticks = [int(t) for t in sizes if int(t) in (10, 20, 40, 60, 80, 120)] or list(sizes)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


def fig01_performance(off_rows, on_rows, outdir):
    if not off_rows:
        print("  [skip] fig01_lipm_performance (no data)"); return
    sizes, groups = _by_size(off_rows)
    keys = _solver_keys(off_rows)
    fig, ax = plt.subplots(figsize=(ps.W2 * 0.62, 2.6))
    for s in ps.SOLVERS:
        if s in keys:
            y = _median_series(sizes, groups, keys[s][0], keys[s][2])
            if not np.all(np.isnan(y)):
                ps.line(ax, s, sizes, y)
    ax.set_yscale("log")
    _x_ticks(ax, sizes)
    ax.set_xlabel(r"QP dimension $n = 2N$")
    ax.set_ylabel("Median solve time (ms)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig01_lipm_performance.{ext}"))
    plt.close(fig)
    print("  fig01_lipm_performance")


def _write_csv(rows, outpath):
    if not rows:
        return
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
    idxs = [0, len(sizes)//2, -1]
    picks = [sizes[i] for i in idxs]

    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.6), sharey=False)
    for ax, nn in zip(axes, picks):
        geo_it_vals = [r.get('Geo_it') for r in rows if int(r.get('QP_n', r.get('n'))) == nn]
        avg_pivot_vals = [r.get('AvgPivot') for r in rows if int(r.get('QP_n', r.get('n'))) == nn]
        geo_hops_vals = [r.get('Geo_hops') for r in rows if int(r.get('QP_n', r.get('n'))) == nn]

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

        qpo_vals = [float(r['qpO_it']) for r in rows if int(r.get('QP_n', r.get('n'))) == nn and r.get('qpO_it') is not None and r.get('qpO_it') >= 0]

        geo = np.array(geo_vals) if geo_vals else np.array([])
        qpo = np.array(qpo_vals) if qpo_vals else np.array([])
        data = [geo, qpo]

        total_samples = sum(int(d.size) for d in data)
        if total_samples == 0:
            geo = np.array([float(r['Geo_ms']) for r in rows if int(r.get('QP_n', r.get('n'))) == nn and r.get('Geo_ms') is not None])
            qpo = np.array([float(r['qpO_ms']) for r in rows if int(r.get('QP_n', r.get('n'))) == nn and r.get('qpO_ms') is not None])
            data = [geo, qpo]
            total_samples = sum(int(d.size) for d in data)
            ylabel = 'Solve time (ms)'

        if total_samples == 0:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
            continue

        enough_for_violin = all(d.size >= 2 for d in data)
        colors = [C_GEO, C_QPO]
        if enough_for_violin:
            parts = ax.violinplot([d for d in data], showmeans=False, showextrema=False)
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
    counts_per_size = {}
    for nn in sizes:
        counts_per_size[nn] = sum(1 for r in rows if int(r.get('QP_n', r.get('n'))) == nn and r.get('Geo_us') is not None)
    nn = max(counts_per_size, key=lambda k: counts_per_size[k])
    geo = sorted([float(r['Geo_ms']) for r in rows if int(r.get('QP_n', r.get('n'))) == nn and r.get('Geo_ms') is not None])
    qpo = sorted([float(r['qpO_ms']) for r in rows if int(r.get('QP_n', r.get('n'))) == nn and r.get('qpO_ms') is not None])
    osqp = sorted([float(r['OSQP_ms']) for r in rows if int(r.get('QP_n', r.get('n'))) == nn and r.get('OSQP_ms') is not None])
    total_samples = len(geo) + len(qpo) + len(osqp)
    if total_samples < 4:
        geo = sorted([float(r['Geo_ms']) for r in rows if r.get('Geo_ms') is not None])
        qpo = sorted([float(r['qpO_ms']) for r in rows if r.get('qpO_ms') is not None])
        osqp = sorted([float(r['OSQP_ms']) for r in rows if r.get('OSQP_ms') is not None])
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
    sizes, groups = _by_size(off_rows)
    keys = _solver_keys(off_rows)
    fig, ax = plt.subplots(figsize=(ps.W2 * 0.62, 2.6))
    for s in ps.ACTIVE_SET:
        if s in keys:
            y = _median_series(sizes, groups, keys[s][1], keys[s][2])
            if not np.all(np.isnan(y)):
                ps.line(ax, s, sizes, np.maximum(y, 1.0))
    ax.set_yscale("log")
    _x_ticks(ax, sizes)
    ax.set_xlabel(r"QP dimension $n = 2N$")
    ax.set_ylabel("Iterations")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig02_lipm_iterations.{ext}"))
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
