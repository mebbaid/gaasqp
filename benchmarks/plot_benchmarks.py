"""
plot_benchmarks.py — random box QPs benchmark.

Generates two figures (PDF + PNG):

  fig01_performance.pdf    GaasQP vs qpOASES wall-clock time vs n
                           (a) diagonal Q   (b) dense Q
  fig02_iterations.pdf     GaasQP active-set pivots and facet-adjacency hops vs n
                           (a) pivots       (b) hops
"""

import re
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          12,
    "axes.titlesize":     12,
    "axes.labelsize":     15,
    "xtick.labelsize":    13,
    "ytick.labelsize":    13,
    "legend.fontsize":    14,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "0.7",
    "legend.fancybox":    False,
    "lines.linewidth":    1.5,
    "lines.markersize":   5,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.4,
    "grid.linestyle":     "--",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

C_GEO_DIAG  = "#2166ac"
C_GEO_DENSE = "#d6604d"
C_QPO       = "#4daf4a"
C_OSQP      = "#ff7f00"
C_BOUND     = "#555555"

# Single GeoQP color for consistency across figures
C_GEO = C_GEO_DIAG

W2    = 7.16
H_FIG = W2 * 0.618 * 0.55


# ---------------------------------------------------------------------------
#  Parsing
# ---------------------------------------------------------------------------

def _numeric(tok):
    try:
        float(tok.rstrip("%x"))
        return True
    except ValueError:
        return False


def _extract_block(lines, start_re, stop_re=r"^={60,}", max_lines=500):
    out, found, skip_banner = [], False, False
    for line in lines:
        if not found:
            if re.search(start_re, line):
                found = True
                skip_banner = False
            continue
        if not skip_banner:
            skip_banner = True
            if line.strip().startswith("==="):
                continue
        if stop_re and re.search(stop_re, line):
            break
        if len(out) >= max_lines:
            break
        out.append(line)
    return out


def _extract_sub_block(lines, sub_re, stop_re=r"^>>>|^={60,}"):
    """Collect lines after the sub_re header until stop_re (blank lines kept)."""
    out, found = [], False
    for line in lines:
        if not found:
            if re.search(sub_re, line):
                found = True
            continue
        if stop_re and re.search(stop_re, line):
            break
        out.append(line)
    return out


def _parse_table(block, extractor):
    rows = []
    for line in block:
        s = line.strip()
        if not s or s.startswith(("===", "---", ">>>", "***")):
            continue
        parts = s.split()
        # Handle merged n+type when label overflows the fixed-width column
        # (e.g., "5affine-dense/refo" → ["5", "affine-dense/refo", ...])
        if parts and not _numeric(parts[0]):
            m = re.match(r'^(\d+)(\S+)$', parts[0])
            if m:
                parts = [m.group(1), m.group(2)] + parts[1:]
        if not parts or not _numeric(parts[0]):
            continue
        try:
            r = extractor(parts)
            if r is not None:
                rows.append(r)
        except (ValueError, IndexError):
            pass
    return rows


def _to_arrays(rows, keys):
    if not rows:
        return {k: np.array([]) for k in keys}
    return {k: np.array([r[k] for r in rows]) for k in keys}


def _filter(rows, key, value):
    return [r for r in rows if r.get(key) == value]


def parse_benchmark(filepath):
    with open(filepath) as fh:
        raw = fh.readlines()

    data = {}

    perf_blk = _extract_block(raw, r"PERFORMANCE.*GeoQP vs qpOASES")
    for tag, sub_re in [
        ("perf_diag",        r"Axis-aligned, diagonal Q"),
        ("perf_dense",       r"Axis-aligned, dense Q"),
        ("perf_affine",            r"Diagonal-T affine image"),
        ("perf_affine_dense",      r"Dense-T affine image, dense Q"),
        ("perf_affine_dense_refo", r"Dense-T affine image, reformulated"),
    ]:
        sub = _extract_sub_block(perf_blk, sub_re)
    
        def _row(p):

            if len(p) < 7:
                return None
            ratio_idx = None
            for idx, tok in enumerate(p):
                if isinstance(tok, str) and tok.endswith('x'):
                    ratio_idx = idx
                    break
            if ratio_idx is None:
                return None

            try:
                geo_us = float(p[2])
                qpo_us = float(p[3])
            except ValueError:
                return None

            qpovh_us = None
            osqp_us = None
            extra_count = max(0, ratio_idx - 4)
            try:
                if extra_count == 0:
                    qpovh_us = None
                elif extra_count == 1:
                    osqp_us = float(p[4])
                else:
                    qpovh_us = float(p[4])
                    try:
                        osqp_us = float(p[5])
                    except (IndexError, ValueError):
                        osqp_us = None
            except ValueError:
                qpovh_us = None
                osqp_us = None

            try:
                avg_pivot = float(p[ratio_idx + 1])
            except (IndexError, ValueError):
                avg_pivot = None

            qpO_iter = None
            qpO_vh_iter = None
            osqp_iter = None
            try:
                qpO_iter = float(p[ratio_idx + 2])
            except (IndexError, ValueError):
                qpO_iter = None

            if qpovh_us is not None:
                try:
                    qpO_vh_iter = float(p[ratio_idx + 3])
                except (IndexError, ValueError):
                    qpO_vh_iter = None
                try:
                    osqp_iter = float(p[ratio_idx + 4])
                except (IndexError, ValueError):
                    osqp_iter = None
            else:
                try:
                    osqp_iter = float(p[ratio_idx + 3])
                except (IndexError, ValueError):
                    osqp_iter = None

            avg_hops = None
            try:
                avg_hops = float(p[-3])
            except (IndexError, ValueError):
                try:
                    avg_hops = float(p[ratio_idx + 3 if qpovh_us is None else ratio_idx + 4])
                except (IndexError, ValueError):
                    avg_hops = None


            return dict(
                n=int(p[0]),
                type=p[1],
                Geo_us=geo_us,
                qpO_us=qpo_us,
                qpO_vh_us=qpovh_us,
                OSQP_us=osqp_us,
                Ratio=float(p[ratio_idx].rstrip('x')),
                AvgPivot=avg_pivot,
                qpO_iter=qpO_iter,
                qpO_vh_iter=qpO_vh_iter,
                OSQP_iter=osqp_iter,
                AvgHops=avg_hops,
            )
        data[tag] = _parse_table(sub, _row)

    hop_blk = _extract_block(raw, r"PHASE II HOP ANALYSIS")
    sub_a = _extract_sub_block(hop_blk, r"\(a\) Hop count vs n")
    data["hops_vs_n"] = _parse_table(sub_a, lambda p: dict(
        n=int(p[0]), Qtype=p[1],
        AvgPivot=float(p[2]), MaxPivot=int(p[3]),
        AvgHops=float(p[4]),  MaxHops=int(p[5]),
        n_solved=int(p[10]),
    ) if len(p) >= 11 else None)

    return data


def _save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    print(f"  {name}")
    plt.close(fig)



def fig01_performance(data, outdir):
    rows_dense = sorted(data.get("perf_dense", []), key=lambda r: r["n"])

    if not rows_dense and not data.get("perf_affine_dense_refo"):
        print("  [skip] fig01_performance (no data)"); return

    fig, axes = plt.subplots(2, 1, figsize=(W2 * 0.6, H_FIG * 2.4))

    ax = axes[0]
    has_data_a = False
    if rows_dense:
        has_data_a = True
        d = _to_arrays(rows_dense, ["n", "Geo_us", "qpO_us"])
        ax.plot(d["n"], d["Geo_us"], "o-",  color=C_GEO_DIAG, label="GaasQP",   zorder=3)
        ax.plot(d["n"], d["qpO_us"], "s--", color=C_QPO,       label="qpOASES", zorder=3)
        osqp_rows = [r for r in rows_dense if r.get("OSQP_us") is not None]
        if osqp_rows:
            do = _to_arrays(osqp_rows, ["n", "OSQP_us"])
            ax.plot(do["n"], do["OSQP_us"], "^:", color=C_OSQP, label="OSQP", zorder=2)
    if has_data_a:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"Dimension $n$")
        ax.set_ylabel(r"Solve time ($\mu$s)")
        ax.set_title(r"(a) Axis-aligned Box, dense $H$")
        ax.legend(fontsize=14)
    else:
        ax.set_visible(False)

    ax = axes[1]
    rows_affine_dense_refo = sorted(data.get("perf_affine_dense_refo", []), key=lambda r: r["n"])
    if rows_affine_dense_refo:
        d = _to_arrays(rows_affine_dense_refo, ["n", "Geo_us", "qpO_us"])
        ax.plot(d["n"], d["Geo_us"], "o-",  color=C_GEO_DIAG, label="GaasQP",   zorder=3)
        ax.plot(d["n"], d["qpO_us"], "s--", color=C_QPO,       label="qpOASES", zorder=3)
        osqp_rows = [r for r in rows_affine_dense_refo if r.get("OSQP_us") is not None]
        if osqp_rows:
            do = _to_arrays(osqp_rows, ["n", "OSQP_us"])
            ax.plot(do["n"], do["OSQP_us"], "^:", color=C_OSQP, label="OSQP", zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"Dimension $n$")
        ax.set_ylabel(r"Solve time ($\mu$s)")
        ax.set_title(r"(b) Affine Image, dense $T$, dense $H$")
        ax.legend(fontsize=14)
    else:
        ax.set_visible(False)

    fig.tight_layout()
    _save(fig, outdir, "fig01_performance")



def fig02_iterations(data, outdir):
    rows_hop = data.get("hops_vs_n", [])
    if not rows_hop:
        print("  [skip] fig02_iterations (no data)"); return

    fig, axes = plt.subplots(2, 1, figsize=(W2 * 0.6, H_FIG * 2.4),
                             gridspec_kw={"hspace": 0.45})

    ax = axes[0]
    sub_dense = sorted(
        [r for r in _filter(rows_hop, "Qtype", "dense") if r.get("n_solved", 0) > 0],
        key=lambda r: r["n"],
    )
    if sub_dense:
        d = _to_arrays(sub_dense, ["n", "AvgPivot"])
        ax.plot(d["n"], d["AvgPivot"], "o-", color=C_GEO_DIAG,
                linewidth=1.6, label="GaasQP", zorder=3)
    perf_dense_rows = sorted(
        [r for r in data.get("perf_dense", []) if r.get("qpO_iter") is not None],
        key=lambda r: r["n"],
    )
    if perf_dense_rows:
        dp = _to_arrays(perf_dense_rows, ["n", "qpO_iter"])
        ax.plot(dp["n"], dp["qpO_iter"], "s--", color=C_QPO,
                linewidth=1.4, label="qpOASES", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.9)
    ax.set_xlabel(r"$n$")
    ax.set_title(r"(a) Axis-aligned Box, dense $H$", pad=4)
    ax.legend(fontsize=14, loc="upper left")
    ax.set_ylabel("Iterations")

    ax = axes[1]
    aff_refo_rows = sorted(
        [r for r in data.get("perf_affine_dense_refo", []) if r.get("AvgPivot") is not None],
        key=lambda r: r["n"],
    )
    if aff_refo_rows:
        dg = _to_arrays(aff_refo_rows, ["n", "AvgPivot"])
        ax.plot(dg["n"], dg["AvgPivot"], "o-", color=C_GEO_DIAG,
                linewidth=1.6, label="GaasQP", zorder=3)
    aff_refo_qpo = sorted(
        [r for r in data.get("perf_affine_dense_refo", []) if r.get("qpO_iter") is not None],
        key=lambda r: r["n"],
    )
    if aff_refo_qpo:
        dp2 = _to_arrays(aff_refo_qpo, ["n", "qpO_iter"])
        ax.plot(dp2["n"], dp2["qpO_iter"], "s--", color=C_QPO,
                linewidth=1.4, label="qpOASES", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.9)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Iterations")
    ax.set_title(r"(b) Affine Image, dense $T$, dense $H$", pad=4)
    ax.legend(fontsize=14, loc="upper left")

    fig.tight_layout()
    _save(fig, outdir, "fig02_iterations")



def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    filepath = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(filepath)), "figures")
    os.makedirs(outdir, exist_ok=True)

    print(f"Parsing:  {filepath}")
    print(f"Output:   {outdir}")
    data = parse_benchmark(filepath)

    for k, v in data.items():
        if isinstance(v, list):
            print(f"  {k:22s}  {len(v):3d} rows")

    print("\nGenerating figures...")
    fig01_performance(data, outdir)
    fig02_iterations(data, outdir)

    print(f"\nDone. 2 figures (PDF + PNG) saved to: {outdir}")


if __name__ == "__main__":
    main()
