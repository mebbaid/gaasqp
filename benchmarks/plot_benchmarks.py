"""
plot_benchmarks.py — Journal-quality figures for GeoQP box benchmark.

Generates two publication-ready figures (PDF + PNG):

  fig01_performance.pdf    GeoQP vs qpOASES wall-clock time vs n
                           (a) diagonal Q   (b) dense Q
  fig02_iterations.pdf     GeoQP active-set pivots and facet-adjacency hops vs n
                           (a) pivots       (b) hops

Usage:
    python plot_benchmarks.py <benchmark_output.txt> [output_dir]

Pipe benchmark output to a file first:
    ./build/benchmarks/benchmark_box > benchmark_results.txt 2>&1
    python benchmarks/plot_benchmarks.py benchmark_results.txt figures/
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
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
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
        ("perf_diag",  r"Axis-aligned, diagonal Q"),
        ("perf_dense", r"Axis-aligned, dense Q"),
    ]:
        sub = _extract_sub_block(perf_blk, sub_re)
        # Format variants (field indices: n type Geo qpO [OSQP_us] ratio Pivot qpO_iter [OSQP_iter] Hops ObjDiff Result)
        #  with OSQP  (12 fields): n type Geo qpO OSQP_us ratio Pivot qpO_iter OSQP_iter Hops ObjDiff Result
        #  without    (10 fields): n type Geo qpO         ratio Pivot qpO_iter           Hops ObjDiff Result
        #  old 9-field:            n type Geo qpO         ratio Pivot                    Hops ObjDiff Result
        def _row(p):
            if len(p) < 7:
                return None
            has_osqp     = len(p) >= 12
            has_qpo_iter = len(p) >= 10
            off = 1 if has_osqp else 0   # shift when OSQP_us column present
            qpo_iter_idx = 6 + off
            osqp_iter_off = 1 if has_osqp else 0  # OSQP_iter comes after qpO_iter when present
            hops_idx = qpo_iter_idx + osqp_iter_off + (1 if has_qpo_iter else 0)
            return dict(
                n=int(p[0]),
                Geo_us=float(p[2]),
                qpO_us=float(p[3]),
                OSQP_us=float(p[4]) if has_osqp else None,
                Ratio=float(p[4 + off].rstrip("x")),
                AvgPivot=float(p[5 + off]),
                qpO_iter=float(p[qpo_iter_idx]) if has_qpo_iter else None,
                OSQP_iter=float(p[qpo_iter_idx + 1]) if has_osqp else None,
                AvgHops=float(p[hops_idx]),
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


# ===========================================================================
#  Fig 1 — Performance: GeoQP vs qpOASES
# ===========================================================================

def fig01_performance(data, outdir):
    pd_ = data.get("perf_diag", [])
    pq  = data.get("perf_dense", [])
    if not pd_ and not pq:
        print("  [skip] fig01_performance (no data)"); return

    fig, axes = plt.subplots(1, 2, figsize=(W2, H_FIG))

    def _panel(ax, rows, title):
        rows = sorted(rows, key=lambda r: r["n"])
        d = _to_arrays(rows, ["n", "Geo_us", "qpO_us"])
        ax.plot(d["n"], d["Geo_us"],  "o-",  color=C_GEO_DIAG, label="GeoQP",   zorder=3)
        ax.plot(d["n"], d["qpO_us"], "s--", color=C_QPO,       label="qpOASES", zorder=3)
        osqp_rows = [r for r in rows if r.get("OSQP_us") is not None]
        if osqp_rows:
            do = _to_arrays(osqp_rows, ["n", "OSQP_us"])
            ax.plot(do["n"], do["OSQP_us"], "^:", color=C_OSQP, label="OSQP", zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"Dimension $n$")
        ax.set_ylabel(r"Solve time ($\mu$s)")
        ax.set_title(title)
        ax.legend()

    if pd_:
        _panel(axes[0], pd_, r"(a) Diagonal $Q$")
    else:
        axes[0].set_visible(False)
    if pq:
        _panel(axes[1], pq, r"(b) Dense $Q$")
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    _save(fig, outdir, "fig01_performance")


# ===========================================================================
#  Fig 2 — Iterations: diagonal Q vs dense Q  (1 × 2)
#
#    (a) Diagonal Q — GeoQP pivots (flat) vs qpOASES (O(n))
#    (b) Dense Q    — GeoQP pivots vs qpOASES
#
#  GeoQP pivots are bounded because hops ≤ 2 (Corollary 2); indicated by
#  an annotation rather than a second row of panels.
# ===========================================================================

def fig02_iterations(data, outdir):
    rows_hop = data.get("hops_vs_n", [])
    if not rows_hop:
        print("  [skip] fig02_iterations (no data)"); return

    fig, axes = plt.subplots(1, 2, figsize=(W2, H_FIG),
                             gridspec_kw={"wspace": 0.30})

    for col_idx, (qtype, col, perf_key, title) in enumerate([
        ("diagonal", C_GEO_DIAG,  "perf_diag",  r"(a) Diagonal $Q$"),
        ("dense",    C_GEO_DENSE, "perf_dense",  r"(b) Dense $Q$"),
    ]):
        ax = axes[col_idx]

        sub = sorted(
            [r for r in _filter(rows_hop, "Qtype", qtype) if r["n_solved"] > 0],
            key=lambda r: r["n"],
        )
        if sub:
            d = _to_arrays(sub, ["n", "AvgPivot", "MaxPivot"])
            ax.plot(d["n"], d["AvgPivot"], "o-", color=col,
                    linewidth=1.6, label="GeoQP (avg)", zorder=3)
            ax.fill_between(d["n"], d["AvgPivot"], d["MaxPivot"],
                            alpha=0.15, color=col, label="GeoQP (max)")

        perf_rows = sorted(
            [r for r in data.get(perf_key, []) if r.get("qpO_iter") is not None],
            key=lambda r: r["n"],
        )
        if perf_rows:
            dp = _to_arrays(perf_rows, ["n", "qpO_iter"])
            ax.plot(dp["n"], dp["qpO_iter"], "s--", color=C_QPO,
                    linewidth=1.4, label="qpOASES (avg)", zorder=2)

        osqp_iter_rows = sorted(
            [r for r in data.get(perf_key, []) if r.get("OSQP_iter") is not None],
            key=lambda r: r["n"],
        )
        if osqp_iter_rows:
            do = _to_arrays(osqp_iter_rows, ["n", "OSQP_iter"])
            ax.plot(do["n"], do["OSQP_iter"], "^:", color=C_OSQP,
                    linewidth=1.4, label="OSQP (avg)", zorder=2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.9)
        ax.set_xlabel(r"$n$")
        ax.set_title(title, pad=4)
        ax.legend(fontsize=7, loc="upper left")
        if col_idx == 0:
            ax.set_ylabel("Iterations")

        # Annotate the hop bound as a text note instead of a second panel
        ax.annotate(r"$\leq\!2$ hops (Cor. 2) $\Rightarrow$ const. GeoQP pivots",
                    xy=(0.97, 0.06), xycoords="axes fraction",
                    fontsize=6.5, color=C_BOUND, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.85))

    fig.tight_layout()
    _save(fig, outdir, "fig02_iterations")


# ===========================================================================
#  MAIN
# ===========================================================================

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
