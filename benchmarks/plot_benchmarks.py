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
        ("perf_affine", r"Diagonal-T affine image"),
    ]:
        sub = _extract_sub_block(perf_blk, sub_re)
        # Format variants (field indices: n type Geo qpO [OSQP_us] ratio Pivot qpO_iter [OSQP_iter] Hops ObjDiff Result)
        #  with OSQP  (12 fields): n type Geo qpO OSQP_us ratio Pivot qpO_iter OSQP_iter Hops ObjDiff Result
        #  without    (10 fields): n type Geo qpO         ratio Pivot qpO_iter           Hops ObjDiff Result
        #  old 9-field:            n type Geo qpO         ratio Pivot                    Hops ObjDiff Result
        def _row(p):
            # Robustly detect new/old table formats. The ratio column contains
            # a trailing 'x' (e.g. '0.627x'); locate it to infer column layout.
            if len(p) < 7:
                return None
            # find ratio token index (first token that ends with 'x')
            ratio_idx = None
            for idx, tok in enumerate(p):
                if isinstance(tok, str) and tok.endswith('x'):
                    ratio_idx = idx
                    break
            if ratio_idx is None:
                return None

            # Core fields
            try:
                geo_us = float(p[2])
                qpo_us = float(p[3])
            except ValueError:
                return None

            # Columns between qpO_us (p[3]) and the ratio token can vary.
            # Common layouts:
            #  - no extra cols: ratio at p[4]
            #  - OSQP present: OSQP_us at p[4], ratio at p[5]
            #  - warm-start + OSQP: qpO_vh_us at p[4], OSQP_us at p[5], ratio at p[6]
            qpovh_us = None
            osqp_us = None
            extra_count = max(0, ratio_idx - 4)
            try:
                if extra_count == 0:
                    qpovh_us = None
                elif extra_count == 1:
                    # single extra column — treat as OSQP_us
                    osqp_us = float(p[4])
                else:
                    # two or more extras — assume qpOASES warm-start then OSQP
                    qpovh_us = float(p[4])
                    # second extra may be OSQP_us
                    try:
                        osqp_us = float(p[5])
                    except (IndexError, ValueError):
                        osqp_us = None
            except ValueError:
                qpovh_us = None
                osqp_us = None

            # Average pivots sits immediately after the ratio token
            try:
                avg_pivot = float(p[ratio_idx + 1])
            except (IndexError, ValueError):
                avg_pivot = None

            # qpOASES iteration count is the token after avg_pivot
            qpO_iter = None
            qpO_vh_iter = None
            osqp_iter = None
            try:
                qpO_iter = float(p[ratio_idx + 2])
            except (IndexError, ValueError):
                qpO_iter = None

            # If qpO_vh was present, an extra iteration column follows
            if qpovh_us is not None:
                try:
                    qpO_vh_iter = float(p[ratio_idx + 3])
                except (IndexError, ValueError):
                    qpO_vh_iter = None
                # OSQP_iter may follow after that
                try:
                    osqp_iter = float(p[ratio_idx + 4])
                except (IndexError, ValueError):
                    osqp_iter = None
            else:
                # No qpO_vh column: OSQP_iter may be immediately after qpO_iter
                try:
                    osqp_iter = float(p[ratio_idx + 3])
                except (IndexError, ValueError):
                    osqp_iter = None

            # Hops token is either after OSQP_iter or after qpO_iter when OSQP absent
            avg_hops = None
            try:
                avg_hops = float(p[-3])
            except (IndexError, ValueError):
                try:
                    avg_hops = float(p[ratio_idx + 3 if qpovh_us is None else ratio_idx + 4])
                except (IndexError, ValueError):
                    avg_hops = None

            # osqp_us already attempted above; keep as is if parsed.

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


# ===========================================================================
#  Fig 1 — Performance: GeoQP vs qpOASES
# ===========================================================================

def fig01_performance(data, outdir):
    # Panel A: dense axis-aligned Q (GeoQP vs qpOASES vs OSQP)
    rows_dense = sorted(data.get("perf_dense", []), key=lambda r: r["n"]) if data.get("perf_dense") else []
    # Panel B: affine-diag entries from perf_affine (T diagonal)
    perf_aff = data.get("perf_affine", [])
    rows_affine_diag = sorted([r for r in perf_aff if "affine-diag" in r.get("type", "").lower()], key=lambda r: r["n"]) if perf_aff else []

    if not rows_dense and not rows_affine_diag:
        print("  [skip] fig01_performance (no data)"); return

    fig, axes = plt.subplots(1, 2, figsize=(W2, H_FIG))

    def _panel(ax, rows, title, geo_color=C_GEO):
        rows = sorted(rows, key=lambda r: r["n"])
        d = _to_arrays(rows, ["n", "Geo_us", "qpO_us"])
        ax.plot(d["n"], d["Geo_us"],  "o-",  color=C_GEO, label="GaasQP",   zorder=3)
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

    if rows_dense:
        _panel(axes[0], rows_dense, r"(a) Axis-aligned Box (dense $H$)")
    else:
        axes[0].set_visible(False)
    if rows_affine_diag:
        _panel(axes[1], rows_affine_diag, r"(b) Affine Image (diagonal $T$, Dense $H$)")
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

    # Panel definitions: (geo_qtype_for_hops, color, perf_key, title, perf_filter_fn)
    panels = [
        ("dense",    C_GEO_DENSE, "perf_dense", r"(a) Axis-aligned Box (dense $H$)", None),
        ("diagonal", C_GEO_DIAG,  "perf_affine", r"(b) Affine Image (diagonal $T$, Dense $H$)", lambda r: "affine-diag" in r.get("type", "").lower()),
    ]

    for col_idx, (geo_qtype, col, perf_key, title, perf_filter) in enumerate(panels):
        ax = axes[col_idx]

        # GeoQP pivots: normally from hops_vs_n matched by Qtype (geo_qtype).
        # For the affine-diag panel the hops_vs_n table contains only box QP
        # entries; use the AvgPivot reported in the PERFORMANCE affine block
        # instead so panel (b) actually plots the affine-diag case.
        if perf_key == "perf_affine" and perf_filter is not None:
            geo_rows = sorted([r for r in data.get(perf_key, []) if perf_filter(r) and r.get("AvgPivot") is not None], key=lambda r: r["n"]) 
            if geo_rows:
                dg = _to_arrays(geo_rows, ["n", "AvgPivot"])
                ax.plot(dg["n"], dg["AvgPivot"], "o-", color=C_GEO,
                    linewidth=1.6, label="GaaQP", zorder=3)
        else:
            sub = sorted(
                [r for r in _filter(rows_hop, "Qtype", geo_qtype) if r.get("n_solved", 0) > 0],
                key=lambda r: r["n"],
            )
            if sub:
                d = _to_arrays(sub, ["n", "AvgPivot", "MaxPivot"])
                ax.plot(d["n"], d["AvgPivot"], "o-", color=C_GEO,
                    linewidth=1.6, label="GaasQP", zorder=3)

        # qpOASES iterations: for perf_affine we may need to filter to affine-diag rows
        perf_rows = sorted(
            [r for r in data.get(perf_key, []) if (r.get("qpO_iter") is not None and (perf_filter(r) if perf_filter else True))],
            key=lambda r: r["n"],
        )
        if perf_rows:
            dp = _to_arrays(perf_rows, ["n", "qpO_iter"])
            ax.plot(dp["n"], dp["qpO_iter"], "s--", color=C_QPO,
                    linewidth=1.4, label="qpOASES", zorder=2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.9)
        ax.set_xlabel(r"$n$")
        ax.set_title(title, pad=4)
        ax.legend(fontsize=7, loc="upper left")
        if col_idx == 0:
            ax.set_ylabel("Iterations")

    fig.tight_layout()
    _save(fig, outdir, "fig02_iterations")


# Fig.3 removed by request: affine-only iterations handled in fig02_iterations


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
    # fig03 removed — affine iteration comparisons included in fig02

    print(f"\nDone. 2 figures (PDF + PNG) saved to: {outdir}")


if __name__ == "__main__":
    main()
