"""
plot_benchmarks.py — Journal-quality figures for GeoQP box benchmark.

Generates publication-ready figures (PDF + PNG) from the output of
benchmark_box, covering every section of the box_antipodal.tex theory:

  fig01_lemma_fastpath.pdf       Lemma — O(n) fast-path, solve time & pivots
  fig02_prop1_p_plus.pdf         Proposition 1 — x* in P_+  margins
  fig03_prop2_halfspaces.pdf     Proposition 2 — x* in P_+ cap H1 cap H2
  fig04_hops_vs_n.pdf            Corollary — pivots & hops vs dimension
  fig05_separation_sweep.pdf     Separation sweep — hops vs ||g||
  fig06_conditioning.pdf         Condition number sweep — hops & pivots vs kappa(Q)
  fig07_affine_image.pdf         Affine image T(B)+c verification
  fig08_timing_breakdown.pdf     Phase-level timing breakdown (stacked bars)
  fig09_performance.pdf          GeoQP vs qpOASES wall-clock comparison
  fig10_hotstart.pdf             Hotstart: fixed Q, varying g

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
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
#  Journal RC defaults (IEEE / Springer single- or double-column)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    7.5,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "0.7",
    "legend.fancybox":    False,
    "lines.linewidth":    1.5,
    "lines.markersize":   5,
    "axes.grid":          True,
    "grid.alpha":         0.30,
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

# Colour palette — colour-blind safe (adapted from ColorBrewer)
C = {
    "diag":     "#2166ac",   # blue
    "dense":    "#d6604d",   # muted red
    "geo":      "#2166ac",
    "qpoases":  "#4daf4a",  # green
    "ph1":      "#66c2a5",   # teal
    "ph2":      "#fc8d62",   # salmon
    "ph3":      "#8da0cb",   # periwinkle
    "bound":    "#333333",   # dark grey
    "affdiag":  "#1b9e77",   # dark teal
    "affdense": "#d95f02",   # dark orange
}

W1 = 3.5   # single-column width (inches)
W2 = 7.16  # double-column width (inches)
GOLDEN = 0.618


# ---------------------------------------------------------------------------
#  Parsing utilities
# ---------------------------------------------------------------------------

def _numeric(tok):
    """Return True if tok looks like a number (possibly with trailing %, x)."""
    try:
        float(tok.rstrip("%x"))
        return True
    except ValueError:
        return False


def _extract_block(lines, start_re, stop_re=r"^={60,}", max_lines=400):
    """Extract lines between start_re match and next stop_re match."""
    out, found, banner_skipped = [], False, False
    for line in lines:
        if not found:
            if re.search(start_re, line):
                found = True
                banner_skipped = False
            continue
        if not banner_skipped:
            banner_skipped = True
            if line.strip().startswith("==="):
                continue
        if stop_re and re.search(stop_re, line):
            break
        if len(out) >= max_lines:
            break
        out.append(line)
    return out


def _parse_table(block, extractor):
    """Apply extractor(fields) to each data line; skip headers/separators."""
    rows = []
    for line in block:
        s = line.strip()
        if not s or s.startswith("===") or s.startswith("---") or s.startswith(">>>"):
            continue
        if s.startswith("***"):
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


def _extract_sub_block(lines, sub_re, stop_re=r"^>>>|^$|^={60,}"):
    """Extract lines within a >>> sub-section."""
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


# ---------------------------------------------------------------------------
#  parse_benchmark — parses the full benchmark_box output
# ---------------------------------------------------------------------------

def parse_benchmark(filepath):
    with open(filepath) as fh:
        raw = fh.readlines()

    data = {}

    # ---- LEMMA ----
    blk = _extract_block(raw, r"LEMMA.*Antipodal Good Vertex")
    data["lemma"] = _parse_table(blk, lambda p: dict(
        n=int(p[0]), Qtype=p[1], Verts=float(p[2]),
        Ph2hops=float(p[7]), Time_us=float(p[8]),
    ) if len(p) >= 9 else None)

    # ---- PROPOSITION 1 ----
    blk = _extract_block(raw, r"PROPOSITION 1.*x\*.*P")
    all_p1 = []
    for line in blk:
        s = line.strip()
        if not s or s.startswith("===") or s.startswith("---") or s.startswith(">>>"):
            continue
        parts = s.split()
        if len(parts) >= 8 and _numeric(parts[0]):
            try:
                all_p1.append(dict(
                    n=int(parts[0]), Qtype=parts[1], scale_g=float(parts[2]),
                    avg_margin=float(parts[3]), min_margin=float(parts[4]),
                    avg_v_norm=float(parts[5]), avg_L=float(parts[6]),
                    n_solved=int(parts[7]),
                ))
            except (ValueError, IndexError):
                pass
    data["prop1"] = all_p1

    # ---- PROPOSITION 2 ----
    blk = _extract_block(raw, r"PROPOSITION 2.*P.*H.*H")
    data["prop2"] = _parse_table(blk, lambda p: dict(
        n=int(p[0]), Qtype=p[1],
        P_margin=float(p[2]), H1_margin=float(p[3]), H2_margin=float(p[4]),
        min_P=float(p[6]), min_H1=float(p[7]), min_H2=float(p[8]),
    ) if len(p) >= 9 else None)

    # ---- PHASE II HOP ANALYSIS ----
    hop_blk = _extract_block(raw, r"PHASE II HOP ANALYSIS")

    # (a) Hop count vs n
    sub_a = _extract_sub_block(hop_blk, r"\(a\) Hop count vs n")
    data["hops_vs_n"] = _parse_table(sub_a, lambda p: dict(
        n=int(p[0]), Qtype=p[1],
        AvgPivot=float(p[2]), MaxPivot=int(p[3]),
        AvgHops=float(p[4]), MaxHops=int(p[5]),
        Ph1Ph2_pct=p[8].rstrip("%"), Time_us=float(p[9]),
        n_solved=int(p[10]),
    ) if len(p) >= 11 else None)

    # (b) Separation sweep
    sub_b = _extract_sub_block(hop_blk, r"\(b\) Separation sweep")
    data["sep_sweep"] = _parse_table(sub_b, lambda p: dict(
        scale_g=float(p[0]), avg_v_norm=float(p[1]),
        AvgPivot=float(p[2]), MaxPivot=int(p[3]),
        AvgHops=float(p[4]), MaxHops=int(p[5]),
        Ph2_us=float(p[6]), n_solved=int(p[7]),
    ) if len(p) >= 8 else None)

    # (c) Condition number sweep — don't stop on blank lines (violation dumps contain them)
    sub_c = _extract_sub_block(hop_blk, r"\(c\) Q condition number",
                               stop_re=r"^>>>|^={60,}")
    data["cond_sweep"] = _parse_table(sub_c, lambda p: dict(
        cond=float(p[0]),
        AvgPivot=float(p[1]), MaxPivot=int(p[2]),
        AvgHops=float(p[3]), MaxHops=int(p[4]),
        ObjDiff=float(p[5]), n_solved=int(p[6]),
    ) if len(p) >= 7 else None)

    # ---- AFFINE IMAGE ----
    aff_blk = _extract_block(raw, r"AFFINE IMAGE")
    sub_diag = _extract_sub_block(aff_blk, r"\(a\) Diagonal T")
    data["affine_diag"] = _parse_table(sub_diag, lambda p: dict(
        n=int(p[0]), Verts=float(p[1]),
        P_margin=float(p[3]), H1_margin=float(p[4]),
        H2_margin=float(p[5]), ObjDiff=float(p[6]),
    ) if len(p) >= 7 else None)

    sub_dense = _extract_sub_block(aff_blk, r"\(b\) Dense T")
    data["affine_dense"] = _parse_table(sub_dense, lambda p: dict(
        n=int(p[0]), AvgVerts=float(p[1]), Ph1iters=float(p[2]),
        Pivots=float(p[3]), Hops=float(p[4]),
        Total_us=float(p[5]), ObjDiff=float(p[6]),
    ) if len(p) >= 7 else None)

    # ---- TIMING BREAKDOWN ----
    tim_blk = _extract_block(raw, r"TIMING BREAKDOWN")
    sub_td = _extract_sub_block(tim_blk, r"Diagonal Q")
    data["timing_diag"] = _parse_table(sub_td, lambda p: dict(
        n=int(p[0]),
        Total_us=float(p[2]), Ph1_us=float(p[3]),
        Ph2_us=float(p[4]), Ph3_us=float(p[5]),
        Ph1_pct=float(p[6].rstrip("%")),
        Ph2_pct=float(p[7].rstrip("%")),
        Ph3_pct=float(p[8].rstrip("%")),
        Pivots=float(p[9]), Hops=float(p[10]),
    ) if len(p) >= 11 else None)

    sub_tq = _extract_sub_block(tim_blk, r"Dense Q")
    data["timing_dense"] = _parse_table(sub_tq, lambda p: dict(
        n=int(p[0]),
        Total_us=float(p[2]), Ph1_us=float(p[3]),
        Ph2_us=float(p[4]), Ph3_us=float(p[5]),
        Ph1_pct=float(p[6].rstrip("%")),
        Ph2_pct=float(p[7].rstrip("%")),
        Ph3_pct=float(p[8].rstrip("%")),
        Pivots=float(p[9]), Hops=float(p[10]),
    ) if len(p) >= 11 else None)

    # ---- PERFORMANCE ----
    perf_blk = _extract_block(raw, r"PERFORMANCE.*GeoQP vs qpOASES")
    for tag, sub_re in [
        ("perf_diag",    r"Axis-aligned, diagonal Q"),
        ("perf_dense",   r"Axis-aligned, dense Q"),
        ("perf_affine",  r"Diagonal-T affine"),
        ("perf_illcond", r"Ill-conditioned"),
    ]:
        sub = _extract_sub_block(perf_blk, sub_re)
        data[tag] = _parse_table(sub, lambda p: dict(
            n=int(p[0]),
            Geo_us=float(p[2]), qpO_us=float(p[3]),
            Ratio=float(p[4].rstrip("x")),
            AvgPivot=float(p[5]), AvgHops=float(p[6]),
            ObjDiff=float(p[7]),
        ) if len(p) >= 8 else None)

    # ---- HOTSTART ----
    hot_blk = _extract_block(raw, r"HOTSTART.*Fixed Q.*Varying g")
    data["hotstart"] = _parse_table(hot_blk, lambda p: dict(
        n=int(p[0]), Qtype=p[1],
        Geo_us=float(p[2]), qpO_us=float(p[3]),
        Ratio=float(p[4].rstrip("x")),
        AvgPivot=float(p[5]), AvgHops=float(p[6]),
        ObjDiff=float(p[7]),
    ) if len(p) >= 8 else None)

    return data


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _to_arrays(rows, keys):
    if not rows:
        return {k: np.array([]) for k in keys}
    return {k: np.array([r[k] for r in rows]) for k in keys}


def _filter(rows, key, value):
    return [r for r in rows if r.get(key) == value]


def _save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    print(f"  {name}")
    plt.close(fig)


def _skip(name, reason="empty data"):
    print(f"  [skip] {name} ({reason})")


def _fig_height(ncols=2):
    return W2 * GOLDEN * 0.52


# ===========================================================================
#  Fig 1 — Lemma: O(n) Fast-Path
# ===========================================================================

def fig01_lemma(data, outdir):
    rows = data["lemma"]
    if not rows:
        _skip("fig01_lemma_fastpath"); return

    fig, axes = plt.subplots(1, 2, figsize=(W2, _fig_height()))

    for qtype, mk, col, lbl in [
        ("diagonal", "o", C["diag"],  r"Diagonal $Q$"),
        ("dense",    "s", C["dense"], r"Dense $Q$"),
    ]:
        sub = sorted(_filter(rows, "Qtype", qtype), key=lambda r: r["n"])
        if not sub:
            continue
        d = _to_arrays(sub, ["n", "Time_us", "Ph2hops"])
        axes[0].plot(d["n"], d["Time_us"], marker=mk, color=col, label=lbl)
        axes[1].plot(d["n"], d["Ph2hops"], marker=mk, color=col, label=lbl)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Dimension $n$")
    axes[0].set_ylabel(r"Solve time ($\mu$s)")
    axes[0].set_title("(a) Wall-clock time")
    axes[0].legend()

    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Dimension $n$")
    axes[1].set_ylabel("Phase-II pivots")
    axes[1].set_title("(b) Active-set pivots")
    axes[1].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    _save(fig, outdir, "fig01_lemma_fastpath")


# ===========================================================================
#  Fig 2 — Proposition 1: x* in P_+ margins
# ===========================================================================

def fig02_prop1(data, outdir):
    rows = data["prop1"]
    if not rows:
        _skip("fig02_prop1_p_plus"); return

    rows_10 = [r for r in rows if abs(r["scale_g"] - 10.0) < 0.1]
    if not rows_10:
        rows_10 = rows

    fig, axes = plt.subplots(1, 2, figsize=(W2, _fig_height()))

    for qtype, mk, col, lbl in [
        ("diagonal", "o", C["diag"],  r"Diagonal $Q$"),
        ("dense",    "s", C["dense"], r"Dense $Q$"),
    ]:
        sub = sorted(_filter(rows_10, "Qtype", qtype), key=lambda r: r["n"])
        if not sub:
            continue
        d = _to_arrays(sub, ["n", "avg_margin", "min_margin"])
        axes[0].plot(d["n"], d["avg_margin"], marker=mk, color=col, label=lbl)
        axes[1].plot(d["n"], d["min_margin"], marker=mk, color=col, label=lbl)

    for ax in axes:
        ax.axhline(0, color=C["bound"], linewidth=0.8, linestyle=":", zorder=0)
        ax.set_xlabel(r"Dimension $n$")

    axes[0].set_ylabel(r"$v^\top(x^* - c_0)$")
    axes[0].set_title(r"(a) Avg $\mathcal{P}_+$ margin ($\|g\|$ scale $= 10$)")
    axes[0].legend()

    axes[1].set_ylabel(r"min margin")
    axes[1].set_title(r"(b) Min $\mathcal{P}_+$ margin")
    axes[1].legend()
    axes[1].set_yscale("symlog", linthresh=1e-2)

    fig.tight_layout()
    _save(fig, outdir, "fig02_prop1_p_plus")


# ===========================================================================
#  Fig 3 — Proposition 2: x* in P_+ cap H1 cap H2
# ===========================================================================

def fig03_prop2(data, outdir):
    rows = data["prop2"]
    if not rows:
        _skip("fig03_prop2_halfspaces"); return

    fig, axes = plt.subplots(1, 2, figsize=(W2, _fig_height()))

    for qtype, mk, col, lbl in [
        ("diagonal", "o", C["diag"],  r"Diagonal $Q$"),
        ("dense",    "s", C["dense"], r"Dense $Q$"),
    ]:
        sub = sorted(_filter(rows, "Qtype", qtype), key=lambda r: r["n"])
        if not sub:
            continue
        d = _to_arrays(sub, ["n", "min_P", "min_H1", "min_H2"])
        axes[0].plot(d["n"], d["min_P"], marker=mk, color=col,
                     label=rf"$\mathcal{{P}}_+$ ({lbl})")
        axes[1].plot(d["n"], d["min_H1"], marker=mk, color=col,
                     linestyle="-",  label=rf"$H_1$ ({lbl})")
        axes[1].plot(d["n"], d["min_H2"], marker=mk, color=col,
                     linestyle="--", label=rf"$H_2$ ({lbl})")

    for ax in axes:
        ax.axhline(0, color=C["bound"], linewidth=0.8, linestyle=":", zorder=0)
        ax.set_xlabel(r"Dimension $n$")
        ax.set_yscale("symlog", linthresh=1e-2)

    axes[0].set_ylabel(r"Min $\mathcal{P}_+$ margin")
    axes[0].set_title(r"(a) $x^* \in \mathcal{P}_+$")
    axes[0].legend(fontsize=7)

    axes[1].set_ylabel("Min margin")
    axes[1].set_title(r"(b) $x^* \in H_1 \cap H_2$")
    axes[1].legend(fontsize=6.5, ncol=2)

    fig.tight_layout()
    _save(fig, outdir, "fig03_prop2_halfspaces")


# ===========================================================================
#  Fig 4 — Corollary: Pivots & Hops vs Dimension
# ===========================================================================

def fig04_hops_vs_n(data, outdir):
    rows = data["hops_vs_n"]
    if not rows:
        _skip("fig04_hops_vs_n"); return

    fig, axes = plt.subplots(1, 2, figsize=(W2, _fig_height()))

    for qtype, mk, col, lbl in [
        ("diagonal", "o", C["diag"],  r"Diagonal $Q$"),
        ("dense",    "s", C["dense"], r"Dense $Q$"),
    ]:
        sub = sorted([r for r in _filter(rows, "Qtype", qtype)
                       if r["n_solved"] > 0], key=lambda r: r["n"])
        if not sub:
            continue
        d = _to_arrays(sub, ["n", "AvgPivot", "MaxPivot", "AvgHops", "MaxHops"])

        axes[0].plot(d["n"], d["AvgPivot"], marker=mk, color=col, label=lbl)
        axes[0].fill_between(d["n"], d["AvgPivot"], d["MaxPivot"],
                             alpha=0.12, color=col)

        axes[1].plot(d["n"], d["AvgHops"], marker=mk, color=col, label=lbl)
        axes[1].fill_between(d["n"], d["AvgHops"], d["MaxHops"],
                             alpha=0.12, color=col)

    axes[1].axhline(2, color=C["bound"], linewidth=1.0, linestyle="--",
                    label=r"Corollary bound ($\leq 2$)", zorder=0)

    axes[0].set_xlabel(r"Dimension $n$")
    axes[0].set_ylabel("Active-set pivots")
    axes[0].set_title("(a) Pivots (avg, shaded = max)")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel(r"Dimension $n$")
    axes[1].set_ylabel("Facet-adjacency hops")
    axes[1].set_title("(b) Hops (avg, shaded = max)")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    _save(fig, outdir, "fig04_hops_vs_n")


# ===========================================================================
#  Fig 5 — Separation Sweep
# ===========================================================================

def fig05_separation(data, outdir):
    rows = data["sep_sweep"]
    if not rows:
        _skip("fig05_separation_sweep"); return

    rows = sorted(rows, key=lambda r: r["scale_g"])
    d = _to_arrays(rows, ["scale_g", "avg_v_norm", "AvgPivot",
                          "MaxPivot", "AvgHops", "MaxHops"])

    fig, axes = plt.subplots(1, 2, figsize=(W2, _fig_height()))

    axes[0].plot(d["scale_g"], d["AvgPivot"], "o-", color=C["diag"],
                 label="Avg pivots")
    axes[0].fill_between(d["scale_g"], d["AvgPivot"], d["MaxPivot"],
                         alpha=0.12, color=C["diag"])
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\|g\|$ scale $\alpha$")
    axes[0].set_ylabel("Pivots")
    axes[0].set_title(r"(a) Pivots ($n\!=\!10$, dense $Q$)")
    axes[0].legend()

    axes[1].plot(d["scale_g"], d["AvgHops"], "s-", color=C["dense"],
                 label="Avg hops")
    axes[1].fill_between(d["scale_g"], d["AvgHops"], d["MaxHops"],
                         alpha=0.12, color=C["dense"])
    axes[1].axhline(2, color=C["bound"], linewidth=1.0, linestyle="--",
                    label=r"Corollary bound ($\leq 2$)", zorder=0)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\|g\|$ scale $\alpha$")
    axes[1].set_ylabel("Facet-adjacency hops")
    axes[1].set_title(r"(b) Hops ($n\!=\!10$, dense $Q$)")
    axes[1].legend()

    fig.tight_layout()
    _save(fig, outdir, "fig05_separation_sweep")


# ===========================================================================
#  Fig 6 — Conditioning
# ===========================================================================

def fig06_conditioning(data, outdir):
    rows = data["cond_sweep"]
    if not rows:
        _skip("fig06_conditioning"); return

    rows = sorted(rows, key=lambda r: r["cond"])
    d = _to_arrays(rows, ["cond", "AvgPivot", "MaxPivot",
                          "AvgHops", "MaxHops", "ObjDiff"])

    fig, axes = plt.subplots(1, 2, figsize=(W2, _fig_height()))

    axes[0].plot(d["cond"], d["AvgPivot"], "o-", color=C["diag"],
                 label="Avg pivots", zorder=3)
    axes[0].fill_between(d["cond"], d["AvgPivot"], d["MaxPivot"],
                         alpha=0.12, color=C["diag"])
    axes[0].plot(d["cond"], d["AvgHops"], "s--", color=C["dense"],
                 label="Avg hops", zorder=3)
    axes[0].fill_between(d["cond"], d["AvgHops"], d["MaxHops"],
                         alpha=0.12, color=C["dense"])
    axes[0].axhline(2, color=C["bound"], linewidth=1.0, linestyle=":",
                    label=r"Hop bound ($\leq 2$)", zorder=0)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\kappa(Q)$")
    axes[0].set_ylabel("Count")
    axes[0].set_title(r"(a) Pivots & hops vs $\kappa(Q)$")
    axes[0].legend(fontsize=7)

    axes[1].plot(d["cond"], d["ObjDiff"], "D-", color=C["dense"], zorder=3)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\kappa(Q)$")
    axes[1].set_ylabel(r"$|f^*_{\mathrm{GeoQP}} - f^*_{\mathrm{ref}}|$")
    axes[1].set_title(r"(b) Objective accuracy vs $\kappa(Q)$")

    fig.tight_layout()
    _save(fig, outdir, "fig06_conditioning")


# ===========================================================================
#  Fig 7 — Affine Image T(B)+c
# ===========================================================================

def fig07_affine(data, outdir):
    rd = data["affine_diag"]
    rn = data["affine_dense"]
    if not rd and not rn:
        _skip("fig07_affine_image"); return

    fig, axes = plt.subplots(1, 2, figsize=(W2, _fig_height()))

    if rd:
        rd = sorted(rd, key=lambda r: r["n"])
        d = _to_arrays(rd, ["n", "P_margin", "H1_margin", "H2_margin", "ObjDiff"])
        axes[0].plot(d["n"], d["P_margin"],  "o-",  color=C["diag"],
                     label=r"$\mathcal{P}_+$")
        axes[0].plot(d["n"], d["H1_margin"], "s--", color=C["dense"],
                     label=r"$H_1$")
        axes[0].plot(d["n"], d["H2_margin"], "D:",  color=C["affdiag"],
                     label=r"$H_2$")
        axes[0].axhline(0, color=C["bound"], linewidth=0.8, linestyle=":", zorder=0)
        axes[0].set_yscale("log")
        axes[0].set_xlabel(r"Dimension $n$")
        axes[0].set_ylabel("Avg margin")
        axes[0].set_title(r"(a) Diagonal $T$: Props 1 & 2")
        axes[0].legend(fontsize=7)

    if rn:
        rn = sorted(rn, key=lambda r: r["n"])
        d = _to_arrays(rn, ["n", "Pivots", "Hops", "Total_us", "ObjDiff"])
        ax = axes[1]
        ax.plot(d["n"], d["Total_us"] / 1000.0, "o-", color=C["diag"],
                label="Total (ms)")
        ax.set_yscale("log")
        ax.set_xlabel(r"Dimension $n$")
        ax.set_ylabel("Time (ms)")
        ax.set_title(r"(b) Dense $T$: BFS Phase I")
        ax_r = ax.twinx()
        ax_r.plot(d["n"], d["Pivots"], "s--", color=C["dense"],
                  label="Ph-II pivots")
        ax_r.set_ylabel("Phase-II pivots", color=C["dense"])
        ax_r.tick_params(axis="y", labelcolor=C["dense"])
        ax_r.spines["right"].set_visible(True)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_r.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

    fig.tight_layout()
    _save(fig, outdir, "fig07_affine_image")


# ===========================================================================
#  Fig 8 — Timing Breakdown
# ===========================================================================

def fig08_timing(data, outdir):
    td = data["timing_diag"]
    tq = data["timing_dense"]
    if not td and not tq:
        _skip("fig08_timing_breakdown"); return

    fig, axes = plt.subplots(1, 2, figsize=(W2, _fig_height()))

    for ax, rows, title in [
        (axes[0], td, r"(a) Diagonal $Q$"),
        (axes[1], tq, r"(b) Dense $Q$"),
    ]:
        if not rows:
            ax.set_visible(False)
            continue
        rows = sorted(rows, key=lambda r: r["n"])
        d = _to_arrays(rows, ["n", "Ph1_us", "Ph2_us", "Ph3_us", "Total_us"])
        n_arr = d["n"]
        x = np.arange(len(n_arr))
        w = 0.65
        ax.bar(x, d["Ph1_us"], width=w, color=C["ph1"],
               label="Phase I", edgecolor="none")
        ax.bar(x, d["Ph2_us"], width=w, bottom=d["Ph1_us"],
               color=C["ph2"], label="Phase II", edgecolor="none")
        ax.bar(x, d["Ph3_us"], width=w,
               bottom=d["Ph1_us"] + d["Ph2_us"],
               color=C["ph3"], label="Phase III", edgecolor="none")
        ax.plot(x, d["Total_us"], "k-", marker=".", markersize=4,
                linewidth=1.0, label="Total", zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(n_arr.astype(int), rotation=45, ha="right")
        ax.set_xlabel(r"Dimension $n$")
        ax.set_ylabel(r"Time ($\mu$s)")
        ax.set_title(title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=7.5)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, outdir, "fig08_timing_breakdown")


# ===========================================================================
#  Fig 9 — Performance: GeoQP vs qpOASES
# ===========================================================================

def fig09_performance(data, outdir):
    pd_ = data["perf_diag"]
    pq  = data["perf_dense"]
    pa  = data["perf_affine"]
    pi  = data["perf_illcond"]

    if not pd_ and not pq:
        _skip("fig09_performance"); return

    fig, axes = plt.subplots(1, 2, figsize=(W2, _fig_height()))

    # (a) Absolute times
    for rows, lbl, col, mk, ls in [
        (pd_, r"GeoQP (diag $Q$)",     C["diag"],    "o", "-"),
        (pq,  r"GeoQP (dense $Q$)",    C["dense"],   "s", "-"),
        (pd_, r"qpOASES (diag $Q$)",   C["qpoases"], "o", "--"),
        (pq,  r"qpOASES (dense $Q$)",  C["qpoases"], "s", "--"),
    ]:
        if not rows:
            continue
        rows_s = sorted(rows, key=lambda r: r["n"])
        d = _to_arrays(rows_s, ["n", "Geo_us", "qpO_us"])
        vals = d["Geo_us"] if "GeoQP" in lbl else d["qpO_us"]
        axes[0].plot(d["n"], vals, marker=mk, color=col,
                     linestyle=ls, label=lbl, markersize=4)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Dimension $n$")
    axes[0].set_ylabel(r"Solve time ($\mu$s)")
    axes[0].set_title("(a) Absolute timing")
    axes[0].legend(fontsize=6, ncol=2)

    # (b) Speed ratio
    for rows, lbl, col, mk, ls in [
        (pd_, r"Diag $Q$",                        C["diag"],     "o", "-"),
        (pq,  r"Dense $Q$",                       C["dense"],    "s", "-"),
        (pa,  r"Affine (diag $T$)",                C["affdiag"],  "D", "--"),
        (pi,  r"Ill-cond ($\kappa\!=\!10^4$)",     C["affdense"], "^", ":"),
    ]:
        if not rows:
            continue
        rows_s = sorted(rows, key=lambda r: r["n"])
        d = _to_arrays(rows_s, ["n", "Ratio"])
        axes[1].plot(d["n"], d["Ratio"], marker=mk, color=col,
                     linestyle=ls, label=lbl, markersize=4)

    axes[1].axhline(1.0, color=C["bound"], linewidth=1.0, linestyle="-",
                    alpha=0.5, zorder=0)
    axes[1].fill_between([1, 1000], 0, 1, alpha=0.04, color="green", zorder=0)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Dimension $n$")
    axes[1].set_ylabel("GeoQP / qpOASES  time ratio")
    axes[1].set_title(r"(b) Speed ratio ($<\!1$: GeoQP faster)")
    axes[1].legend(fontsize=6.5)
    axes[1].set_ylim(bottom=0)

    fig.tight_layout()
    _save(fig, outdir, "fig09_performance")


# ===========================================================================
#  Fig 10 — Hotstart
# ===========================================================================

def fig10_hotstart(data, outdir):
    rows = data["hotstart"]
    if not rows:
        _skip("fig10_hotstart"); return

    fig, axes = plt.subplots(1, 2, figsize=(W2, _fig_height()))

    for qtype, mk, col, lbl in [
        ("diagonal", "o", C["diag"],  r"Diagonal $Q$"),
        ("dense",    "s", C["dense"], r"Dense $Q$"),
    ]:
        sub = sorted(_filter(rows, "Qtype", qtype), key=lambda r: r["n"])
        if not sub:
            continue
        d = _to_arrays(sub, ["n", "Geo_us", "qpO_us", "Ratio"])
        axes[0].plot(d["n"], d["Geo_us"], marker=mk, color=col,
                     linestyle="-",  label=rf"GeoQP ({lbl})")
        axes[0].plot(d["n"], d["qpO_us"], marker=mk, color=col,
                     linestyle="--", label=rf"qpOASES ({lbl})")
        axes[1].plot(d["n"], d["Ratio"], marker=mk, color=col, label=lbl)

    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Dimension $n$")
    axes[0].set_ylabel(r"Solve time ($\mu$s)")
    axes[0].set_title("(a) Hotstart timing")
    axes[0].legend(fontsize=6, ncol=2)

    axes[1].axhline(1.0, color=C["bound"], linewidth=1.0, linestyle="-",
                    alpha=0.5, zorder=0)
    axes[1].fill_between([1, 200], 0, 1, alpha=0.04, color="green", zorder=0)
    axes[1].set_xlabel(r"Dimension $n$")
    axes[1].set_ylabel("GeoQP / qpOASES")
    axes[1].set_title(r"(b) Speed ratio ($<\!1$: GeoQP faster)")
    axes[1].legend(fontsize=7)
    axes[1].set_ylim(bottom=0)

    fig.tight_layout()
    _save(fig, outdir, "fig10_hotstart")


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
    fig01_lemma(data, outdir)
    fig02_prop1(data, outdir)
    fig03_prop2(data, outdir)
    fig04_hops_vs_n(data, outdir)
    fig05_separation(data, outdir)
    fig06_conditioning(data, outdir)
    fig07_affine(data, outdir)
    fig08_timing(data, outdir)
    fig09_performance(data, outdir)
    fig10_hotstart(data, outdir)

    print(f"\nDone. 10 figures (PDF + PNG) saved to: {outdir}")


if __name__ == "__main__":
    main()
