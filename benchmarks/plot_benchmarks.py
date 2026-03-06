"""
plot_benchmarks.py — figures for GeoQP benchmark results.

Generates 12 figures (PDF + PNG) covering every section of benchmark_results.txt:

  fig01_prop5_fastpath.pdf      Proposition 5 — O(n) box initialization
  fig02_lemma2_qdist.pdf        Lemma 2 — farthest-good-vertex quality
  fig03_prop1_alignment.pdf     Proposition 1 — exit-face alignment
  fig04_thm1_hops.pdf           Theorem 1 — Phase-II hop counts
  fig05_cor1_kappa.pdf          Corollary 1 — kappa-well-conditioned bound
  fig06_kkt_quality.pdf         KKT / solution quality vs qpOASES
  fig07_performance.pdf         Wall-clock comparison vs OSQP / qpOASES
  fig08_timing_breakdown.pdf    Phase-level timing breakdown
  fig09_hotstart.pdf            Hotstart benchmark
  fig10_conditioning.pdf        Conditioning stress test
  fig11_speedup_summary.pdf     Speed-ratio summary (all problem families)
  fig12_ph2_fraction.pdf        Phase-II cost share vs problem size

Usage:
    python plot_benchmarks.py benchmark_results.txt [output_dir]
"""

import re
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
#  Journal RC defaults  (IEEE / Springer single- or double-column)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.7",
    "lines.linewidth":    1.4,
    "lines.markersize":   5,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linewidth":     0.5,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

C = {
    "geo":     "#1f78b4",
    "osqp":    "#e31a1c",
    "qpoases": "#33a02c",
    "diag":    "#1f78b4",
    "dense":   "#ff7f00",
    "box":     "#1f78b4",
    "poly":    "#e31a1c",
    "mpc":     "#33a02c",
    "dmpc":    "#984ea3",
    "setup":   "#984ea3",
    "ph1":     "#4daf4a",
    "ph2":     "#377eb8",
    "ph3":     "#e41a1c",
}

W1 = 3.5   # single-column width (inches)
W2 = 7.1   # double-column width (inches)

_FILEPATH = None   # set in main()


# ---------------------------------------------------------------------------
#  Parsing utilities
# ---------------------------------------------------------------------------

def _is_sep(line):
    s = line.strip()
    return s.startswith("===") or s.startswith("---") or s == ""


def _is_sub(line):
    return line.strip().startswith(">>>")


def _numeric(tok):
    try:
        float(tok.rstrip("%x"))
        return True
    except ValueError:
        return False


def _clean(tok):
    return tok.rstrip("%x")


def _extract_block(all_lines, start_re, stop_re=None, max_lines=300):
    """
    Collect lines starting immediately after the line matching start_re.

    Key feature: if the very first line encountered after the start match
    is a '===' banner (the closing banner of a section title), it is skipped
    rather than triggering stop_re.  This lets callers use stop_re=r'^={60,}'
    to stop at the NEXT section boundary without being fooled by the one
    immediately following the matched title line.
    """
    out = []
    found = False
    banner_skipped = not True   # becomes True once we've handled the post-title ===
    for line in all_lines:
        if not found:
            if re.search(start_re, line):
                found = True
                banner_skipped = False
            continue

        # Single-shot: skip a === banner that immediately follows the match
        if not banner_skipped:
            banner_skipped = True
            if line.strip().startswith("==="):
                continue   # drop the closing === of the section header
            # otherwise fall through and process this line normally

        if stop_re and re.search(stop_re, line):
            break
        if len(out) >= max_lines:
            break
        out.append(line)
    return out


def _rows(block, extractor):
    """Apply extractor(parts_list) to every data line in block."""
    rows = []
    for line in block:
        if _is_sep(line) or _is_sub(line):
            continue
        p = line.split()
        if not p or not _numeric(p[0]):
            continue
        try:
            r = extractor(p)
            if r is not None:
                rows.append(r)
        except (ValueError, IndexError):
            pass
    return rows


# ---------------------------------------------------------------------------
#  parse_all
# ---------------------------------------------------------------------------

def parse_all(filepath):
    with open(filepath) as fh:
        raw = fh.readlines()

    def blk(start, stop=r"^={60,}", mx=300):
        return _extract_block(raw, start, stop, mx)

    # ---- Proposition 5 ----
    b = blk(r"PROPOSITION 5: Box Polytope")
    # tokens: n Qtype Verts bound% ObjDiff Ph2hops Time_ms ...
    df5 = pd.DataFrame(_rows(b, lambda p: dict(
        n=int(p[0]), Qtype=p[1],
        Verts=float(_clean(p[2])),
        ObjDiff=float(_clean(p[4])),
        Ph2hops=float(_clean(p[5])),
        Time_ms=float(_clean(p[6])),
    ) if len(p) >= 7 else None))

    # ---- Lemma 2 ----
    b = blk(r"LEMMA 2: Farthest Good Vertex")
    # tokens: n Type QDistRatio AvgObjDiff NSolved Result
    dflem2 = pd.DataFrame(_rows(b, lambda p: dict(
        n=int(p[0]),
        QDistRatio=float(p[2]),
        AvgObjDiff=float(_clean(p[3])),
        NSolved=int(p[4]),
    ) if len(p) >= 5 else None))

    # ---- Proposition 1 — box ----
    b = blk(r"Box polytopes \(varying n", r"General polytopes")
    # tokens: n Type L D eps min_th th_ok Result
    dfp1b = pd.DataFrame(_rows(b, lambda p: dict(
        n=int(p[0]),
        L=float(p[2]), D=float(p[3]),
        eps=float(p[4]), min_th=float(p[5]),
    ) if len(p) >= 6 else None))

    # ---- Proposition 1 — general polytope ----
    b = blk(r"General polytopes \(m=2n\)", r"Proposition 1 overall")
    dfp1p = pd.DataFrame(_rows(b, lambda p: dict(
        n=int(p[0]),
        L=float(p[2]), D=float(p[3]),
        eps=float(p[4]), min_th=float(p[5]),
    ) if len(p) >= 6 else None))

    # ---- Theorem 1 (a): hop bound ----
    b = blk(r"Phase II hops <= adjacency bound", r"L sweep")
    # tokens: n Qtype AvgHops AdjBound(text) MaxHops Bounded(text) Solved(text)
    dft1a = pd.DataFrame(_rows(b, lambda p: dict(
        n=int(p[0]), Qtype=p[1],
        AvgHops=float(p[2]),
        MaxHops=float(p[4]),
    ) if len(p) >= 5 else None))

    # ---- Theorem 1 (b): L sweep ----
    b = blk(r"L sweep: hops vs separation", r"General polytopes -- Phase II")
    # tokens: scale_g avg_L AvgHops MaxHops avg_wP L>=w? Solved
    dft1b = pd.DataFrame(_rows(b, lambda p: dict(
        scale_g=float(p[0]),
        avg_L=float(p[1]),
        AvgHops=float(p[2]),
        MaxHops=float(p[3]),
        avg_wP=float(p[4]),
        L_ge_w=p[5],
    ) if len(p) >= 6 else None))

    # ---- Theorem 1 (c): general polytopes ----
    b = blk(r"General polytopes -- Phase II hop count", r"Theorem 1 overall")
    # tokens: n m AvgHops MaxHops AvgActv MaxOD Solved(text) NumErr
    dft1c = pd.DataFrame(_rows(b, lambda p: dict(
        n=int(p[0]), m=int(p[1]),
        AvgHops=float(p[2]), MaxHops=float(p[3]),
        AvgActv=float(p[4]), MaxOD=float(_clean(p[5])),
    ) if len(p) >= 6 else None))

    # ---- Corollary 1 ----
    b = blk(r"COROLLARY 1: Well-Conditioned")
    # tokens: n kappa avg_L L_kappa min_th neg_kL th_ok(text) AvgHops Result
    dfc1 = pd.DataFrame(_rows(b, lambda p: dict(
        n=int(p[0]), kappa=float(p[1]),
        avg_L=float(p[2]), L_kappa=float(p[3]),
        min_th=float(p[4]), neg_kL=float(p[5]),
        AvgHops=float(p[7]),
    ) if len(p) >= 8 else None))

    # ---- KKT ----
    b = blk(r"KKT / SOLUTION QUALITY")
    # tokens: n Type MaxPFeas MaxObjDiff NSolved Result
    dfkkt = pd.DataFrame(_rows(b, lambda p: dict(
        n=int(p[0]), Type=p[1],
        MaxPFeas=float(_clean(p[2])),
        MaxObjDiff=float(_clean(p[3])),
    ) if len(p) >= 4 else None))

    # ---- Performance — Box ----
    b = blk(r"^>>> Box QP\s*$", r">>> Polytope QP")
    # tokens: n Type Geo OSQP qpO Geo/OSQP Geo/qpO Hops ObjDiff Result
    dfperfb = pd.DataFrame(_rows(b, lambda p: dict(
        n=int(p[0]),
        Geo_ms=float(p[2]), OSQP_ms=float(p[3]), qpO_ms=float(p[4]),
        Geo_OSQP=float(_clean(p[5])), Geo_qpO=float(_clean(p[6])),
        GeoHops=float(p[7]),
    ) if len(p) >= 8 else None))

    # ---- Performance — Polytope ----
    b = blk(r"Polytope QP \(m=2n\)", r"^={60,}")
    dfperfp = pd.DataFrame(_rows(b, lambda p: dict(
        n=int(p[0]),
        Geo_ms=float(p[2]), OSQP_ms=float(p[3]), qpO_ms=float(p[4]),
        Geo_OSQP=float(_clean(p[5])), Geo_qpO=float(_clean(p[6])),
        GeoHops=float(p[7]),
    ) if len(p) >= 8 else None))

    # ---- Timing breakdown (per subsection) ----
    # tokens: n Type Total Setup Ph1 Ph2 Ph3 S% Ph1% Ph2% Ph3% Hops us/hop
    def _tim_extractor(p):
        if len(p) < 13:
            return None
        return dict(
            n=int(p[0]),
            Total_us=float(p[2]), Setup_us=float(p[3]),
            Ph1_us=float(p[4]),   Ph2_us=float(p[5]),
            Ph3_us=float(p[6]),   Hops=float(p[11]),
        )

    dftim = {}
    tim_subs = [
        ("diag_box",  r"Box QP \(diagonal Q\)",  r"Box QP \(dense Q\)"),
        ("dense_box", r"Box QP \(dense Q\)",      r"General Polytope QP"),
        ("polytope",  r"General Polytope QP",     r"MPC QP \(dense Q\)"),
        ("mpc_dense", r"MPC QP \(dense Q\)",      r"MPC QP \(diagonal Q\)"),
        ("mpc_diag",  r"MPC QP \(diagonal Q\)",   r"^={60,}"),
    ]
    for key, start, stop in tim_subs:
        b = _extract_block(raw, start, stop, 20)
        dftim[key] = pd.DataFrame(_rows(b, _tim_extractor))

    # ---- Hotstart ----
    b = blk(r"HOTSTART BENCHMARK")
    # tokens: n Geo OSQP qpO Geo/OSQP Geo/qpO AvgHops ObjDiff Result
    dfhot = pd.DataFrame(_rows(b, lambda p: dict(
        n=int(p[0]),
        Geo_ms=float(p[1]), OSQP_ms=float(p[2]), qpO_ms=float(p[3]),
        Geo_OSQP=float(_clean(p[4])), Geo_qpO=float(_clean(p[5])),
        AvgHops=float(p[6]),
    ) if len(p) >= 7 else None))

    # ---- Conditioning ----
    b = blk(r"CONDITIONING STRESS TEST")
    # tokens: n cond AvgHops MaxHops ObjDiff Time_ms Solved
    dfcond = pd.DataFrame(_rows(b, lambda p: dict(
        n=int(p[0]), cond=float(p[1]),
        AvgHops=float(p[2]), MaxHops=float(p[3]),
        ObjDiff=float(_clean(p[4])), Time_ms=float(p[5]),
    ) if len(p) >= 6 else None))

    return dict(
        prop5=df5, lemma2=dflem2,
        prop1_box=dfp1b, prop1_poly=dfp1p,
        thm1a=dft1a, thm1b=dft1b, thm1c=dft1c,
        cor1=dfc1, kkt=dfkkt,
        perf_box=dfperfb, perf_poly=dfperfp,
        tim=dftim, hotstart=dfhot, conditioning=dfcond,
    )


# ---------------------------------------------------------------------------
#  Save helper
# ---------------------------------------------------------------------------

def _save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    print(f"  [{name}]")
    plt.close(fig)


def _skip(name, reason="empty"):
    print(f"  [skip {name}] {reason}")


# ===========================================================================
#  Fig 1 — Proposition 5: O(n) box fast-path  (2 panels)
# ===========================================================================

def fig1_prop5(d, outdir):
    df = d["prop5"]
    if df.empty:
        _skip("fig01"); return

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))

    for qtype, mk, col, lbl in [
        ("diag",  "o", C["diag"],  "Diagonal $Q$"),
        ("dense", "s", C["dense"], "Dense $Q$"),
    ]:
        sub = df[df["Qtype"] == qtype].sort_values("n")
        if sub.empty:
            continue
        axes[0].plot(sub["n"], sub["Time_ms"],   marker=mk, color=col, label=lbl)
        axes[1].plot(sub["n"], sub["Ph2hops"],   marker=mk, color=col, label=lbl)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Dimension $n$")
    axes[0].set_ylabel("Solve time (ms)")
    axes[0].set_title("(a) Wall-clock time")
    axes[0].legend()

    axes[1].set_xlabel("Dimension $n$")
    axes[1].set_ylabel("Phase-II hops")
    axes[1].set_title("(b) Active-set iterations")
    axes[1].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("Proposition 5 — Box Polytope $O(n)$ Initialization", y=1.02)
    fig.tight_layout()
    _save(fig, outdir, "fig01_prop5_fastpath")


# ===========================================================================
#  Fig 2 — Lemma 2: Q-distance ratio  (1 panel)
# ===========================================================================

def fig2_lemma2(d, outdir):
    df = d["lemma2"].dropna(subset=["n", "QDistRatio"])
    if df.empty:
        _skip("fig02"); return

    fig, ax = plt.subplots(figsize=(W1, 2.4))
    ax.bar(df["n"].astype(int), df["QDistRatio"],
           color=C["geo"], width=1.0, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--",
               label="Optimal (ratio $= 1$)", zorder=4)
    ax.set_ylim(0.0, 1.12)
    ax.set_xlabel("Dimension $n$")
    ax.set_ylabel(r"$\|v^{*} - \bar{x}\|_Q^2 \;/\; \max_{v} \|v - \bar{x}\|_Q^2$",
                  fontsize=7.5)
    ax.set_title("Lemma 2 — Farthest Good Vertex Quality")
    ax.legend()
    fig.tight_layout()
    _save(fig, outdir, "fig02_lemma2_qdist")


# ===========================================================================
#  Fig 3 — Proposition 1: exit-face alignment  (2 panels)
# ===========================================================================

def fig3_prop1(d, outdir):
    dfb = d["prop1_box"].dropna(subset=["eps", "min_th", "n"])
    dfp = d["prop1_poly"].dropna(subset=["eps", "min_th", "n"])
    if dfb.empty and dfp.empty:
        _skip("fig03"); return

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.7))

    def _panel(ax, df, title):
        if df.empty:
            return
        neg_eps = -df["eps"].values
        min_th  =  df["min_th"].values
        sc = ax.scatter(neg_eps, min_th, c=df["n"].values,
                        cmap="viridis", s=35, zorder=3)
        lo = float(min(neg_eps.min(), min_th.min())) - 0.05
        hi = float(max(neg_eps.max(), min_th.max())) + 0.05
        rng = [lo, hi]
        ax.plot(rng, rng, "k--", linewidth=1.0,
                label=r"$\theta_i = -\epsilon$ (bound)")
        ax.fill_between(rng, rng, [hi, hi], alpha=0.08, color="green",
                        label="Feasible region")
        ax.set_xlim(rng); ax.set_ylim(rng)
        ax.set_xlabel(r"$-\epsilon = -D/L$")
        ax.set_ylabel(r"$\min_i \theta_i$  (measured)")
        ax.set_title(title)
        ax.legend(fontsize=7)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("$n$", rotation=0, labelpad=6)

    _panel(axes[0], dfb, "(a) Box polytopes")
    _panel(axes[1], dfp, "(b) General polytopes ($m=2n$)")

    fig.suptitle(
        r"Proposition 1 — Exit-Face Alignment:  $\theta_i > -\epsilon$ "
        r"when  $L \geq D/\epsilon$",
        y=1.02)
    fig.tight_layout()
    _save(fig, outdir, "fig03_prop1_alignment")


# ===========================================================================
#  Fig 4 — Theorem 1: Phase-II hop counts  (3 panels)
# ===========================================================================

def fig4_thm1(d, outdir):
    dfa  = d["thm1a"]
    dft1b = d["thm1b"]
    dfc  = d["thm1c"]

    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.7))

    # (a) box hops vs n
    if not dfa.empty:
        for qtype, mk, col, lbl in [
            ("diag",  "o", C["diag"],  "Diagonal $Q$"),
            ("dense", "s", C["dense"], "Dense $Q$"),
        ]:
            sub = dfa[dfa["Qtype"] == qtype].sort_values("n")
            if sub.empty:
                continue
            axes[0].plot(sub["n"], sub["AvgHops"],
                         marker=mk, color=col, label=lbl)
            axes[0].fill_between(sub["n"], sub["AvgHops"], sub["MaxHops"],
                                 alpha=0.15, color=col)
    axes[0].set_xlabel("Dimension $n$")
    axes[0].set_ylabel("Phase-II hops")
    axes[0].set_title("(a) Box polytopes (avg $\\pm$ max)")
    axes[0].legend(fontsize=7.5)

    # (b) L sweep
    if not dft1b.empty:
        ax = axes[1]
        ax.plot(dft1b["scale_g"], dft1b["AvgHops"],
                marker="o", color=C["geo"], label="Avg hops")
        ax.fill_between(dft1b["scale_g"], dft1b["AvgHops"], dft1b["MaxHops"],
                        alpha=0.2, color=C["geo"], label="Max hops")
        enough = dft1b[dft1b["L_ge_w"] == "yes"]
        if not enough.empty:
            ax.axvline(enough["scale_g"].min(), color="gray", linestyle=":",
                       linewidth=1.3,
                       label=r"$L \geq w_{\mathcal{P}}(\nu)$")
        ax.set_xscale("log")
        ax.set_xlabel(r"Scale $\alpha$ (controls $L$)")
        ax.set_ylabel("Phase-II hops")
        ax.set_title(r"(b) $L$ sweep, $n=8$ box")
        ax.legend(fontsize=7)

    # (c) general polytopes
    if not dfc.empty:
        sub = dfc.dropna(subset=["n", "AvgHops", "MaxHops"])
        axes[2].bar(sub["n"].astype(int), sub["AvgHops"],
                    color=C["poly"], width=4, edgecolor="white",
                    linewidth=0.5, label="Avg hops", zorder=3)
        axes[2].errorbar(sub["n"], sub["AvgHops"],
                         yerr=(sub["MaxHops"] - sub["AvgHops"]),
                         fmt="none", ecolor="black",
                         capsize=3, elinewidth=1.0, label="Max hops")
        axes[2].set_xlabel("Dimension $n$")
        axes[2].set_ylabel("Phase-II hops")
        axes[2].set_title("(c) General polytopes")
        axes[2].legend(fontsize=7)

    fig.suptitle("Theorem 1 — Phase-II Hop Counts", y=1.02)
    fig.tight_layout()
    _save(fig, outdir, "fig04_thm1_hops")


# ===========================================================================
#  Fig 5 — Corollary 1: kappa-well-conditioned bound  (2 panels)
# ===========================================================================

def fig5_cor1(d, outdir):
    df = d["cor1"].dropna(subset=["n", "kappa", "min_th", "neg_kL", "AvgHops"])
    if df.empty:
        _skip("fig05"); return

    ns = sorted(df["n"].unique().astype(int))
    cmap_n = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.7))

    # (a) scatter: min_theta vs -kappa/L
    sc = axes[0].scatter(df["neg_kL"], df["min_th"],
                          c=df["n"], cmap="plasma", s=25, zorder=3)
    lo = float(min(df["neg_kL"].min(), df["min_th"].min())) - 0.02
    hi = float(max(df["neg_kL"].max(), df["min_th"].max())) + 0.02
    rng = [lo, hi]
    axes[0].plot(rng, rng, "k--", linewidth=1.0,
                 label=r"$\theta_i = -\kappa/L$ (bound)")
    axes[0].fill_between(rng, rng, [hi, hi], alpha=0.08, color="green",
                          label="Feasible region")
    axes[0].set_xlim(rng); axes[0].set_ylim(rng)
    axes[0].set_xlabel(r"$-\kappa/L$  (theoretical lower bound)")
    axes[0].set_ylabel(r"$\min_i \theta_i$  (measured)")
    axes[0].set_title("(a) Bound tightness")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.colorbar(sc, ax=axes[0], label="$n$", pad=0.02)

    # (b) avg hops vs kappa
    for i, n in enumerate(ns):
        sub = df[df["n"] == n].sort_values("kappa")
        axes[1].plot(sub["kappa"], sub["AvgHops"],
                     marker="o", color=cmap_n(i), label=f"$n={n}$")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\kappa$  (polytope width bound)")
    axes[1].set_ylabel("Avg Phase-II hops")
    axes[1].set_title(r"(b) Hops vs $\kappa$")
    axes[1].legend(fontsize=7)

    fig.suptitle(
        r"Corollary 1 — $\kappa$-Well-Conditioned Bound:  "
        r"$\theta_i > -\kappa/L$  when  $L \geq \kappa/\epsilon$",
        y=1.02)
    fig.tight_layout()
    _save(fig, outdir, "fig05_cor1_kappa")


# ===========================================================================
#  Fig 6 — KKT / solution quality  (2 panels)
# ===========================================================================

def fig6_kkt(d, outdir):
    df = d["kkt"]
    if df is None or df.empty:
        _skip("fig06"); return

    type_style = {
        "diag-box":  ("Diag box",    C["diag"],  "o"),
        "dense-box": ("Dense box",   C["dense"], "s"),
        "polytope":  ("Polytope",    C["poly"],  "D"),
        "MPC-h=5":   ("MPC $h=5$",  C["mpc"],   "^"),
        "MPC-h=10":  ("MPC $h=10$", C["mpc"],   "v"),
        "MPC-h=20":  ("MPC $h=20$", C["mpc"],   "P"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.7))
    for col, ylabel, title, ax in [
        ("MaxObjDiff",
         r"$|f^{*}_{\mathrm{GeoQP}} - f^{*}_{\mathrm{ref}}|$",
         "(a) Objective gap",
         axes[0]),
        ("MaxPFeas",
         "Max primal infeasibility",
         "(b) Primal feasibility",
         axes[1]),
    ]:
        for t in df["Type"].unique():
            sub = df[df["Type"] == t].sort_values("n")
            lbl, col_c, mk = type_style.get(t, (t, "gray", "o"))
            ax.plot(sub["n"], sub[col], marker=mk, color=col_c,
                    label=lbl, linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("Dimension $n$")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title)
        ax.legend(fontsize=6.5, ncol=2)

    fig.suptitle("KKT Solution Quality  (vs. qpOASES reference)", y=1.02)
    fig.tight_layout()
    _save(fig, outdir, "fig06_kkt_quality")


# ===========================================================================
#  Fig 7 — Wall-clock performance comparison  (2 panels)
# ===========================================================================

def fig7_performance(d, outdir):
    dfb = d["perf_box"].dropna(subset=["n", "Geo_ms", "OSQP_ms"])
    dfp = d["perf_poly"].dropna(subset=["n", "Geo_ms", "OSQP_ms"])

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.7))

    def _panel(ax, df, title):
        if df.empty:
            return
        df = df.sort_values("n")
        ax.plot(df["n"], df["Geo_ms"],
                marker="o", color=C["geo"],     label="GeoQP")
        ax.plot(df["n"], df["OSQP_ms"],
                marker="s", color=C["osqp"],    label="OSQP")
        dfq = df[df["qpO_ms"] > 0]
        if not dfq.empty:
            ax.plot(dfq["n"], dfq["qpO_ms"],
                    marker="^", color=C["qpoases"], label="qpOASES")
        ax.set_yscale("log")
        ax.set_xlabel("Dimension $n$")
        ax.set_ylabel("Solve time (ms)")
        ax.set_title(title)
        ax.legend(fontsize=7.5)

    _panel(axes[0], dfb, "(a) Box QP")
    _panel(axes[1], dfp, "(b) General polytope  ($m = 2n$)")

    fig.suptitle("Performance Comparison:  GeoQP vs. OSQP vs. qpOASES", y=1.02)
    fig.tight_layout()
    _save(fig, outdir, "fig07_performance")


# ===========================================================================
#  Fig 8 — Phase-level timing breakdown (stacked bars)
# ===========================================================================

def fig8_timing(d, outdir):
    tim = d["tim"]
    keys   = ["diag_box", "dense_box", "polytope", "mpc_dense", "mpc_diag"]
    titles = [
        "Box\n(diag $Q$)",
        "Box\n(dense $Q$)",
        "Polytope\n($m=2n$)",
        "MPC\n(dense $Q$)",
        "MPC\n(diag $Q$)",
    ]
    ph_cols   = ["Setup_us", "Ph1_us", "Ph2_us", "Ph3_us"]
    ph_labels = ["Setup", "Phase I", "Phase II", "Phase III"]
    ph_colors = [C["setup"], C["ph1"], C["ph2"], C["ph3"]]

    fig, axes = plt.subplots(1, 5, figsize=(W2, 2.7))

    for ax, key, title in zip(axes, keys, titles):
        df = tim.get(key, pd.DataFrame())
        if df is None or df.empty:
            ax.set_visible(False)
            continue
        df = df.dropna(subset=["n", "Total_us"]).sort_values("n")
        x_pos = np.arange(len(df))
        bottoms = np.zeros(len(df))
        for col, lbl, col_c in zip(ph_cols, ph_labels, ph_colors):
            if col not in df.columns:
                continue
            vals = df[col].fillna(0).values
            ax.bar(x_pos, vals, bottom=bottoms, label=lbl,
                   color=col_c, edgecolor="none", width=0.7)
            bottoms += vals
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df["n"].astype(int).tolist(),
                           rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("$n$", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel(r"Time ($\mu$s)", fontsize=8)
        ax.set_title(title, fontsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ph_colors]
    fig.legend(handles, ph_labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=4,
               framealpha=0.9, fontsize=8)
    fig.suptitle("GeoQP Phase-Level Timing Breakdown", y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, outdir, "fig08_timing_breakdown")


# ===========================================================================
#  Fig 9 — Hotstart benchmark  (2 panels)
# ===========================================================================

def fig9_hotstart(d, outdir):
    df = d["hotstart"].dropna(subset=["n", "Geo_ms", "OSQP_ms", "qpO_ms"])
    if df.empty:
        _skip("fig09"); return
    df = df.sort_values("n")

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))

    axes[0].plot(df["n"], df["Geo_ms"],  marker="o",
                 color=C["geo"],     label="GeoQP")
    axes[0].plot(df["n"], df["OSQP_ms"], marker="s",
                 color=C["osqp"],    label="OSQP")
    axes[0].plot(df["n"], df["qpO_ms"],  marker="^",
                 color=C["qpoases"], label="qpOASES")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Dimension $n$")
    axes[0].set_ylabel("Solve time (ms)")
    axes[0].set_title("(a) Timing")
    axes[0].legend()

    if "Geo_OSQP" in df.columns:
        axes[1].plot(df["n"], df["Geo_OSQP"],
                     marker="s", color=C["osqp"],    label="GeoQP / OSQP")
        axes[1].plot(df["n"], df["Geo_qpO"],
                     marker="^", color=C["qpoases"], label="GeoQP / qpOASES")
        axes[1].axhline(1.0, color="gray", linestyle="--",
                        linewidth=1.0, label="Break-even")
        axes[1].set_xlabel("Dimension $n$")
        axes[1].set_ylabel("Speed ratio  ($< 1$: GeoQP is faster)")
        axes[1].set_title("(b) Speed ratio")
        axes[1].legend()

    fig.suptitle(r"Hotstart:  Fixed $Q, A$,  Varying $g$", y=1.02)
    fig.tight_layout()
    _save(fig, outdir, "fig09_hotstart")


# ===========================================================================
#  Fig 10 — Conditioning stress test  (2 panels)
# ===========================================================================

def fig10_conditioning(d, outdir):
    df = d["conditioning"].dropna(
        subset=["n", "cond", "AvgHops", "MaxHops", "Time_ms"])
    if df.empty:
        _skip("fig10"); return

    ns = sorted(df["n"].unique().astype(int))
    cmap_n = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6))
    for i, n in enumerate(ns):
        sub = df[df["n"] == n].sort_values("cond")
        col = cmap_n(i)
        lbl = f"$n={n}$"
        axes[0].plot(sub["cond"], sub["AvgHops"],
                     marker="o", color=col, label=lbl)
        axes[0].fill_between(sub["cond"], sub["AvgHops"], sub["MaxHops"],
                              alpha=0.10, color=col)
        axes[1].plot(sub["cond"], sub["Time_ms"],
                     marker="o", color=col, label=lbl)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"Condition number $\kappa(Q)$")
    axes[0].set_ylabel("Phase-II hops  (avg, shaded max)")
    axes[0].set_title(r"(a) Iterations vs. $\kappa(Q)$")
    axes[0].legend(fontsize=7)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Solve time (ms)")
    axes[1].set_title(r"(b) Runtime vs. $\kappa(Q)$")
    axes[1].legend(fontsize=7)

    fig.suptitle(r"Conditioning Stress Test:  Varying $\kappa(Q)$", y=1.02)
    fig.tight_layout()
    _save(fig, outdir, "fig10_conditioning")


# ===========================================================================
#  Fig 11 — Speed-ratio summary across all problem families  (2 panels)
# ===========================================================================

def fig11_speedup(d, outdir):
    dfb = d["perf_box"].dropna(subset=["n", "Geo_OSQP", "Geo_qpO"])
    dfp = d["perf_poly"].dropna(subset=["n", "Geo_OSQP", "Geo_qpO"])
    dfh = d["hotstart"].dropna(subset=["n", "Geo_OSQP", "Geo_qpO"])

    datasets = [
        (dfb, "Box QP",   C["box"],  "o", "-"),
        (dfp, "Polytope", C["poly"], "s", "--"),
        (dfh, "Hotstart", C["mpc"],  "^", ":"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))
    for ax, ratio_col, versus in [
        (axes[0], "Geo_OSQP",  "OSQP"),
        (axes[1], "Geo_qpO",   "qpOASES"),
    ]:
        for df_src, lbl, col, mk, ls in datasets:
            if df_src.empty:
                continue
            df2 = df_src.sort_values("n")
            vals = df2[ratio_col]
            if vals.abs().max() < 1e-9:
                continue
            ax.plot(df2["n"], vals, marker=mk, color=col,
                    linestyle=ls, label=lbl)
        ax.axhline(1.0, color="black", linestyle="-",
                   linewidth=0.9, label="Break-even")
        ax.set_xlabel("Dimension $n$")
        ax.set_ylabel(f"GeoQP / {versus}  time ratio")
        ax.set_title(f"vs. {versus}")
        ax.legend(fontsize=7.5)

    fig.suptitle("Speed-Ratio Summary  (ratio $< 1$: GeoQP faster)", y=1.02)
    fig.tight_layout()
    _save(fig, outdir, "fig11_speedup_summary")


# ===========================================================================
#  Fig 12 — Phase-II cost share as problem size grows  (2 panels)
# ===========================================================================

def fig12_ph2_fraction(d, outdir):
    tim = d["tim"]
    label_style = {
        "diag_box":  ("Diagonal box", C["diag"],  "o"),
        "dense_box": ("Dense box",    C["dense"], "s"),
        "polytope":  ("Polytope",     C["poly"],  "D"),
        "mpc_dense": ("MPC (dense)",  C["mpc"],   "^"),
        "mpc_diag":  ("MPC (diag)",   C["dmpc"],  "v"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))
    for key, (lbl, col, mk) in label_style.items():
        df = tim.get(key, pd.DataFrame())
        if df is None or df.empty:
            continue
        df = df.dropna(subset=["n", "Ph2_us", "Ph1_us", "Total_us"]
                       ).sort_values("n")
        tot = df["Total_us"].replace(0, np.nan)
        axes[0].plot(df["n"], df["Ph2_us"] / tot * 100,
                     marker=mk, color=col, label=lbl)
        axes[1].plot(df["n"], df["Ph1_us"] / tot * 100,
                     marker=mk, color=col, label=lbl)

    for ax, title, ylabel in [
        (axes[0], "(a) Phase-II (active-set) share",  "Phase-II fraction (\\%)"),
        (axes[1], "(b) Phase-I (init) share",          "Phase-I fraction (\\%)"),
    ]:
        ax.set_xlabel("Dimension $n$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.set_ylim(0, 105)

    fig.suptitle("GeoQP Phase Cost Share vs. Problem Size", y=1.02)
    fig.tight_layout()
    _save(fig, outdir, "fig12_ph2_fraction")


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    global _FILEPATH
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    _FILEPATH = sys.argv[1]
    outdir = (sys.argv[2] if len(sys.argv) > 2
              else os.path.dirname(os.path.abspath(_FILEPATH)))
    os.makedirs(outdir, exist_ok=True)

    print(f"Parsing  : {_FILEPATH}")
    print(f"Output   : {outdir}")
    d = parse_all(_FILEPATH)

    # Diagnostics
    for k, v in d.items():
        if isinstance(v, pd.DataFrame):
            print(f"  {k:22s}  {len(v)} rows")
        else:
            for kk, vv in v.items():
                print(f"  {k}/{kk:16s}  {len(vv)} rows")

    print("\nGenerating figures ...")
    fig1_prop5(d, outdir)
    fig2_lemma2(d, outdir)
    fig3_prop1(d, outdir)
    fig4_thm1(d, outdir)
    fig5_cor1(d, outdir)
    fig6_kkt(d, outdir)
    fig7_performance(d, outdir)
    fig8_timing(d, outdir)
    fig9_hotstart(d, outdir)
    fig10_conditioning(d, outdir)
    fig11_speedup(d, outdir)
    fig12_ph2_fraction(d, outdir)

    print(f"\nDone.  12 figures (PDF + PNG) saved to:  {outdir}")


if __name__ == "__main__":
    main()
