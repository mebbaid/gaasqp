"""
plot_benchmarks.py — random box QP benchmark figures.

Usage:  python plot_benchmarks.py <benchmark_box output> [outdir]

Reads the machine-readable PERF lines printed by benchmark_box
(one per trial and solver) and writes, as PDF + PNG:

  fig01_performance   median solve time vs n, every solver
                      (a) axis-aligned box, dense H   (b) affine image, dense T, dense H
  fig02_iterations    median iterations vs n, active-set solvers only
                      (GaasQP pivots, qpOASES / qpOASES-B working-set changes,
                      DAQP iterations); interior-point and ADMM iterations
                      measure a different kind of step and are left out
  box_summary.csv     per (family, n, solver): median / quartile time, median
                      iterations, failures, max objective difference vs qpOASES
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps  # noqa: E402

PANELS = [
    ("box/denseQ", r"(a) Axis-aligned box, dense $H$"),
    ("affine-dense/refo", r"(b) Affine image, dense $T$, dense $H$"),
]


def parse(path):
    """PERF,<label>,<n>,<trial>,<solver>,<ok>,<time_us>,<iters>,<obj_diff>,<viol>,<pivots>,<hops>"""
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("PERF,"):
                continue
            p = line.strip().split(",")
            if len(p) < 12:
                continue
            rows.append(dict(label=p[1], n=int(p[2]), trial=int(p[3]), solver=p[4],
                             ok=p[5] == "1", time_us=float(p[6]), iters=float(p[7]),
                             obj_diff=float(p[8]), viol=float(p[9])))
    return rows


def group(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["label"], r["n"], r["solver"])].append(r)
    return g


def series(g, label, solver, key):
    ns = sorted({n for (lb, n, s) in g if lb == label and s == solver})
    xs, ys = [], []
    for n in ns:
        vals = [r[key] for r in g[(label, n, solver)] if r["ok"]]
        if vals:
            xs.append(n)
            ys.append(float(np.median(vals)))
    return np.array(xs), np.array(ys)


def fig01_performance(g, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(ps.W2, 2.55), sharey=True)
    for ax, (label, title) in zip(axes, PANELS):
        for solver in ps.SOLVERS:
            x, y = series(g, label, solver, "time_us")
            if len(x):
                ps.line(ax, solver, x, y)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Dimension $n$")
        ax.set_title(title)
    axes[0].set_ylabel(r"Median solve time ($\mu$s)")
    ps.shared_legend(fig, axes, ncol=4)
    fig.tight_layout()
    save(fig, outdir, "fig01_performance")


def fig02_iterations(g, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(ps.W2, 2.55), sharey=True)
    for ax, (label, title) in zip(axes, PANELS):
        # On box QPs QProblem and QProblemB take the same working-set changes;
        # draw one line then, instead of hiding one under the other.
        xq, yq = series(g, label, "qpOASES", "iters")
        xb, yb = series(g, label, "qpOASES-B", "iters")
        same_qpo = len(xq) == len(xb) and np.array_equal(xq, xb) and np.allclose(yq, yb)
        for solver in ps.ACTIVE_SET:
            if solver == "qpOASES-B" and same_qpo:
                continue
            x, y = series(g, label, solver, "iters")
            if len(x):
                kw = {"label": "qpOASES (QProblem = QProblemB)"} if solver == "qpOASES" and same_qpo else {}
                ps.line(ax, solver, x, np.maximum(y, 1.0), **kw)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Dimension $n$")
        ax.set_title(title)
    axes[0].set_ylabel("Median iterations")
    ps.shared_legend(fig, axes, ncol=4)
    fig.tight_layout()
    save(fig, outdir, "fig02_iterations")


def write_summary(g, outdir):
    path = os.path.join(outdir, "box_summary.csv")
    labels = sorted({k[0] for k in g})
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "n", "solver", "trials", "failures", "time_us_median",
                    "time_us_q25", "time_us_q75", "iters_median", "max_obj_diff", "max_viol"])
        for label in labels:
            for n in sorted({k[1] for k in g if k[0] == label}):
                for solver in ps.SOLVERS:
                    rs = g.get((label, n, solver))
                    if not rs:
                        continue
                    ok = [r for r in rs if r["ok"]]
                    t = [r["time_us"] for r in ok]
                    od = [r["obj_diff"] for r in ok if np.isfinite(r["obj_diff"])]
                    vi = [r["viol"] for r in ok if np.isfinite(r["viol"])]
                    w.writerow([label, n, solver, len(rs), len(rs) - len(ok),
                                f"{np.median(t):.3f}" if t else "",
                                f"{np.percentile(t, 25):.3f}" if t else "",
                                f"{np.percentile(t, 75):.3f}" if t else "",
                                f"{np.median([r['iters'] for r in ok]):.1f}" if ok else "",
                                f"{max(od):.2e}" if od else "",
                                f"{max(vi):.2e}" if vi else ""])
    print(f"  {os.path.basename(path)}")


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  {name}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(path)), "figures")
    os.makedirs(outdir, exist_ok=True)
    ps.apply()
    rows = parse(path)
    if not rows:
        sys.exit(f"no PERF lines in {path}")
    g = group(rows)
    print(f"Parsing:  {path}  ({len(rows)} PERF rows)")
    print(f"Output:   {outdir}")
    fig01_performance(g, outdir)
    fig02_iterations(g, outdir)
    write_summary(g, outdir)


if __name__ == "__main__":
    main()
