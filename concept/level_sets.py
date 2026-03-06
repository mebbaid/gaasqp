"""
level_sets.py — GeoQP: Farthest-Good-Vertex Theorem Visualisation
==================================================================

Fixed ill-conditioned dense Q (thin, rotated ellipses) and a convex polytope.
The unconstrained minimiser x_u is moved progressively further away across
four panels, so we can verify:

  THEOREM (informal):  The Q-farthest GOOD vertex v* has a Newton ray that
  exits the polytope through the face containing the Q-optimal solution x*.

A vertex v is GOOD if the Newton direction d = x_u - v  (= -H^{-1} grad f(v))
points *into* the polytope interior from v, i.e. for every active constraint
i at v:   A[i] @ (x_u - v)  <=  0.

Bad vertices immediately violate an active face — their Newton step length
is zero and they contribute nothing.

Usage:
    python concept/level_sets.py
Outputs:
    concept/level_sets_theorem.{pdf,png}
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Q matrix: dense, ill-conditioned (thin, elongated level sets)
#     H = R D R^T  with condition number ≈ 36, major axis tilted 30°
# ─────────────────────────────────────────────────────────────────────────────
_theta = np.radians(30)
_R = np.array([[np.cos(_theta), -np.sin(_theta)],
               [np.sin(_theta),  np.cos(_theta)]])
H = _R @ np.diag([18.0, 0.5]) @ _R.T      # dense PD,  cond(H) ≈ 36

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Polytope  (convex, CCW vertex order)
# ─────────────────────────────────────────────────────────────────────────────
POLY_VERTS = np.array([
    # ── right / top side (unchanged — x_u is far to the LEFT) ──────────────
    [ 1.0, -1.5],   # 0  bottom
    [ 5.0, -0.5],   # 1  right-bottom
    [ 5.5,  2.5],   # 2  right-top
    [ 2.5,  4.0],   # 3  top
    # ── left side: 5 faces pointing toward x_u ──────────────────────────────
    [-0.5,  3.0],   # 4  upper-left
    [-1.8,  2.2],   # 5  left-upper  (new face)
    [-2.2,  1.2],   # 6  leftmost    (new face)
    [-2.0,  0.1],   # 7  left-lower  (new face)
    [-1.0, -0.9],   # 8  lower-left  (new face)
], dtype=float)
# Convexity check (CCW cross-products, all > 0):
# c0=11.5  c1=9.75  c2=7.5  c3=1.1  c4=0.98  c5=0.64  c6=0.9  c7=1.4  c8=4.4  ✓


def _poly_halfspaces(verts):
    """(A, b) s.t.  Ax <= b  <=>  x is inside the CCW polygon `verts`."""
    n = len(verts)
    A = np.zeros((n, 2))
    b = np.zeros(n)
    for i in range(n):
        v0, v1 = verts[i], verts[(i + 1) % n]
        edge   = v1 - v0
        # outward normal for CCW ordering: rotate edge 90° clockwise
        normal = np.array([edge[1], -edge[0]])
        A[i]   = normal
        b[i]   = normal @ v0
    return A, b


A_P, b_P   = _poly_halfspaces(POLY_VERTS)
_centroid   = POLY_VERTS.mean(axis=0)
assert np.all(A_P @ _centroid < b_P), "centroid must be strictly inside polytope"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def q_dist_sq(v, x_u):
    """Squared Q-distance  (v - x_u)^T H (v - x_u)."""
    d = v - x_u
    return float(d @ H @ d)


def active_set(v, tol=1e-5):
    """Indices of active constraints at vertex v."""
    return np.where(np.abs(A_P @ v - b_P) < tol)[0]


def is_good_vertex(v, x_u, tol=1e-5):
    """
    True iff the Newton direction  d = x_u - v  points INTO the polytope
    from vertex v  (i.e. the active faces at v are not immediately violated).

    For each active constraint i:  A[i] @ (x_u - v)  <=  0.
    """
    act = active_set(v, tol)
    if len(act) == 0:
        return True
    d = x_u - v
    return bool(np.all((A_P[act] @ d) <= tol))


def ray_polytope_exit(v, x_u, tol=1e-8):
    """
    First time the ray  p(t) = v + t*(x_u - v),  t in (0, 1]
    crosses a non-active polytope facet.

    For a GOOD vertex the ray enters the interior and exits on the far side.
    For a BAD  vertex there is no valid exit (returns t=0, face=-1).

    Returns (exit_point, t_exit, face_index).
    """
    d   = x_u - v
    res = b_P - A_P @ v          # residuals >= 0 (v is inside or on boundary)
    Ad  = A_P @ d

    t_exit = np.inf
    face   = -1
    for i in range(len(b_P)):
        # Non-active constraint (res[i] > 0) whose feasibility tightens (Ad[i] > 0)
        if Ad[i] > tol and res[i] > tol:
            t_i = res[i] / Ad[i]
            if t_i < t_exit:
                t_exit = t_i
                face   = i

    if np.isinf(t_exit) or t_exit > 1.0 + tol:
        return v.copy(), 0.0, -1

    t_clamped = min(t_exit, 1.0)
    return v + t_clamped * d, t_clamped, face


def q_optimal_with_face(x_u):
    """
    Exact analytical Q-projection of x_u onto each face (edge) of the
    polygon, returning the face-point with the smallest Q-distance and its
    face index.  Works for any 2-D convex polygon; no external solver needed.

    For face i:  x(t) = v_i + t*(v_{i+1}-v_i),  t in [0,1]
    Minimise  (x(t)-x_u)^T H (x(t)-x_u)  over t  analytically.
    """
    best_d    = np.inf
    best_x    = _centroid.copy()
    best_face = -1
    n = len(POLY_VERTS)
    for i in range(n):
        v0   = POLY_VERTS[i]
        v1   = POLY_VERTS[(i + 1) % n]
        edge = v1 - v0
        e    = v0 - x_u                    # offset: v0 relative to x_u
        dHd  = float(edge @ H @ edge)
        if abs(dHd) < 1e-14:
            continue
        t_star = float(-(e @ H @ edge) / dHd)
        t_star = float(np.clip(t_star, 0.0, 1.0))
        x_face = v0 + t_star * edge
        d = q_dist_sq(x_face, x_u)
        if d < best_d:
            best_d    = d
            best_x    = x_face.copy()
            best_face = i
    return best_x, best_face


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Parameterize x_u along the FIXED direction ν so the theorem is tested
#     correctly.  The theorem requires L = ||x_u - x_hat|| >= w_P(ν).
#
#     Key insight: start from a reference x_u to the left, compute x_hat as
#     its H-projection onto the polytope.  δ = x_u_ref - x_hat is then
#     automatically H-perpendicular to the active face, so:
#       • ν = δ / ||δ||  is the fixed Euclidean direction for all panels
#       • x_hat stays fixed (H-projection of x_u = x_hat + t·δ is always x_hat)
#       • L(t) = t·||δ||  scales linearly with t
#     We choose t values so L/w spans below (0.25, 0.75) and above (1.5, 3.0)
#     the theorem threshold  L* = w_P(ν).
# ─────────────────────────────────────────────────────────────────────────────

# Reference x_u: outside the polytope on its left side
_x_u_ref             = np.array([-5.0, 0.5])
# Compute H-projection to obtain x_hat and the active face F*
_x_hat, _ref_face    = q_optimal_with_face(_x_u_ref)
# δ is H-perpendicular to face F* by construction of the H-projection
_delta       = _x_u_ref - _x_hat        # displacement: x_hat → x_u_ref

# ν = H·δ / ||H·δ||  (M-weighted gradient direction, = theorem's ν)
# This lies in the normal cone of x_hat by the H-projection KKT conditions.
_H_delta     = H @ _delta
_nu_unit     = _H_delta / np.linalg.norm(_H_delta)

# L = ||x_u - x_hat||_H  (H-norm distance, the L in the theorem)
_delta_Hnorm = float(np.sqrt(_delta @ H @ _delta))  # L per unit scale

# Directional width  w_P(ν) = max_v <ν,v> − min_v <ν,v>  (in ν direction)
_nu_dots = POLY_VERTS @ _nu_unit
_w       = float(_nu_dots.max() - _nu_dots.min())

# Theorem threshold: L(t) = t·||δ||_H >= w_P(ν)  ⟺  t >= t_thresh
_t_thresh = _w / _delta_Hnorm

print(f"ref face F* = {_ref_face},  x_hat = {_x_hat},  nu (H-weighted) = {_nu_unit}")
print(f"w_P(nu) = {_w:.3f},  ||delta||_H = {_delta_Hnorm:.3f},  t_thresh = {_t_thresh:.3f}")

# Four panels: L/w = 0.25, 0.75, 1.5, 3.0  (two below, two above threshold)
_L_OVER_W = [0.25, 0.75, 1.5, 3.0]
X_U_LIST  = [_x_hat + f * _t_thresh * _delta for f in _L_OVER_W]

SUBTITLES = [
    rf"(a)  $L/w = {_L_OVER_W[0]:.2f}$  [below]",
    rf"(b)  $L/w = {_L_OVER_W[1]:.2f}$  [below]",
    rf"(c)  $L/w = {_L_OVER_W[2]:.2f}$  [above]",
    rf"(d)  $L/w = {_L_OVER_W[3]:.2f}$  [above]",
]

# Fixed viewport for ALL panels — always keeps the polytope fully in view.
# Ellipses are framed relative to this window so curvature is always visible.
_POLY_PAD  = 2.5    # padding around polytope (in data units)
_VPORT_XLIM = (POLY_VERTS[:, 0].min() - _POLY_PAD,
               POLY_VERTS[:, 0].max() + _POLY_PAD)
_VPORT_YLIM = (POLY_VERTS[:, 1].min() - _POLY_PAD,
               POLY_VERTS[:, 1].max() + _POLY_PAD)

# Pre-compute the H eigensystem once for parametric ellipse drawing
_EIG_VALS, _EIG_VECS = np.linalg.eigh(H)   # eigenvalues ascending

# ── Colour palette ───────────────────────────────────────────────────────────
C_GOOD     = "#27ae60"
C_BAD      = "#95a5a6"
C_BEST     = "#8e44ad"
C_OPTIMAL  = "#f1c40f"
C_EXIT     = "#e67e22"
C_XU       = "#2980b9"
C_POLYTOPE = "#c0392b"


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 5.  Draw one panel
# ─────────────────────────────────────────────────────────────────────────────

def _ellipse_xy(x_u, z_level, n_pts=800):
    """Parametric Q-ellipse {x : (x-xu)^T H (x-xu) = z_level} as (x_arr, y_arr).

    H = V D V^T  (eigh, eigenvalues ascending).
    p(t) = x_u + sqrt(z/d0)*cos(t)*v0 + sqrt(z/d1)*sin(t)*v1
    d0 is the SMALL eigenvalue (long semi-axis = elongated direction).
    d1 is the LARGE eigenvalue (short semi-axis).
    """
    t  = np.linspace(0, 2 * np.pi, n_pts, endpoint=True)
    a0 = np.sqrt(z_level / _EIG_VALS[0])   # long semi-axis
    a1 = np.sqrt(z_level / _EIG_VALS[1])   # short semi-axis
    xy = (x_u[:, None]
          + a0 * np.cos(t) * _EIG_VECS[:, 0:1]
          + a1 * np.sin(t) * _EIG_VECS[:, 1:2])
    return xy[0], xy[1]


def _draw_panel(ax, x_u, subtitle):
    xlim = _VPORT_XLIM
    ylim = _VPORT_YLIM

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title(subtitle, fontsize=9.5)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.6)

    # ── Q-level sets — parametric ellipses, levels from viewport corners ──────
    # Anchoring z-levels to the viewport ensures we always see fully curved,
    # properly shaped (thin, rotated) ellipses — not just straight-line slices.
    corners = np.array([
        [xlim[0], ylim[0]], [xlim[0], ylim[1]],
        [xlim[1], ylim[0]], [xlim[1], ylim[1]],
        [(xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2],
    ])
    z_vals = np.array([q_dist_sq(c, x_u) for c in corners])
    z_hi   = z_vals.max()
    z_lo   = max(z_vals.min() * 0.04, 1e-2)
    levels = np.geomspace(z_lo, z_hi, 16)

    cmap  = plt.get_cmap("Blues")
    n_lev = len(levels)
    for k, z in enumerate(levels):
        alpha = 0.22 + 0.55 * (k / (n_lev - 1))
        lw    = 0.50 + 0.90 * (k / (n_lev - 1))
        color = cmap(0.28 + 0.62 * (k / (n_lev - 1)))
        ex, ey = _ellipse_xy(x_u, z)
        ax.plot(ex, ey, color=color, alpha=alpha, linewidth=lw,
                zorder=1, solid_capstyle="round", clip_on=True)

    # ── Polytope ──────────────────────────────────────────────────────────────
    poly_patch = MplPolygon(
        POLY_VERTS, closed=True, fill=True,
        edgecolor=C_POLYTOPE, facecolor=C_POLYTOPE, alpha=0.18,
        linewidth=1.8, zorder=2,
    )
    ax.add_patch(poly_patch)

    # ── Classify vertices ─────────────────────────────────────────────────────
    good_mask = np.array([is_good_vertex(v, x_u) for v in POLY_VERTS])
    q_dists   = np.array([q_dist_sq(v, x_u)      for v in POLY_VERTS])

    best_idx = -1
    if good_mask.any():
        best_idx = int(np.argmax(np.where(good_mask, q_dists, -np.inf)))

    for i, v in enumerate(POLY_VERTS):
        if i == best_idx:
            continue
        color  = C_GOOD if good_mask[i] else C_BAD
        marker = "o"    if good_mask[i] else "s"
        ax.scatter(*v, color=color, marker=marker, s=38, zorder=5,
                   edgecolors="black", linewidths=0.6)

    # ── Q-optimal x* ──────────────────────────────────────────────────────────
    x_opt, opt_face = q_optimal_with_face(x_u)
    ax.scatter(*x_opt, color=C_OPTIMAL, marker="*", s=160, zorder=8,
               edgecolors="black", linewidths=0.7)

    # ── Newton ray from farthest good vertex ──────────────────────────────────
    exact_match    = False
    adjacent_match = False
    exit_face_val  = -1
    obj_gap        = None

    if best_idx >= 0:
        v_best = POLY_VERTS[best_idx]
        exit_pt, t_exit, exit_face = ray_polytope_exit(v_best, x_u)
        exit_face_val = exit_face

        f_exit  = 0.5 * float((exit_pt - x_u) @ H @ (exit_pt - x_u))
        f_opt   = 0.5 * float((x_opt   - x_u) @ H @ (x_opt   - x_u))
        obj_gap = f_exit - f_opt

        n_faces = len(POLY_VERTS)
        # Level 1: exact same face
        exact_match   = (exit_face == opt_face) and (t_exit > 1e-3)
        # Level 2: adjacent face (exit and optimal share a vertex)
        adjacent_match = (abs(exit_face - opt_face) <= 1
                          or abs(exit_face - opt_face) >= n_faces - 1) and not exact_match
        theorem_holds = exact_match or adjacent_match

        if t_exit > 1e-3:
            ax.annotate(
                "", xy=exit_pt, xytext=v_best,
                arrowprops=dict(arrowstyle="-|>", color=C_BEST,
                                lw=2.2, mutation_scale=13),
                zorder=7,
            )
        d_vec = x_u - exit_pt
        norm  = np.linalg.norm(d_vec)
        if norm > 1e-9:
            d_unit = d_vec / norm
            far    = exit_pt + d_unit * 60.0
            ax.plot([exit_pt[0], far[0]], [exit_pt[1], far[1]],
                    "--", color=C_BEST, lw=1.1, alpha=0.50, zorder=6,
                    clip_on=True)

        ax.scatter(*v_best, color=C_BEST, marker="D", s=95, zorder=9,
                   edgecolors="black", linewidths=0.9)
        if t_exit > 1e-3:
            ax.scatter(*exit_pt, color=C_EXIT, marker="^", s=85, zorder=8,
                       edgecolors="black", linewidths=0.7)

        # ── Highlight optimal face (thick yellow) and exit face (thick orange)
        def _face_line(face_idx, color, lw, zorder, ls="-"):
            va = POLY_VERTS[face_idx]
            vb = POLY_VERTS[(face_idx + 1) % len(POLY_VERTS)]
            ax.plot([va[0], vb[0]], [va[1], vb[1]],
                    color=color, lw=lw, zorder=zorder,
                    linestyle=ls, solid_capstyle="round")

        _face_line(opt_face,  C_OPTIMAL, 3.5, 4)          # optimal face: yellow
        if exit_face >= 0 and exit_face != opt_face:
            _face_line(exit_face, C_EXIT,    2.5, 4, "--") # exit face: orange dashed

        # ── Verdict annotation ─────────────────────────────────────────────
        if exact_match:
            thm_color, thm_edge = "#d5f5e3", "#1e8449"
            thm_text = "[OK] Exact face"
        elif adjacent_match:
            thm_color, thm_edge = "#fef9e7", "#d4ac0d"
            thm_text = "[~]  Adjacent face"
        else:
            thm_color, thm_edge = "#fadbd8", "#922b21"
            thm_text = "[X]  Wrong side"
        ax.text(0.03, 0.03,
                (f"{thm_text}\n"
                 f"Exit f{exit_face}, Opt f{opt_face}\n"
                 f"Obj gap: {obj_gap:+.3f}"),
                transform=ax.transAxes, fontsize=7, va="bottom",
                bbox=dict(facecolor=thm_color, edgecolor=thm_edge,
                          alpha=0.93, pad=3, boxstyle="round,pad=0.3"),
                zorder=10)

    # ── x_u marker / annotation ───────────────────────────────────────────────
    x_u_in_view = (xlim[0] <= x_u[0] <= xlim[1]
                   and ylim[0] <= x_u[1] <= ylim[1])
    if x_u_in_view:
        ax.scatter(*x_u, color=C_XU, marker="x", s=120, linewidths=2.4,
                   zorder=8)
        ax.text(x_u[0] + 0.2, x_u[1] + 0.35, r"$x_u$",
                color=C_XU, fontsize=9, zorder=9)
    else:
        dx  = x_u[0] - _centroid[0]
        dy  = x_u[1] - _centroid[1]
        dn  = np.sqrt(dx**2 + dy**2)
        ux, uy = dx / dn, dy / dn
        ex_lbl = float(np.clip(_centroid[0] + ux * 3.0,
                                xlim[0] + 0.4, xlim[1] - 0.4))
        ey_lbl = float(np.clip(_centroid[1] + uy * 3.0,
                                ylim[0] + 0.4, ylim[1] - 0.4))
        dist   = np.linalg.norm(x_u - _centroid)
        ax.annotate(
            "",
            xy=(ex_lbl, ey_lbl),
            xytext=(ex_lbl - ux * 1.2, ey_lbl - uy * 1.2),
            arrowprops=dict(arrowstyle="-|>", color=C_XU, lw=1.5),
            zorder=8,
        )
        ax.text(ex_lbl - ux * 1.7, ey_lbl - uy * 1.7,
                f"$x_u$\n({dist:.0f} away)",
                color=C_XU, fontsize=7.5, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.88, edgecolor=C_XU,
                          pad=2, boxstyle="round"),
                zorder=9)

    # ── Q-distance labels on vertices ─────────────────────────────────────────
    for i, v in enumerate(POLY_VERTS):
        dq = np.sqrt(q_dists[i])
        ax.text(v[0] + 0.27, v[1] + 0.27, f"{dq:.1f}",
                fontsize=6.2, color="black", alpha=0.70, zorder=10)

    return exact_match, adjacent_match, exit_face_val, opt_face, obj_gap

# 6.  Build the figure
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   8.5,
    "xtick.labelsize":  7.5,
    "ytick.labelsize":  7.5,
    "legend.fontsize":  8.5,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

fig, axes = plt.subplots(1, 4, figsize=(17, 5.6))

print(f"Running panels ...  (w_P(nu) = {_w:.3f},  threshold L* = {_w:.3f})")
for ax, x_u, subtitle, lw_ratio in zip(axes, X_U_LIST, SUBTITLES, _L_OVER_W):
    exact, adjacent, ef, of, gap = _draw_panel(ax, x_u, subtitle)
    L_val = lw_ratio * _w   # L = L/w * w
    if gap is None:
        status = "no good vertex"
    else:
        lvl = "EXACT" if exact else ("ADJACENT" if adjacent else "WRONG-SIDE")
        status = (f"[{lvl}]  exit_face={ef}  opt_face={of}  "
                  f"obj_gap={gap:+.4f}  L={L_val:.2f}  L/w={lw_ratio:.2f}")
    print(f"  {subtitle:38s}  {status}")

# ── Shared legend ─────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], marker="D", color="w", markerfacecolor=C_BEST,
           markersize=9, markeredgecolor="black",
           label=r"$v^*$ — farthest good vertex"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_GOOD,
           markersize=8, markeredgecolor="black", label="Good vertex"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_BAD,
           markersize=8, markeredgecolor="black", label="Bad vertex"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor=C_OPTIMAL,
           markersize=13, markeredgecolor="black", label=r"$x^*$ — Q-optimal"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=C_EXIT,
           markersize=9, markeredgecolor="black",
           label=r"$x_{\mathrm{ph1}}$ — Phase-I exit"),
    Line2D([0], [0], linestyle="-", color=C_BEST, lw=2.0,
           label=r"Newton ray  ($v^* \!\to x^*$ direction)"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center", bbox_to_anchor=(0.5, -0.04),
    ncol=6, framealpha=0.95, edgecolor="0.7", fontsize=8.5,
)

fig.suptitle(
    r"GeoQP — Farthest-Good-Vertex Theorem:  $x_u = \hat{x} + L\nu$ along fixed $\nu$"
    r"  (ill-conditioned $Q$,  $\kappa(Q)\approx 36$,  $30°$ tilt)"
    "\n"
    r"Panels (a,b): $L < w_{\mathcal{P}}(\nu)$ [theorem not guaranteed];  "
    r"Panels (c,d): $L \geq w_{\mathcal{P}}(\nu)$ [theorem applies].  "
    r"Numbers = $\|v_i - x_u\|_Q$",
    y=1.01, fontsize=9.5,
)

fig.tight_layout(rect=[0, 0.07, 1, 1])

import os
os.makedirs("concept", exist_ok=True)
for ext in ("pdf", "png"):
    path = f"concept/level_sets_theorem.{ext}"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    print(f"Saved  {path}")