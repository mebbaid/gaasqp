import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap


def _sh_clip(polygon, halfplane_pt, halfplane_normal):
    """Sutherland-Hodgman: clip convex polygon (Nx2) to halfplane n·(x-p)≥0."""
    def _inside(pt):
        return float(np.dot(halfplane_normal, pt - halfplane_pt)) >= -1e-10
    def _isect(a, b):
        d = b - a
        denom = float(np.dot(halfplane_normal, d))
        if abs(denom) < 1e-14:
            return a.copy()
        t = float(np.dot(halfplane_normal, halfplane_pt - a)) / denom
        return a + t * d
    out = []
    n = len(polygon)
    for i in range(n):
        c, nxt = polygon[i], polygon[(i + 1) % n]
        if _inside(c):
            out.append(c)
            if not _inside(nxt):
                out.append(_isect(c, nxt))
        elif _inside(nxt):
            out.append(_isect(c, nxt))
    return np.array(out, dtype=float) if out else np.zeros((0, 2), dtype=float)


def make_H(theta_deg, lam_max=50.0, lam_min=1):
    """Rotated PD Hessian:  H = R(θ) diag(λ) R(θ)ᵀ"""
    th = theta_deg * np.pi / 180
    R  = np.array([[np.cos(th), -np.sin(th)],
                   [np.sin(th),  np.cos(th)]])
    D  = np.diag([lam_max, lam_min])
    return R @ D @ R.T

CASES = [
    dict(label='θ = 30°',  H=make_H(30),  x_star=np.array([-2.8,  2.5])),
    dict(label='θ = 90°', H=make_H(90), x_star=np.array([-2.8,  2.5])),
    dict(label='θ = -100°', H=make_H(-100), x_star=np.array([-2.8,  2.5])),
]

box = dict(xlo=-1.5, xhi=1.5, ylo=-1.5, yhi=1.5)

NF_P1   = 100
NF_P2   = 80
NF_HOLD = 35
NF_FADE = 20
NF_CASE = NF_P1 + NF_P2 + NF_HOLD + NF_FADE

N_CASES        = len(CASES)
N_FRAMES_TOTAL = NF_CASE * N_CASES

NX, NY = 800, 800
xs_box = np.linspace(box['xlo'], box['xhi'], NX)
ys_box = np.linspace(box['ylo'], box['yhi'], NY)
XX_box, YY_box = np.meshgrid(xs_box, ys_box)

def compute_case(H, x_star):
    eigvals, eigvecs = np.linalg.eigh(H)
    Hinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    def ellipse_pts(t, n=600):
        theta = np.linspace(0, 2*np.pi, n)
        circ  = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        return x_star + t * (Hinv_sqrt @ circ.T).T

    def cost_grid(XX, YY):
        d0 = XX - x_star[0];  d1 = YY - x_star[1]
        return 0.5*(H[0,0]*d0**2 + 2*H[0,1]*d0*d1 + H[1,1]*d1**2)

    ZZ = cost_grid(XX_box, YY_box)

    c_near   = float(ZZ.min())
    near_idx = np.unravel_index(ZZ.argmin(), ZZ.shape)
    x_near   = np.array([XX_box[near_idx], YY_box[near_idx]])
    t_near   = np.sqrt(2 * c_near)

    c_far    = float(ZZ.max())
    far_idx  = np.unravel_index(ZZ.argmax(), ZZ.shape)
    x_far    = np.array([XX_box[far_idx], YY_box[far_idx]])
    t_far    = np.sqrt(2 * c_far)

    t_end    = t_far * 1.08

    # Pre-compute negative outward normals: −∇f(x) = −H(x − x*)
    neg_grad_near = -(H @ (x_near - x_star))
    neg_grad_far  = -(H @ (x_far  - x_star))

    # Farthest antipodal good vertex (Lemma: v good iff x*_i strictly inside active bound)
    _ell = np.array([box['xlo'], box['ylo']])
    _u   = np.array([box['xhi'], box['yhi']])
    _good_verts = []
    for _v0 in [_ell[0], _u[0]]:
        for _v1 in [_ell[1], _u[1]]:
            _v  = np.array([_v0, _v1])
            _ok = True
            for _i, (_vi, _li, _ui) in enumerate(zip(_v, _ell, _u)):
                if np.isclose(_vi, _li) and x_star[_i] <= _li:
                    _ok = False; break
                if np.isclose(_vi, _ui) and x_star[_i] >= _ui:
                    _ok = False; break
            if _ok:
                _good_verts.append(_v)
    if _good_verts:
        _gv_costs = [0.5 * ((v - x_star) @ H @ (v - x_star)) for v in _good_verts]
        _best     = int(np.argmax(_gv_costs))   # farthest in Q-norm = highest f(v)
        x_good    = _good_verts[_best]
        t_good    = np.sqrt(2.0 * _gv_costs[_best])
    else:
        x_good = x_star.copy()
        t_good = 0.0
    neg_grad_good = -(H @ (x_good - x_star))

    # H_1: normal to level-set f(v̂) at v̂  =  -∇f(v̂)  (proved to contain x*)
    # H_2: tangent to level-set f(v̂) at v̂  =  90° rotation of ∇f(v̂),
    #       oriented so that x* lies on the positive side.
    _grad_good = -neg_grad_good              # ∇f(v̂) = H(v̂ - x*)
    _t_ccw     = np.array([-_grad_good[1],  _grad_good[0]])   # CCW rotation
    if np.dot(_t_ccw, x_star - x_good) >= 0:
        h2_normal = _t_ccw
    else:
        h2_normal = -_t_ccw

    # Ray exit: first box-boundary crossing of r(t)=x_good+t*(x_star-x_good) for t>0
    _d_ray  = x_star - x_good
    _ts_exit = []
    for _i in range(2):
        if abs(_d_ray[_i]) > 1e-10:
            for _bnd in [_ell[_i], _u[_i]]:
                _t = (_bnd - x_good[_i]) / _d_ray[_i]
                if _t > 1e-9:
                    _pt = x_good + _t * _d_ray
                    if all(_ell[_j]-1e-9 <= _pt[_j] <= _u[_j]+1e-9 for _j in range(2)):
                        _ts_exit.append(_t)
    _t_exit = min(_ts_exit) if _ts_exit else 1.0
    x_exit  = x_good + _t_exit * _d_ray

    # P_+ half-box: normalized-coordinate split
    # n_i = s_i / h_i,  s_i = sign(x_star_i - c0_i),  h_i = (u_i - l_i)/2
    # P_+ = {x in P : sum_i n_i*(x_i - c0_i) >= 0}
    _c0_pp    = np.array([0.5*(box['xlo']+box['xhi']), 0.5*(box['ylo']+box['yhi'])])
    _h_pp     = 0.5 * (_u - _ell)                       # half-widths
    _s_pp     = np.sign(x_star - _c0_pp)                # ±1 per coordinate
    _v_pp     = _s_pp / _h_pp                            # normalized normal
    _box_rect = np.array([[box['xlo'], box['ylo']], [box['xhi'], box['ylo']],
                          [box['xhi'], box['yhi']], [box['xlo'], box['yhi']]])
    _pp = _sh_clip(_box_rect, _c0_pp, _v_pp)
    p_plus_poly = _pp if len(_pp) >= 3 else _box_rect

    t_p1   = np.linspace(0.0,    t_near, NF_P1)
    t_p2   = np.linspace(t_near, t_end,  NF_P2)
    t_hold = np.full(NF_HOLD, t_end)
    t_fade = np.full(NF_FADE, t_end)
    t_all  = np.concatenate([t_p1, t_p2, t_hold, t_fade])

    return dict(
        H=H, x_star=x_star, eigvals=eigvals, eigvecs=eigvecs,
        Hinv_sqrt=Hinv_sqrt, ellipse_pts=ellipse_pts, cost_grid=cost_grid,
        ZZ=ZZ, x_near=x_near, t_near=t_near, x_far=x_far, t_far=t_far,
        t_end=t_end, t_all=t_all, c_near=c_near, c_far=c_far,
        neg_grad_near=neg_grad_near, neg_grad_far=neg_grad_far,
        x_good=x_good, t_good=t_good, neg_grad_good=neg_grad_good,
        h2_normal=h2_normal,
        x_exit=x_exit, p_plus_poly=p_plus_poly,
    )

geo = [compute_case(c['H'], c['x_star']) for c in CASES]

# Fixed display length for arrows (data units)
ARROW_LEN = 0.9

def scaled_vec(vec):
    """Normalise vec to ARROW_LEN for display."""
    norm = np.linalg.norm(vec)
    return (ARROW_LEN / norm) * vec if norm > 1e-10 else np.zeros(2)


cx  = 0.5*(box['xlo']+box['xhi']);  cy  = 0.5*(box['ylo']+box['yhi'])
hw  = 0.5*(box['xhi']-box['xlo']);  hh  = 0.5*(box['yhi']-box['ylo'])
xlim = (cx - hw*1.25 - 1.3,  cx + hw*1.25 + 1.3)
ylim = (cy - hh*1.25 - 1.3,  cy + hh*1.25 + 1.3)

fig, ax = plt.subplots(figsize=(7, 7), facecolor='#0d1117')
ax.set_facecolor('#0d1117')
ax.set_xlim(*xlim);  ax.set_ylim(*ylim);  ax.set_aspect('equal')
ax.tick_params(colors='#8b949e')
for sp in ax.spines.values(): sp.set_color('#30363d')
ax.xaxis.label.set_color('#8b949e');  ax.yaxis.label.set_color('#8b949e')
ax.set_xlabel('$x_1$', fontsize=13);  ax.set_ylabel('$x_2$', fontsize=13)
ax.set_title(r'Quadratic cost — $-\nabla f$ at nearest & farthest contacts',
             color='#e6edf3', fontsize=13, pad=12)

# Feasible box (static)
bw = box['xhi']-box['xlo'];  bh_box = box['yhi']-box['ylo']
feas_patch = patches.FancyBboxPatch(
    (box['xlo'], box['ylo']), bw, bh_box,
    boxstyle="square,pad=0", linewidth=2.2,
    edgecolor='#58a6ff', facecolor='#58a6ff12', zorder=3)
ax.add_patch(feas_patch)
ax.text(box['xhi']+0.07, box['yhi']+0.07, 'Feasible set',
        color='#58a6ff', fontsize=10, ha='left', va='bottom', zorder=5)

# x* glow
x_star_all = CASES[0]['x_star']
N_GLOW = 5
glow_artists = []
for i in range(N_GLOW):
    g, = ax.plot(*x_star_all, 'o', color='#79c0ff',
                 alpha=0.05+0.04*(N_GLOW-i)/N_GLOW,
                 markersize=18+22*i, zorder=6, markeredgewidth=0)
    glow_artists.append(g)
droplet_dot, = ax.plot(*x_star_all, 'o', color='#cae8ff',
                       markersize=7, markeredgecolor='#ffffff',
                       markeredgewidth=0.8, zorder=7)
ax.annotate('$x^*$ (unconstrained min)',
            xy=x_star_all, xytext=x_star_all + np.array([0.3, -0.5]),
            color='#cae8ff', fontsize=10, zorder=8)

# Background cost contour
bg_contour_collection = [None]

# Expanding ellipse
ellipse_line, = ax.plot([], [], '-', lw=1.8, alpha=0.92, color='white', zorder=9)
ellipse_fill  = plt.Polygon([[0,0]], closed=True,
                             facecolor='white', edgecolor='none',
                             alpha=0.07, zorder=8)
ax.add_patch(ellipse_fill)

# Nearest (green star)
near_dot,  = ax.plot([], [], '*', color='#3fb950', markersize=15,
                     markeredgecolor='#ffffff', markeredgewidth=0.6, zorder=12)
near_text  = ax.text(0, 0, 'Constrained\noptimum\n(nearest)',
                     color='#3fb950', fontsize=9.5, visible=False,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117bb',
                               edgecolor='#3fb95055', lw=0.8), zorder=13)
ring_near, = ax.plot([], [], '--', color='#3fb950', lw=1.2, alpha=0.5, zorder=7)

# FancyArrowPatch for -∇f at nearest point
near_arrow = FancyArrowPatch(
    posA=(0, 0), posB=(0, 0),
    arrowstyle='->', color='#3fb950',
    linewidth=2.2, mutation_scale=16, zorder=14)
ax.add_patch(near_arrow)
near_arrow.set_visible(False)

near_arrow_label = ax.text(
    0, 0, r'$-\nabla f$', color='#3fb950', fontsize=10, visible=False,
    bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117cc',
              edgecolor='#3fb95066', lw=0.8), zorder=15)

# Farthest (red diamond)
far_dot,   = ax.plot([], [], 'D', color='#f85149', markersize=11,
                     markeredgecolor='#ffffff', markeredgewidth=0.6, zorder=12)
far_text   = ax.text(0, 0, 'Farthest\nbox point',
                     color='#f85149', fontsize=9.5, visible=False,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117bb',
                               edgecolor='#f8514955', lw=0.8), zorder=13)
ring_far,  = ax.plot([], [], '--', color='#f85149', lw=1.2, alpha=0.5, zorder=7)

# FancyArrowPatch for -∇f at farthest point
far_arrow = FancyArrowPatch(
    posA=(0, 0), posB=(0, 0),
    arrowstyle='->', color='#f85149',
    linewidth=2.2, mutation_scale=16, zorder=14)
ax.add_patch(far_arrow)
far_arrow.set_visible(False)

far_arrow_label = ax.text(
    0, 0, r'$-\nabla f$', color='#f85149', fontsize=10, visible=False,
    bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117cc',
              edgecolor='#f8514966', lw=0.8), zorder=15)

# Good vertex (antipodal, purple pentagon)
good_dot,  = ax.plot([], [], 'p', color='#d2a8ff', markersize=14,
                     markeredgecolor='#ffffff', markeredgewidth=0.6, zorder=12)
good_text  = ax.text(0, 0, 'Antipodal\ngood vertex',
                     color='#d2a8ff', fontsize=9.5, visible=False,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117bb',
                               edgecolor='#d2a8ff55', lw=0.8), zorder=13)
ring_good, = ax.plot([], [], '--', color='#d2a8ff', lw=1.2, alpha=0.5, zorder=7)

# Certificate region: {x ∈ P : f(x) ≤ f(v)} — ellipse at f(v) clipped to the box
cert_fill = plt.Polygon([[0, 0]], closed=True,
                        facecolor='#39d0d8', edgecolor='#39d0d8',
                        alpha=0.0, lw=1.5, zorder=4)
ax.add_patch(cert_fill)
cert_fill.set_clip_path(feas_patch)   # clip to feasible box automatically

# Cone wedge between -∇f and (x*-v) directions, rooted at good vertex
cone_fill = plt.Polygon([[0, 0]], closed=True,
                        facecolor='#ffa657', edgecolor='none',
                        alpha=0.0, zorder=3)
ax.add_patch(cone_fill)

# P_+ half-box (translucent blue)
p_plus_fill = plt.Polygon([[0, 0]], closed=True,
                           facecolor='#388bfd', edgecolor='#388bfd',
                           alpha=0.0, lw=0.0, zorder=3)
ax.add_patch(p_plus_fill)

# Cone ∩ P_+ intersection (bright green highlight)
cone_p_plus_fill = plt.Polygon([[0, 0]], closed=True,
                                facecolor='#7ee787', edgecolor='#7ee787',
                                alpha=0.0, lw=1.0, zorder=6)
ax.add_patch(cone_p_plus_fill)

# Ray from good vertex to exit point
ray_line, = ax.plot([], [], '--', color='#e3b341', lw=1.8, alpha=0.0, zorder=8)

# Exit point marker
exit_dot,  = ax.plot([], [], 'X', color='#ff7b72', markersize=13,
                     markeredgecolor='#ffffff', markeredgewidth=0.7, zorder=12)
exit_dot.set_visible(False)
exit_text  = ax.text(0, 0, 'Ray exit\n(last face\ncrossing)',
                     color='#ff7b72', fontsize=9, visible=False,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117bb',
                               edgecolor='#ff7b7255', lw=0.8), zorder=13)

good_arrow = FancyArrowPatch(
    posA=(0, 0), posB=(0, 0),
    arrowstyle='->', color='#d2a8ff',
    linewidth=2.2, mutation_scale=16, zorder=14)
ax.add_patch(good_arrow)
good_arrow.set_visible(False)

good_arrow_label = ax.text(
    0, 0, r'$-\nabla f$', color='#d2a8ff', fontsize=10, visible=False,
    bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117cc',
              edgecolor='#d2a8ff66', lw=0.8), zorder=15)

# Arrow from good vertex toward x* (direction to unconstrained min)
good_to_xstar_arrow = FancyArrowPatch(
    posA=(0, 0), posB=(0, 0),
    arrowstyle='->', color='#e3b341',
    linewidth=2.2, mutation_scale=16, zorder=14)
ax.add_patch(good_to_xstar_arrow)
good_to_xstar_arrow.set_visible(False)

good_to_xstar_label = ax.text(
    0, 0, r'$t(\hat{v})$', color='#e3b341', fontsize=10, visible=False,
    bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117cc',
              edgecolor='#e3b34166', lw=0.8), zorder=15)

# Phase / cost / case labels
phase_label = ax.text(0.03, 0.97, '', transform=ax.transAxes,
                      color='#e6edf3', fontsize=10, va='top',
                      fontfamily='monospace', zorder=10)
cost_label  = ax.text(0.03, 0.91, '', transform=ax.transAxes,
                      color='#8b949e', fontsize=9.5, va='top',
                      fontfamily='monospace', zorder=10)
case_badge  = ax.text(0.97, 0.97, '', transform=ax.transAxes,
                      color='#ffa657', fontsize=11, va='top', ha='right',
                      fontweight='bold', fontfamily='monospace', zorder=10)

# Progress bar
bar_y    = ylim[0] + 0.03*(ylim[1]-ylim[0])
x_bar_lo = xlim[0] + 0.04*(xlim[1]-xlim[0])
x_bar_hi = xlim[0] + 0.96*(xlim[1]-xlim[0])
bar_span = x_bar_hi - x_bar_lo
ax.axhline(bar_y, color='#30363d', lw=4, zorder=10)
bar_seg1, = ax.plot([], [], '-', color='#3fb950', lw=4, zorder=11, solid_capstyle='butt')
bar_seg2, = ax.plot([], [], '-', color='#f85149', lw=4, zorder=11, solid_capstyle='butt')

def bar_x(frac):
    return x_bar_lo + np.clip(frac, 0, 1) * bar_span


XX_bg, YY_bg = np.meshgrid(
    np.linspace(xlim[0], xlim[1], 350),
    np.linspace(ylim[0], ylim[1], 350))

cmap_bg = LinearSegmentedColormap.from_list(
    'bg', ['#0d1117', '#0d2a3a', '#0a3d5c', '#0d5278'], N=256)

_last_case_drawn = [-1]

def redraw_background(case_idx):
    if case_idx == _last_case_drawn[0]:
        return
    _last_case_drawn[0] = case_idx
    if bg_contour_collection[0] is not None:
        for coll in bg_contour_collection[0].collections:
            coll.remove()
    g = geo[case_idx]
    ZZ_bg = g['cost_grid'](XX_bg, YY_bg)
    levels_bg = np.linspace(0, g['c_far']*1.5, 28)
    cf = ax.contourf(XX_bg, YY_bg, ZZ_bg, levels=levels_bg,
                     cmap=cmap_bg, alpha=0.5, zorder=1)
    bg_contour_collection[0] = cf


ALL_DYNAMIC = (ellipse_line, ellipse_fill,
               near_dot, far_dot, near_text, far_text,
               ring_near, ring_far,
               near_arrow, near_arrow_label,
               far_arrow, far_arrow_label,
               good_dot, good_text, ring_good, cert_fill,
               cone_fill, p_plus_fill, cone_p_plus_fill,
               ray_line, exit_dot, exit_text,
               good_arrow, good_arrow_label,
               good_to_xstar_arrow, good_to_xstar_label,
               phase_label, cost_label, case_badge,
               bar_seg1, bar_seg2)

def init():
    ellipse_line.set_data([], [])
    ellipse_fill.set_xy([[0, 0]])
    near_dot.set_data([], []);    far_dot.set_data([], [])
    near_text.set_visible(False); far_text.set_visible(False)
    ring_near.set_data([], []);   ring_far.set_data([], [])
    near_arrow.set_visible(False); near_arrow_label.set_visible(False)
    far_arrow.set_visible(False);  far_arrow_label.set_visible(False)
    good_dot.set_data([], [])
    good_text.set_visible(False)
    ring_good.set_data([], [])
    cert_fill.set_xy([[0, 0]])
    cert_fill.set_alpha(0.0)
    cone_fill.set_xy([[0, 0]])
    cone_fill.set_alpha(0.0)
    p_plus_fill.set_xy([[0, 0]])
    p_plus_fill.set_alpha(0.0)
    cone_p_plus_fill.set_xy([[0, 0]])
    cone_p_plus_fill.set_alpha(0.0)
    ray_line.set_data([], [])
    ray_line.set_alpha(0.0)
    exit_dot.set_data([], [])
    exit_dot.set_visible(False)
    exit_text.set_visible(False)
    good_arrow.set_visible(False)
    good_arrow_label.set_visible(False)
    good_to_xstar_arrow.set_visible(False)
    good_to_xstar_label.set_visible(False)
    phase_label.set_text('');  cost_label.set_text('');  case_badge.set_text('')
    bar_seg1.set_data([], []);  bar_seg2.set_data([], [])
    return (*ALL_DYNAMIC, *glow_artists)


def update(frame):
    case_idx = frame // NF_CASE
    local_f  = frame %  NF_CASE
    case_idx = min(case_idx, N_CASES - 1)
    g        = geo[case_idx]

    redraw_background(case_idx)

    t = g['t_all'][min(local_f, NF_CASE - 1)]

    in_p1   = local_f < NF_P1
    in_p2   = NF_P1 <= local_f < NF_P1 + NF_P2
    in_hold = NF_P1 + NF_P2 <= local_f < NF_P1 + NF_P2 + NF_HOLD
    in_fade = local_f >= NF_P1 + NF_P2 + NF_HOLD

    near_revealed = local_f >= NF_P1 - 1
    far_revealed  = local_f >= NF_P1 + NF_P2 - 1

    if in_fade:
        fade_frac  = (local_f - (NF_P1 + NF_P2 + NF_HOLD)) / NF_FADE
        alpha_main = max(1.0 - fade_frac * 1.5, 0.0)
    else:
        alpha_main = 1.0

    # ── ellipse ───────────────────────────────────────────────
    if t > 1e-6 and alpha_main > 0:
        pts = g['ellipse_pts'](t)
        ellipse_line.set_data(pts[:, 0], pts[:, 1])
        ellipse_fill.set_xy(pts)
    else:
        ellipse_line.set_data([], [])
        ellipse_fill.set_xy([[0, 0]])

    t_near = g['t_near'];  t_far = g['t_far']
    if t <= t_near:
        r = min(t / t_near, 1.0) if t_near > 0 else 1.0
        color = (0.10+0.15*r, 0.79+0.04*r, 0.63-0.31*r)
    else:
        r = min((t - t_near) / (t_far - t_near), 1.0) if t_far > t_near else 1.0
        color = (0.25+0.72*r, 0.73-0.41*r, 0.31+0.00*r)
    ellipse_line.set_color((*color, alpha_main))
    ellipse_line.set_alpha(alpha_main)
    ellipse_fill.set_facecolor((*color, 0.07*alpha_main))

    # ── labels ────────────────────────────────────────────────
    c_now = 0.5 * t**2
    cost_label.set_text(f'f  =  {c_now:.3f}')
    cost_label.set_alpha(alpha_main)
    case_badge.set_text(f'Case {case_idx+1}/3: {CASES[case_idx]["label"]}')
    case_badge.set_alpha(alpha_main)

    if in_p1:
        phase_label.set_text('Phase 1 — searching nearest contact …')
    elif in_p2:
        phase_label.set_text('Phase 2 — searching farthest contact …')
    elif in_hold:
        phase_label.set_text('Both contacts identified ✓')
    else:
        phase_label.set_text('⟳ next case …')
    phase_label.set_alpha(alpha_main)

    # ── glow pulse ────────────────────────────────────────────
    pulse = 1.0 + 0.11*np.sin(frame * 0.27)
    for i, g_ in enumerate(glow_artists):
        g_.set_markersize((18+22*i) * pulse)
        g_.set_alpha((0.05+0.04*(N_GLOW-i)/N_GLOW) * alpha_main)
    droplet_dot.set_alpha(alpha_main)

    _cx = 0.5*(box['xlo'] + box['xhi'])

    # ── nearest point + −∇f arrow ─────────────────────────────
    if near_revealed and alpha_main > 0:
        flash_n = 1.0 + 0.4*np.exp(-0.12*(local_f-(NF_P1-1))) \
                      * abs(np.sin((local_f-(NF_P1-1))*0.9))
        near_dot.set_data([g['x_near'][0]], [g['x_near'][1]])
        near_dot.set_markersize(15 * flash_n)
        near_dot.set_alpha(alpha_main)

        dx = 0.14 if g['x_near'][0] < _cx else -0.60
        dy = -0.32 if g['x_near'][1] > 0 else 0.18
        near_text.set_position((g['x_near'][0]+dx, g['x_near'][1]+dy))
        near_text.set_visible(True)
        near_text.set_alpha(alpha_main)

        ring_pts = g['ellipse_pts'](t_near)
        ring_near.set_data(ring_pts[:, 0], ring_pts[:, 1])
        ring_near.set_alpha(0.5 * alpha_main)

        # −∇f arrow via FancyArrowPatch
        vec_n = scaled_vec(g['neg_grad_near'])
        base_n = tuple(g['x_near'])
        tip_n  = tuple(g['x_near'] + vec_n)
        near_arrow.set_positions(base_n, tip_n)
        near_arrow.set_alpha(alpha_main)
        near_arrow.set_visible(True)

        label_pos_n = g['x_near'] + vec_n * 1.18
        near_arrow_label.set_position(label_pos_n)
        near_arrow_label.set_alpha(alpha_main)
        near_arrow_label.set_visible(True)
    else:
        near_dot.set_data([], [])
        near_text.set_visible(False)
        ring_near.set_data([], [])
        near_arrow.set_visible(False)
        near_arrow_label.set_visible(False)

    # ── farthest point + −∇f arrow ────────────────────────────
    if far_revealed and alpha_main > 0:
        flash_f = 1.0 + 0.4*np.exp(-0.12*(local_f-(NF_P1+NF_P2-1))) \
                      * abs(np.sin((local_f-(NF_P1+NF_P2-1))*0.9))
        far_dot.set_data([g['x_far'][0]], [g['x_far'][1]])
        far_dot.set_markersize(11 * flash_f)
        far_dot.set_alpha(alpha_main)

        dx = 0.14 if g['x_far'][0] < _cx else -0.52
        dy = 0.18 if g['x_far'][1] > 0 else -0.30
        far_text.set_position((g['x_far'][0]+dx, g['x_far'][1]+dy))
        far_text.set_visible(True)
        far_text.set_alpha(alpha_main)

        ring_pts = g['ellipse_pts'](t_far)
        ring_far.set_data(ring_pts[:, 0], ring_pts[:, 1])
        ring_far.set_alpha(0.5 * alpha_main)

        # −∇f arrow via FancyArrowPatch
        vec_f = scaled_vec(g['neg_grad_far'])
        base_f = tuple(g['x_far'])
        tip_f  = tuple(g['x_far'] + vec_f)
        far_arrow.set_positions(base_f, tip_f)
        far_arrow.set_alpha(alpha_main)
        far_arrow.set_visible(True)

        label_pos_f = g['x_far'] + vec_f * 1.18
        far_arrow_label.set_position(label_pos_f)
        far_arrow_label.set_alpha(alpha_main)
        far_arrow_label.set_visible(True)
    else:
        far_dot.set_data([], [])
        far_text.set_visible(False)
        ring_far.set_data([], [])
        far_arrow.set_visible(False)
        far_arrow_label.set_visible(False)

    # ── antipodal good vertex + −∇f arrow ─────────────────────
    if near_revealed and alpha_main > 0:
        flash_g = 1.0 + 0.4*np.exp(-0.12*(local_f-(NF_P1-1))) \
                      * abs(np.sin((local_f-(NF_P1-1))*0.9))
        good_dot.set_data([g['x_good'][0]], [g['x_good'][1]])
        good_dot.set_markersize(14 * flash_g)
        good_dot.set_alpha(alpha_main)

        dx = 0.14 if g['x_good'][0] < _cx else -0.72
        dy = 0.18 if g['x_good'][1] > 0   else -0.42
        good_text.set_position((g['x_good'][0]+dx, g['x_good'][1]+dy))
        good_text.set_visible(True)
        good_text.set_alpha(alpha_main)

        ring_pts_g = g['ellipse_pts'](g['t_good'])
        ring_good.set_data(ring_pts_g[:, 0], ring_pts_g[:, 1])
        ring_good.set_alpha(0.5 * alpha_main)
        cert_fill.set_xy(ring_pts_g)
        cert_fill.set_alpha(0.18 * alpha_main)

        # ── ray from v to exit point ───────────────────────────
        x_exit = g['x_exit']
        ray_line.set_data([g['x_good'][0], x_exit[0]], [g['x_good'][1], x_exit[1]])
        ray_line.set_alpha(0.85 * alpha_main)
        exit_dot.set_data([x_exit[0]], [x_exit[1]])
        exit_dot.set_alpha(alpha_main)
        exit_dot.set_visible(True)
        _edx = 0.12 if x_exit[0] < _cx else -0.52
        _edy = 0.14 if x_exit[1] < 0   else -0.46
        exit_text.set_position((x_exit[0] + _edx, x_exit[1] + _edy))
        exit_text.set_alpha(alpha_main)
        exit_text.set_visible(True)

        # ── cone between −∇f and H_2 direction (v → v_opp) ─────────────
        ng      = g['neg_grad_good']
        ng_hat  = ng / np.linalg.norm(ng) if np.linalg.norm(ng) > 1e-10 else ng
        h2_vec  = g['h2_normal']
        h2_hat  = h2_vec / np.linalg.norm(h2_vec) if np.linalg.norm(h2_vec) > 1e-10 else h2_vec
        L_cone  = ARROW_LEN * 1.6
        N_fan   = 40
        fan_pts = [g['x_good']]
        for _k in range(N_fan + 1):
            _s    = _k / N_fan
            _dir  = (1 - _s) * ng_hat + _s * h2_hat
            _dnrm = np.linalg.norm(_dir)
            if _dnrm > 1e-10:
                _dir /= _dnrm
            fan_pts.append(g['x_good'] + L_cone * _dir)
        cone_fill.set_xy(np.array(fan_pts))
        cone_fill.set_alpha(0.18 * alpha_main)

        p_plus_fill.set_xy(g['p_plus_poly'])
        p_plus_fill.set_alpha(0.12 * alpha_main)

        # ── P_+ ∩ H_1 ∩ H_2 intersection ──────────────────────────────────
        _c0_anim = np.array([0.5*(box['xlo']+box['xhi']), 0.5*(box['ylo']+box['yhi'])])
        _h_anim  = np.array([0.5*(box['xhi']-box['xlo']), 0.5*(box['yhi']-box['ylo'])])
        _vnorm_anim = np.sign(g['x_star'] - _c0_anim) / _h_anim
        _box_rect_anim = np.array([[box['xlo'], box['ylo']], [box['xhi'], box['ylo']],
                                   [box['xhi'], box['yhi']], [box['xlo'], box['yhi']]])
        _clip = _sh_clip(_box_rect_anim, _c0_anim, _vnorm_anim)        # clip to P_+
        if len(_clip) >= 3:
            _clip = _sh_clip(_clip, g['x_good'], g['neg_grad_good'])   # clip to H_1
        if len(_clip) >= 3:
            _clip = _sh_clip(_clip, g['x_good'], g['h2_normal'])       # clip to H_2
        if len(_clip) >= 3:
            cone_p_plus_fill.set_xy(_clip)
            cone_p_plus_fill.set_alpha(0.35 * alpha_main)
        else:
            cone_p_plus_fill.set_xy([[0, 0]])
            cone_p_plus_fill.set_alpha(0.0)

        vec_g  = scaled_vec(g['neg_grad_good'])
        base_g = tuple(g['x_good'])
        tip_g  = tuple(g['x_good'] + vec_g)
        good_arrow.set_positions(base_g, tip_g)
        good_arrow.set_alpha(alpha_main)
        good_arrow.set_visible(True)

        label_pos_g = g['x_good'] + vec_g * 1.18
        good_arrow_label.set_position(label_pos_g)
        good_arrow_label.set_alpha(alpha_main)
        good_arrow_label.set_visible(True)

        # H_2 arrow: from good vertex toward opposite corner
        vec_h2 = scaled_vec(g['h2_normal'])
        good_to_xstar_arrow.set_positions(tuple(g['x_good']), tuple(g['x_good'] + vec_h2))
        good_to_xstar_arrow.set_alpha(alpha_main)
        good_to_xstar_arrow.set_visible(True)
        label_pos_h2 = g['x_good'] + vec_h2 * 1.18
        good_to_xstar_label.set_position(label_pos_h2)
        good_to_xstar_label.set_alpha(alpha_main)
        good_to_xstar_label.set_visible(True)
    else:
        good_dot.set_data([], [])
        good_text.set_visible(False)
        ring_good.set_data([], [])
        cert_fill.set_xy([[0, 0]])
        cert_fill.set_alpha(0.0)
        cone_fill.set_xy([[0, 0]])
        cone_fill.set_alpha(0.0)
        p_plus_fill.set_xy([[0, 0]])
        p_plus_fill.set_alpha(0.0)
        cone_p_plus_fill.set_xy([[0, 0]])
        cone_p_plus_fill.set_alpha(0.0)
        ray_line.set_data([], [])
        ray_line.set_alpha(0.0)
        exit_dot.set_data([], [])
        exit_dot.set_visible(False)
        exit_text.set_visible(False)
        good_arrow.set_visible(False)
        good_arrow_label.set_visible(False)
        good_to_xstar_arrow.set_visible(False)
        good_to_xstar_label.set_visible(False)

    # ── progress bar ──────────────────────────────────────────
    p1_end   = NF_P1 / NF_CASE
    p2_start = NF_P1 / NF_CASE
    p2_end   = (NF_P1 + NF_P2) / NF_CASE

    frac1 = min(local_f / NF_P1, 1.0)
    bar_seg1.set_data([x_bar_lo, bar_x(frac1 * p1_end)], [bar_y, bar_y])
    bar_seg1.set_alpha(alpha_main)

    if local_f >= NF_P1:
        frac2 = min((local_f - NF_P1) / NF_P2, 1.0)
        bar_seg2.set_data([bar_x(p2_start),
                           bar_x(p2_start + frac2*(p2_end - p2_start))],
                          [bar_y, bar_y])
        bar_seg2.set_alpha(alpha_main)
    else:
        bar_seg2.set_data([], [])

    return (*ALL_DYNAMIC, *glow_artists, droplet_dot)


anim = FuncAnimation(fig, update, frames=N_FRAMES_TOTAL,
                     init_func=init, interval=50, blit=True)

plt.tight_layout()


def save_journal_figures(dpi=300):
    """Save the final hold-frame of each case as a white-background journal figure."""
    _cx = 0.5 * (box['xlo'] + box['xhi'])
    for case_idx in range(N_CASES):
        g = geo[case_idx]

        fig_j, ax_j = plt.subplots(figsize=(7, 7), facecolor='white')
        ax_j.set_facecolor('white')
        ax_j.set_xlim(*xlim)
        ax_j.set_ylim(*ylim)
        ax_j.set_aspect('equal')
        ax_j.tick_params(colors='#444444')
        for sp in ax_j.spines.values():
            sp.set_color('#bbbbbb')
        ax_j.set_xlabel('$x_1$', fontsize=13, color='#333333')
        ax_j.set_ylabel('$x_2$', fontsize=13, color='#333333')
        # ax_j.set_title(
        #     rf'Case {case_idx + 1}: {CASES[case_idx]["label"]}',
        #     color='#111111', fontsize=14, pad=10)

        # Background cost contour
        ZZ_bg_j = g['cost_grid'](XX_bg, YY_bg)
        levels_j = np.linspace(0, g['c_far'] * 1.5, 28)
        ax_j.contourf(XX_bg, YY_bg, ZZ_bg_j, levels=levels_j,
                      cmap='YlOrBr_r', alpha=0.22, zorder=1)
        ax_j.contour(XX_bg, YY_bg, ZZ_bg_j, levels=levels_j,
                     colors='#aaaaaa', alpha=0.30, linewidths=0.4, zorder=2)

        # Feasible box
        bw = box['xhi'] - box['xlo']
        bh_j = box['yhi'] - box['ylo']
        feas_j = patches.FancyBboxPatch(
            (box['xlo'], box['ylo']), bw, bh_j,
            boxstyle="square,pad=0", linewidth=2.2,
            edgecolor='#0969da', facecolor='#dbeafe', zorder=3)
        ax_j.add_patch(feas_j)
        ax_j.text(box['xhi'] + 0.07, box['yhi'] + 0.07, 'Feasible set',
                  color='#0969da', fontsize=10, ha='left', va='bottom', zorder=5)

        # x* marker
        ax_j.plot(*g['x_star'], 'o', color='#0969da', alpha=0.15,
                  markersize=26, zorder=6, markeredgewidth=0)
        ax_j.plot(*g['x_star'], 'o', color='#0550ae', markersize=7,
                  markeredgecolor='#000000', markeredgewidth=0.8, zorder=7)
        ax_j.annotate(r'$x^\star$',
                      xy=g['x_star'], xytext=g['x_star'] + np.array([0.28, -0.45]),
                      color='#0550ae', fontsize=10, zorder=8)

        # Nearest point
        x_near = g['x_near']
        ring_n = g['ellipse_pts'](g['t_near'])
        ax_j.plot(ring_n[:, 0], ring_n[:, 1], '--', color='#1a7f37',
                  lw=1.4, alpha=0.7, zorder=7)
        ax_j.plot(*x_near, '*', color='#1a7f37', markersize=15,
                  markeredgecolor='#000000', markeredgewidth=0.6, zorder=12)
        dx_n = 0.14 if x_near[0] < _cx else -0.60
        dy_n = -0.32 if x_near[1] > 0 else 0.18
        ax_j.text(x_near[0] + dx_n, x_near[1] + dy_n,
                  'Constrained\noptimum', color='#1a7f37', fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffffee',
                            edgecolor='#1a7f3755', lw=0.8), zorder=13)
        vec_n = scaled_vec(g['neg_grad_near'])
        ax_j.add_patch(FancyArrowPatch(
            posA=tuple(x_near), posB=tuple(x_near + vec_n),
            arrowstyle='->', color='#1a7f37', linewidth=2.2,
            mutation_scale=16, zorder=14))
        ax_j.text(*(x_near + vec_n * 1.22), r'$-\nabla f$',
                  color='#1a7f37', fontsize=10,
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffffee',
                            edgecolor='#1a7f3766', lw=0.8), zorder=15)

        # Farthest point
        x_far = g['x_far']
        ring_f = g['ellipse_pts'](g['t_far'])
        ax_j.plot(ring_f[:, 0], ring_f[:, 1], '--', color='#cf222e',
                  lw=1.4, alpha=0.7, zorder=7)
        ax_j.plot(*x_far, 'D', color='#cf222e', markersize=11,
                  markeredgecolor='#000000', markeredgewidth=0.6, zorder=12)
        dx_f = 0.14 if x_far[0] < _cx else -0.52
        dy_f = 0.18 if x_far[1] > 0 else -0.30
        ax_j.text(x_far[0] + dx_f, x_far[1] + dy_f,
                  'Farthest Q point', color='#cf222e', fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffffee',
                            edgecolor='#cf222e55', lw=0.8), zorder=13)
        vec_f = scaled_vec(g['neg_grad_far'])
        ax_j.add_patch(FancyArrowPatch(
            posA=tuple(x_far), posB=tuple(x_far + vec_f),
            arrowstyle='->', color='#cf222e', linewidth=2.2,
            mutation_scale=16, zorder=14))
        ax_j.text(*(x_far + vec_f * 1.22), r'$-\nabla f$',
                  color='#cf222e', fontsize=10,
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffffee',
                            edgecolor='#cf222e66', lw=0.8), zorder=15)

        # Good (antipodal) vertex
        x_good = g['x_good']
        ring_g = g['ellipse_pts'](g['t_good'])
        ax_j.plot(ring_g[:, 0], ring_g[:, 1], '--', color='#8250df',
                  lw=1.4, alpha=0.7, zorder=7)
        ax_j.fill(ring_g[:, 0], ring_g[:, 1], facecolor='#8250df',
                  alpha=0.08, zorder=4)
        ax_j.plot(*x_good, 'p', color='#8250df', markersize=14,
                  markeredgecolor='#000000', markeredgewidth=0.6, zorder=12)
        dx_g = 0.14 if x_good[0] < _cx else -0.72
        dy_g = 0.18 if x_good[1] > 0 else -0.42
        ax_j.text(x_good[0] - 0.5* dx_g, x_good[1] + dy_g,
                  'Antipodal\ngood vertex', color='#8250df', fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffffee',
                            edgecolor='#8250df55', lw=0.8), zorder=13)
        vec_g = scaled_vec(g['neg_grad_good'])
        ax_j.add_patch(FancyArrowPatch(
            posA=tuple(x_good), posB=tuple(x_good + vec_g),
            arrowstyle='->', color='#8250df', linewidth=2.2,
            mutation_scale=16, zorder=14))
        ax_j.text(*(x_good + vec_g * 1.22), r'$-\nabla f$',
                  color='#8250df', fontsize=10,
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffffee',
                            edgecolor='#8250df66', lw=0.8), zorder=15)

        # Cone between −∇f and H_2 direction
        ng_j     = g['neg_grad_good']
        ng_hat_j = ng_j / np.linalg.norm(ng_j) if np.linalg.norm(ng_j) > 1e-10 else ng_j
        h2_hat_j = g['h2_normal'] / np.linalg.norm(g['h2_normal'])
        fan_pts_j = [x_good]
        for _k in range(41):
            _s = _k / 40
            _dir = (1 - _s) * ng_hat_j + _s * h2_hat_j
            _dn = np.linalg.norm(_dir)
            if _dn > 1e-10:
                _dir /= _dn
            fan_pts_j.append(x_good + ARROW_LEN * 1.6 * _dir)
        ax_j.add_patch(plt.Polygon(
            np.array(fan_pts_j), closed=True,
            facecolor='#e3b341', edgecolor='none', alpha=0.28, zorder=3))
        # H_2 arrow (v → v_opp)
        vec_h2_j = scaled_vec(g['h2_normal'])
        ax_j.add_patch(FancyArrowPatch(
            posA=tuple(x_good), posB=tuple(x_good + vec_h2_j),
            arrowstyle='->', color='#e3b341', linewidth=2.2,
            mutation_scale=16, zorder=14))
        ax_j.text(*(x_good + vec_h2_j * 1.22), r'$t(\hat{v})$',
                  color='#e3b341', fontsize=10,
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffffee',
                            edgecolor='#e3b34166', lw=0.8), zorder=15)

        # P_+ half-box fill (light blue)
        ax_j.add_patch(plt.Polygon(
            g['p_plus_poly'], closed=True,
            facecolor='#388bfd', edgecolor='#388bfd', alpha=0.12, lw=0.0, zorder=4))

        # P_+ ∩ H_1 ∩ H_2 intersection
        _c0_j = np.array([0.5*(box['xlo']+box['xhi']), 0.5*(box['ylo']+box['yhi'])])
        _h_j  = np.array([0.5*(box['xhi']-box['xlo']), 0.5*(box['yhi']-box['ylo'])])
        _vnorm_j = np.sign(g['x_star'] - _c0_j) / _h_j
        _box_rect_j = np.array([[box['xlo'], box['ylo']], [box['xhi'], box['ylo']],
                                 [box['xhi'], box['yhi']], [box['xlo'], box['yhi']]])
        _clip_j = _sh_clip(_box_rect_j, _c0_j, _vnorm_j)               # clip to P_+
        if len(_clip_j) >= 3:
            _clip_j = _sh_clip(_clip_j, x_good, g['neg_grad_good'])    # clip to H_1
        if len(_clip_j) >= 3:
            _clip_j = _sh_clip(_clip_j, x_good, g['h2_normal'])        # clip to H_2
        if len(_clip_j) >= 3:
            ax_j.add_patch(plt.Polygon(
                _clip_j, closed=True,
                facecolor='#1a7f37', edgecolor='#1a7f37', alpha=0.35, lw=1.0, zorder=6))
            _label_pt = _clip_j.mean(axis=0)
            ax_j.text(_label_pt[0], _label_pt[1],
                      r'$\mathcal{P}_+\!\cap H_1\!\cap H_2$', color='#1a7f37', fontsize=8,
                      ha='center', va='center',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffffee',
                                edgecolor='#1a7f3755', lw=0.8), zorder=13)
        
        # Ray + exit point
        x_exit = g['x_exit']
        ax_j.plot([x_good[0], x_exit[0]], [x_good[1], x_exit[1]],
                  '--', color='#b68206', lw=1.8, alpha=0.85, zorder=8)
        ax_j.plot(*x_exit, 'X', color='#cf222e', markersize=13,
                  markeredgecolor='#000000', markeredgewidth=0.7, zorder=12)
        _edx = 0.12 if x_exit[0] < _cx else -0.52
        _edy = 0.14 if x_exit[1] < 0 else -0.46
        ax_j.text(x_exit[0] + _edx, x_exit[1] + _edy,
                  'Ray exit', color='#cf222e', fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffffee',
                            edgecolor='#cf222e55', lw=0.8), zorder=13)

        # Face labels: F_0 on exit face, F* on minimizer face
        _tol = 1e-4
        def _face_mid_norm(pt):
            if abs(pt[1] - box['yhi']) < _tol:
                return np.array([0.5*(box['xlo']+box['xhi']), box['yhi']]), np.array([0.,  1.])
            elif abs(pt[1] - box['ylo']) < _tol:
                return np.array([0.5*(box['xlo']+box['xhi']), box['ylo']]), np.array([0., -1.])
            elif abs(pt[0] - box['xhi']) < _tol:
                return np.array([box['xhi'], 0.5*(box['ylo']+box['yhi'])]), np.array([ 1., 0.])
            else:
                return np.array([box['xlo'], 0.5*(box['ylo']+box['yhi'])]), np.array([-1., 0.])

        _face_off = 0.24
        _fm0, _fn0 = _face_mid_norm(x_exit)
        ax_j.text(*(_fm0 + _face_off * _fn0), r'$F_0$',
                  color='#cf222e', fontsize=12, ha='center', va='center',
                  fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffffffee',
                            edgecolor='#cf222e66', lw=1.0), zorder=16)

        _fms, _fns = _face_mid_norm(x_near)
        ax_j.text(*(_fms + _face_off * _fns), r'$F^\star$',
                  color='#1a7f37', fontsize=12, ha='center', va='center',
                  fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.25', facecolor='#ffffffee',
                            edgecolor='#1a7f3766', lw=1.0), zorder=16)

        fig_j.tight_layout()
        fname = f'case_{case_idx + 1}_journal.png'
        fig_j.savefig(fname, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f'Saved {fname}')
        plt.close(fig_j)


save_journal_figures()

# anim.save('quadratic_animation.gif', writer='pillow', fps=25, dpi=120)
plt.show()