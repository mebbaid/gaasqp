import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap


def make_H(theta_deg, lam_max=50.0, lam_min=1.0):
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
                     init_func=init, interval=40, blit=True)

plt.tight_layout()

# anim.save('quadratic_animation.gif', writer='pillow', fps=25, dpi=120)
plt.show()