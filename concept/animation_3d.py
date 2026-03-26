"""
Quadratic Cost + Feasible Cube — 3D Animation
==============================================
3D analogue of the 2D animation:
  - Quadratic cost  f(x) = ½(x−x*)ᵀ H (x−x*)  with rotated PD Hessian
  - Feasible set: axis-aligned cube  [-1.5, 1.5]³
  - Expanding ellipsoid level sets sweep outward from x*
  - Phase 1: first contact with cube  → GREEN star  (constrained optimum)
  - Phase 2: last contact             → RED diamond  (farthest cube point)
  - At each contact: −∇f arrow (FancyArrowPatch3D via mpl_toolkits)
  - Loops over 3 differently-rotated Hessians, fades between cases

Dependencies: numpy, matplotlib, scipy
    pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial.transform import Rotation

# ══════════════════════════════════════════════════════════════
# Arrow3D helper  (FancyArrowPatch projected into 3D)
# ══════════════════════════════════════════════════════════════
class Arrow3D(FancyArrowPatch):
    """A 3-D arrow drawn as a FancyArrowPatch projected onto the 2-D canvas."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def set_verts3d(self, xs, ys, zs):
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)

    def draw(self, renderer):
        self.do_3d_projection()
        super().draw(renderer)

# ══════════════════════════════════════════════════════════════
# 1.  Problem definitions
# ══════════════════════════════════════════════════════════════
def make_H(theta_deg, phi_deg, lam_max=50.0, lam_mid=8.0, lam_min=1.0):
    """3-D rotated PD Hessian: H = R diag(λ) Rᵀ"""
    r = Rotation.from_euler('zy', [theta_deg, phi_deg], degrees=True)
    R = r.as_matrix()
    D = np.diag([lam_max, lam_mid, lam_min])
    return R @ D @ R.T

CASES = [
    dict(label='θ=30°,φ=20°',   H=make_H(30,  20),  x_star=np.array([-2.8,  4.0,  1.5])),
    dict(label='θ=90°,φ=90°',  H=make_H(90,  90),  x_star=np.array([-2.8,  4.0,  1.5])),
    dict(label='θ=160°,φ=-30°', H=make_H(160, -30), x_star=np.array([-2.8,  4.0,  1.5])),
]

box = dict(xlo=-1.5, xhi=1.5, ylo=-1.5, yhi=1.5, zlo=-1.5, zhi=1.5)

NF_P1   = 100
NF_P2   = 80
NF_HOLD = 40
NF_FADE = 25
NF_CASE = NF_P1 + NF_P2 + NF_HOLD + NF_FADE

N_CASES        = len(CASES)
N_FRAMES_TOTAL = NF_CASE * N_CASES

# ══════════════════════════════════════════════════════════════
# 2.  Pre-compute per-case geometry
# ══════════════════════════════════════════════════════════════
NFACE = 120   # grid resolution per face for finding near/far

def sample_cube_faces(box, n=NFACE):
    """Return (M,3) array of points uniformly sampled on all 6 cube faces."""
    lo = np.array([box['xlo'], box['ylo'], box['zlo']])
    hi = np.array([box['xhi'], box['yhi'], box['zhi']])
    pts = []
    for fixed_dim in range(3):
        dims = [d for d in range(3) if d != fixed_dim]
        for fixed_val in (lo[fixed_dim], hi[fixed_dim]):
            a = np.linspace(lo[dims[0]], hi[dims[0]], n)
            b = np.linspace(lo[dims[1]], hi[dims[1]], n)
            AA, BB = np.meshgrid(a, b)
            block = np.zeros((n*n, 3))
            block[:, fixed_dim]  = fixed_val
            block[:, dims[0]]    = AA.ravel()
            block[:, dims[1]]    = BB.ravel()
            pts.append(block)
    return np.vstack(pts)

CUBE_PTS = sample_cube_faces(box)

def compute_case(H, x_star):
    eigvals, eigvecs = np.linalg.eigh(H)
    Hinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    def cost(x):
        d = x - x_star
        return 0.5 * float(d @ H @ d)

    def ellipsoid_surface(t, n=38):
        """Return (n,n,3) array of surface points for ellipsoid at scale t."""
        u = np.linspace(0, 2*np.pi, n)
        v = np.linspace(0,   np.pi, n)
        uu, vv = np.meshgrid(u, v)
        sphere = np.stack([np.cos(uu)*np.sin(vv),
                           np.sin(uu)*np.sin(vv),
                           np.cos(vv)], axis=-1)          # (n,n,3)
        pts = x_star + t * (Hinv_sqrt @ sphere.reshape(-1,3).T).T
        return pts.reshape(n, n, 3)

    # Cost at every cube-face sample
    diffs  = CUBE_PTS - x_star           # (M,3)
    costs  = 0.5 * np.einsum('mi,ij,mj->m', diffs, H, diffs)

    near_idx = int(np.argmin(costs))
    far_idx  = int(np.argmax(costs))
    x_near   = CUBE_PTS[near_idx]
    x_far    = CUBE_PTS[far_idx]
    c_near   = float(costs[near_idx])
    c_far    = float(costs[far_idx])
    t_near   = np.sqrt(2 * c_near)
    t_far    = np.sqrt(2 * c_far)
    t_end    = t_far * 1.08

    neg_grad_near = -(H @ (x_near - x_star))
    neg_grad_far  = -(H @ (x_far  - x_star))

    t_p1   = np.linspace(0.0,    t_near, NF_P1)
    t_p2   = np.linspace(t_near, t_end,  NF_P2)
    t_hold = np.full(NF_HOLD, t_end)
    t_fade = np.full(NF_FADE, t_end)
    t_all  = np.concatenate([t_p1, t_p2, t_hold, t_fade])

    return dict(
        H=H, x_star=x_star,
        ellipsoid_surface=ellipsoid_surface,
        x_near=x_near, t_near=t_near, c_near=c_near,
        x_far=x_far,  t_far=t_far,   c_far=c_far,
        t_end=t_end, t_all=t_all,
        neg_grad_near=neg_grad_near,
        neg_grad_far=neg_grad_far,
    )

print('Pre-computing geometry …', end=' ', flush=True)
geo = [compute_case(c['H'], c['x_star']) for c in CASES]
print('done.')

ARROW_LEN = 0.85

def scaled_vec(vec):
    n = np.linalg.norm(vec)
    return (ARROW_LEN / n) * vec if n > 1e-10 else np.zeros(3)

# ══════════════════════════════════════════════════════════════
# 3.  Cube geometry helpers
# ══════════════════════════════════════════════════════════════
def cube_corners(b):
    xlo,xhi = b['xlo'],b['xhi']
    ylo,yhi = b['ylo'],b['yhi']
    zlo,zhi = b['zlo'],b['zhi']
    return np.array([
        [xlo,ylo,zlo],[xhi,ylo,zlo],[xhi,yhi,zlo],[xlo,yhi,zlo],
        [xlo,ylo,zhi],[xhi,ylo,zhi],[xhi,yhi,zhi],[xlo,yhi,zhi],
    ])

CORNERS = cube_corners(box)
CUBE_EDGES = [(0,1),(1,2),(2,3),(3,0),
              (4,5),(5,6),(6,7),(7,4),
              (0,4),(1,5),(2,6),(3,7)]

FACE_INDICES = [
    [0,1,2,3],  # bottom  z=lo
    [4,5,6,7],  # top     z=hi
    [0,1,5,4],  # front   y=lo
    [2,3,7,6],  # back    y=hi
    [0,3,7,4],  # left    x=lo
    [1,2,6,5],  # right   x=hi
]

# ══════════════════════════════════════════════════════════════
# 4.  Figure setup
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(8, 8), facecolor='#0d1117')
ax  = fig.add_subplot(111, projection='3d', facecolor='#0d1117')

ax.set_xlim(-3.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_zlim(-2.5, 2.5)
ax.set_xlabel('$x_1$', color='#8b949e', fontsize=12, labelpad=4)
ax.set_ylabel('$x_2$', color='#8b949e', fontsize=12, labelpad=4)
ax.set_zlabel('$x_3$', color='#8b949e', fontsize=12, labelpad=4)
ax.tick_params(colors='#8b949e', labelsize=8)
ax.xaxis.pane.fill = False;  ax.yaxis.pane.fill = False;  ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#1e2a38')
ax.yaxis.pane.set_edgecolor('#1e2a38')
ax.zaxis.pane.set_edgecolor('#1e2a38')
ax.grid(True, color='#1e2a38', linewidth=0.5)
ax.set_title('Quadratic cost — 3D ellipsoid sweeping cube',
             color='#e6edf3', fontsize=13, pad=10)
ax.view_init(elev=22, azim=-50)

# ── Static: cube wireframe ────────────────────────────────────
for e in CUBE_EDGES:
    p0, p1 = CORNERS[e[0]], CORNERS[e[1]]
    ax.plot3D(*zip(p0, p1), color='#58a6ff', alpha=0.55, lw=1.4)

# ── Static: cube semi-transparent faces ───────────────────────
face_verts = [[CORNERS[i] for i in fi] for fi in FACE_INDICES]
cube_faces = Poly3DCollection(face_verts,
    facecolor='#58a6ff', alpha=0.04,
    edgecolor='#58a6ff', linewidth=0.5)
ax.add_collection3d(cube_faces)

# ── Static: x* marker ────────────────────────────────────────
x_star_all = CASES[0]['x_star']
xstar_dot, = ax.plot3D(*x_star_all, 'o', color='#cae8ff',
                       markersize=8, markeredgecolor='#ffffff',
                       markeredgewidth=0.8, zorder=10)
ax.text(x_star_all[0]+0.1, x_star_all[1]+0.1, x_star_all[2]+0.25,
        r'$\bar{x}$', color='#cae8ff', fontsize=11)

# ══════════════════════════════════════════════════════════════
# 5.  Dynamic artists
# ══════════════════════════════════════════════════════════════

# Ellipsoid surface (updated each frame)
_dummy_surf = np.zeros((38,38,3))
ellipsoid_surf = [ax.plot_surface(
    _dummy_surf[...,0], _dummy_surf[...,1], _dummy_surf[...,2],
    color='white', alpha=0.0, linewidth=0, antialiased=True, zorder=5)]

# Nearest contact
near_dot, = ax.plot3D([], [], [], '*', color='#3fb950', markersize=13,
                      markeredgecolor='#ffffff', markeredgewidth=0.5, zorder=15)
near_arrow = Arrow3D([0,0],[0,0],[0,0],
    mutation_scale=14, lw=2.0, arrowstyle='->', color='#3fb950', zorder=16)
ax.add_artist(near_arrow)
near_arrow.set_visible(False)

# Farthest contact
far_dot, = ax.plot3D([], [], [], 'D', color='#f85149', markersize=9,
                     markeredgecolor='#ffffff', markeredgewidth=0.5, zorder=15)
far_arrow = Arrow3D([0,0],[0,0],[0,0],
    mutation_scale=14, lw=2.0, arrowstyle='->', color='#f85149', zorder=16)
ax.add_artist(far_arrow)
far_arrow.set_visible(False)

# Text overlays (axes-fraction coordinates via fig.text)
phase_txt  = fig.text(0.03, 0.96, '', color='#e6edf3',  fontsize=10,
                      fontfamily='monospace', va='top')
cost_txt   = fig.text(0.03, 0.92, '', color='#8b949e',  fontsize=9.5,
                      fontfamily='monospace', va='top')
case_txt   = fig.text(0.97, 0.96, '', color='#ffa657',  fontsize=11,
                      fontfamily='monospace', va='top', ha='right', fontweight='bold')
near_lbl   = fig.text(0.03, 0.10, '', color='#3fb950',  fontsize=9.5, va='top')
far_lbl    = fig.text(0.03, 0.06, '', color='#f85149',  fontsize=9.5, va='top')

# ══════════════════════════════════════════════════════════════
# 6.  Update function
# ══════════════════════════════════════════════════════════════

def update(frame):
    case_idx = min(frame // NF_CASE, N_CASES - 1)
    local_f  = frame % NF_CASE
    g        = geo[case_idx]

    t = g['t_all'][min(local_f, NF_CASE - 1)]

    in_p1    = local_f < NF_P1
    in_p2    = NF_P1 <= local_f < NF_P1 + NF_P2
    in_hold  = NF_P1 + NF_P2 <= local_f < NF_P1 + NF_P2 + NF_HOLD
    in_fade  = local_f >= NF_P1 + NF_P2 + NF_HOLD

    near_revealed = local_f >= NF_P1 - 1
    far_revealed  = local_f >= NF_P1 + NF_P2 - 1

    if in_fade:
        fade_frac  = (local_f - (NF_P1 + NF_P2 + NF_HOLD)) / NF_FADE
        alpha_main = max(1.0 - fade_frac * 1.5, 0.0)
    else:
        alpha_main = 1.0

    # ── ellipsoid surface ─────────────────────────────────────
    ellipsoid_surf[0].remove()
    if t > 0.05 and alpha_main > 0:
        S = g['ellipsoid_surface'](t)
        t_near = g['t_near'];  t_far = g['t_far']
        if t <= t_near:
            r = min(t / t_near, 1.0) if t_near > 0 else 1.0
            col = (0.10+0.15*r, 0.79+0.04*r, 0.63-0.31*r)
        else:
            r = min((t - t_near) / (t_far - t_near), 1.0) if t_far > t_near else 1.0
            col = (0.25+0.72*r, 0.73-0.41*r, 0.31)
        ellipsoid_surf[0] = ax.plot_surface(
            S[...,0], S[...,1], S[...,2],
            color=col, alpha=0.18 * alpha_main,
            linewidth=0, antialiased=True, zorder=5)
    else:
        ellipsoid_surf[0] = ax.plot_surface(
            _dummy_surf[...,0], _dummy_surf[...,1], _dummy_surf[...,2],
            color='white', alpha=0.0, linewidth=0)

    # ── text overlays ─────────────────────────────────────────
    c_now = 0.5 * t**2
    cost_txt.set_text(f'f  =  {c_now:.2f}')
    cost_txt.set_alpha(alpha_main)
    case_txt.set_text(f'Case {case_idx+1}/3: {CASES[case_idx]["label"]}')
    case_txt.set_alpha(alpha_main)

    if in_p1:
        phase_txt.set_text('Phase 1 — expanding to nearest contact …')
    elif in_p2:
        phase_txt.set_text('Phase 2 — expanding to farthest contact …')
    elif in_hold:
        phase_txt.set_text('Both contacts identified ✓')
    else:
        phase_txt.set_text('⟳  next case …')
    phase_txt.set_alpha(alpha_main)

    # ── nearest contact ───────────────────────────────────────
    if near_revealed and alpha_main > 0:
        flash_n = 1.0 + 0.4*np.exp(-0.12*(local_f-(NF_P1-1))) \
                      * abs(np.sin((local_f-(NF_P1-1))*0.9))
        xn = g['x_near']
        near_dot.set_data_3d([xn[0]], [xn[1]], [xn[2]])
        near_dot.set_markersize(13 * flash_n)
        near_dot.set_alpha(alpha_main)

        vec_n = scaled_vec(g['neg_grad_near'])
        near_arrow.set_verts3d(
            [xn[0], xn[0]+vec_n[0]],
            [xn[1], xn[1]+vec_n[1]],
            [xn[2], xn[2]+vec_n[2]])
        near_arrow.set_alpha(alpha_main)
        near_arrow.set_visible(True)

        near_lbl.set_text(f'x_near = ({xn[0]:.2f}, {xn[1]:.2f}, {xn[2]:.2f})'
                          r'   $-\nabla f$ shown')
        near_lbl.set_alpha(alpha_main)
    else:
        near_dot.set_data_3d([], [], [])
        near_arrow.set_visible(False)
        near_lbl.set_text('')

    # ── farthest contact ──────────────────────────────────────
    if far_revealed and alpha_main > 0:
        flash_f = 1.0 + 0.4*np.exp(-0.12*(local_f-(NF_P1+NF_P2-1))) \
                      * abs(np.sin((local_f-(NF_P1+NF_P2-1))*0.9))
        xf = g['x_far']
        far_dot.set_data_3d([xf[0]], [xf[1]], [xf[2]])
        far_dot.set_markersize(9 * flash_f)
        far_dot.set_alpha(alpha_main)

        vec_f = scaled_vec(g['neg_grad_far'])
        far_arrow.set_verts3d(
            [xf[0], xf[0]+vec_f[0]],
            [xf[1], xf[1]+vec_f[1]],
            [xf[2], xf[2]+vec_f[2]])
        far_arrow.set_alpha(alpha_main)
        far_arrow.set_visible(True)

        far_lbl.set_text(f'x_far  = ({xf[0]:.2f}, {xf[1]:.2f}, {xf[2]:.2f})'
                         r'   $-\nabla f$ shown')
        far_lbl.set_alpha(alpha_main)
    else:
        far_dot.set_data_3d([], [], [])
        far_arrow.set_visible(False)
        far_lbl.set_text('')

    # ── slow camera rotation ──────────────────────────────────
    # ax.view_init(elev=22, azim=-50 + frame * 0.10)

    return []


anim = FuncAnimation(fig, update, frames=N_FRAMES_TOTAL,
                     interval=45, blit=False)

plt.tight_layout()

# ─────────────────────────────────────────────────────────────
# Save / show
# ─────────────────────────────────────────────────────────────
# To save as GIF (requires Pillow):
#   anim.save('quadratic_3d.gif', writer='pillow', fps=22, dpi=120)
#
# To save as MP4 (requires ffmpeg):
#   anim.save('quadratic_3d.mp4', writer='ffmpeg', fps=22, dpi=150)

plt.show()