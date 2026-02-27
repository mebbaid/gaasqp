import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path

# ------------------------------------------------------------
# Utility: Q-metric projection by brute force boundary sampling
# ------------------------------------------------------------
def q_projection_to_polygon(H, x_u, poly_pts, samples=1000):
    pts = []

    for i in range(len(poly_pts)):
        p1 = poly_pts[i]
        p2 = poly_pts[(i+1) % len(poly_pts)]
        t_vals = np.linspace(0, 1, samples//len(poly_pts))
        for t in t_vals:
            pts.append((1-t)*p1 + t*p2)

    pts = np.array(pts)
    diffs = pts - x_u
    vals = np.einsum('ij,jk,ik->i', diffs, H, diffs)
    return pts[np.argmin(vals)]

# ------------------------------------------------------------
# Plot function
# ------------------------------------------------------------
def plot_qp_with_constraints(ax, H, q, poly_pts, v, title):

    # Grid
    x = np.linspace(-8, 8, 400)
    y = np.linspace(-8, 8, 400)
    X, Y = np.meshgrid(x, y)

    Z = 0.5*(H[0,0]*X**2 + 2*H[0,1]*X*Y + H[1,1]*Y**2) + q[0]*X + q[1]*Y
    ax.contour(X, Y, Z, levels=40, cmap='viridis', alpha=0.6)

    # Polygon
    polygon = Polygon(poly_pts, closed=True, fill=True,
                      edgecolor='red', facecolor='red', alpha=0.2, linewidth=2)
    ax.add_patch(polygon)

    # Unconstrained minimizer
    x_u = -np.linalg.solve(H, q)

    # Newton direction from vertex v
    grad_v = H @ v + q
    nu = -np.linalg.solve(H, grad_v)

    # Ray
    ray = np.array([v + t*nu for t in np.linspace(0, 3, 200)])
    ax.plot(ray[:,0], ray[:,1], 'k--', linewidth=2, label='Newton Ray')

    # Q-projection
    x_proj = q_projection_to_polygon(H, x_u, poly_pts)

    # Plot points
    ax.plot(x_u[0], x_u[1], 'bx', markersize=10, label='x_u')
    ax.plot(v[0], v[1], 'ko', label='Vertex v')
    ax.plot(x_proj[0], x_proj[1], 'go', markersize=8, label='Q-Projection')

    # Direction arrow
    ax.arrow(v[0], v[1], nu[0], nu[1],
             head_width=0.4, length_includes_head=True,
             color='black')

    # Formatting
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize='small')

# ------------------------------------------------------------
# Skewed polytope (counterexample-style)
# ------------------------------------------------------------
poly_pts = np.array([
    [2, -1],
    [6, 0],
    [5, 4],
    [1, 3],
    [-1, 1]
])

v = poly_pts[2]   # choose starting vertex

# ------------------------------------------------------------
# Different Hessian cases
# ------------------------------------------------------------
q = np.array([0.0, 0.0])   # x_u = (0,0)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Mild conditioning
H1 = np.array([[2.0, 0.0], [0.0, 1.0]])
plot_qp_with_constraints(axes[0,0], H1, q, poly_pts, v,
                         "Well-Conditioned")

# 2. Rotated ellipses
H2 = np.array([[2.0, 1.5], [1.5, 3.0]])
plot_qp_with_constraints(axes[0,1], H2, q, poly_pts, v,
                         "Rotated Ellipses")

# 3. Ill-conditioned rotated
H3 = np.array([[8.0, 2.5], [2.5, 1.0]])
plot_qp_with_constraints(axes[1,0], H3, q, poly_pts, v,
                         "Ill-Conditioned Rotated")

# 4. Extremely anisotropic
H4 = np.array([[10.0, 3.0], [3.0, 1.0]])
plot_qp_with_constraints(axes[1,1], H4, q, poly_pts, poly_pts[1],
                         "Extreme Anisotropy")

plt.tight_layout()
plt.show()