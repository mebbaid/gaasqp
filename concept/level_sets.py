import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

def plot_qp_with_constraints(ax, H, q, center, title):
    # 1. Create Grid
    x = np.linspace(-7, 7, 200)
    y = np.linspace(-7, 7, 200)
    X, Y = np.meshgrid(x, y)
    
    # 2. Evaluate Objective: 0.5 * x.T * H * x + q.T * x
    # Note: Global minimum is at x* = -inv(H) @ q. 
    # With q=[0,0], global min is at (0,0).
    Z = 0.5 * (H[0,0]*X**2 + (H[0,1] + H[1,0])*X*Y + H[1,1]*Y**2) + q[0]*X + q[1]*Y
    
    # 3. Plot Contours
    cp = ax.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    ax.clabel(cp, inline=True, fontsize=8)
    
    # 4. Define Hexagonal Feasible Region (Convex Compact Polytope)
    # We place the hexagon at 'center' so it doesn't contain (0,0)
    hexagon = RegularPolygon(center, numVertices=6, radius=2, 
                            orientation=np.radians(30), 
                            facecolor='none', edgecolor='red', 
                            linewidth=2.5, label='Feasible Region (Hexagon)')
    ax.add_patch(hexagon)
    
    # Fill the feasible region
    # For visualization, we use the same hexagon logic to fill it
    fill_hex = RegularPolygon(center, numVertices=6, radius=2, 
                             orientation=np.radians(30), 
                             facecolor='red', alpha=0.15)
    ax.add_patch(fill_hex)

    # 5. Mark the Global Minimum (Infeasible)
    ax.plot(0, 0, 'bx', markersize=10, markeredgewidth=2, label='Global Min (Infeasible)')

    # Formatting
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize='small')

# --- Define Hessian Cases ---
q = np.array([0, 0]) # Global minimum at (0,0)
hex_center = (4, 3)  # Shifted so (0,0) is outside

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Case 1: Positive Definite (Standard Ellipses)
H1 = np.array([[2.0, 0.0], [0.0, 1.0]])
plot_qp_with_constraints(axes[0,0], H1, q, hex_center, "Pos. Definite: Touching at Edge/Vertex")

# Case 2: Positive Definite (Dense/Rotated)
H2 = np.array([[2.0, 1.2], [1.2, 2.0]])
plot_qp_with_constraints(axes[0,1], H2, q, hex_center, "Rotated: Axis Alignment Shifted")

# Case 3: Ill-Conditioned (Sharp Curvature)
H3 = np.array([[5.0, 0.0], [0.0, 0.2]])
plot_qp_with_constraints(axes[1,0], H3, q, hex_center, "Ill-Conditioned: Long Narrow Level Sets")

# Case 4: Positive Semidefinite (Valley)
H4 = np.array([[1.0, 1.0], [1.0, 1.0]])
plot_qp_with_constraints(axes[1,1], H4, q, hex_center, "Semidefinite: Optimal Set is a Line Segment")

plt.tight_layout()
plt.show()