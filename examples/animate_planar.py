#!/usr/bin/env python3
"""
Animated Visualization for 2D Planar QP Comparison

This script creates an animation showing how GeoQP, qpOASES, and OSQP
approach the solution from different starting points.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

def create_polytope_vertices():
    """Create vertices of the polytope."""
    candidates = [
        (2, 1),    # x=2, x+y=3
        (1, 2),    # y=2, x+y=3
        (2, -1),   # x=2, y=-1
        (-1, 2),   # x=-1, y=2
        (-1, 1),   # x=-1, x+y=0
        (1, -1),   # y=-1, x+y=0
    ]
    
    def is_feasible(x, y, tol=1e-6):
        return (x <= 2 + tol and y <= 2 + tol and 
                x >= -1 - tol and y >= -1 - tol and
                x + y <= 3 + tol and x + y >= -tol)
    
    vertices = [(x, y) for x, y in candidates if is_feasible(x, y)]
    cx = np.mean([v[0] for v in vertices])
    cy = np.mean([v[1] for v in vertices])
    vertices.sort(key=lambda v: np.arctan2(v[1] - cy, v[0] - cx))
    return vertices

def plot_level_sets(ax, x_bar, levels=[0.5, 1, 2, 4, 8]):
    """Plot circular level sets of the objective function."""
    theta = np.linspace(0, 2*np.pi, 100)
    for level in levels:
        r = np.sqrt(2 * level)
        x = x_bar[0] + r * np.cos(theta)
        y = x_bar[1] + r * np.sin(theta)
        ax.plot(x, y, 'b-', alpha=0.15, linewidth=0.5)

# Problem setup
x_bar = np.array([3.0, 3.0])
solution = np.array([1.5, 1.5])

# Define solver trajectories
# GeoQP: x_bar → good_vertex → ray intersection → solution
# The geometric approach traces from x_bar toward polytope
geoqp_trajectory = [
    x_bar,                      # Start at unconstrained minimum
    np.array([2.0, 2.0]),       # Ray hits corner (x=2, y=2)
    np.array([1.5, 1.5]),       # Face hop to optimal face, project
]

# qpOASES: Starts from a feasible point, walks along constraints
# Parametric active set method - starts inside and moves to boundary
qpoases_trajectory = [
    np.array([0.0, 0.0]),       # Start from origin (feasible)
    np.array([1.0, 1.0]),       # Move toward unconstrained min
    np.array([1.5, 1.5]),       # Hit optimal constraint, done
]

# OSQP: ADMM iterations - many small steps through interior
# Simulating ADMM behavior with intermediate points
osqp_trajectory = [
    np.array([0.0, 0.0]),       # Start from zero
]
# Generate smooth ADMM-like path toward solution
for i in range(1, 20):
    t = i / 19.0
    # ADMM typically has oscillation and gradual convergence
    damping = 1 - np.exp(-3 * t)
    oscillation = 0.1 * np.exp(-2 * t) * np.sin(8 * np.pi * t)
    point = (1 - damping) * np.array([0.0, 0.0]) + damping * solution
    point += oscillation * np.array([1.0, -1.0])
    osqp_trajectory.append(point)
osqp_trajectory.append(solution)

def create_animation():
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot static elements
    vertices = create_polytope_vertices()
    poly = Polygon(vertices, fill=True, facecolor='lightblue', 
                   edgecolor='navy', linewidth=2, alpha=0.4)
    ax.add_patch(poly)
    
    plot_level_sets(ax, x_bar, levels=[0.5, 1, 2, 4, 6, 8, 10])
    
    # Constraint lines
    x_range = np.linspace(-2, 4, 100)
    ax.plot(x_range, 3 - x_range, 'g--', alpha=0.3, linewidth=1)
    ax.plot(x_range, -x_range, 'g--', alpha=0.3, linewidth=1)
    
    # Key points
    ax.plot(*x_bar, 'r*', markersize=20, label='x_bar (unconstrained min)', zorder=10)
    ax.plot(*solution, 'g*', markersize=20, label='Optimal solution', zorder=10)
    
    # Annotations
    ax.annotate('x̄ = (3, 3)\nUnconstrained\nMinimum', xy=x_bar, 
                xytext=(3.3, 2.5), fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    ax.annotate('x* = (1.5, 1.5)\nOptimal', xy=solution,
                xytext=(0.3, 0.8), fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
    
    # Initialize trajectory lines and markers
    geoqp_line, = ax.plot([], [], 'r-', linewidth=3, alpha=0.8, label='GeoQP (geometric)')
    geoqp_marker, = ax.plot([], [], 'ro', markersize=15, zorder=15)
    
    qpoases_line, = ax.plot([], [], 'b-', linewidth=3, alpha=0.8, label='qpOASES (active set)')
    qpoases_marker, = ax.plot([], [], 'bs', markersize=12, zorder=15)
    
    osqp_line, = ax.plot([], [], 'm-', linewidth=2, alpha=0.6, label='OSQP (ADMM)')
    osqp_marker, = ax.plot([], [], 'm^', markersize=10, zorder=15)
    
    # Add iteration counter text
    iter_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlim(-2, 4.5)
    ax.set_ylim(-2, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x₁', fontsize=14)
    ax.set_ylabel('x₂', fontsize=14)
    ax.set_title('GeoQP vs qpOASES vs OSQP: Solver Trajectories\n(Geometric Face-Hopping vs Active Set vs ADMM)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    # Calculate max frames needed
    max_frames = max(len(geoqp_trajectory), len(qpoases_trajectory), len(osqp_trajectory))
    total_frames = max_frames + 10  # Extra frames to hold at end
    
    def interpolate_trajectory(trajectory, frame, total):
        """Interpolate position along trajectory at given frame."""
        n_points = len(trajectory)
        if n_points == 1:
            return trajectory[0], [trajectory[0]]
        
        # Map frame to trajectory progress
        progress = min(frame / (total - 10), 1.0) * (n_points - 1)
        idx = int(progress)
        frac = progress - idx
        
        if idx >= n_points - 1:
            return trajectory[-1], trajectory
        
        # Interpolate between points
        current = (1 - frac) * trajectory[idx] + frac * trajectory[idx + 1]
        
        # Return path up to current point
        path = trajectory[:idx+1] + [current]
        return current, path
    
    def animate(frame):
        # Pause at beginning
        if frame < 5:
            effective_frame = 0
        else:
            effective_frame = frame - 5
        
        # GeoQP trajectory
        geoqp_pos, geoqp_path = interpolate_trajectory(geoqp_trajectory, effective_frame, total_frames)
        geoqp_xs = [p[0] for p in geoqp_path]
        geoqp_ys = [p[1] for p in geoqp_path]
        geoqp_line.set_data(geoqp_xs, geoqp_ys)
        geoqp_marker.set_data([geoqp_pos[0]], [geoqp_pos[1]])
        
        # qpOASES trajectory
        qpoases_pos, qpoases_path = interpolate_trajectory(qpoases_trajectory, effective_frame, total_frames)
        qpoases_xs = [p[0] for p in qpoases_path]
        qpoases_ys = [p[1] for p in qpoases_path]
        qpoases_line.set_data(qpoases_xs, qpoases_ys)
        qpoases_marker.set_data([qpoases_pos[0]], [qpoases_pos[1]])
        
        # OSQP trajectory
        osqp_pos, osqp_path = interpolate_trajectory(osqp_trajectory, effective_frame, total_frames)
        osqp_xs = [p[0] for p in osqp_path]
        osqp_ys = [p[1] for p in osqp_path]
        osqp_line.set_data(osqp_xs, osqp_ys)
        osqp_marker.set_data([osqp_pos[0]], [osqp_pos[1]])
        
        # Update iteration counter
        geoqp_iter = min(len(geoqp_trajectory)-1, int(effective_frame / (total_frames-10) * (len(geoqp_trajectory)-1)))
        qpoases_iter = min(len(qpoases_trajectory)-1, int(effective_frame / (total_frames-10) * (len(qpoases_trajectory)-1)))
        osqp_iter = min(len(osqp_trajectory)-1, int(effective_frame / (total_frames-10) * (len(osqp_trajectory)-1)))
        
        iter_text.set_text(f'Iterations:\n'
                          f'  GeoQP:   {geoqp_iter:2d} (face hops)\n'
                          f'  qpOASES: {qpoases_iter:2d} (working set)\n'
                          f'  OSQP:    {osqp_iter:2d} (ADMM)')
        
        return geoqp_line, geoqp_marker, qpoases_line, qpoases_marker, osqp_line, osqp_marker, iter_text
    
    anim = FuncAnimation(fig, animate, frames=total_frames + 5, 
                         interval=100, blit=True, repeat=True)
    
    # Save animation
    print("Saving animation...")
    writer = PillowWriter(fps=10)
    anim.save('geoqp_solver_animation.gif', writer=writer, dpi=100)
    print("Animation saved to geoqp_solver_animation.gif")
    
    # Also save final frame as static image
    animate(total_frames + 4)
    fig.savefig('geoqp_solver_comparison.png', dpi=150, bbox_inches='tight')
    print("Static image saved to geoqp_solver_comparison.png")
    
    plt.show()

if __name__ == '__main__':
    create_animation()
