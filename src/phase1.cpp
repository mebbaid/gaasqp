#include "geoqp/internal.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

namespace geoqp {
namespace internal {

Phase1Solver::Phase1Solver(
    const double* A, const double* b,
    const std::vector<double>& x_bar,
    int m, int n,
    const Config& config
) : A_(A), b_(b), x_bar_(x_bar), m_(m), n_(n), config_(config) {}

bool Phase1Solver::is_direction_feasible(
    const std::vector<double>& v,
    const std::vector<double>& d,
    const std::vector<int>& active_set,
    double tol
) const {
    // Direction d is feasible at v if <a_i, d> <= 0 for all active constraints
    for (int idx : active_set) {
        double dot_product = 0.0;
        for (int j = 0; j < n_; ++j) {
            dot_product += A_[idx * n_ + j] * d[j];
        }
        if (dot_product > tol) {
            return false;
        }
    }
    return true;
}

int Phase1Solver::find_most_violated_constraint(
    const std::vector<double>& v,
    const std::vector<double>& d,
    const std::vector<int>& active_set
) const {
    // Find constraint i in active_set where <a_i, d> is most positive
    int most_violated = -1;
    double max_violation = 0.0;
    
    for (int idx : active_set) {
        double dot_product = 0.0;
        for (int j = 0; j < n_; ++j) {
            dot_product += A_[idx * n_ + j] * d[j];
        }
        if (dot_product > max_violation) {
            max_violation = dot_product;
            most_violated = idx;
        }
    }
    
    return most_violated;
}

std::vector<double> Phase1Solver::pivot_to_neighbor(
    const std::vector<double>& v,
    int constraint_to_drop
) {
    // Find a neighboring vertex by dropping one constraint
    // Get current active constraints
    Polytope polytope(A_, b_, m_, n_);
    std::vector<int> active = polytope.get_active_constraints(v, 1e-8);
    
    // Remove the constraint to drop
    std::vector<int> new_active;
    for (int idx : active) {
        if (idx != constraint_to_drop) {
            new_active.push_back(idx);
        }
    }
    
    if (new_active.size() < static_cast<size_t>(n_ - 1)) {
        // Not enough constraints - this shouldn't happen at a vertex
        return v;
    }
    
    // We now have n-1 constraints. We need to find which new constraint becomes active
    // Move in the direction that improves toward x_bar
    
    std::vector<double> direction(n_);
    for (int j = 0; j < n_; ++j) {
        direction[j] = x_bar_[j] - v[j];
    }
    
    // Find the first constraint we hit when moving in this direction
    double min_t = std::numeric_limits<double>::infinity();
    int new_constraint = -1;
    
    for (int i = 0; i < m_; ++i) {
        // Skip if already in new_active
        bool already_active = false;
        for (int idx : new_active) {
            if (idx == i) {
                already_active = true;
                break;
            }
        }
        if (already_active) continue;
        
        // Compute intersection parameter: v + t*d hits constraint i
        // <a_i, v + t*d> = b_i
        // t = (b_i - <a_i, v>) / <a_i, d>
        
        double av = 0.0, ad = 0.0;
        for (int j = 0; j < n_; ++j) {
            av += A_[i * n_ + j] * v[j];
            ad += A_[i * n_ + j] * direction[j];
        }
        
        if (std::abs(ad) < 1e-14) {
            continue; // Direction parallel to constraint
        }
        
        double t = (b_[i] - av) / ad;
        
        if (t > 1e-10 && t < min_t) {
            min_t = t;
            new_constraint = i;
        }
    }
    
    if (new_constraint == -1) {
        // No new constraint found - problem may be unbounded
        // Return current vertex
        return v;
    }
    
    // Add new constraint to active set
    new_active.push_back(new_constraint);
    
    // Solve for new vertex defined by new_active constraints
    if (new_active.size() != static_cast<size_t>(n_)) {
        // Need exactly n constraints
        return v;
    }
    
    std::vector<double> A_active(n_ * n_);
    std::vector<double> b_active(n_);
    std::vector<double> new_vertex(n_);
    
    for (int k = 0; k < n_; ++k) {
        int constraint_idx = new_active[k];
        b_active[k] = b_[constraint_idx];
        for (int j = 0; j < n_; ++j) {
            A_active[k * n_ + j] = A_[constraint_idx * n_ + j];
        }
    }
    
    try {
        LinearAlgebra::solve_linear_system(A_active.data(), b_active.data(), 
                                          new_vertex.data(), n_);
        
        // Verify feasibility
        for (int i = 0; i < m_; ++i) {
            double lhs = 0.0;
            for (int j = 0; j < n_; ++j) {
                lhs += A_[i * n_ + j] * new_vertex[j];
            }
            if (lhs > b_[i] + 1e-6) {
                // Not feasible
                return v;
            }
        }
        
        return new_vertex;
    } catch (...) {
        return v;
    }
}

std::vector<double> Phase1Solver::find_good_vertex(
    const std::vector<double>& initial_vertex,
    int& iterations
) {
    std::vector<double> current_vertex = initial_vertex;
    iterations = 0;
    
    // For 1D problems, check if the initial vertex is already good
    // In 1D, a vertex with n=1 active constraint is by definition a boundary point
    if (n_ == 1) {
        // Just verify the vertex is on the boundary
        for (int i = 0; i < m_; ++i) {
            double lhs = A_[i] * current_vertex[0];
            if (std::abs(lhs - b_[i]) < 1e-8) {
                // This constraint is active - this is a valid vertex
                return current_vertex;
            }
        }
        // If no constraint is active, we're at an interior point (shouldn't happen)
        // Move to nearest boundary
        double min_dist = std::numeric_limits<double>::infinity();
        double best_x = current_vertex[0];
        for (int i = 0; i < m_; ++i) {
            if (std::abs(A_[i]) > 1e-14) {
                double x_on_boundary = b_[i] / A_[i];
                // Check if this is feasible
                bool feasible = true;
                for (int j = 0; j < m_; ++j) {
                    if (A_[j] * x_on_boundary > b_[j] + 1e-8) {
                        feasible = false;
                        break;
                    }
                }
                if (feasible) {
                    double dist = std::abs(x_on_boundary - x_bar_[0]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_x = x_on_boundary;
                    }
                }
            }
        }
        return std::vector<double>{best_x};
    }
    
    const int max_iter = config_.max_simplex_iterations;
    
    while (iterations < max_iter) {
        // Compute direction to x_bar
        std::vector<double> direction(n_);
        for (int j = 0; j < n_; ++j) {
            direction[j] = x_bar_[j] - current_vertex[j];
        }
        
        // Get active constraints
        Polytope polytope(A_, b_, m_, n_);
        std::vector<int> active = polytope.get_active_constraints(current_vertex, 1e-8);
        
        // Check if direction is feasible
        if (is_direction_feasible(current_vertex, direction, active, config_.tolerance)) {
            // Good vertex found!
            if (config_.verbosity >= 2) {
                // Vertex is good
            }
            break;
        }
        
        // Find most violated constraint
        int violated = find_most_violated_constraint(current_vertex, direction, active);
        
        if (violated == -1) {
            // No violated constraint found - shouldn't happen
            break;
        }
        
        // Pivot to neighbor by dropping this constraint
        std::vector<double> next_vertex = pivot_to_neighbor(current_vertex, violated);
        
        // Check if we made progress
        double dist_old = 0.0, dist_new = 0.0;
        for (int j = 0; j < n_; ++j) {
            double diff_old = current_vertex[j] - x_bar_[j];
            double diff_new = next_vertex[j] - x_bar_[j];
            dist_old += diff_old * diff_old;
            dist_new += diff_new * diff_new;
        }
        
        if (dist_new >= dist_old - 1e-10) {
            // Not making progress - might be cycling
            if (config_.verbosity >= 1) {
                // Warning about potential cycling
            }
            // Try to continue anyway
        }
        
        current_vertex = next_vertex;
        iterations++;
    }
    
    if (iterations >= max_iter) {
        if (config_.verbosity >= 1) {
            // Max iterations reached in Phase I
        }
    }
    
    return current_vertex;
}

} // namespace internal
} // namespace geoqp
