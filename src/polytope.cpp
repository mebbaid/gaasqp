#include "geoqp/internal.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace geoqp {
namespace internal {

Polytope::Polytope(const double* A, const double* b, int m, int n)
    : A_(A), b_(b), m_(m), n_(n) {}

std::vector<double> Polytope::find_initial_vertex() const {
    // Special case for 1D problems - find tightest constraint
    if (n_ == 1) {
        double min_bound = std::numeric_limits<double>::infinity();
        for (int i = 0; i < m_; ++i) {
            if (std::abs(A_[i]) > 1e-14) {
                // Constraint: a_i * x <= b_i
                // If a_i > 0: x <= b_i / a_i
                // If a_i < 0: x >= b_i / a_i (or -x <= -b_i / a_i)
                double bound = b_[i] / A_[i];
                if (A_[i] > 0 && bound < min_bound) {
                    min_bound = bound;
                }
            }
        }
        if (std::isfinite(min_bound)) {
            return std::vector<double>{min_bound};
        }
        // No upper bound found - check for feasibility at 0
        return std::vector<double>{0.0};
    }
    
    // Use a simple Phase-I LP approach to find any feasible vertex
    // We solve: min sum(slack_i) subject to A*x + slack >= b, slack >= 0
    // Here we use a simpler heuristic: start from origin and project
    
    // Try the "analytic center" approach for initial point
    std::vector<double> x(n_, 0.0);
    
    // Check if origin is feasible
    bool origin_feasible = true;
    for (int i = 0; i < m_; ++i) {
        double lhs = 0.0;
        for (int j = 0; j < n_; ++j) {
            lhs += A_[i * n_ + j] * x[j];
        }
        if (lhs > b_[i] + 1e-10) {
            origin_feasible = false;
            break;
        }
    }
    
    if (origin_feasible) {
        // Need to move to a vertex - find tight constraints
        std::vector<int> active = get_active_constraints(x, 1e-8);
        if (active.size() >= static_cast<size_t>(n_)) {
            return x; // Already at vertex
        }
    }
    
    // Use a simple heuristic: find the "Chebyshev center"
    // For now, use a simpler approach: find point that satisfies all constraints
    // by solving a feasibility LP
    
    // Simple heuristic: move along negative gradient of violated constraints
    std::vector<double> point(n_, 0.0);
    const int max_iter = 100;
    const double step_size = 0.1;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        double max_violation = 0.0;
        std::vector<double> direction(n_, 0.0);
        
        for (int i = 0; i < m_; ++i) {
            double lhs = 0.0;
            for (int j = 0; j < n_; ++j) {
                lhs += A_[i * n_ + j] * point[j];
            }
            double violation = lhs - b_[i];
            
            if (violation > max_violation) {
                max_violation = violation;
            }
            
            if (violation > 1e-8) {
                // Move in direction that reduces this constraint
                for (int j = 0; j < n_; ++j) {
                    direction[j] -= A_[i * n_ + j] * violation;
                }
            }
        }
        
        if (max_violation < 1e-8) {
            // Feasible point found - now move to vertex
            break;
        }
        
        // Normalize direction
        double norm = 0.0;
        for (int j = 0; j < n_; ++j) {
            norm += direction[j] * direction[j];
        }
        if (norm > 1e-14) {
            norm = std::sqrt(norm);
            for (int j = 0; j < n_; ++j) {
                point[j] += step_size * direction[j] / norm;
            }
        }
    }
    
    // Now find a vertex by identifying n linearly independent tight constraints
    // Use a greedy approach: start from point and "snap" to constraints
    
    std::vector<int> active_constraints;
    std::vector<double> vertex = point;
    
    // Identify constraints that are nearly active
    std::vector<std::pair<double, int>> distances;
    for (int i = 0; i < m_; ++i) {
        double lhs = 0.0;
        for (int j = 0; j < n_; ++j) {
            lhs += A_[i * n_ + j] * vertex[j];
        }
        double dist = b_[i] - lhs;
        if (dist >= -1e-8) {
            distances.push_back({dist, i});
        }
    }
    
    std::sort(distances.begin(), distances.end());
    
    // Select first n constraints to define vertex
    for (size_t i = 0; i < distances.size() && active_constraints.size() < static_cast<size_t>(n_); ++i) {
        active_constraints.push_back(distances[i].second);
    }
    
    if (active_constraints.size() < static_cast<size_t>(n_)) {
        throw std::runtime_error("Could not find initial vertex - polytope may be unbounded or degenerate");
    }
    
    // Solve for vertex defined by these n constraints
    // A_active * x = b_active
    std::vector<double> A_active(n_ * n_);
    std::vector<double> b_active(n_);
    
    for (int i = 0; i < n_; ++i) {
        int constraint_idx = active_constraints[i];
        b_active[i] = b_[constraint_idx];
        for (int j = 0; j < n_; ++j) {
            A_active[i * n_ + j] = A_[constraint_idx * n_ + j];
        }
    }
    
    try {
        LinearAlgebra::solve_linear_system(A_active.data(), b_active.data(), vertex.data(), n_);
    } catch (...) {
        // If system is singular, constraints are not independent
        // Fall back to original point
        return point;
    }
    
    // Verify vertex is feasible
    for (int i = 0; i < m_; ++i) {
        double lhs = 0.0;
        for (int j = 0; j < n_; ++j) {
            lhs += A_[i * n_ + j] * vertex[j];
        }
        if (lhs > b_[i] + 1e-6) {
            // Not feasible - return the point we had
            return point;
        }
    }
    
    return vertex;
}

std::vector<std::vector<double>> Polytope::enumerate_vertices() const {
    // Only for very small problems
    // This is a stub - full vertex enumeration is complex
    throw std::runtime_error("Vertex enumeration not implemented");
}

bool Polytope::is_vertex(const std::vector<double>& x, double tol) const {
    auto active = get_active_constraints(x, tol);
    return active.size() >= static_cast<size_t>(n_);
}

std::vector<int> Polytope::get_active_constraints(
    const std::vector<double>& x,
    double tol
) const {
    std::vector<int> active;
    for (int i = 0; i < m_; ++i) {
        double lhs = 0.0;
        for (int j = 0; j < n_; ++j) {
            lhs += A_[i * n_ + j] * x[j];
        }
        if (std::abs(lhs - b_[i]) <= tol) {
            active.push_back(i);
        }
    }
    return active;
}

std::vector<std::vector<double>> Polytope::get_neighbors(
    const std::vector<double>& vertex
) const {
    // Find adjacent vertices by pivoting (dropping one constraint)
    // This is a stub - full implementation would use simplex pivoting
    std::vector<std::vector<double>> neighbors;
    
    auto active = get_active_constraints(vertex, 1e-8);
    
    // Try dropping each active constraint and solving for new vertex
    for (size_t drop_idx = 0; drop_idx < active.size(); ++drop_idx) {
        // Create new active set without this constraint
        std::vector<int> new_active;
        for (size_t i = 0; i < active.size(); ++i) {
            if (i != drop_idx) {
                new_active.push_back(active[i]);
            }
        }
        
        if (new_active.size() < static_cast<size_t>(n_ - 1)) {
            continue; // Need at least n-1 constraints
        }
        
        // Need to add one more constraint - try each non-active constraint
        for (int i = 0; i < m_; ++i) {
            bool is_active = false;
            for (int a : new_active) {
                if (a == i) {
                    is_active = true;
                    break;
                }
            }
            if (!is_active) {
                std::vector<int> candidate_active = new_active;
                candidate_active.push_back(i);
                
                if (candidate_active.size() == static_cast<size_t>(n_)) {
                    // Try to solve for this vertex
                    std::vector<double> A_active(n_ * n_);
                    std::vector<double> b_active(n_);
                    std::vector<double> new_vertex(n_);
                    
                    for (int k = 0; k < n_; ++k) {
                        int constraint_idx = candidate_active[k];
                        b_active[k] = b_[constraint_idx];
                        for (int j = 0; j < n_; ++j) {
                            A_active[k * n_ + j] = A_[constraint_idx * n_ + j];
                        }
                    }
                    
                    try {
                        LinearAlgebra::solve_linear_system(A_active.data(), b_active.data(), 
                                                          new_vertex.data(), n_);
                        
                        // Check if feasible
                        bool feasible = true;
                        for (int j = 0; j < m_; ++j) {
                            double lhs = 0.0;
                            for (int k = 0; k < n_; ++k) {
                                lhs += A_[j * n_ + k] * new_vertex[k];
                            }
                            if (lhs > b_[j] + 1e-8) {
                                feasible = false;
                                break;
                            }
                        }
                        
                        if (feasible) {
                            neighbors.push_back(new_vertex);
                        }
                    } catch (...) {
                        // Singular system - skip
                    }
                }
            }
        }
    }
    
    return neighbors;
}

} // namespace internal
} // namespace geoqp
