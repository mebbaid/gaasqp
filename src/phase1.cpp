#include "geoqp/internal.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <set>

namespace geoqp {
namespace internal {

Phase1Solver::Phase1Solver(
    const double* Q, const double* A, const double* b,
    const std::vector<double>& x_bar,
    int m, int n,
    const Config& config
) : Q_(Q), A_(A), b_(b), x_bar_(x_bar), m_(m), n_(n), config_(config) {}

double Phase1Solver::compute_q_weighted_distance_squared(
    const std::vector<double>& v) const {
    // Compute (v - x_bar)^T Q (v - x_bar)
    // This is the squared Mahalanobis distance in the Q-metric
    
    std::vector<double> diff(n_);
    for (int i = 0; i < n_; ++i) {
        diff[i] = v[i] - x_bar_[i];
    }
    
    // Compute Q * diff
    std::vector<double> Q_diff(n_);
    LinearAlgebra::matvec(Q_, diff.data(), Q_diff.data(), n_, n_);
    
    // Compute diff^T * Q_diff
    return LinearAlgebra::dot(diff.data(), Q_diff.data(), n_);
}

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
    Polytope polytope(A_, b_, m_, n_);
    std::vector<int> active = polytope.get_active_constraints(v, 1e-8);
    
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
    
    // Move in the direction that improves toward x_bar_

    std::vector<double> direction(n_);
    for (int j = 0; j < n_; ++j) {
        direction[j] = x_bar_[j] - v[j];
    }
    
    double min_t = std::numeric_limits<double>::infinity();
    int new_constraint = -1;
    
    for (int i = 0; i < m_; ++i) {
        bool already_active = false;
        for (int idx : new_active) {
            if (idx == i) {
                already_active = true;
                break;
            }
        }
        if (already_active) continue;
        
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
        return v;
    }
    
    new_active.push_back(new_constraint);
    
    if (new_active.size() != static_cast<size_t>(n_)) {
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
        
        for (int i = 0; i < m_; ++i) {
            double lhs = 0.0;
            for (int j = 0; j < n_; ++j) {
                lhs += A_[i * n_ + j] * new_vertex[j];
            }
            if (lhs > b_[i] + 1e-6) {
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
    iterations = 0;
    
    // Special handling for 1D problems
    if (n_ == 1) {
        // Find all good vertices (boundary points where direction to x_bar is feasible)
        double max_dist = -1.0;
        double best_x = initial_vertex[0];
        
        for (int i = 0; i < m_; ++i) {
            if (std::abs(A_[i]) > 1e-14) {
                double x_on_boundary = b_[i] / A_[i];
                
                // Check feasibility
                bool feasible = true;
                for (int j = 0; j < m_; ++j) {
                    if (A_[j] * x_on_boundary > b_[j] + 1e-8) {
                        feasible = false;
                        break;
                    }
                }
                
                if (feasible) {
                    // Check if this is a good vertex (direction to x_bar is feasible)
                    double direction = x_bar_[0] - x_on_boundary;
                    bool is_good = (A_[i] * direction <= 1e-8);
                    
                    if (is_good) {
                        double dist = std::abs(x_on_boundary - x_bar_[0]);
                        if (dist > max_dist) {
                            max_dist = dist;
                            best_x = x_on_boundary;
                        }
                    }
                }
            }
        }
        return std::vector<double>{best_x};
    }
    
    // For higher dimensions: enumerate good vertices and select farthest
    // Pre-allocate to reduce dynamic allocations
    std::vector<std::vector<double>> good_vertices;
    good_vertices.reserve(std::min(100, m_)); // Reserve reasonable capacity
    
    std::set<std::vector<double>> visited;
    
    const int max_iter = config_.max_simplex_iterations;
    
    // Pre-allocate direction vector to avoid repeated allocations
    std::vector<double> direction(n_);
    
    // Start with initial vertex
    Polytope polytope(A_, b_, m_, n_);
    std::vector<double> current_vertex = initial_vertex;
    visited.insert(current_vertex);
    
    // Check if initial vertex is good
    for (int j = 0; j < n_; ++j) {
        direction[j] = x_bar_[j] - current_vertex[j];
    }
    std::vector<int> active = polytope.get_active_constraints(current_vertex, 1e-8);
    if (is_direction_feasible(current_vertex, direction, active, config_.tolerance)) {
        good_vertices.push_back(current_vertex);
    }
    
    // Explore neighboring vertices via simplex pivoting
    while (iterations < max_iter && !good_vertices.empty()) {
        bool found_new_vertex = false;
        
        for (int idx : active) {
            std::vector<double> next_vertex = pivot_to_neighbor(current_vertex, idx);
            
            // Check if this is a new vertex
            if (visited.find(next_vertex) == visited.end()) {
                visited.insert(next_vertex);
                found_new_vertex = true;
                
                // Check if this vertex is good (reuse direction vector)
                for (int j = 0; j < n_; ++j) {
                    direction[j] = x_bar_[j] - next_vertex[j];
                }
                std::vector<int> next_active = polytope.get_active_constraints(next_vertex, 1e-8);
                
                if (is_direction_feasible(next_vertex, direction, next_active, config_.tolerance)) {
                    good_vertices.push_back(next_vertex);
                }
                
                current_vertex = std::move(next_vertex); // Move instead of copy
                active = std::move(next_active);
                break;
            }
        }
        
        if (!found_new_vertex) {
            break;
        }
        
        iterations++;
    }
    
    // If no good vertices found via exploration, use modified simplex
    if (good_vertices.empty()) {
        current_vertex = initial_vertex;
        iterations = 0;
        
        while (iterations < max_iter) {
            for (int j = 0; j < n_; ++j) {
                direction[j] = x_bar_[j] - current_vertex[j];
            }
            
            active = polytope.get_active_constraints(current_vertex, 1e-8);
            
            if (is_direction_feasible(current_vertex, direction, active, config_.tolerance)) {
                good_vertices.push_back(current_vertex);
                break;
            }
            
            int violated = find_most_violated_constraint(current_vertex, direction, active);
            if (violated == -1) {
                break;
            }
            
            current_vertex = pivot_to_neighbor(current_vertex, violated);
            iterations++;
        }
    }
    
    if (good_vertices.empty()) {
        return initial_vertex;
    }
    
    // Select the farthest good vertex from x_bar using Q-weighted distance
    double max_distance_sq = -1.0;
    size_t farthest_idx = 0;
    
    for (size_t i = 0; i < good_vertices.size(); ++i) {
        // Compute Q-weighted squared distance: (v - x_bar)^T Q (v - x_bar)
        double dist_sq = compute_q_weighted_distance_squared(good_vertices[i]);
        if (dist_sq > max_distance_sq) {
            max_distance_sq = dist_sq;
            farthest_idx = i;
        }
    }
    
    if (config_.verbosity >= 1) {
        // Log: Selected vertex with Q-weighted distance = sqrt(max_distance_sq)
    }
    
    return std::move(good_vertices[farthest_idx]); // Move instead of copy
}

} // namespace internal
} // namespace geoqp
