#include "geoqp/internal.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstring>

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
    std::vector<double> diff(n_);
    for (int i = 0; i < n_; ++i) {
        diff[i] = v[i] - x_bar_[i];
    }
    std::vector<double> Q_diff(n_);
    LinearAlgebra::matvec(Q_, diff.data(), Q_diff.data(), n_, n_);
    return LinearAlgebra::dot(diff.data(), Q_diff.data(), n_);
}

double Phase1Solver::compute_euclidean_distance_squared(
    const std::vector<double>& v) const {
    double dist_sq = 0.0;
    for (int i = 0; i < n_; ++i) {
        double d = v[i] - x_bar_[i];
        dist_sq += d * d;
    }
    return dist_sq;
}

bool Phase1Solver::is_good_vertex(
    const std::vector<double>& v,
    const std::vector<int>& active_set) const {
    // A vertex v is "good" if: a_i^T (x_bar - v) <= 0 for all active constraints i
    // This means the ray from v toward x_bar doesn't immediately leave the polytope
    for (int idx : active_set) {
        const double* a_row = A_ + idx * n_;
        double dot = 0.0;
        for (int j = 0; j < n_; ++j) {
            dot += a_row[j] * (x_bar_[j] - v[j]);
        }
        if (dot > config_.tolerance) {
            return false;
        }
    }
    return true;
}

std::vector<double> Phase1Solver::compute_vertex_from_active_set(
    const std::vector<int>& active_indices) const {
    if (static_cast<int>(active_indices.size()) != n_) {
        return std::vector<double>();
    }
    
    std::vector<double> A_active(n_ * n_);
    std::vector<double> b_active(n_);
    
    for (int i = 0; i < n_; ++i) {
        int idx = active_indices[i];
        b_active[i] = b_[idx];
        std::memcpy(A_active.data() + i * n_, A_ + idx * n_, n_ * sizeof(double));
    }
    
    std::vector<double> vertex(n_);
    try {
        LinearAlgebra::solve_linear_system(A_active.data(), b_active.data(), vertex.data(), n_);
    } catch (...) {
        return std::vector<double>();
    }
    
    // Verify feasibility
    for (int i = 0; i < m_; ++i) {
        const double* a_row = A_ + i * n_;
        double ax = LinearAlgebra::dot(a_row, vertex.data(), n_);
        if (ax > b_[i] + 1e-6) {
            return std::vector<double>();
        }
    }
    
    return vertex;
}

std::vector<double> Phase1Solver::find_good_vertex(
    const std::vector<double>& initial_vertex,
    int& iterations,
    int& vertices_explored
) {
    iterations = 0;
    vertices_explored = 0;
    
    Polytope polytope(A_, b_, m_, n_);
    
    // Get initial vertex's active set
    std::vector<int> current_active = polytope.get_active_constraints(initial_vertex, 1e-8);
    std::vector<double> current_vertex = initial_vertex;
    vertices_explored = 1;
    
    // Track best good vertex found
    std::vector<double> best_good_vertex;
    double best_distance_sq = -1.0;
    
    if (is_good_vertex(current_vertex, current_active)) {
        best_good_vertex = current_vertex;
        best_distance_sq = compute_euclidean_distance_squared(current_vertex);
    }
    
    // Simplex-like iteration: pivot toward vertices farther from x_bar
    const int max_iter = config_.max_simplex_iterations;
    
    while (iterations < max_iter) {
        iterations++;
        
        // Ensure we have enough active constraints
        if (current_active.size() < static_cast<size_t>(n_)) {
            break;
        }
        
        // Direction away from x_bar (we want to find farthest good vertex)
        std::vector<double> direction(n_);
        for (int j = 0; j < n_; ++j) {
            direction[j] = current_vertex[j] - x_bar_[j];
        }
        
        // Find best pivot: drop constraint where a_i^T d is most negative
        int best_drop = -1;
        double best_ratio = 0.0;
        
        for (int k = 0; k < n_ && k < static_cast<int>(current_active.size()); ++k) {
            int idx = current_active[k];
            const double* a_row = A_ + idx * n_;
            double a_dot_d = LinearAlgebra::dot(a_row, direction.data(), n_);
            
            if (a_dot_d < -config_.tolerance && a_dot_d < best_ratio) {
                best_ratio = a_dot_d;
                best_drop = k;
            }
        }
        
        if (best_drop < 0) {
            break;  // No improving pivot
        }
        
        // Build reduced active set (drop one constraint)
        std::vector<int> reduced_active;
        for (int k = 0; k < n_ && k < static_cast<int>(current_active.size()); ++k) {
            if (k != best_drop) {
                reduced_active.push_back(current_active[k]);
            }
        }
        
        // Find blocking constraint when moving away from x_bar
        double min_t = std::numeric_limits<double>::infinity();
        int blocking = -1;
        
        for (int c = 0; c < m_; ++c) {
            bool in_reduced = false;
            for (int a : reduced_active) {
                if (a == c) { in_reduced = true; break; }
            }
            if (in_reduced) continue;
            
            const double* a_row = A_ + c * n_;
            double a_dot_d = LinearAlgebra::dot(a_row, direction.data(), n_);
            double a_dot_x = LinearAlgebra::dot(a_row, current_vertex.data(), n_);
            
            if (a_dot_d > config_.tolerance) {
                double t = (b_[c] - a_dot_x) / a_dot_d;
                if (t > config_.tolerance && t < min_t) {
                    min_t = t;
                    blocking = c;
                }
            }
        }
        
        if (blocking < 0 || !std::isfinite(min_t)) {
            break;
        }
        
        // New active set
        std::vector<int> new_active = reduced_active;
        new_active.push_back(blocking);
        
        if (new_active.size() != static_cast<size_t>(n_)) {
            break;
        }
        
        std::vector<double> new_vertex = compute_vertex_from_active_set(new_active);
        if (new_vertex.empty()) {
            break;
        }
        
        vertices_explored++;
        
        double new_dist_sq = compute_euclidean_distance_squared(new_vertex);
        
        if (is_good_vertex(new_vertex, new_active)) {
            if (new_dist_sq > best_distance_sq) {
                best_good_vertex = new_vertex;
                best_distance_sq = new_dist_sq;
            }
        }
        
        // Continue only if making progress
        double current_dist_sq = compute_euclidean_distance_squared(current_vertex);
        if (new_dist_sq <= current_dist_sq + config_.tolerance) {
            break;
        }
        
        current_vertex = std::move(new_vertex);
        current_active = std::move(new_active);
    }
    
    if (best_good_vertex.empty()) {
        return initial_vertex;
    }
    return best_good_vertex;
}

bool Phase1Solver::is_direction_feasible(
    const std::vector<double>& v,
    const std::vector<double>& d,
    const std::vector<int>& active_set,
    double tol
) const {
    for (int idx : active_set) {
        const double* a_row = A_ + idx * n_;
        double dot = LinearAlgebra::dot(a_row, d.data(), n_);
        if (dot > tol) return false;
    }
    return true;
}

int Phase1Solver::find_most_violated_constraint(
    const std::vector<double>& v,
    const std::vector<double>& d,
    const std::vector<int>& active_set
) const {
    int most_violated = -1;
    double max_violation = 0.0;
    for (int idx : active_set) {
        const double* a_row = A_ + idx * n_;
        double dot = LinearAlgebra::dot(a_row, d.data(), n_);
        if (dot > max_violation) {
            max_violation = dot;
            most_violated = idx;
        }
    }
    return most_violated;
}

std::vector<double> Phase1Solver::pivot_to_neighbor(
    const std::vector<double>& v,
    int constraint_to_drop
) {
    return v;
}

} // namespace internal
} // namespace geoqp
