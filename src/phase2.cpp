#include "geoqp/internal.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

namespace geoqp {
namespace internal {

Phase2Solver::Phase2Solver(
    const double* A, const double* b,
    const std::vector<double>& x_bar,
    int m, int n,
    const Config& config
) : A_(A), b_(b), x_bar_(x_bar), m_(m), n_(n), config_(config) {}

double Phase2Solver::compute_intersection_parameter(
    const std::vector<double>& v,
    int constraint_idx
) const {
    // Compute t where ray r(t) = v + t*(x_bar - v) hits constraint i
    // <a_i, v + t*(x_bar - v)> = b_i
    // <a_i, v> + t*<a_i, x_bar - v> = b_i
    // t = (b_i - <a_i, v>) / <a_i, x_bar - v>
    
    double av = 0.0;  // <a_i, v>
    double ad = 0.0;  // <a_i, x_bar - v>
    
    for (int j = 0; j < n_; ++j) {
        double a_ij = A_[constraint_idx * n_ + j];
        av += a_ij * v[j];
        ad += a_ij * (x_bar_[j] - v[j]);
    }
    
    // Check if ray is parallel to constraint
    if (std::abs(ad) < 1e-14) {
        // Ray parallel to constraint
        // Check if already on constraint
        if (std::abs(av - b_[constraint_idx]) < 1e-10) {
            return 0.0; // Already on constraint
        } else {
            return std::numeric_limits<double>::infinity(); // Never intersects
        }
    }
    
    double t = (b_[constraint_idx] - av) / ad;
    return t;
}

std::vector<int> Phase2Solver::find_active_face(
    const std::vector<double>& v
) {
    // Ray: r(t) = v + t*(x_bar - v) for t in [0, 1]
    
    // Collect all constraints and their intersection parameters
    std::vector<std::pair<double, int>> intersections;
    
    for (int i = 0; i < m_; ++i) {
        double t = compute_intersection_parameter(v, i);
        
        // Only consider intersections in (0, 1]
        // We use a small epsilon to avoid numerical issues at t=0
        if (t > 1e-10 && t <= 1.0 + 1e-10) {
            intersections.push_back({t, i});
        }
    }
    
    if (intersections.empty()) {
        // No intersections found in [0,1]
        // This could mean x_bar is actually in the polytope interior
        // Or we need to extend the ray beyond t=1
        // Check all constraints at larger t values
        for (int i = 0; i < m_; ++i) {
            double t = compute_intersection_parameter(v, i);
            if (t > 1.0 + 1e-10) {
                intersections.push_back({t, i});
            }
        }
        
        if (intersections.empty()) {
            // Still no intersections - return empty active set
            return std::vector<int>();
        }
    }
    
    // Find minimum t
    double t_min = std::numeric_limits<double>::infinity();
    for (const auto& pair : intersections) {
        if (pair.first < t_min) {
            t_min = pair.first;
        }
    }
    
    // Collect all constraints that are hit at t_min (within tolerance)
    std::vector<int> active_face;
    const double tol = 1e-8;
    
    for (const auto& pair : intersections) {
        if (std::abs(pair.first - t_min) <= tol) {
            active_face.push_back(pair.second);
        }
    }
    
    if (config_.verbosity >= 2) {
        // Log: Found active face with N constraints at t = t_min
    }
    
    return active_face;
}

} // namespace internal
} // namespace geoqp
