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
    
    // Pre-allocate to reduce allocations
    std::vector<std::pair<double, int>> intersections;
    intersections.reserve(m_); // Reserve max possible size
    
    // Single pass through constraints - compute all intersections
    for (int i = 0; i < m_; ++i) {
        double t = compute_intersection_parameter(v, i);
        
        // Only consider intersections in (0, 1]
        if (t > 1e-10 && t <= 1.0 + 1e-10) {
            intersections.emplace_back(t, i);
        }
    }
    
    if (intersections.empty()) {
        // No intersections in [0,1] - extend search
        for (int i = 0; i < m_; ++i) {
            double t = compute_intersection_parameter(v, i);
            if (t > 1.0 + 1e-10) {
                intersections.emplace_back(t, i);
            }
        }
        
        if (intersections.empty()) {
            return std::vector<int>();
        }
    }
    
    // Find minimum t with single pass
    double t_min = intersections[0].first;
    for (size_t i = 1; i < intersections.size(); ++i) {
        if (intersections[i].first < t_min) {
            t_min = intersections[i].first;
        }
    }
    
    // Collect all constraints at t_min
    std::vector<int> active_face;
    active_face.reserve(n_); // At most n constraints can be simultaneously active
    
    const double tol = 1e-8;
    for (const auto& pair : intersections) {
        if (std::abs(pair.first - t_min) <= tol) {
            active_face.push_back(pair.second);
        }
    }
    
    return active_face;
}

} // namespace internal
} // namespace geoqp
