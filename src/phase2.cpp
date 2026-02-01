#include "geoqp/internal.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <set>
#include <cstring>
#include <iostream>

namespace geoqp {
namespace internal {

Phase2Solver::Phase2Solver(
    const double* Q, const double* A, const double* b,
    const std::vector<double>& x_bar,
    int m, int n,
    const Config& config
) : Q_(Q), A_(A), b_(b), x_bar_(x_bar), m_(m), n_(n), config_(config) {
    // Pre-compute Cholesky factorization of Q for fast projections
    Q_chol_.resize(n_ * n_);
    try {
        LinearAlgebra::cholesky(Q_, Q_chol_.data(), n_);
    } catch (...) {
        // If Cholesky fails, we'll use general solver
        Q_chol_.clear();
    }
}

double Phase2Solver::compute_intersection_parameter(
    const std::vector<double>& v,
    int constraint_idx
) const {
    // Compute t where ray r(t) = v + t*(x_bar - v) hits constraint i
    // <a_i, v + t*(x_bar - v)> = b_i
    // t = (b_i - <a_i, v>) / <a_i, x_bar - v>
    
    const double* a_row = A_ + constraint_idx * n_;
    double av = 0.0;  // <a_i, v>
    double ad = 0.0;  // <a_i, x_bar - v>
    
    // Unrolled for performance
    int j = 0;
    for (; j + 3 < n_; j += 4) {
        av += a_row[j] * v[j] + a_row[j+1] * v[j+1] +
              a_row[j+2] * v[j+2] + a_row[j+3] * v[j+3];
        ad += a_row[j] * (x_bar_[j] - v[j]) + a_row[j+1] * (x_bar_[j+1] - v[j+1]) +
              a_row[j+2] * (x_bar_[j+2] - v[j+2]) + a_row[j+3] * (x_bar_[j+3] - v[j+3]);
    }
    for (; j < n_; ++j) {
        av += a_row[j] * v[j];
        ad += a_row[j] * (x_bar_[j] - v[j]);
    }
    
    if (std::abs(ad) < 1e-14) {
        return (std::abs(av - b_[constraint_idx]) < 1e-10) ? 
               0.0 : std::numeric_limits<double>::infinity();
    }
    
    return (b_[constraint_idx] - av) / ad;
}

std::vector<int> Phase2Solver::find_initial_face(
    const std::vector<double>& v
) {
    // Ray: r(t) = v + t*(x_bar - v) for t in [0, 1]
    // Find first blocking face (smallest t >= 0 where we hit a constraint)
    // Note: t=0 means v is already on the constraint
    
    // First, collect constraints that v is on (t ≈ 0)
    std::vector<int> active_at_v;
    for (int i = 0; i < m_; ++i) {
        const double* a_row = A_ + i * n_;
        double av = LinearAlgebra::dot(a_row, v.data(), n_);
        if (std::abs(av - b_[i]) < 1e-8) {
            active_at_v.push_back(i);
        }
    }
    
    // If v is on constraints and x_bar violates them, those are the blocking constraints
    std::vector<int> blocking_at_v;
    for (int i : active_at_v) {
        const double* a_row = A_ + i * n_;
        double a_xbar = LinearAlgebra::dot(a_row, x_bar_.data(), n_);
        if (a_xbar > b_[i] + 1e-8) {
            // x_bar violates this constraint - it's blocking immediately
            blocking_at_v.push_back(i);
        }
    }
    
    if (!blocking_at_v.empty()) {
        // The initial face is the constraints that v is on and x_bar violates
        return blocking_at_v;
    }
    
    // Otherwise, find first blocking constraint along the ray (t > 0)
    double t_min = std::numeric_limits<double>::infinity();
    
    // Single pass to find minimum t
    for (int i = 0; i < m_; ++i) {
        double t = compute_intersection_parameter(v, i);
        // We want positive t (moving from v toward x_bar)
        // t should be in (0, inf) - if t > 1, x_bar is feasible for this constraint
        if (t > 1e-10 && t < t_min) {
            t_min = t;
        }
    }
    
    // If t_min > 1, then x_bar is feasible (no blocking before reaching x_bar)
    // But we should have checked this earlier! Let's verify x_bar feasibility
    if (t_min > 1.0 + 1e-8) {
        // Check if x_bar is actually feasible
        bool x_bar_feasible = true;
        for (int i = 0; i < m_; ++i) {
            const double* a_row = A_ + i * n_;
            double ax = LinearAlgebra::dot(a_row, x_bar_.data(), n_);
            if (ax > b_[i] + config_.tolerance) {
                x_bar_feasible = false;
                break;
            }
        }
        
        if (x_bar_feasible) {
            return std::vector<int>(); // x_bar is optimal
        }
        
        // x_bar is not feasible but t_min > 1 - this means v is not a good starting point
        // Fall through to use t_min anyway (clamp to t <= something reasonable)
    }
    
    if (!std::isfinite(t_min)) {
        // No intersection found at all - this shouldn't happen if v is in the polytope
        return std::vector<int>();
    }
    
    // Collect all constraints at t_min (handles degeneracy)
    std::vector<int> initial_face;
    initial_face.reserve(n_);
    
    const double tol = 1e-8;
    for (int i = 0; i < m_; ++i) {
        double t = compute_intersection_parameter(v, i);
        if (std::abs(t - t_min) <= tol) {
            initial_face.push_back(i);
        }
    }
    
    return initial_face;
}

double Phase2Solver::compute_projection_distance_sq(
    const std::vector<int>& face
) const {
    // Project x_bar onto face and return squared distance
    // x* = x_bar - A_F^T (A_F A_F^T)^{-1} (A_F x_bar - b_F)
    
    const int k = static_cast<int>(face.size());
    
    if (k == 0) {
        // Empty face means projection to x_bar - only valid if x_bar is feasible
        for (int i = 0; i < m_; ++i) {
            const double* a_row = A_ + i * n_;
            double ax = LinearAlgebra::dot(a_row, x_bar_.data(), n_);
            if (ax > b_[i] + config_.tolerance) {
                return std::numeric_limits<double>::infinity(); // x_bar infeasible
            }
        }
        return 0.0; // x_bar is feasible and optimal
    }
    
    // Build A_F (k x n) and b_F (k)
    std::vector<double> A_F(k * n_);
    std::vector<double> b_F(k);
    
    for (int i = 0; i < k; ++i) {
        const int idx = face[i];
        b_F[i] = b_[idx];
        std::memcpy(A_F.data() + i * n_, A_ + idx * n_, n_ * sizeof(double));
    }
    
    // Compute A_F * x_bar
    std::vector<double> Ax(k);
    LinearAlgebra::matvec(A_F.data(), x_bar_.data(), Ax.data(), k, n_);
    
    // residual = A_F * x_bar - b_F
    for (int i = 0; i < k; ++i) {
        Ax[i] -= b_F[i];
    }
    
    // Compute Gram matrix G = A_F * A_F^T (k x k)
    std::vector<double> G(k * k);
    for (int i = 0; i < k; ++i) {
        const double* row_i = A_F.data() + i * n_;
        for (int j = i; j < k; ++j) {
            const double* row_j = A_F.data() + j * n_;
            double sum = LinearAlgebra::dot(row_i, row_j, n_);
            G[i * k + j] = sum;
            G[j * k + i] = sum; // Symmetric
        }
    }
    
    // Solve G * lambda = residual
    std::vector<double> lambda(k);
    try {
        LinearAlgebra::solve_linear_system(G.data(), Ax.data(), lambda.data(), k);
    } catch (...) {
        return std::numeric_limits<double>::infinity(); // Singular - invalid face
    }
    
    // Compute projection: x* = x_bar - A_F^T * lambda
    std::vector<double> x_star(n_);
    std::memcpy(x_star.data(), x_bar_.data(), n_ * sizeof(double));
    
    for (int i = 0; i < k; ++i) {
        const double lambda_i = lambda[i];
        const double* a_row = A_F.data() + i * n_;
        for (int j = 0; j < n_; ++j) {
            x_star[j] -= a_row[j] * lambda_i;
        }
    }
    
    // Check dual feasibility: all lambda >= 0
    for (int i = 0; i < k; ++i) {
        if (lambda[i] < -config_.tolerance) {
            return std::numeric_limits<double>::infinity(); // Not optimal face
        }
    }
    
    // Check primal feasibility of x_star
    int max_violation_idx = -1;
    double max_violation = 0.0;
    for (int i = 0; i < m_; ++i) {
        const double* a_row = A_ + i * n_;
        double ax = LinearAlgebra::dot(a_row, x_star.data(), n_);
        double violation = ax - b_[i];
        if (violation > max_violation) {
            max_violation = violation;
            max_violation_idx = i;
        }
        if (ax > b_[i] + config_.tolerance) {
            if (config_.verbosity >= 3) {
                std::cout << "  Face {";
                for (int idx : face) std::cout << idx << " ";
                std::cout << "}: violates constraint " << i << " by " << violation << "\n";
            }
            return std::numeric_limits<double>::infinity(); // Infeasible projection
        }
    }
    
    // Compute ||x* - x_bar||^2
    double dist_sq = 0.0;
    for (int j = 0; j < n_; ++j) {
        double d = x_star[j] - x_bar_[j];
        dist_sq += d * d;
    }
    
    if (config_.verbosity >= 3) {
        std::cout << "  Face {";
        for (int idx : face) std::cout << idx << " ";
        std::cout << "}: feasible, dist_sq=" << dist_sq << "\n";
    }
    
    return dist_sq;
}

std::vector<std::vector<int>> Phase2Solver::get_neighbor_faces(
    const std::vector<int>& face
) const {
    std::vector<std::vector<int>> neighbors;
    const int k = static_cast<int>(face.size());
    
    // Create set for fast lookup
    std::set<int> face_set(face.begin(), face.end());
    
    // Neighbors by dropping one constraint (if k > 0)
    if (k > 0) {
        for (int i = 0; i < k; ++i) {
            std::vector<int> neighbor;
            neighbor.reserve(k - 1);
            for (int j = 0; j < k; ++j) {
                if (i != j) neighbor.push_back(face[j]);
            }
            neighbors.push_back(std::move(neighbor));
        }
    }
    
    // Neighbors by adding one constraint (if k < n)
    if (k < n_) {
        for (int c = 0; c < m_; ++c) {
            if (face_set.find(c) == face_set.end()) {
                std::vector<int> neighbor = face;
                neighbor.push_back(c);
                std::sort(neighbor.begin(), neighbor.end());
                neighbors.push_back(std::move(neighbor));
            }
        }
    }
    
    return neighbors;
}

// Compute projection and lambdas for a face
// Returns true if feasible, fills x_star and lambda
bool Phase2Solver::compute_projection_and_lambda(
    const std::vector<int>& face,
    std::vector<double>& x_star,
    std::vector<double>& lambda
) const {
    const int k = static_cast<int>(face.size());
    x_star = x_bar_;
    lambda.resize(k);
    
    if (k == 0) {
        // Empty face - check if x_bar is feasible
        for (int i = 0; i < m_; ++i) {
            const double* a_row = A_ + i * n_;
            double ax = LinearAlgebra::dot(a_row, x_bar_.data(), n_);
            if (ax > b_[i] + config_.tolerance) {
                return false;
            }
        }
        return true;
    }
    
    // Build A_F and b_F
    std::vector<double> A_F(k * n_);
    std::vector<double> b_F(k);
    
    for (int i = 0; i < k; ++i) {
        const int idx = face[i];
        b_F[i] = b_[idx];
        std::memcpy(A_F.data() + i * n_, A_ + idx * n_, n_ * sizeof(double));
    }
    
    // Compute residual
    std::vector<double> Ax(k);
    LinearAlgebra::matvec(A_F.data(), x_bar_.data(), Ax.data(), k, n_);
    for (int i = 0; i < k; ++i) Ax[i] -= b_F[i];
    
    // Compute Gram matrix
    std::vector<double> G(k * k);
    for (int i = 0; i < k; ++i) {
        const double* row_i = A_F.data() + i * n_;
        for (int j = i; j < k; ++j) {
            const double* row_j = A_F.data() + j * n_;
            double sum = LinearAlgebra::dot(row_i, row_j, n_);
            G[i * k + j] = sum;
            G[j * k + i] = sum;
        }
    }
    
    // Solve for lambda
    try {
        LinearAlgebra::solve_linear_system(G.data(), Ax.data(), lambda.data(), k);
    } catch (...) {
        return false;
    }
    
    // Compute x_star = x_bar - A_F^T * lambda
    for (int i = 0; i < k; ++i) {
        const double lambda_i = lambda[i];
        const double* a_row = A_F.data() + i * n_;
        for (int j = 0; j < n_; ++j) {
            x_star[j] -= a_row[j] * lambda_i;
        }
    }
    
    // Check primal feasibility
    for (int i = 0; i < m_; ++i) {
        const double* a_row = A_ + i * n_;
        double ax = LinearAlgebra::dot(a_row, x_star.data(), n_);
        if (ax > b_[i] + config_.tolerance) {
            return false;
        }
    }
    
    return true;
}

std::vector<int> Phase2Solver::find_optimal_face(
    const std::vector<int>& initial_face,
    int& iterations
) {
    iterations = 0;
    
    if (initial_face.empty()) {
        return initial_face;
    }
    
    std::vector<int> current_face = initial_face;
    const int max_iter = config_.max_iterations;
    
    while (iterations < max_iter) {
        iterations++;
        
        std::vector<double> x_star, lambda;
        bool feasible = compute_projection_and_lambda(current_face, x_star, lambda);
        
        if (feasible) {
            // Check if any lambda is negative - only those constraints can potentially be dropped
            bool improved = false;
            
            for (size_t i = 0; i < current_face.size(); ++i) {
                if (lambda[i] < -config_.tolerance) {
                    // Try dropping this constraint
                    std::vector<int> smaller_face;
                    for (size_t j = 0; j < current_face.size(); ++j) {
                        if (i != j) smaller_face.push_back(current_face[j]);
                    }
                    
                    std::vector<double> x2, lambda2;
                    if (compute_projection_and_lambda(smaller_face, x2, lambda2)) {
                        // Check if distance is smaller
                        double d1 = 0.0, d2 = 0.0;
                        for (int j = 0; j < n_; ++j) {
                            double diff1 = x_star[j] - x_bar_[j];
                            double diff2 = x2[j] - x_bar_[j];
                            d1 += diff1 * diff1;
                            d2 += diff2 * diff2;
                        }
                        
                        if (d2 < d1 - config_.tolerance) {
                            current_face = std::move(smaller_face);
                            improved = true;
                            break;
                        }
                    }
                }
            }
            
            if (!improved) {
                // All lambdas >= 0 or no improvement found - optimal
                if (config_.verbosity >= 3) {
                    std::cout << "  Optimal face with " << current_face.size() << " constraints\n";
                }
                return current_face;
            }
        } else {
            // Infeasible projection - add all violated constraints
            std::set<int> face_set(current_face.begin(), current_face.end());
            std::vector<int> violated;
            
            for (int i = 0; i < m_; ++i) {
                if (face_set.find(i) != face_set.end()) continue;
                
                const double* a_row = A_ + i * n_;
                double ax = LinearAlgebra::dot(a_row, x_star.data(), n_);
                if (ax > b_[i] + config_.tolerance) {
                    violated.push_back(i);
                }
            }
            
            if (violated.empty()) {
                if (config_.verbosity >= 2) {
                    std::cout << "  Warning: no violated constraint but projection infeasible\n";
                }
                return current_face;
            }
            
            for (int c : violated) {
                current_face.push_back(c);
            }
            std::sort(current_face.begin(), current_face.end());
            
            if (config_.verbosity >= 3) {
                std::cout << "  Added " << violated.size() << " violated constraints\n";
            }
        }
    }
    
    if (config_.verbosity >= 1) {
        std::cout << "Warning: Face exploration reached max iterations\n";
    }
    
    return current_face;
}

} // namespace internal
} // namespace geoqp
