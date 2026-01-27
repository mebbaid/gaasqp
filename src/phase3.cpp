#include "geoqp/internal.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace geoqp {
namespace internal {

Phase3Solver::Phase3Solver(
    const double* A, const double* b,
    const std::vector<double>& x_bar,
    int m, int n,
    const Config& config
) : A_(A), b_(b), x_bar_(x_bar), m_(m), n_(n), config_(config) {}

std::vector<double> Phase3Solver::project_onto_face(
    const std::vector<int>& active_set,
    std::vector<double>& lambda_out
) {
    // Project x_bar onto face defined by active constraints
    // Face: { x : A_active * x = b_active }
    // 
    // Projection: x* = argmin ||x - x_bar||^2 subject to A_active * x = b_active
    //
    // KKT system:
    // [ I      A_active^T ] [ x*     ]   [ x_bar    ]
    // [ A_active    0     ] [ lambda ] = [ b_active ]
    //
    // Solution via elimination:
    // x* = x_bar - A_active^T * (A_active * A_active^T)^{-1} * (A_active * x_bar - b_active)
    // lambda = (A_active * A_active^T)^{-1} * (A_active * x_bar - b_active)
    
    int k = static_cast<int>(active_set.size());
    
    if (k == 0) {
        // No active constraints - return x_bar
        lambda_out.clear();
        return x_bar_;
    }
    
    // Build A_active (k x n) and b_active (k)
    std::vector<double> A_active(k * n_);
    std::vector<double> b_active(k);
    
    for (int i = 0; i < k; ++i) {
        int constraint_idx = active_set[i];
        b_active[i] = b_[constraint_idx];
        for (int j = 0; j < n_; ++j) {
            A_active[i * n_ + j] = A_[constraint_idx * n_ + j];
        }
    }
    
    // Compute A_active * x_bar
    std::vector<double> Ax_bar(k);
    LinearAlgebra::matvec(A_active.data(), x_bar_.data(), Ax_bar.data(), k, n_);
    
    // Compute residual: A_active * x_bar - b_active
    std::vector<double> residual(k);
    for (int i = 0; i < k; ++i) {
        residual[i] = Ax_bar[i] - b_active[i];
    }
    
    // Compute Gram matrix: G = A_active * A_active^T (k x k)
    std::vector<double> G(k * k, 0.0);
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            double sum = 0.0;
            for (int l = 0; l < n_; ++l) {
                sum += A_active[i * n_ + l] * A_active[j * n_ + l];
            }
            G[i * k + j] = sum;
        }
    }
    
    // Solve G * lambda = residual
    lambda_out.resize(k);
    try {
        LinearAlgebra::solve_linear_system(G.data(), residual.data(), lambda_out.data(), k);
    } catch (const std::exception& e) {
        throw NumericalException("Failed to solve for Lagrange multipliers in Phase III: " + 
                                std::string(e.what()));
    }
    
    // Compute A_active^T * lambda
    std::vector<double> AT_lambda(n_, 0.0);
    for (int j = 0; j < n_; ++j) {
        for (int i = 0; i < k; ++i) {
            AT_lambda[j] += A_active[i * n_ + j] * lambda_out[i];
        }
    }
    
    // Compute x* = x_bar - A_active^T * lambda
    std::vector<double> x_star(n_);
    for (int j = 0; j < n_; ++j) {
        x_star[j] = x_bar_[j] - AT_lambda[j];
    }
    
    // Verify that x* satisfies the constraints (for debugging)
    if (config_.verbosity >= 2) {
        double max_error = 0.0;
        for (int i = 0; i < k; ++i) {
            double lhs = 0.0;
            for (int j = 0; j < n_; ++j) {
                lhs += A_active[i * n_ + j] * x_star[j];
            }
            double error = std::abs(lhs - b_active[i]);
            max_error = std::max(max_error, error);
        }
        // Log: Projection error: max_error
    }
    
    return x_star;
}

} // namespace internal
} // namespace geoqp
