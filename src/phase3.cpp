#include "geoqp/internal.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>

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
    // x* = x_bar - A_active^T * (A_active * A_active^T)^{-1} * (A_active * x_bar - b_active)
    
    const int k = static_cast<int>(active_set.size());
    
    if (k == 0) {
        lambda_out.clear();
        return x_bar_;
    }
    
    // Pre-allocate all needed arrays
    std::vector<double> A_active(k * n_);
    std::vector<double> b_active(k);
    std::vector<double> Ax_bar(k);
    std::vector<double> residual(k);
    std::vector<double> G(k * k);
    std::vector<double> AT_lambda(n_);
    std::vector<double> x_star(n_);
    
    // Build A_active and b_active
    for (int i = 0; i < k; ++i) {
        const int constraint_idx = active_set[i];
        b_active[i] = b_[constraint_idx];
        const double* A_row = A_ + constraint_idx * n_;
        double* A_active_row = A_active.data() + i * n_;
        std::memcpy(A_active_row, A_row, n_ * sizeof(double));
    }
    
    // Compute A_active * x_bar using optimized matvec
    LinearAlgebra::matvec(A_active.data(), x_bar_.data(), Ax_bar.data(), k, n_);
    
    // Compute residual: A_active * x_bar - b_active
    for (int i = 0; i < k; ++i) {
        residual[i] = Ax_bar[i] - b_active[i];
    }
    
    // Compute Gram matrix: G = A_active * A_active^T (k x k)
    // Optimize: compute only upper triangle, then fill lower
    for (int i = 0; i < k; ++i) {
        const double* row_i = A_active.data() + i * n_;
        for (int j = i; j < k; ++j) {
            const double* row_j = A_active.data() + j * n_;
            double sum = LinearAlgebra::dot(row_i, row_j, n_);
            G[i * k + j] = sum;
            if (i != j) {
                G[j * k + i] = sum; // Symmetric
            }
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
    std::fill(AT_lambda.begin(), AT_lambda.end(), 0.0);
    for (int i = 0; i < k; ++i) {
        const double lambda_i = lambda_out[i];
        const double* A_row = A_active.data() + i * n_;
        for (int j = 0; j < n_; ++j) {
            AT_lambda[j] += A_row[j] * lambda_i;
        }
    }
    
    // Compute x* = x_bar - A_active^T * lambda
    for (int j = 0; j < n_; ++j) {
        x_star[j] = x_bar_[j] - AT_lambda[j];
    }
    
    return x_star;
}

} // namespace internal
} // namespace geoqp
