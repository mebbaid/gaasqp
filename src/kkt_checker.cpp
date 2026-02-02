#include "geoqp/internal.hpp"
#include <cmath>
#include <algorithm>

namespace geoqp {
namespace internal {

KKTChecker::KKTChecker(
    const double* Q, const double* g,
    const double* A, const double* b,
    int m, int n,
    double tol
) : Q_(Q), g_(g), A_(A), b_(b), m_(m), n_(n), tol_(tol) {}

bool KKTChecker::check(
    const std::vector<double>& x,
    const std::vector<double>& lambda,
    const std::vector<int>& active_set
) const {
    // Check dual feasibility: lambda >= 0 for active constraints
    for (size_t i = 0; i < active_set.size(); ++i) {
        int constraint_idx = active_set[i];
        if (lambda[constraint_idx] < -tol_) {
            return false;
        }
    }
    
    for (int i = 0; i < m_; ++i) {
        double lhs = 0.0;
        for (int j = 0; j < n_; ++j) {
            lhs += A_[i * n_ + j] * x[j];
        }
        if (lhs > b_[i] + tol_) {
            return false;
        }
    }
    
    for (int idx : active_set) {
        double lhs = 0.0;
        for (int j = 0; j < n_; ++j) {
            lhs += A_[idx * n_ + j] * x[j];
        }
        if (std::abs(lhs - b_[idx]) > tol_) {
            return false;
        }
    }

    std::vector<double> Qx(n_);
    LinearAlgebra::matvec(Q_, x.data(), Qx.data(), n_, n_);
    
    std::vector<double> grad(n_);
    for (int j = 0; j < n_; ++j) {
        grad[j] = Qx[j] + g_[j];
    }
    
    std::vector<double> dual_contribution(n_, 0.0);
    for (size_t i = 0; i < active_set.size(); ++i) {
        int constraint_idx = active_set[i];
        for (int j = 0; j < n_; ++j) {
            dual_contribution[j] += lambda[constraint_idx] * A_[constraint_idx * n_ + j];
        }
    }
    
    double residual = 0.0;
    double grad_norm = 0.0;
    for (int j = 0; j < n_; ++j) {
        double diff = grad[j] + dual_contribution[j];  // KKT: Q*x + g + A^T*λ = 0
        residual += diff * diff;
        grad_norm += grad[j] * grad[j];
    }
    residual = std::sqrt(residual);
    grad_norm = std::sqrt(grad_norm);
    
    // Use relative tolerance: residual / (1 + ||gradient||) <= tol
    // This is more robust than absolute tolerance
    double relative_residual = residual / (1.0 + grad_norm);
    double scaled_tol = std::sqrt(tol_);  // Relax tolerance slightly
    
    return relative_residual <= scaled_tol;
}

double KKTChecker::compute_residual(
    const std::vector<double>& x,
    const std::vector<double>& lambda,
    const std::vector<int>& active_set
) const {
    std::vector<double> Qx(n_);
    LinearAlgebra::matvec(Q_, x.data(), Qx.data(), n_, n_);
    
    std::vector<double> grad(n_);
    for (int j = 0; j < n_; ++j) {
        grad[j] = Qx[j] + g_[j];
    }
    
    std::vector<double> dual_contribution(n_, 0.0);
    for (size_t i = 0; i < active_set.size(); ++i) {
        int constraint_idx = active_set[i];
        for (int j = 0; j < n_; ++j) {
            dual_contribution[j] += lambda[constraint_idx] * A_[constraint_idx * n_ + j];
        }
    }
    
    double residual = 0.0;
    for (int j = 0; j < n_; ++j) {
        double diff = grad[j] + dual_contribution[j];  // KKT: Q*x + g + A^T*λ = 0
        residual += diff * diff;
    }
    
    return std::sqrt(residual);
}

} // namespace internal
} // namespace geoqp
