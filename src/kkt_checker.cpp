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
    // KKT conditions:
    // 1. Stationarity: Q*x + g = sum_i lambda_i * a_i (for active constraints)
    // 2. Primal feasibility: A*x <= b
    // 3. Dual feasibility: lambda >= 0
    // 4. Complementarity: lambda_i * (a_i^T*x - b_i) = 0
    
    // Check dual feasibility: lambda >= 0
    for (size_t i = 0; i < active_set.size(); ++i) {
        if (lambda[i] < -tol_) {
            return false;
        }
    }
    
    // Check primal feasibility: A*x <= b
    for (int i = 0; i < m_; ++i) {
        double lhs = 0.0;
        for (int j = 0; j < n_; ++j) {
            lhs += A_[i * n_ + j] * x[j];
        }
        if (lhs > b_[i] + tol_) {
            return false;
        }
    }
    
    // Check complementarity: for active constraints, a_i^T*x should equal b_i
    for (int idx : active_set) {
        double lhs = 0.0;
        for (int j = 0; j < n_; ++j) {
            lhs += A_[idx * n_ + j] * x[j];
        }
        if (std::abs(lhs - b_[idx]) > tol_) {
            return false;
        }
    }
    
    // Check stationarity: Q*x + g = A_active^T * lambda
    // Compute gradient: Q*x + g
    std::vector<double> Qx(n_);
    LinearAlgebra::matvec(Q_, x.data(), Qx.data(), n_, n_);
    
    std::vector<double> grad(n_);
    for (int j = 0; j < n_; ++j) {
        grad[j] = Qx[j] + g_[j];
    }
    
    // Compute A_active^T * lambda
    std::vector<double> dual_contribution(n_, 0.0);
    for (size_t i = 0; i < active_set.size(); ++i) {
        int constraint_idx = active_set[i];
        for (int j = 0; j < n_; ++j) {
            dual_contribution[j] += lambda[i] * A_[constraint_idx * n_ + j];
        }
    }
    
    // Check if grad = dual_contribution
    double residual = 0.0;
    for (int j = 0; j < n_; ++j) {
        double diff = grad[j] - dual_contribution[j];
        residual += diff * diff;
    }
    residual = std::sqrt(residual);
    
    return residual <= tol_;
}

double KKTChecker::compute_residual(
    const std::vector<double>& x,
    const std::vector<double>& lambda,
    const std::vector<int>& active_set
) const {
    // Compute stationarity residual: ||Q*x + g - A_active^T * lambda||
    
    // Compute gradient: Q*x + g
    std::vector<double> Qx(n_);
    LinearAlgebra::matvec(Q_, x.data(), Qx.data(), n_, n_);
    
    std::vector<double> grad(n_);
    for (int j = 0; j < n_; ++j) {
        grad[j] = Qx[j] + g_[j];
    }
    
    // Compute A_active^T * lambda
    std::vector<double> dual_contribution(n_, 0.0);
    for (size_t i = 0; i < active_set.size(); ++i) {
        int constraint_idx = active_set[i];
        for (int j = 0; j < n_; ++j) {
            dual_contribution[j] += lambda[i] * A_[constraint_idx * n_ + j];
        }
    }
    
    // Compute residual
    double residual = 0.0;
    for (int j = 0; j < n_; ++j) {
        double diff = grad[j] - dual_contribution[j];
        residual += diff * diff;
    }
    
    return std::sqrt(residual);
}

} // namespace internal
} // namespace geoqp
