#include "geoqp/internal.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace geoqp {
namespace internal {

Projector::Projector(
    const double* Q, const double* A, const double* b,
    const std::vector<double>& x_bar,
    const std::vector<double>& Q_chol,
    int m, int n,
    const Config& config
) : Q_(Q), A_(A), b_(b), x_bar_(x_bar), Q_chol_(Q_chol), m_(m), n_(n), config_(config) {}

std::vector<double> Projector::project_onto_face(
    const std::vector<int>& active_set,
    std::vector<double>& lambda_out
) {
    // Minimize (1/2) x^T Q x + g^T x subject to A_active * x = b_active
    // 
    // Using the KKT conditions:
    //   Q x + A^T λ = -g
    //   A x = b
    //
    // Since x_bar = -Q^{-1} g, we have -g = Q x_bar
    // So: Q x + A^T λ = Q x_bar
    //     x = x_bar - Q^{-1} A^T λ
    //
    // Substituting into A x = b:
    //     A (x_bar - Q^{-1} A^T λ) = b
    //     A x_bar - A Q^{-1} A^T λ = b
    //     (A Q^{-1} A^T) λ = A x_bar - b
    //
    // So: λ = (A Q^{-1} A^T)^{-1} (A x_bar - b)
    //     x = x_bar - Q^{-1} A^T λ
    
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
    std::vector<double> S(k * k);         // Schur complement: A Q^{-1} A^T
    std::vector<double> Q_inv_AT(n_ * k); // Q^{-1} A^T (n x k matrix)
    std::vector<double> x_star(n_);
    
    // Build A_active and b_active
    for (int i = 0; i < k; ++i) {
        const int constraint_idx = active_set[i];
        b_active[i] = b_[constraint_idx];
        const double* A_row = A_ + constraint_idx * n_;
        double* A_active_row = A_active.data() + i * n_;
        std::memcpy(A_active_row, A_row, n_ * sizeof(double));
    }
    
    // Compute A_active * x_bar
    LinearAlgebra::matvec(A_active.data(), x_bar_.data(), Ax_bar.data(), k, n_);
    
    // Compute residual: A_active * x_bar - b_active
    for (int i = 0; i < k; ++i) {
        residual[i] = Ax_bar[i] - b_active[i];
    }
    
    // Compute Q^{-1} A^T by solving Q * (Q^{-1} A^T) = A^T for each column
    // Q^{-1} A^T is n x k, and A^T's j-th column is the j-th row of A_active
    for (int j = 0; j < k; ++j) {
        // Get j-th column of A^T (= j-th row of A_active)
        std::vector<double> a_col(n_);
        for (int i = 0; i < n_; ++i) {
            a_col[i] = A_active[j * n_ + i];
        }
        
        // Solve Q * y = a_col for y using pre-computed Cholesky if available
        std::vector<double> y(n_);
        
        if (!Q_chol_.empty()) {
            LinearAlgebra::solve_cholesky(Q_chol_.data(), a_col.data(), y.data(), n_);
        } else {
            // Fallback to general solve
            std::vector<double> Q_copy(n_ * n_);
            std::memcpy(Q_copy.data(), Q_, n_ * n_ * sizeof(double));
            
            try {
                LinearAlgebra::solve_linear_system(Q_copy.data(), a_col.data(), y.data(), n_);
            } catch (const std::exception& e) {
                throw NumericalException("Failed to compute Q^{-1} A^T in Phase III: " + 
                                        std::string(e.what()));
            }
        }
        
        // Store as j-th column of Q^{-1} A^T
        for (int i = 0; i < n_; ++i) {
            Q_inv_AT[i * k + j] = y[i];
        }
    }
    
    // Compute Schur complement: S = A * (Q^{-1} A^T) = A_active * Q_inv_AT
    // S is k x k
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            double sum = 0.0;
            for (int l = 0; l < n_; ++l) {
                sum += A_active[i * n_ + l] * Q_inv_AT[l * k + j];
            }
            S[i * k + j] = sum;
        }
    }
    
    // Solve S * lambda = residual
    lambda_out.resize(k);
    try {
        LinearAlgebra::solve_linear_system(S.data(), residual.data(), lambda_out.data(), k);
    } catch (const std::exception& e) {
        throw NumericalException("Failed to solve for Lagrange multipliers in Phase III: " + 
                                std::string(e.what()));
    }
    
    // Compute x* = x_bar - Q^{-1} A^T * lambda
    // Q^{-1} A^T * lambda = sum over j of lambda[j] * (Q^{-1} A^T)[:, j]
    for (int i = 0; i < n_; ++i) {
        double correction = 0.0;
        for (int j = 0; j < k; ++j) {
            correction += Q_inv_AT[i * k + j] * lambda_out[j];
        }
        x_star[i] = x_bar_[i] - correction;
    }
    
    return x_star;
}

} // namespace internal
} // namespace geoqp
