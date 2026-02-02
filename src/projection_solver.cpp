

#include "geoqp/internal.hpp"
#include <cstring>

namespace geoqp {
namespace internal {

bool ProjectionSolver::solve(
    const std::vector<int>& active_set,
    const double* Q,
    const double* A,
    const double* b,
    const std::vector<double>& x_bar,
    const std::vector<double>& Q_chol,
    int m,
    int n,
    std::vector<double>& x_star_out,
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
        x_star_out = x_bar;
        return true;
    }
    
    // Pre-allocate all needed arrays
    std::vector<double> A_active(k * n);
    std::vector<double> b_active(k);
    std::vector<double> residual(k);
    std::vector<double> Q_inv_AT(n * k); // Q^{-1} A^T (n x k matrix)
    std::vector<double> S(k * k);        // Schur complement
    
    // Build A_active and b_active
    for (int i = 0; i < k; ++i) {
        const int constraint_idx = active_set[i];
        b_active[i] = b[constraint_idx];
        const double* A_row = A + constraint_idx * n;
        std::memcpy(A_active.data() + i * n, A_row, n * sizeof(double));
    }
    
    // Compute residual: A_active * x_bar - b_active
    LinearAlgebra::matvec(A_active.data(), x_bar.data(), residual.data(), k, n);
    for (int i = 0; i < k; ++i) {
        residual[i] -= b_active[i];
    }
    
    // Compute Q^{-1} A^T by solving Q * y = a_col for each constraint
    for (int j = 0; j < k; ++j) {
        // Get j-th row of A_active as a column vector
        std::vector<double> a_col(n);
        for (int i = 0; i < n; ++i) {
            a_col[i] = A_active[j * n + i];
        }
        
        // Solve Q * y = a_col for y using pre-computed Cholesky if available
        std::vector<double> y(n);
        
        if (!Q_chol.empty()) {
            LinearAlgebra::solve_cholesky(Q_chol.data(), a_col.data(), y.data(), n);
        } else {
            // Fallback to general solve
            std::vector<double> Q_copy(n * n);
            std::memcpy(Q_copy.data(), Q, n * n * sizeof(double));
            
            try {
                LinearAlgebra::solve_linear_system(Q_copy.data(), a_col.data(), y.data(), n);
            } catch (...) {
                return false;
            }
        }
        
        // Store as j-th column of Q^{-1} A^T
        for (int i = 0; i < n; ++i) {
            Q_inv_AT[i * k + j] = y[i];
        }
    }
    
    // Compute Schur complement: S = A_active * Q^{-1} A^T
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            double sum = 0.0;
            for (int l = 0; l < n; ++l) {
                sum += A_active[i * n + l] * Q_inv_AT[l * k + j];
            }
            S[i * k + j] = sum;
        }
    }
    
    // Solve S * lambda = residual
    lambda_out.resize(k);
    try {
        LinearAlgebra::solve_linear_system(S.data(), residual.data(), lambda_out.data(), k);
    } catch (...) {
        return false;
    }
    
    // Compute x* = x_bar - Q^{-1} A^T * lambda
    x_star_out.resize(n);
    for (int i = 0; i < n; ++i) {
        double correction = 0.0;
        for (int j = 0; j < k; ++j) {
            correction += Q_inv_AT[i * k + j] * lambda_out[j];
        }
        x_star_out[i] = x_bar[i] - correction;
    }
    
    return true;
}

} // namespace internal
} // namespace geoqp
