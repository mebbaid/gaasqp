#include "geoqp/internal.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace geoqp {
namespace internal {

void LinearAlgebra::matvec(const double* A, const double* x, double* y, int m, int n) {
    for (int i = 0; i < m; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

double LinearAlgebra::dot(const double* x, const double* y, int n) {
    double result = 0.0;
    for (int i = 0; i < n; ++i) {
        result += x[i] * y[i];
    }
    return result;
}

double LinearAlgebra::norm(const double* x, int n) {
    return std::sqrt(dot(x, x, n));
}

void LinearAlgebra::cholesky(const double* A, double* L, int n) {
    // A = L L^T
    std::fill(L, L + n * n, 0.0);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            
            if (i == j) {
                double val = A[i * n + i] - sum;
                if (val <= 0.0) {
                    throw NumericalException("Matrix is not positive definite");
                }
                L[i * n + j] = std::sqrt(val);
            } else {
                L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
            }
        }
    }
}

void LinearAlgebra::solve_cholesky(const double* L, const double* b, double* x, int n) {
    // Solve L L^T x = b
    // First solve L y = b (forward substitution)
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L[i * n + j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i * n + i];
    }
    
    // Then solve L^T x = y (backward substitution)
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += L[j * n + i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i * n + i];
    }
}

void LinearAlgebra::solve_pd_system(const double* Q, const double* b, double* x, int n) {
    // Solve Q x = b where Q is positive definite
    // Use Cholesky decomposition
    std::vector<double> L(n * n);
    cholesky(Q, L.data(), n);
    solve_cholesky(L.data(), b, x, n);
}

void LinearAlgebra::solve_linear_system(const double* A, const double* b, double* x, int n) {
    // Simple Gaussian elimination with partial pivoting
    // Copy A and b since we'll modify them
    std::vector<double> A_copy(n * n);
    std::vector<double> b_copy(n);
    std::memcpy(A_copy.data(), A, n * n * sizeof(double));
    std::memcpy(b_copy.data(), b, n * sizeof(double));
    
    // Forward elimination
    for (int k = 0; k < n; ++k) {
        // Find pivot
        int pivot = k;
        double max_val = std::abs(A_copy[k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            double val = std::abs(A_copy[i * n + k]);
            if (val > max_val) {
                max_val = val;
                pivot = i;
            }
        }
        
        if (max_val < 1e-14) {
            throw NumericalException("Singular matrix in linear system");
        }
        
        // Swap rows
        if (pivot != k) {
            for (int j = 0; j < n; ++j) {
                std::swap(A_copy[k * n + j], A_copy[pivot * n + j]);
            }
            std::swap(b_copy[k], b_copy[pivot]);
        }
        
        // Eliminate
        for (int i = k + 1; i < n; ++i) {
            double factor = A_copy[i * n + k] / A_copy[k * n + k];
            for (int j = k; j < n; ++j) {
                A_copy[i * n + j] -= factor * A_copy[k * n + j];
            }
            b_copy[i] -= factor * b_copy[k];
        }
    }
    
    // Back substitution
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += A_copy[i * n + j] * x[j];
        }
        x[i] = (b_copy[i] - sum) / A_copy[i * n + i];
    }
}

} // namespace internal
} // namespace geoqp
