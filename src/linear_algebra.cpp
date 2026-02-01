#include "geoqp/internal.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace geoqp {
namespace internal {

void LinearAlgebra::matvec(const double* A, const double* x, double* y, int m, int n) {
    // Optimized matrix-vector multiplication with better cache locality
    for (int i = 0; i < m; ++i) {
        const double* row = A + i * n;
        double sum = 0.0;
        
        // Unroll loop for better performance (4-way unrolling)
        int j = 0;
        for (; j + 3 < n; j += 4) {
            sum += row[j] * x[j] + row[j+1] * x[j+1] + 
                   row[j+2] * x[j+2] + row[j+3] * x[j+3];
        }
        
        // Handle remaining elements
        for (; j < n; ++j) {
            sum += row[j] * x[j];
        }
        
        y[i] = sum;
    }
}

double LinearAlgebra::dot(const double* x, const double* y, int n) {
    double result = 0.0;
    
    // Unroll for better performance
    int i = 0;
    for (; i + 3 < n; i += 4) {
        result += x[i] * y[i] + x[i+1] * y[i+1] + 
                  x[i+2] * y[i+2] + x[i+3] * y[i+3];
    }
    
    for (; i < n; ++i) {
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
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L[i * n + j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i * n + i];
    }
    
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
    std::vector<double> L(n * n);
    cholesky(Q, L.data(), n);
    solve_cholesky(L.data(), b, x, n);
}

void LinearAlgebra::solve_linear_system(const double* A, const double* b, double* x, int n) {
    std::vector<double> A_copy(n * n);
    std::vector<double> b_copy(n);
    std::memcpy(A_copy.data(), A, n * n * sizeof(double));
    std::memcpy(b_copy.data(), b, n * sizeof(double));
    
    for (int k = 0; k < n; ++k) {
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
        
        if (pivot != k) {
            for (int j = 0; j < n; ++j) {
                std::swap(A_copy[k * n + j], A_copy[pivot * n + j]);
            }
            std::swap(b_copy[k], b_copy[pivot]);
        }
        
        for (int i = k + 1; i < n; ++i) {
            double factor = A_copy[i * n + k] / A_copy[k * n + k];
            for (int j = k; j < n; ++j) {
                A_copy[i * n + j] -= factor * A_copy[k * n + j];
            }
            b_copy[i] -= factor * b_copy[k];
        }
    }
    
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
