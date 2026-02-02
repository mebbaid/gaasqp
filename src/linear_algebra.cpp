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

void LinearAlgebra::forward_sub(const double* L, const double* b, double* x, int n) {
    // Solve L * x = b where L is lower triangular
    // This computes x directly (x[i] depends on x[0]...x[i-1])
    for (int i = 0; i < n; ++i) {
        const double* Li = L + i * n; // start of row i
        double sum = 0.0;
        int j = 0;
        // 4-way unrolling over strictly lower part of row i
        for (; j + 3 < i; j += 4) {
            sum += Li[j]     * x[j]
                 + Li[j + 1] * x[j + 1]
                 + Li[j + 2] * x[j + 2]
                 + Li[j + 3] * x[j + 3];
        }
        for (; j < i; ++j) {
            sum += Li[j] * x[j];
        }
        const double Lii = Li[i];
        x[i] = (b[i] - sum) / Lii;
    }
}

void LinearAlgebra::backward_sub(const double* L, const double* b, double* x, int n) {
    // Solve L^T * x = b where L is lower triangular (so L^T is upper triangular)
    // This accesses L column-wise to get L^T row-wise
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        int j = i + 1;
        // Access column i below the diagonal with unit-stride across rows
        // j iterates rows; element is L[j * n + i]
        for (; j + 3 < n; j += 4) {
            sum += L[(j)     * n + i] * x[j]
                 + L[(j + 1) * n + i] * x[j + 1]
                 + L[(j + 2) * n + i] * x[j + 2]
                 + L[(j + 3) * n + i] * x[j + 3];
        }
        for (; j < n; ++j) {
            sum += L[j * n + i] * x[j];
        }
        const double Lii = L[i * n + i];
        x[i] = (b[i] - sum) / Lii;
    }
}

void LinearAlgebra::solve_cholesky(const double* L, const double* b, double* x, int n) {
    // Solve (L * L^T) * x = b in two steps:
    // 1. Forward substitution: L * y = b
    // 2. Backward substitution: L^T * x = y
    std::vector<double> y(n);
    forward_sub(L, b, y.data(), n);
    backward_sub(L, y.data(), x, n);
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

bool LinearAlgebra::cholesky_rank1_update(double* L, const double* v, int k) {
    // Given L such that L*L^T = S, compute L' such that L'*L'^T = S + v*v^T
    // Uses the standard rank-1 update algorithm (Gill, Golub, Murray, Saunders)
    std::vector<double> w(v, v + k);
    
    for (int i = 0; i < k; ++i) {
        double Lii = L[i * k + i];
        double wi = w[i];
        double r = std::sqrt(Lii * Lii + wi * wi);
        
        if (r < 1e-14) return false; // Numerical failure
        
        double c = r / Lii;
        double s = wi / Lii;
        L[i * k + i] = r;
        
        // Update remaining elements in this column
        for (int j = i + 1; j < k; ++j) {
            double Lji = L[j * k + i];
            w[j] = w[j] - s * Lji;
            L[j * k + i] = (Lji + s * w[j]) / c;
        }
    }
    return true;
}

bool LinearAlgebra::cholesky_extend(
    const double* L, int k,
    const double* s, double sigma,
    double* L_new
) {
    // Given L (k x k) such that L*L^T = S,
    // extend to L' ((k+1) x (k+1)) for [S, s; s^T, sigma]
    // 
    // The new Cholesky factor has form:
    // L' = [L,     0  ]
    //      [l^T, l_new]
    // where L * l = s  =>  l = L^{-1} s (forward solve)
    // and l_new = sqrt(sigma - ||l||^2)
    
    const int k1 = k + 1;
    
    // Copy L to top-left of L_new
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j <= i; ++j) {
            L_new[i * k1 + j] = L[i * k + j];
        }
        // Zero out the new column for rows < k
        L_new[i * k1 + k] = 0.0;
    }
    
    // Solve L * l = s for l (forward substitution)
    std::vector<double> l(k);
    for (int i = 0; i < k; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L[i * k + j] * l[j];
        }
        l[i] = (s[i] - sum) / L[i * k + i];
    }
    
    // Store l as the last row of L_new (bottom-left)
    for (int j = 0; j < k; ++j) {
        L_new[k * k1 + j] = l[j];
    }
    
    // Compute l_new = sqrt(sigma - ||l||^2)
    double l_norm_sq = 0.0;
    for (int i = 0; i < k; ++i) {
        l_norm_sq += l[i] * l[i];
    }
    
    double diag_sq = sigma - l_norm_sq;
    if (diag_sq < 1e-14) {
        return false; // Matrix would be singular or not positive definite
    }
    
    L_new[k * k1 + k] = std::sqrt(diag_sq);
    return true;
}

bool LinearAlgebra::cholesky_shrink(
    const double* L, int k, int idx,
    double* L_new
) {
    // Remove row/column idx from Cholesky factor using Givens rotations
    // This is O(k^2) instead of O(k^3)!
    //
    // Strategy:
    // 1. Permute row/col idx to the end (using Givens to maintain triangular form)
    // 2. Truncate the last row/column
    
    if (k <= 1 || idx < 0 || idx >= k) {
        return false;
    }
    
    const int k1 = k - 1;
    
    // Special case: deleting the last row/column (no rotation needed)
    if (idx == k1) {
        for (int i = 0; i < k1; ++i) {
            for (int j = 0; j <= i; ++j) {
                L_new[i * k1 + j] = L[i * k + j];
            }
        }
        return true;
    }
    
    // Copy L to working array (we'll modify it in-place)
    std::vector<double> L_work(k * k);
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j <= i; ++j) {
            L_work[i * k + j] = L[i * k + j];
        }
    }
    
    // Permute row/col idx to position k-1 using Givens rotations
    for (int i = idx; i < k1; ++i) {
        // Apply Givens rotation to eliminate L_work[i+1, i]
        
        double a = L_work[i * k + i];      // Diagonal element
        double b = L_work[(i+1) * k + i];  // Element to eliminate
        
        if (std::abs(b) < 1e-14) {
            continue; // Already zero
        }
        
        // Compute Givens rotation parameters
        double r = std::sqrt(a * a + b * b);
        if (r < 1e-14) {
            return false; // Singular
        }
        
        double c = a / r;
        double s = b / r;
        
        // Apply rotation to rows i and i+1
        for (int j = 0; j <= i + 1 && j < k; ++j) {
            double L_i_j = (j <= i) ? L_work[i * k + j] : 0.0;
            double L_ip1_j = (j <= i + 1) ? L_work[(i+1) * k + j] : 0.0;
            
            double new_i = c * L_i_j + s * L_ip1_j;
            double new_ip1 = -s * L_i_j + c * L_ip1_j;
            
            if (j <= i) L_work[i * k + j] = new_i;
            if (j <= i + 1) L_work[(i+1) * k + j] = new_ip1;
        }
        
        // Update later rows
        for (int row = i + 2; row < k; ++row) {
            double L_row_i = (i < row) ? L_work[row * k + i] : 0.0;
            double L_row_ip1 = (i + 1 < row) ? L_work[row * k + i + 1] : 0.0;
            
            if (i < row) L_work[row * k + i] = c * L_row_i + s * L_row_ip1;
            if (i + 1 < row) L_work[row * k + i + 1] = -s * L_row_i + c * L_row_ip1;
        }
    }
    
    // Extract the first (k-1) x (k-1) block
    for (int i = 0; i < k1; ++i) {
        for (int j = 0; j <= i; ++j) {
            L_new[i * k1 + j] = L_work[i * k + j];
        }
    }
    
    return true;
}

} // namespace internal
} // namespace geoqp
