#include "geoqp/geoqp.hpp"
#include "geoqp/internal.hpp"
#include <cmath>
#include <limits>

namespace geoqp {

bool validate_problem(const Problem& problem) {
    // Check dimensions
    if (problem.n <= 0 || problem.m < 0) {
        return false;
    }
    
    // Check sizes
    if (problem.Q.size() != static_cast<size_t>(problem.n * problem.n)) {
        return false;
    }
    if (problem.g.size() != static_cast<size_t>(problem.n)) {
        return false;
    }
    if (problem.A.size() != static_cast<size_t>(problem.m * problem.n)) {
        return false;
    }
    if (problem.b.size() != static_cast<size_t>(problem.m)) {
        return false;
    }
    
    // Check Q is symmetric
    for (int i = 0; i < problem.n; ++i) {
        for (int j = i + 1; j < problem.n; ++j) {
            double diff = std::abs(problem.Q[i * problem.n + j] - 
                                  problem.Q[j * problem.n + i]);
            if (diff > 1e-10) {
                return false;
            }
        }
    }
    
    // Check Q is positive definite (simple check: diagonal elements > 0)
    // More rigorous: compute eigenvalues or try Cholesky
    for (int i = 0; i < problem.n; ++i) {
        if (problem.Q[i * problem.n + i] <= 0.0) {
            return false;
        }
    }
    
    return true;
}

bool is_positive_definite(const std::vector<double>& Q, int n) {
    try {
        std::vector<double> L(n * n);
        internal::LinearAlgebra::cholesky(Q.data(), L.data(), n);
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<double> compute_unconstrained_minimum(
    const std::vector<double>& Q,
    const std::vector<double>& g,
    int n
) {
    std::vector<double> x_bar(n);
    std::vector<double> neg_g(n);
    
    // Compute -g
    for (int i = 0; i < n; ++i) {
        neg_g[i] = -g[i];
    }
    
    // Solve Q x_bar = -g
    internal::LinearAlgebra::solve_pd_system(Q.data(), neg_g.data(), x_bar.data(), n);
    
    return x_bar;
}

bool is_feasible(
    const std::vector<double>& x,
    const std::vector<double>& A,
    const std::vector<double>& b,
    int m, int n,
    double tol
) {
    const double* x_ptr = x.data();
    const double* A_ptr = A.data();
    
    for (int i = 0; i < m; ++i) {
        // Compute dot product inline for better performance
        const double* row = A_ptr + i * n;
        double lhs = 0.0;
        
        // Unrolled loop
        int j = 0;
        for (; j + 3 < n; j += 4) {
            lhs += row[j] * x_ptr[j] + row[j+1] * x_ptr[j+1] +
                   row[j+2] * x_ptr[j+2] + row[j+3] * x_ptr[j+3];
        }
        for (; j < n; ++j) {
            lhs += row[j] * x_ptr[j];
        }
        
        if (lhs > b[i] + tol) {
            return false;
        }
    }
    return true;
}

double compute_objective(
    const std::vector<double>& x,
    const std::vector<double>& Q,
    const std::vector<double>& g,
    int n
) {
    // Compute (1/2) x^T Q x + g^T x
    
    // First: Q x
    std::vector<double> Qx(n);
    internal::LinearAlgebra::matvec(Q.data(), x.data(), Qx.data(), n, n);
    
    // x^T Q x
    double quadratic = internal::LinearAlgebra::dot(x.data(), Qx.data(), n);
    
    // g^T x
    double linear = internal::LinearAlgebra::dot(g.data(), x.data(), n);
    
    return 0.5 * quadratic + linear;
}

} // namespace geoqp
