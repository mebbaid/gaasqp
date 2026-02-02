#include "geoqp/internal.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <set>
#include <cstring>
#include <iostream>

namespace geoqp {
namespace internal {

FaceHopper::FaceHopper(
    const double* Q, const double* A, const double* b,
    const std::vector<double>& x_bar,
    int m, int n,
    const Config& config
) : Q_(Q), A_(A), b_(b), x_bar_(x_bar), m_(m), n_(n), config_(config) {
    Q_chol_.resize(n_ * n_);
    try {
        LinearAlgebra::cholesky(Q_, Q_chol_.data(), n_);
    } catch (...) {
        Q_chol_.clear();
    }
    
    // Detect diagonal/box constraints (common in MPC)
    has_diagonal_constraints_ = true;
    for (int i = 0; i < m_ && has_diagonal_constraints_; ++i) {
        const double* a_row = A_ + i * n_;
        int nonzero_count = 0;
        int nonzero_idx = -1;
        for (int j = 0; j < n_; ++j) {
            if (std::abs(a_row[j]) > 1e-10) {
                nonzero_count++;
                nonzero_idx = j;
                if (std::abs(std::abs(a_row[j]) - 1.0) > 1e-10) {
                    has_diagonal_constraints_ = false;
                    break;
                }
            }
        }
        if (nonzero_count != 1) {
            has_diagonal_constraints_ = false;
        }
    }
    
    if (has_diagonal_constraints_ && config_.verbosity >= 2) {
        std::cout << "  Detected diagonal/box constraints - using fast path\n";
    }
}

double FaceHopper::compute_intersection_parameter(
    const std::vector<double>& v,
    int constraint_idx
) const {
    const double* a_row = A_ + constraint_idx * n_;
    double av = 0.0;
    double ad = 0.0;
    
    int j = 0;
    for (; j + 3 < n_; j += 4) {
        av += a_row[j] * v[j] + a_row[j+1] * v[j+1] +
              a_row[j+2] * v[j+2] + a_row[j+3] * v[j+3];
        ad += a_row[j] * (x_bar_[j] - v[j]) + a_row[j+1] * (x_bar_[j+1] - v[j+1]) +
              a_row[j+2] * (x_bar_[j+2] - v[j+2]) + a_row[j+3] * (x_bar_[j+3] - v[j+3]);
    }
    for (; j < n_; ++j) {
        av += a_row[j] * v[j];
        ad += a_row[j] * (x_bar_[j] - v[j]);
    }
    
    if (std::abs(ad) < 1e-14) {
        return (std::abs(av - b_[constraint_idx]) < 1e-10) ? 
               0.0 : std::numeric_limits<double>::infinity();
    }
    
    return (b_[constraint_idx] - av) / ad;
}

std::vector<int> FaceHopper::find_initial_face(
    const std::vector<double>& v
) {
    std::vector<int> active_at_v;
    for (int i = 0; i < m_; ++i) {
        const double* a_row = A_ + i * n_;
        double av = LinearAlgebra::dot(a_row, v.data(), n_);
        if (std::abs(av - b_[i]) < 1e-8) {
            active_at_v.push_back(i);
        }
    }
    
    std::vector<int> blocking_at_v;
    for (int i : active_at_v) {
        const double* a_row = A_ + i * n_;
        double a_xbar = LinearAlgebra::dot(a_row, x_bar_.data(), n_);
        if (a_xbar > b_[i] + 1e-8) {
            blocking_at_v.push_back(i);
        }
    }
    
    if (!blocking_at_v.empty()) {
        return blocking_at_v;
    }
    
    double t_min = std::numeric_limits<double>::infinity();
    for (int i = 0; i < m_; ++i) {
        double t = compute_intersection_parameter(v, i);
        if (t > 1e-10 && t < t_min) {
            t_min = t;
        }
    }
    
    if (t_min > 1.0 + 1e-8) {
        bool x_bar_feasible = true;
        for (int i = 0; i < m_; ++i) {
            const double* a_row = A_ + i * n_;
            double ax = LinearAlgebra::dot(a_row, x_bar_.data(), n_);
            if (ax > b_[i] + config_.tolerance) {
                x_bar_feasible = false;
                break;
            }
        }
        
        if (x_bar_feasible) {
            return std::vector<int>();
        }
    }
    
    if (!std::isfinite(t_min)) {
        return std::vector<int>();
    }
    
    std::vector<int> initial_face;
    initial_face.reserve(n_);
    const double tol = 1e-8;
    for (int i = 0; i < m_; ++i) {
        double t = compute_intersection_parameter(v, i);
        if (std::abs(t - t_min) <= tol) {
            initial_face.push_back(i);
        }
    }
    
    return initial_face;
}

bool FaceHopper::compute_projection_and_lambda(
    const std::vector<int>& face,
    std::vector<double>& x_star,
    std::vector<double>& lambda
) const {
    const int k = static_cast<int>(face.size());
    x_star = x_bar_;
    lambda.resize(k);
    
    if (k == 0) {
        for (int i = 0; i < m_; ++i) {
            const double* a_row = A_ + i * n_;
            double ax = LinearAlgebra::dot(a_row, x_bar_.data(), n_);
            if (ax > b_[i] + config_.tolerance) {
                return false;
            }
        }
        return true;
    }
    
    // Fast path for diagonal/box constraints
    if (has_diagonal_constraints_) {
        
        std::vector<int> constrained_vars(k);
        std::vector<double> signs(k);
        std::vector<double> bounds(k);
        
        for (int i = 0; i < k; ++i) {
            const int idx = face[i];
            const double* a_row = A_ + idx * n_;
            bounds[i] = b_[idx];
            for (int j = 0; j < n_; ++j) {
                if (std::abs(a_row[j]) > 1e-10) {
                    constrained_vars[i] = j;
                    signs[i] = a_row[j];
                    break;
                }
            }
        }
        
        std::vector<double> residual(k);
        for (int i = 0; i < k; ++i) {
            residual[i] = signs[i] * x_bar_[constrained_vars[i]] - bounds[i];
        }
        
        std::vector<double> Q_inv_columns(n_ * k);
        for (int j = 0; j < k; ++j) {
            std::vector<double> rhs(n_, 0.0);
            rhs[constrained_vars[j]] = signs[j];
            
            std::vector<double> col(n_);
            if (!Q_chol_.empty()) {
                LinearAlgebra::solve_cholesky(Q_chol_.data(), rhs.data(), col.data(), n_);
            } else {
                std::vector<double> Q_copy(n_ * n_);
                std::memcpy(Q_copy.data(), Q_, n_ * n_ * sizeof(double));
                try {
                    LinearAlgebra::solve_linear_system(Q_copy.data(), rhs.data(), col.data(), n_);
                } catch (...) {
                    return false;
                }
            }
            std::memcpy(Q_inv_columns.data() + j * n_, col.data(), n_ * sizeof(double));
        }
        
        std::vector<double> S(k * k);
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                S[i * k + j] = Q_inv_columns[j * n_ + constrained_vars[i]] * signs[i];
            }
        }
        
        try {
            LinearAlgebra::solve_linear_system(S.data(), residual.data(), lambda.data(), k);
        } catch (...) {
            return false;
        }
        
        x_star = x_bar_;
        for (int i = 0; i < n_; ++i) {
            double correction = 0.0;
            for (int j = 0; j < k; ++j) {
                correction += Q_inv_columns[j * n_ + i] * lambda[j];
            }
            x_star[i] -= correction;
        }
        
        for (int i = 0; i < m_; ++i) {
            const double* a_row = A_ + i * n_;
            double ax = LinearAlgebra::dot(a_row, x_star.data(), n_);
            if (ax > b_[i] + config_.tolerance) {
                return false;
            }
        }
        
        return true;
    }
    
    // General path for arbitrary constraints
    std::vector<double> A_F(k * n_);
    std::vector<double> b_F(k);
    
    for (int i = 0; i < k; ++i) {
        const int idx = face[i];
        b_F[i] = b_[idx];
        std::memcpy(A_F.data() + i * n_, A_ + idx * n_, n_ * sizeof(double));
    }
    
    // Compute residual: A_F * x_bar - b_F
    std::vector<double> residual(k);
    LinearAlgebra::matvec(A_F.data(), x_bar_.data(), residual.data(), k, n_);
    for (int i = 0; i < k; ++i) residual[i] -= b_F[i];
    
    // Compute Q^{-1} A_F^T (n x k matrix)
    // For each column j: solve Q * y = A_F[j,:] for y
    // Use pre-computed Cholesky factorization if available
    std::vector<double> Q_inv_AT(n_ * k);
    
    for (int j = 0; j < k; ++j) {
        // Get j-th row of A_F as a column vector
        std::vector<double> a_col(n_);
        for (int i = 0; i < n_; ++i) {
            a_col[i] = A_F[j * n_ + i];
        }
        
        // Solve Q * y = a_col for y using Cholesky
        std::vector<double> y(n_);
        if (!Q_chol_.empty()) {
            LinearAlgebra::solve_cholesky(Q_chol_.data(), a_col.data(), y.data(), n_);
        } else {
            std::vector<double> Q_copy(n_ * n_);
            std::memcpy(Q_copy.data(), Q_, n_ * n_ * sizeof(double));
            try {
                LinearAlgebra::solve_linear_system(Q_copy.data(), a_col.data(), y.data(), n_);
            } catch (...) {
                return false;
            }
        }
        
        // Store as j-th column of Q_inv_AT
        for (int i = 0; i < n_; ++i) {
            Q_inv_AT[i * k + j] = y[i];
        }
    }
    
    std::vector<double> S(k * k);
    for (int i = 0; i < k; ++i) {
        for (int j = i; j < k; ++j) {
            double sum = 0.0;
            for (int l = 0; l < n_; ++l) {
                sum += A_F[i * n_ + l] * Q_inv_AT[l * k + j];
            }
            S[i * k + j] = sum;
            S[j * k + i] = sum;
        }
    }
    
    try {
        LinearAlgebra::solve_linear_system(S.data(), residual.data(), lambda.data(), k);
    } catch (...) {
        return false;
    }
    
    for (int i = 0; i < n_; ++i) {
        double correction = 0.0;
        for (int j = 0; j < k; ++j) {
            correction += Q_inv_AT[i * k + j] * lambda[j];
        }
        x_star[i] = x_bar_[i] - correction;
    }
    
    for (int i = 0; i < m_; ++i) {
        const double* a_row = A_ + i * n_;
        double ax = LinearAlgebra::dot(a_row, x_star.data(), n_);
        if (ax > b_[i] + config_.tolerance) {
            return false;
        }
    }
    
    return true;
}

std::vector<int> FaceHopper::find_optimal_face(
    const std::vector<int>& initial_face,
    int& iterations
) {
    iterations = 0;
    
    if (initial_face.empty()) {
        return initial_face;
    }
    
    std::vector<int> current_face = initial_face;
    const int max_iter = config_.max_iterations;
    
    while (iterations < max_iter) {
        iterations++;
        
        std::vector<double> x_star, lambda;
        bool feasible = compute_projection_and_lambda(current_face, x_star, lambda);
        
        if (feasible) {
            bool improved = false;
            
            for (size_t i = 0; i < current_face.size(); ++i) {
                if (lambda[i] < -config_.tolerance) {
                    std::vector<int> smaller_face;
                    for (size_t j = 0; j < current_face.size(); ++j) {
                        if (i != j) smaller_face.push_back(current_face[j]);
                    }
                    
                    std::vector<double> x2, lambda2;
                    if (compute_projection_and_lambda(smaller_face, x2, lambda2)) {
                        // Early dual feasibility check
                        bool dual_feasible = true;
                        for (double l : lambda2) {
                            if (l < -config_.tolerance) {
                                dual_feasible = false;
                                break;
                            }
                        }
                        
                        if (!dual_feasible) {
                            continue;
                        }
                        
                        // Compute Q-weighted distance with early exit
                        double d1 = 0.0, d2 = 0.0;
                        bool early_exit = false;
                        
                        // Precompute differences once
                        std::vector<double> diff1(n_), diff2(n_);
                        for (int j = 0; j < n_; ++j) {
                            diff1[j] = x_star[j] - x_bar_[j];
                            diff2[j] = x2[j] - x_bar_[j];
                        }
                        
                        // Compute Q-weighted norms incrementally
                        for (int j = 0; j < n_ && !early_exit; ++j) {
                            const double* Q_row = Q_ + j * n_;
                            double row_contrib1 = 0.0, row_contrib2 = 0.0;
                            for (int k = 0; k < n_; ++k) {
                                row_contrib1 += Q_row[k] * diff1[k];
                                row_contrib2 += Q_row[k] * diff2[k];
                            }
                            d1 += diff1[j] * row_contrib1;
                            d2 += diff2[j] * row_contrib2;
                            
                            // Early exit if d2 clearly worse
                            if (j % 10 == 9 && d2 > d1 * 1.1) {
                                early_exit = true;
                            }
                        }
                        
                        if (!early_exit && d2 < d1 - config_.tolerance) {
                            current_face = std::move(smaller_face);
                            improved = true;
                            break;
                        }
                    }
                }
            }
            
            if (!improved) {
                if (config_.verbosity >= 3) {
                    std::cout << "  Optimal face with " << current_face.size() << " constraints\n";
                }
                return current_face;
            }
        } else {
            std::set<int> face_set(current_face.begin(), current_face.end());
            std::vector<int> violated;
            
            for (int i = 0; i < m_; ++i) {
                if (face_set.find(i) != face_set.end()) continue;
                
                const double* a_row = A_ + i * n_;
                double ax = LinearAlgebra::dot(a_row, x_star.data(), n_);
                if (ax > b_[i] + config_.tolerance) {
                    violated.push_back(i);
                }
            }
            
            if (violated.empty()) {
                if (config_.verbosity >= 2) {
                    std::cout << "  Warning: no violated constraint but projection infeasible\n";
                }
                return current_face;
            }
            
            for (int c : violated) {
                current_face.push_back(c);
            }
            std::sort(current_face.begin(), current_face.end());
            
            if (config_.verbosity >= 3) {
                std::cout << "  Added " << violated.size() << " violated constraints\n";
            }
        }
    }
    
    if (config_.verbosity >= 1) {
        std::cout << "Warning: Face exploration reached max iterations\n";
    }
    
    return current_face;
}

} // namespace internal
} // namespace geoqp
