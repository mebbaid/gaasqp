#include "geoqp/internal.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <set>
#include <cstring>
#include <iostream>
#include <chrono>

namespace geoqp {
namespace internal {

// We will use internal::Timer and aggregate timings into timing_stats_

FaceHopper::FaceHopper(
    const double* Q, const double* A, const double* b,
    const std::vector<double>& x_bar,
    int m, int n,
    const Config& config
) : Q_(Q), A_(A), b_(b), x_bar_(x_bar), m_(m), n_(n), config_(config),
    constraint_cache_(m, n) {  // Initialize cache
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
    
    if (config_.use_caching && !Q_chol_.empty() && config_.verbosity >= 2) {
        std::cout << "  Incremental Schur complement caching enabled\n";
    }
}

double FaceHopper::compute_intersection_parameter(
    const std::vector<double>& v,
    int constraint_idx
) const {
    Timer t;
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
        double res = (std::abs(av - b_[constraint_idx]) < 1e-10) ? 0.0 : std::numeric_limits<double>::infinity();
        if (config_.timing_verbosity > 0) {
            timing_stats_.compute_intersection_total += t.elapsed_ms();
            ++timing_stats_.compute_intersection_calls;
        }
        return res;
    }
    double out = (b_[constraint_idx] - av) / ad;
    if (config_.timing_verbosity > 0) {
        timing_stats_.compute_intersection_total += t.elapsed_ms();
        ++timing_stats_.compute_intersection_calls;
    }
    return out;
}

std::vector<int> FaceHopper::find_initial_face(
    const std::vector<double>& v
) {
    Timer t;
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
        if (config_.timing_verbosity > 0) {
            timing_stats_.find_initial_face_total += t.elapsed_ms();
            ++timing_stats_.find_initial_face_calls;
        }
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
            if (config_.timing_verbosity > 0) {
                timing_stats_.find_initial_face_total += t.elapsed_ms();
                ++timing_stats_.find_initial_face_calls;
            }
            return std::vector<int>();
        }
    }
    
    if (!std::isfinite(t_min)) {
        if (config_.timing_verbosity > 0) {
            timing_stats_.find_initial_face_total += t.elapsed_ms();
            ++timing_stats_.find_initial_face_calls;
        }
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
    
    if (config_.timing_verbosity > 0) {
        timing_stats_.find_initial_face_total += t.elapsed_ms();
        ++timing_stats_.find_initial_face_calls;
    }

    return initial_face;
}

bool FaceHopper::compute_projection_and_lambda(
    const std::vector<int>& face,
    std::vector<double>& x_star,
    std::vector<double>& lambda
) const {
    Timer t;
    
    // Use cached path if enabled and Q_chol is available
    if (config_.use_caching && !Q_chol_.empty() && !has_diagonal_constraints_) {
        bool success = compute_projection_cached(face, x_star, lambda);
        
        // Verify feasibility
        if (success) {
            for (int i = 0; i < m_; ++i) {
                const double* a_row = A_ + i * n_;
                double ax = LinearAlgebra::dot(a_row, x_star.data(), n_);
                if (ax > b_[i] + config_.tolerance) {
                    success = false;
                    break;
                }
            }
        }
        
        if (config_.timing_verbosity > 0) {
            timing_stats_.compute_projection_total += t.elapsed_ms();
            ++timing_stats_.compute_projection_calls;
        }
        
        if (success) return true;
        // Fall through to original path if cached version fails
    }
    
    // Original implementation follows...
    const int k = static_cast<int>(face.size());
    x_star = x_bar_;
    lambda.resize(k);
    
    if (k == 0) {
        for (int i = 0; i < m_; ++i) {
            const double* a_row = A_ + i * n_;
            double ax = LinearAlgebra::dot(a_row, x_bar_.data(), n_);
            if (ax > b_[i] + config_.tolerance) {
                if (config_.timing_verbosity > 0) {
                    timing_stats_.compute_projection_total += t.elapsed_ms();
                    ++timing_stats_.compute_projection_calls;
                }
                return false;
            }
        }
        if (config_.timing_verbosity > 0) {
            timing_stats_.compute_projection_total += t.elapsed_ms();
            ++timing_stats_.compute_projection_calls;
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
                    if (config_.timing_verbosity > 0) {
                        timing_stats_.compute_projection_total += t.elapsed_ms();
                        ++timing_stats_.compute_projection_calls;
                    }
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
            if (config_.timing_verbosity > 0) {
                timing_stats_.compute_projection_total += t.elapsed_ms();
                ++timing_stats_.compute_projection_calls;
            }
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
                if (config_.timing_verbosity > 0) {
                    timing_stats_.compute_projection_total += t.elapsed_ms();
                    ++timing_stats_.compute_projection_calls;
                }
                return false;
            }
        }
        
        if (config_.timing_verbosity > 0) {
            timing_stats_.compute_projection_total += t.elapsed_ms();
            ++timing_stats_.compute_projection_calls;
        }
        return true;
    }
    
    // General path for arbitrary constraints - use shared solver
    bool success = ProjectionSolver::solve(
        face, Q_, A_, b_, x_bar_, Q_chol_,
        m_, n_, x_star, lambda
    );
    
    if (!success) {
        if (config_.timing_verbosity > 0) {
            timing_stats_.compute_projection_total += t.elapsed_ms();
            ++timing_stats_.compute_projection_calls;
        }
        return false;
    }
    
    // Verify feasibility
    for (int i = 0; i < m_; ++i) {
        const double* a_row = A_ + i * n_;
        double ax = LinearAlgebra::dot(a_row, x_star.data(), n_);
        if (ax > b_[i] + config_.tolerance) {
            if (config_.timing_verbosity > 0) {
                timing_stats_.compute_projection_total += t.elapsed_ms();
                ++timing_stats_.compute_projection_calls;
            }
            return false;
        }
    }
    
    if (config_.timing_verbosity > 0) {
        timing_stats_.compute_projection_total += t.elapsed_ms();
        ++timing_stats_.compute_projection_calls;
    }
    
    return true;
}

bool FaceHopper::compute_projection_cached(
    const std::vector<int>& face,
    std::vector<double>& x_star,
    std::vector<double>& lambda
) const {
    const int k = static_cast<int>(face.size());
    
    // Handle empty face
    if (k == 0) {
        x_star = x_bar_;
        lambda.clear();
        return true;
    }
    
    // Build residual: A_F * x_bar - b_F
    std::vector<double> residual(k);
    for (int i = 0; i < k; ++i) {
        const int idx = face[i];
        const double* a_row = A_ + idx * n_;
        residual[i] = LinearAlgebra::dot(a_row, x_bar_.data(), n_) - b_[idx];
    }
    
    // Initialize Schur manager if needed
    if (!schur_manager_.initialize(face, A_, Q_chol_.data(), n_, constraint_cache_)) {
        return false;
    }
    
    // Solve S * lambda = residual using cached factorization
    if (!schur_manager_.solve(residual, lambda)) {
        return false;
    }
    
    // Compute x_star = x_bar - Q^{-1} A_F^T lambda
    // Use cached z vectors: x_star = x_bar - sum_j(lambda[j] * L_Q^{-T} * z[j])
    std::vector<double> w(n_, 0.0);
    for (int j = 0; j < k; ++j) {
        const auto& z_j = constraint_cache_.get_z(face[j], A_, Q_chol_.data());
        for (int i = 0; i < n_; ++i) {
            w[i] += lambda[j] * z_j[i];
        }
    }
    
    // Solve L_Q^T * correction = w
    std::vector<double> correction(n_);
    LinearAlgebra::backward_sub(Q_chol_.data(), w.data(), correction.data(), n_);
    
    // Apply correction
    x_star.resize(n_);
    for (int i = 0; i < n_; ++i) {
        x_star[i] = x_bar_[i] - correction[i];
    }
    
    return true;
}

std::vector<int> FaceHopper::find_optimal_face_cached(
    const std::vector<int>& initial_face,
    int& iterations,
    Timer& t
) const {
    iterations = 0;
    const int max_iter = config_.max_iterations;
    
    // Initialize Schur manager with initial face
    if (!schur_manager_.initialize(initial_face, A_, Q_chol_.data(), n_, constraint_cache_)) {
        // Fall back to non-cached if initialization fails
        return initial_face;
    }
    
    while (iterations < max_iter) {
        iterations++;
        const auto& current_face = schur_manager_.get_active_constraints();
        const int k = schur_manager_.size();
        
        // Compute residual: A_F * x_bar - b_F  
        std::vector<double> residual(k);
        for (int i = 0; i < k; ++i) {
            const int idx = current_face[i];
            const double* a_row = A_ + idx * n_;
            residual[i] = LinearAlgebra::dot(a_row, x_bar_.data(), n_) - b_[idx];
        }
        
        // Solve for lambda using cached Schur complement
        std::vector<double> lambda;
        if (!schur_manager_.solve(residual, lambda)) {
            break; // Numerical failure
        }
        
        // Compute x_star using cached z vectors
        std::vector<double> w(n_, 0.0);
        for (int j = 0; j < k; ++j) {
            const auto& z_j = constraint_cache_.get_z(current_face[j], A_, Q_chol_.data());
            for (int i = 0; i < n_; ++i) {
                w[i] += lambda[j] * z_j[i];
            }
        }
        
        std::vector<double> correction(n_);
        LinearAlgebra::backward_sub(Q_chol_.data(), w.data(), correction.data(), n_);
        
        std::vector<double> x_star(n_);
        for (int i = 0; i < n_; ++i) {
            x_star[i] = x_bar_[i] - correction[i];
        }
        
        // Check primal feasibility
        bool primal_feasible = true;
        int most_violated = -1;
        double max_violation = 0.0;
        
        std::set<int> active_set(current_face.begin(), current_face.end());
        for (int i = 0; i < m_; ++i) {
            if (active_set.count(i)) continue;
            
            const double* a_row = A_ + i * n_;
            double ax = LinearAlgebra::dot(a_row, x_star.data(), n_);
            double violation = ax - b_[i];
            
            if (violation > max_violation + config_.tolerance) {
                max_violation = violation;
                most_violated = i;
                primal_feasible = false;
            }
        }
        
        if (!primal_feasible && most_violated >= 0) {
            // Collect all significantly violated constraints
            std::vector<int> to_add;
            for (int i = 0; i < m_; ++i) {
                if (active_set.count(i)) continue;
                
                const double* a_row = A_ + i * n_;
                double ax = LinearAlgebra::dot(a_row, x_star.data(), n_);
                if (ax > b_[i] + config_.tolerance) {
                    to_add.push_back(i);
                }
            }
            
            // Add all violated constraints incrementally
            bool all_added = true;
            for (int c : to_add) {
                if (!schur_manager_.add_constraint(c, A_, Q_chol_.data(), n_, constraint_cache_)) {
                    all_added = false;
                    break;
                }
            }
            
            if (!all_added) break; // Numerical failure
            continue;
        }
        
        // Primal feasible - check dual feasibility
        int drop_idx = -1;
        double min_lambda = -config_.tolerance;
        for (int i = 0; i < k; ++i) {
            if (lambda[i] < min_lambda) {
                min_lambda = lambda[i];
                drop_idx = i;
            }
        }
        
        if (drop_idx >= 0) {
            // Remove constraint with negative multiplier incrementally
            if (!schur_manager_.remove_constraint(drop_idx)) {
                break; // Numerical failure
            }
            continue;
        }
        
        // Both primal and dual feasible - optimal!
        if (config_.timing_verbosity > 0) {
            timing_stats_.find_optimal_face_total += t.elapsed_ms();
            ++timing_stats_.find_optimal_face_calls;
            std::cout << "FaceHopper timing summary:\n";
            std::cout << "  compute_intersection: " << timing_stats_.compute_intersection_calls
                      << " calls, " << timing_stats_.compute_intersection_total << " ms\n";
            std::cout << "  find_initial_face: " << timing_stats_.find_initial_face_calls
                      << " calls, " << timing_stats_.find_initial_face_total << " ms\n";
            std::cout << "  compute_projection: " << timing_stats_.compute_projection_calls
                      << " calls, " << timing_stats_.compute_projection_total << " ms\n";
            std::cout << "  find_optimal_face: " << timing_stats_.find_optimal_face_calls
                      << " calls, " << timing_stats_.find_optimal_face_total << " ms\n";
        }
        return schur_manager_.get_active_constraints();
    }
    
    // Max iterations or numerical failure
    if (config_.timing_verbosity > 0) {
        timing_stats_.find_optimal_face_total += t.elapsed_ms();
        ++timing_stats_.find_optimal_face_calls;
        std::cout << "FaceHopper timing summary:\n";
        std::cout << "  compute_intersection: " << timing_stats_.compute_intersection_calls
                  << " calls, " << timing_stats_.compute_intersection_total << " ms\n";
        std::cout << "  find_initial_face: " << timing_stats_.find_initial_face_calls
                  << " calls, " << timing_stats_.find_initial_face_total << " ms\n";
        std::cout << "  compute_projection: " << timing_stats_.compute_projection_calls
                  << " calls, " << timing_stats_.compute_projection_total << " ms\n";
        std::cout << "  find_optimal_face: " << timing_stats_.find_optimal_face_calls
                  << " calls, " << timing_stats_.find_optimal_face_total << " ms\n";
    }
    return schur_manager_.get_active_constraints();
}

std::vector<int> FaceHopper::find_optimal_face(
    const std::vector<int>& initial_face,
    int& iterations
) {
    Timer t;
    iterations = 0;
    
    if (initial_face.empty()) {
        if (config_.timing_verbosity > 0) {
            timing_stats_.find_optimal_face_total += t.elapsed_ms();
            ++timing_stats_.find_optimal_face_calls;
            std::cout << "FaceHopper timing summary:\n";
            std::cout << "  compute_intersection: " << timing_stats_.compute_intersection_calls
                      << " calls, " << timing_stats_.compute_intersection_total << " ms\n";
            std::cout << "  find_initial_face: " << timing_stats_.find_initial_face_calls
                      << " calls, " << timing_stats_.find_initial_face_total << " ms\n";
            std::cout << "  compute_projection: " << timing_stats_.compute_projection_calls
                      << " calls, " << timing_stats_.compute_projection_total << " ms\n";
            std::cout << "  find_optimal_face: " << timing_stats_.find_optimal_face_calls
                      << " calls, " << timing_stats_.find_optimal_face_total << " ms\n";
        }
        return initial_face;
    }
    
    // Use stateful caching if enabled
    if (config_.use_caching && !Q_chol_.empty() && !has_diagonal_constraints_) {
        return find_optimal_face_cached(initial_face, iterations, t);
    }
    
    // Original implementation
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
                if (config_.timing_verbosity > 0) {
                    timing_stats_.find_optimal_face_total += t.elapsed_ms();
                    ++timing_stats_.find_optimal_face_calls;
                    // print aggregate summary
                    std::cout << "FaceHopper timing summary:\n";
                    std::cout << "  compute_intersection: " << timing_stats_.compute_intersection_calls
                              << " calls, " << timing_stats_.compute_intersection_total << " ms\n";
                    std::cout << "  find_initial_face: " << timing_stats_.find_initial_face_calls
                              << " calls, " << timing_stats_.find_initial_face_total << " ms\n";
                    std::cout << "  compute_projection: " << timing_stats_.compute_projection_calls
                              << " calls, " << timing_stats_.compute_projection_total << " ms\n";
                    std::cout << "  find_optimal_face: " << timing_stats_.find_optimal_face_calls
                              << " calls, " << timing_stats_.find_optimal_face_total << " ms\n";
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
                if (config_.timing_verbosity > 0) {
                    timing_stats_.find_optimal_face_total += t.elapsed_ms();
                    ++timing_stats_.find_optimal_face_calls;
                    std::cout << "FaceHopper timing summary:\n";
                    std::cout << "  compute_intersection: " << timing_stats_.compute_intersection_calls
                              << " calls, " << timing_stats_.compute_intersection_total << " ms\n";
                    std::cout << "  find_initial_face: " << timing_stats_.find_initial_face_calls
                              << " calls, " << timing_stats_.find_initial_face_total << " ms\n";
                    std::cout << "  compute_projection: " << timing_stats_.compute_projection_calls
                              << " calls, " << timing_stats_.compute_projection_total << " ms\n";
                    std::cout << "  find_optimal_face: " << timing_stats_.find_optimal_face_calls
                              << " calls, " << timing_stats_.find_optimal_face_total << " ms\n";
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
    if (config_.timing_verbosity > 0) {
        timing_stats_.find_optimal_face_total += t.elapsed_ms();
        ++timing_stats_.find_optimal_face_calls;
        std::cout << "FaceHopper timing summary:\n";
        std::cout << "  compute_intersection: " << timing_stats_.compute_intersection_calls
                  << " calls, " << timing_stats_.compute_intersection_total << " ms\n";
        std::cout << "  find_initial_face: " << timing_stats_.find_initial_face_calls
                  << " calls, " << timing_stats_.find_initial_face_total << " ms\n";
        std::cout << "  compute_projection: " << timing_stats_.compute_projection_calls
                  << " calls, " << timing_stats_.compute_projection_total << " ms\n";
        std::cout << "  find_optimal_face: " << timing_stats_.find_optimal_face_calls
                  << " calls, " << timing_stats_.find_optimal_face_total << " ms\n";
    }

    return current_face;
}

} // namespace internal
} // namespace geoqp
