#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
#include <algorithm>
#include <set>
#include <iostream>

namespace geoqp {
namespace internal {

namespace {

double compute_q_distance(
    const std::vector<double>& x,
    const std::vector<double>& x_bar,
    const double* GEOQP_RESTRICT Q,
    int n
) {
    const double *GEOQP_RESTRICT x_ptr = x.data();
    const double *GEOQP_RESTRICT xbar_ptr = x_bar.data();
    
    double distance = 0.0;
    
    for (int i = 0; i < n; ++i) {
        const double *GEOQP_RESTRICT Q_row = Q + i * n;
        const double diff_i = x_ptr[i] - xbar_ptr[i];
        
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;
        
        int j = 0;
        for (; j + 3 < n; j += 4) {
            double diff_j0 = x_ptr[j] - xbar_ptr[j];
            double diff_j1 = x_ptr[j + 1] - xbar_ptr[j + 1];
            double diff_j2 = x_ptr[j + 2] - xbar_ptr[j + 2];
            double diff_j3 = x_ptr[j + 3] - xbar_ptr[j + 3];
            
            sum0 += Q_row[j] * diff_j0;
            sum1 += Q_row[j + 1] * diff_j1;
            sum2 += Q_row[j + 2] * diff_j2;
            sum3 += Q_row[j + 3] * diff_j3;
        }
        
        double row_contrib = (sum0 + sum1) + (sum2 + sum3);
        
        for (; j < n; ++j) {
            row_contrib += Q_row[j] * (x_ptr[j] - xbar_ptr[j]);
        }
        
        distance += diff_i * row_contrib;
    }
    
    return distance;
}

bool compute_projection_with_schur(
    const SchurComplementManager& schur_manager,
    ConstraintCache& constraint_cache,
    const std::vector<double>& x_bar,
    const double* GEOQP_RESTRICT A,
    const double* GEOQP_RESTRICT b,
    const double* GEOQP_RESTRICT Q_chol,
    int n,
    std::vector<double>& x_star,
    std::vector<double>& lambda,
    std::vector<double>& ws_residual,  
    std::vector<double>& ws_w,         
    std::vector<double>& ws_correction 
) {
    const auto& current_face = schur_manager.get_active_constraints();
    const int k = schur_manager.size();
    
    // Ensure the residual workspace has exactly k entries (must match Schur size)
    if (static_cast<int>(ws_residual.size()) != k) ws_residual.resize(k);
    if (static_cast<int>(ws_w.size()) < n) ws_w.resize(n);
    if (static_cast<int>(ws_correction.size()) < n) ws_correction.resize(n);
    
    const double *GEOQP_RESTRICT xbar_ptr = x_bar.data();
    
    for (int i = 0; i < k; ++i) {
        const int idx = current_face[i];
        const double *GEOQP_RESTRICT a_row = A + idx * n;
        ws_residual[i] = LinearAlgebra::dot(a_row, xbar_ptr, n) - b[idx];
    }
    
    if (!schur_manager.solve(ws_residual, lambda)) {
        return false;
    }
    
    std::fill(ws_w.begin(), ws_w.begin() + n, 0.0);
    
    double *GEOQP_RESTRICT w_ptr = ws_w.data();
    const double *GEOQP_RESTRICT lambda_ptr = lambda.data();
    
    for (int j = 0; j < k; ++j) {
        const auto& z_j = constraint_cache.get_z(current_face[j], A, Q_chol);
        const double *GEOQP_RESTRICT z_ptr = z_j.data();
        const double lambda_j = lambda_ptr[j];
        
        int i = 0;
        for (; i + 3 < n; i += 4) {
            w_ptr[i] += lambda_j * z_ptr[i];
            w_ptr[i + 1] += lambda_j * z_ptr[i + 1];
            w_ptr[i + 2] += lambda_j * z_ptr[i + 2];
            w_ptr[i + 3] += lambda_j * z_ptr[i + 3];
        }
        for (; i < n; ++i) {
            w_ptr[i] += lambda_j * z_ptr[i];
        }
    }
    
    LinearAlgebra::backward_sub(Q_chol, w_ptr, ws_correction.data(), n);
    
    x_star.resize(n);
    double *GEOQP_RESTRICT xstar_ptr = x_star.data();
    const double *GEOQP_RESTRICT corr_ptr = ws_correction.data();
    
    for (int i = 0; i < n; ++i) {
        xstar_ptr[i] = xbar_ptr[i] - corr_ptr[i];
    }
    
    return true;
}

bool check_primal_feasibility(
    const std::vector<double>& x_star,
    const std::vector<int>& current_face,
    const double* GEOQP_RESTRICT A,
    const double* GEOQP_RESTRICT b,
    int m,
    int n,
    double tolerance,
    std::vector<int>& to_add
) {
    to_add.clear();
    std::set<int> active_set(current_face.begin(), current_face.end());
    
    const double *GEOQP_RESTRICT x_ptr = x_star.data();
    
    for (int i = 0; i < m; ++i) {
        if (active_set.count(i)) continue;
        
        const double *GEOQP_RESTRICT a_row = A + i * n;
        
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;
        
        int j = 0;
        for (; j + 3 < n; j += 4) {
            sum0 += a_row[j] * x_ptr[j];
            sum1 += a_row[j + 1] * x_ptr[j + 1];
            sum2 += a_row[j + 2] * x_ptr[j + 2];
            sum3 += a_row[j + 3] * x_ptr[j + 3];
        }
        
        double ax = (sum0 + sum1) + (sum2 + sum3);
        
        for (; j < n; ++j) {
            ax += a_row[j] * x_ptr[j];
        }
        
        if (ax > b[i] + tolerance) {
            to_add.push_back(i);
        }
    }
    
    return to_add.empty();
}

} // anonymous namespace

std::vector<int> hopping_strategy_batch(
    const std::vector<int>& initial_face,
    int& iterations,
    Timer& t,
    const double* Q,
    const double* A,
    const double* b,
    const std::vector<double>& x_bar,
    int m,
    int n,
    const Config& config,
    ConstraintCache& constraint_cache,
    SchurComplementManager& schur_manager,
    const double* Q_chol
) {
    iterations = 0;
    const int max_iter = config.max_iterations;
    
    if (!schur_manager.initialize(initial_face, A, Q_chol, n, constraint_cache)) {
        return initial_face;
    }
    
    std::vector<double> ws_residual(m);
    std::vector<double> ws_w(n);
    std::vector<double> ws_correction(n);
    
    while (iterations < max_iter) {
        iterations++;
        const auto& current_face = schur_manager.get_active_constraints();
        const int k = schur_manager.size();
        
        std::vector<double> x_star, lambda;
        if (!compute_projection_with_schur(schur_manager, constraint_cache, x_bar,
                                          A, b, Q_chol, n, x_star, lambda,
                                          ws_residual, ws_w, ws_correction)) {
            break;
        }
        
        std::vector<int> to_add;
        bool primal_feasible = check_primal_feasibility(x_star, current_face, A, b,
                                                        m, n, config.constraint_tolerance, to_add);
        
        if (!primal_feasible) {
            bool all_added = true;
            for (int c : to_add) {
                if (!schur_manager.add_constraint(c, A, Q_chol, n, constraint_cache)) {
                    all_added = false;
                    break;
                }
            }
            if (!all_added) break;
            continue;
        }
        
        struct ConstraintSensitivity {
            int index;
            double sensitivity;
            double lambda;
        };
        
        std::vector<ConstraintSensitivity> candidates;
        for (int i = 0; i < k; ++i) {
            if (lambda[i] < -config.constraint_tolerance) {
                double L_ii = schur_manager.get_L_diagonal(i);
                if (L_ii < 1e-12) continue;
                
                double sensitivity = (lambda[i] * lambda[i]) / (2.0 * L_ii * L_ii);
                candidates.push_back({i, sensitivity, lambda[i]});
            }
        }
        
        if (!candidates.empty()) {
            std::sort(candidates.begin(), candidates.end(),
                     [](const ConstraintSensitivity& a, const ConstraintSensitivity& b) {
                         return a.sensitivity > b.sensitivity;
                     });
            
            const int max_batch_size = 10;  // Safety limit
            double max_sensitivity = candidates[0].sensitivity;
            double threshold = 0.5 * max_sensitivity;
            
            std::vector<int> to_drop;
            for (const auto& cand : candidates) {
                if (to_drop.size() < max_batch_size && cand.sensitivity >= threshold) {
                    to_drop.push_back(cand.index);
                }
            }
            
            if (!to_drop.empty()) {
                std::sort(to_drop.rbegin(), to_drop.rend());
                bool all_removed = true;
                for (int idx : to_drop) {
                    if (!schur_manager.remove_constraint(idx)) {
                        all_removed = false;
                        break;
                    }
                }
                if (!all_removed) break;
                
                if (config.verbosity >= 3) {
                    std::cout << "  [BATCH] Dropped " << to_drop.size() 
                              << " of " << candidates.size() << " negative constraints"
                              << " (max sens: " << max_sensitivity << ")\n";
                }
                continue;
            }
        }
        
        if (config.verbosity >= 3) {
            std::cout << "  [BATCH] Optimal face with " << current_face.size() << " constraints\n";
        }
        return schur_manager.get_active_constraints();
    }
    
    return schur_manager.get_active_constraints();
}

std::vector<int> hopping_strategy_sensitivity(
    const std::vector<int>& initial_face,
    int& iterations,
    Timer& t,
    const double* Q,
    const double* A,
    const double* b,
    const std::vector<double>& x_bar,
    int m,
    int n,
    const Config& config,
    ConstraintCache& constraint_cache,
    SchurComplementManager& schur_manager,
    const double* Q_chol
) {
    iterations = 0;
    const int max_iter = config.max_iterations;

    if (!schur_manager.initialize(initial_face, A, Q_chol, n, constraint_cache)) {
        return initial_face;
    }

    std::vector<double> ws_residual(m);
    std::vector<double> ws_w(n);
    std::vector<double> ws_correction(n);

    while (iterations < max_iter) {
        iterations++;
        const auto& current_face = schur_manager.get_active_constraints();
        const int k = schur_manager.size();

        std::vector<double> x_star, lambda;
        if (!compute_projection_with_schur(schur_manager, constraint_cache, x_bar,
                                          A, b, Q_chol, n, x_star, lambda,
                                          ws_residual, ws_w, ws_correction)) {
            break;
        }

        std::vector<int> to_add;
        bool primal_feasible = check_primal_feasibility(x_star, current_face, A, b,
                                                        m, n, config.constraint_tolerance, to_add);

        if (!primal_feasible) {
            bool all_added = true;
            for (int c : to_add) {
                if (!schur_manager.add_constraint(c, A, Q_chol, n, constraint_cache)) {
                    all_added = false;
                    break;
                }
            }
            if (!all_added) break;
            continue;
        }

        int best_drop_idx = -1;
        double best_sensitivity = 0.0;

        for (int i = 0; i < k; ++i) {
            if (lambda[i] < -config.constraint_tolerance) {
                double L_ii = schur_manager.get_L_diagonal(i);
                if (L_ii < 1e-12) continue;
                double sensitivity = (lambda[i] * lambda[i]) / (2.0 * L_ii * L_ii);
                if (sensitivity > best_sensitivity) {
                    best_sensitivity = sensitivity;
                    best_drop_idx = i;
                }
            }
        }

        if (best_drop_idx >= 0) {
            if (!schur_manager.remove_constraint(best_drop_idx)) break;
            if (config.verbosity >= 3)
                std::cout << "  [SENSITIVITY] Dropped constraint with sensitivity "
                          << best_sensitivity << "\n";
            continue;
        }

        if (config.verbosity >= 3)
            std::cout << "  [SENSITIVITY] Optimal face with " << current_face.size() << " constraints\n";
        return schur_manager.get_active_constraints();
    }

    return schur_manager.get_active_constraints();
}

// hopping_strategy_neighbors uses the same single-drop sensitivity criterion.
std::vector<int> hopping_strategy_neighbors(
    const std::vector<int>& initial_face,
    int& iterations,
    Timer& t,
    const double* Q,
    const double* A,
    const double* b,
    const std::vector<double>& x_bar,
    int m,
    int n,
    const Config& config,
    ConstraintCache& constraint_cache,
    SchurComplementManager& schur_manager,
    const double* Q_chol
) {
    return hopping_strategy_sensitivity(initial_face, iterations, t, Q, A, b,
                                        x_bar, m, n, config, constraint_cache,
                                        schur_manager, Q_chol);
}

} // namespace internal
} // namespace geoqp
