#include "geoqp/internal.hpp"
#include <algorithm>
#include <set>
#include <iostream>

namespace geoqp {
namespace internal {

/**
 * @brief Helper functions for face hopping strategies
 * 
 * These implement different geometric active set methods:
 * - Batch: Drop top-K constraints by sensitivity (adaptive batching)
 * - Sensitivity: Drop single constraint with best predicted improvement
 * - Neighbors: Analytical lookahead using Schur complement diagonal
 */

namespace {

// Helper: Compute Q-weighted distance ||x - x_bar||_Q^2
double compute_q_distance(
    const std::vector<double>& x,
    const std::vector<double>& x_bar,
    const double* Q,
    int n
) {
    std::vector<double> diff(n);
    for (int i = 0; i < n; ++i) {
        diff[i] = x[i] - x_bar[i];
    }
    
    double distance = 0.0;
    for (int i = 0; i < n; ++i) {
        const double* Q_row = Q + i * n;
        double row_contrib = 0.0;
        for (int j = 0; j < n; ++j) {
            row_contrib += Q_row[j] * diff[j];
        }
        distance += diff[i] * row_contrib;
    }
    
    return distance;
}

// Helper: Compute projection and x_star using Schur manager
bool compute_projection_with_schur(
    const SchurComplementManager& schur_manager,
    ConstraintCache& constraint_cache,
    const std::vector<double>& x_bar,
    const double* A,
    const double* b,
    const double* Q_chol,
    int n,
    std::vector<double>& x_star,
    std::vector<double>& lambda
) {
    const auto& current_face = schur_manager.get_active_constraints();
    const int k = schur_manager.size();
    
    // Compute residual: A_F * x_bar - b_F
    std::vector<double> residual(k);
    for (int i = 0; i < k; ++i) {
        const int idx = current_face[i];
        const double* a_row = A + idx * n;
        residual[i] = LinearAlgebra::dot(a_row, x_bar.data(), n) - b[idx];
    }
    
    // Solve for lambda
    if (!schur_manager.solve(residual, lambda)) {
        return false;
    }
    
    // Compute x_star = x_bar - Q^{-1} A_F^T lambda
    std::vector<double> w(n, 0.0);
    for (int j = 0; j < k; ++j) {
        const auto& z_j = constraint_cache.get_z(current_face[j], A, Q_chol);
        for (int i = 0; i < n; ++i) {
            w[i] += lambda[j] * z_j[i];
        }
    }
    
    std::vector<double> correction(n);
    LinearAlgebra::backward_sub(Q_chol, w.data(), correction.data(), n);
    
    x_star.resize(n);
    for (int i = 0; i < n; ++i) {
        x_star[i] = x_bar[i] - correction[i];
    }
    
    return true;
}

// Helper: Check primal feasibility and collect violated constraints
bool check_primal_feasibility(
    const std::vector<double>& x_star,
    const std::vector<int>& current_face,
    const double* A,
    const double* b,
    int m,
    int n,
    double tolerance,
    std::vector<int>& to_add
) {
    to_add.clear();
    std::set<int> active_set(current_face.begin(), current_face.end());
    
    for (int i = 0; i < m; ++i) {
        if (active_set.count(i)) continue;
        
        const double* a_row = A + i * n;
        double ax = LinearAlgebra::dot(a_row, x_star.data(), n);
        if (ax > b[i] + tolerance) {
            to_add.push_back(i);
        }
    }
    
    return to_add.empty();
}

} // anonymous namespace

// Strategy 1: Batch Deactivation
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
    
    while (iterations < max_iter) {
        iterations++;
        const auto& current_face = schur_manager.get_active_constraints();
        const int k = schur_manager.size();
        
        // Compute projection
        std::vector<double> x_star, lambda;
        if (!compute_projection_with_schur(schur_manager, constraint_cache, x_bar,
                                          A, b, Q_chol, n, x_star, lambda)) {
            break;
        }
        
        // Check primal feasibility
        std::vector<int> to_add;
        bool primal_feasible = check_primal_feasibility(x_star, current_face, A, b,
                                                        m, n, config.tolerance, to_add);
        
        if (!primal_feasible) {
            // Add all violated constraints
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
        
        // BATCH DEACTIVATION: Drop top constraints by sensitivity
        // Compute sensitivities for all negative lambda constraints
        struct ConstraintSensitivity {
            int index;
            double sensitivity;
            double lambda;
        };
        
        std::vector<ConstraintSensitivity> candidates;
        for (int i = 0; i < k; ++i) {
            if (lambda[i] < -config.tolerance) {
                double L_ii = schur_manager.get_L_diagonal(i);
                if (L_ii < 1e-12) continue;
                
                // Predicted objective improvement
                double sensitivity = (lambda[i] * lambda[i]) / (2.0 * L_ii * L_ii);
                candidates.push_back({i, sensitivity, lambda[i]});
            }
        }
        
        if (!candidates.empty()) {
            // Sort by sensitivity (descending - best improvements first)
            std::sort(candidates.begin(), candidates.end(),
                     [](const ConstraintSensitivity& a, const ConstraintSensitivity& b) {
                         return a.sensitivity > b.sensitivity;
                     });
            
            // Strategy: Drop top N constraints or all above threshold
            const int max_batch_size = 10;  // Safety limit
            double max_sensitivity = candidates[0].sensitivity;
            double threshold = 0.5 * max_sensitivity;
            
            std::vector<int> to_drop;
            for (const auto& cand : candidates) {
                // Drop if: (1) in top max_batch_size AND (2) above threshold
                if (to_drop.size() < max_batch_size && cand.sensitivity >= threshold) {
                    to_drop.push_back(cand.index);
                }
            }
            
            if (!to_drop.empty()) {
                // Remove in reverse order to maintain indices
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
        
        // Optimal!
        if (config.verbosity >= 3) {
            std::cout << "  [BATCH] Optimal face with " << current_face.size() << " constraints\n";
        }
        return schur_manager.get_active_constraints();
    }
    
    return schur_manager.get_active_constraints();
}

// Strategy 2: Sensitivity Analysis
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
    
    while (iterations < max_iter) {
        iterations++;
        const auto& current_face = schur_manager.get_active_constraints();
        const int k = schur_manager.size();
        
        // Compute projection
        std::vector<double> x_star, lambda;
        if (!compute_projection_with_schur(schur_manager, constraint_cache, x_bar,
                                          A, b, Q_chol, n, x_star, lambda)) {
            break;
        }
        
        // Check primal feasibility
        std::vector<int> to_add;
        bool primal_feasible = check_primal_feasibility(x_star, current_face, A, b,
                                                        m, n, config.tolerance, to_add);
        
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
        
        // SENSITIVITY ANALYSIS: Drop constraint with best improvement
        // Exact: Δf = -lambda[i]^2 / (2 * [S^-1]_ii)
        // Approximation: Δf ≈ -lambda[i]^2 / (2 * L_ii^2)
        // where L_ii is diagonal of Cholesky factor of Schur complement
        int best_drop_idx = -1;
        double best_sensitivity = 0.0;
        
        for (int i = 0; i < k; ++i) {
            if (lambda[i] < -config.tolerance) {
                // Get diagonal of Schur Cholesky factor
                double L_ii = schur_manager.get_L_diagonal(i);
                if (L_ii < 1e-12) continue; // Avoid division by near-zero
                
                // Predicted objective improvement: Δf ≈ lambda^2 / (2 * L_ii^2)
                double sensitivity = (lambda[i] * lambda[i]) / (2.0 * L_ii * L_ii);
                
                if (sensitivity > best_sensitivity) {
                    best_sensitivity = sensitivity;
                    best_drop_idx = i;
                }
            }
        }
        
        if (best_drop_idx >= 0) {
            if (!schur_manager.remove_constraint(best_drop_idx)) {
                break;
            }
            
            if (config.verbosity >= 3) {
                std::cout << "  [SENSITIVITY] Dropped constraint with sensitivity " 
                          << best_sensitivity << "\n";
            }
            continue;
        }
        
        // Optimal!
        if (config.verbosity >= 3) {
            std::cout << "  [SENSITIVITY] Optimal face with " << current_face.size() << " constraints\n";
        }
        return schur_manager.get_active_constraints();
    }
    
    return schur_manager.get_active_constraints();
}

// Strategy 3: Face Neighbor Exploration
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
    iterations = 0;
    const int max_iter = config.max_iterations;
    
    if (!schur_manager.initialize(initial_face, A, Q_chol, n, constraint_cache)) {
        return initial_face;
    }
    
    while (iterations < max_iter) {
        iterations++;
        const auto& current_face = schur_manager.get_active_constraints();
        const int k = schur_manager.size();
        
        // Compute projection
        std::vector<double> x_star, lambda;
        if (!compute_projection_with_schur(schur_manager, constraint_cache, x_bar,
                                          A, b, Q_chol, n, x_star, lambda)) {
            break;
        }
        
        // Check primal feasibility
        std::vector<int> to_add;
        bool primal_feasible = check_primal_feasibility(x_star, current_face, A, b,
                                                        m, n, config.tolerance, to_add);
        
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
        
        // ANALYTICAL NEIGHBOR EVALUATION
        // For dropping constraint i, the new objective improvement is:
        // Δf_i = lambda[i]^2 / (2 * [S^-1]_ii) ≈ lambda[i]^2 / (2 * L_ii^2)
        // This is the steepest descent direction in the dual space.
        
        int best_drop_idx = -1;
        double best_improvement = 0.0;
        
        for (int i = 0; i < k; ++i) {
            if (lambda[i] >= -config.tolerance) {
                continue; // Only drop negative lambda constraints
            }
            
            // Get diagonal of Schur Cholesky factor
            double L_ii = schur_manager.get_L_diagonal(i);
            if (L_ii < 1e-12) continue;
            
            // Predicted objective improvement when dropping constraint i
            double improvement = (lambda[i] * lambda[i]) / (2.0 * L_ii * L_ii);
            
            if (improvement > best_improvement) {
                best_improvement = improvement;
                best_drop_idx = i;
            }
        }
        
        if (best_drop_idx >= 0) {
            if (!schur_manager.remove_constraint(best_drop_idx)) {
                break;
            }
            
            if (config.verbosity >= 3) {
                std::cout << "  [NEIGHBORS] Dropped constraint with improvement " 
                          << best_improvement << "\n";
            }
            continue;
        }
        
        // Optimal!
        if (config.verbosity >= 3) {
            std::cout << "  [NEIGHBORS] Optimal face with " << current_face.size() << " constraints\n";
        }
        return schur_manager.get_active_constraints();
    }
    
    return schur_manager.get_active_constraints();
}

} // namespace internal
} // namespace geoqp
