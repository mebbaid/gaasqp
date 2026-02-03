#include <geoqp/geoqp.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <cmath>

// Create realistic MPC problem matching benchmark_osqp.cpp Test 2
geoqp::Problem makeMPCQP(int horizon) {
    const int n_states = 4;
    const int n_inputs = 2;
    const int n = n_inputs * horizon;
    
    geoqp::Problem prob;
    prob.n = n;
    prob.m = 2 * n; // Box constraints: -1 <= u <= 1
    
    // Build state-to-input propagation matrix Su (n_states*horizon x n)
    // This creates dense coupling via state dynamics
    std::vector<double> Su(n_states * horizon * n, 0.0);
    for (int k = 0; k < horizon; ++k) {
        for (int j = 0; j <= k; ++j) {
            double decay = std::pow(0.9, k - j);
            for (int i = 0; i < n_states; ++i) {
                for (int l = 0; l < n_inputs; ++l) {
                    if (i % n_inputs == l) {
                        Su[(k * n_states + i) * n + (j * n_inputs + l)] = decay * 0.05;
                    }
                }
            }
        }
    }
    
    // State cost matrix Q (n_states*horizon x n_states*horizon, diagonal)
    std::vector<double> Q_state(n_states * horizon * n_states * horizon, 0.0);
    for (int i = 0; i < n_states * horizon; ++i) {
        Q_state[i * n_states * horizon + i] = 1.0;
    }
    
    // Input cost matrix R (n x n, diagonal)
    std::vector<double> R(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        R[i * n + i] = 0.1;
    }
    
    // Compute H = Su^T * Q_state * Su + R
    // This creates a dense Hessian due to state propagation
    prob.Q.resize(n * n, 0.0);
    
    // First: SuT_Q = Su^T * Q_state (n x n_states*horizon)
    std::vector<double> SuT_Q(n * n_states * horizon, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n_states * horizon; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n_states * horizon; ++k) {
                sum += Su[k * n + i] * Q_state[k * n_states * horizon + j];
            }
            SuT_Q[i * n_states * horizon + j] = sum;
        }
    }
    
    // Second: H = SuT_Q * Su + R (n x n)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n_states * horizon; ++k) {
                sum += SuT_Q[i * n_states * horizon + k] * Su[k * n + j];
            }
            prob.Q[i * n + j] = sum + R[i * n + j];
        }
    }
    
    // Make H symmetric (numerical precision)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double avg = 0.5 * (prob.Q[i * n + j] + prob.Q[j * n + i]);
            prob.Q[i * n + j] = avg;
            prob.Q[j * n + i] = avg;
        }
    }
    
    // Linear term: random gradient (matching benchmark_osqp.cpp)
    std::mt19937 rng(42);
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    prob.g.resize(n);
    for (int i = 0; i < n; ++i) {
        prob.g[i] = dist(rng);
    }
    
    // Box constraints: -1 <= u <= 1
    prob.A.resize(prob.m * n, 0.0);
    prob.b.resize(prob.m);
    
    for (int i = 0; i < n; ++i) {
        // Upper bound: u[i] <= 1
        prob.A[i * n + i] = 1.0;
        prob.b[i] = 1.0;
        
        // Lower bound: -u[i] <= 1 (i.e., u[i] >= -1)
        prob.A[(n + i) * n + i] = -1.0;
        prob.b[n + i] = 1.0;
    }
    
    return prob;
}

struct BenchmarkResult {
    std::string strategy_name;
    double solve_time_ms;
    double phase2_time_ms;
    int iterations;
    int phase2_iterations;
    int active_constraints;
    double objective_value;
    bool success;
};

BenchmarkResult runBenchmark(const geoqp::Problem& prob, geoqp::FaceHoppingStrategy strategy, const std::string& name, int verbosity = 0) {
    BenchmarkResult result;
    result.strategy_name = name;
    
    geoqp::Config config;
    config.verbosity = verbosity;
    config.use_caching = true;
    config.face_hopping_strategy = strategy;
    
    geoqp::Solver solver(config);
    
    auto start = std::chrono::high_resolution_clock::now();
    geoqp::Solution sol = solver.solve(prob);
    auto end = std::chrono::high_resolution_clock::now();
    
    result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.phase2_time_ms = sol.phase2_time * 1000.0;
    result.iterations = sol.iterations;
    result.phase2_iterations = sol.phase2_iterations;
    result.active_constraints = sol.num_active_constraints;
    result.objective_value = sol.objective_value;
    result.success = (sol.status == geoqp::Status::SUCCESS);
    
    return result;
}

void printResults(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n";
    std::cout << std::string(120, '=') << "\n";
    std::cout << std::setw(20) << "Strategy" 
              << std::setw(15) << "Solve (ms)"
              << std::setw(15) << "Phase2 (ms)"
              << std::setw(12) << "Total Iter"
              << std::setw(12) << "Phase2 Iter"
              << std::setw(15) << "Active Cons"
              << std::setw(18) << "Objective"
              << std::setw(10) << "Status" << "\n";
    std::cout << std::string(120, '-') << "\n";
    
    for (const auto& r : results) {
        std::cout << std::setw(20) << r.strategy_name
                  << std::setw(15) << std::fixed << std::setprecision(3) << r.solve_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(3) << r.phase2_time_ms
                  << std::setw(12) << r.iterations
                  << std::setw(12) << r.phase2_iterations
                  << std::setw(15) << r.active_constraints
                  << std::setw(18) << std::scientific << std::setprecision(6) << r.objective_value
                  << std::setw(10) << (r.success ? "SUCCESS" : "FAIL") << "\n";
    }
    std::cout << std::string(120, '=') << "\n";
}

void compareStrategies(int horizon) {
    std::cout << "\n=== MPC Problem: Horizon = " << horizon << " (n = " << 2*horizon << ") ===\n";
    
    geoqp::Problem prob = makeMPCQP(horizon);
    
    std::vector<BenchmarkResult> results;
    
    // Test all strategies
    results.push_back(runBenchmark(prob, geoqp::FaceHoppingStrategy::STANDARD, "Standard (Dantzig)"));
    results.push_back(runBenchmark(prob, geoqp::FaceHoppingStrategy::BATCH, "Batch Deactivation"));
    results.push_back(runBenchmark(prob, geoqp::FaceHoppingStrategy::SENSITIVITY, "Sensitivity"));
    results.push_back(runBenchmark(prob, geoqp::FaceHoppingStrategy::NEIGHBORS, "Face Neighbors"));
    
    printResults(results);
    
    // Compute speedups relative to standard
    double standard_time = results[0].phase2_time_ms;
    std::cout << "\nPhase II Speedups vs Standard:\n";
    for (size_t i = 1; i < results.size(); ++i) {
        double speedup = standard_time / results[i].phase2_time_ms;
        std::cout << "  " << std::setw(20) << std::left << results[i].strategy_name << ": " 
                  << std::fixed << std::setprecision(2) << speedup << "x";
        if (speedup > 1.05) {
            std::cout << " ✓ FASTER";
        } else if (speedup < 0.95) {
            std::cout << " ✗ SLOWER";
        } else {
            std::cout << " ~ SAME";
        }
        std::cout << "\n";
    }
    
    // Check iteration reduction
    int standard_iters = results[0].phase2_iterations;
    std::cout << "\nPhase II Iteration Reduction vs Standard:\n";
    for (size_t i = 1; i < results.size(); ++i) {
        int reduction = standard_iters - results[i].phase2_iterations;
        double pct = (standard_iters > 0) ? 100.0 * reduction / standard_iters : 0.0;
        std::cout << "  " << std::setw(20) << std::left << results[i].strategy_name << ": " 
                  << reduction << " iterations (" << std::fixed << std::setprecision(1) << pct << "%)";
        if (reduction > 0) {
            std::cout << " ✓";
        }
        std::cout << "\n";
    }
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     Face Hopping Strategy Ablation Study                      ║\n";
    std::cout << "║     Comparing Geometric vs Standard Active Set Methods        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    // Test different problem sizes
    std::vector<int> horizons = {5, 10, 20, 50};
    
    for (int h : horizons) {
        compareStrategies(h);
        std::cout << "\n";
    }
    
    std::cout << "\nStrategy Descriptions:\n";
    std::cout << "  Standard (Dantzig):   Drop constraint with most negative lambda (classic)\n";
    std::cout << "  Batch Deactivation:   Drop ALL negative lambda constraints in one iteration\n";
    std::cout << "  Sensitivity:          Drop constraint with best predicted improvement\n";
    std::cout << "  Face Neighbors:       Evaluate all adjacent faces, pick best one\n";
    std::cout << "\n";
    
    return 0;
}
