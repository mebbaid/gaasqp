/**
 * Simple timing test - single MPC problem with detailed timing
 */
#include "geoqp/geoqp.hpp"
#include <iostream>
#include <random>
#include <chrono>

geoqp::Problem makeMPCProblem(int horizon) {
    std::mt19937 rng(42);
    const int nu = 2;
    const int n = nu * horizon;
    const int m = 2 * n;
    
    geoqp::Problem p;
    p.n = n;
    p.m = m;
    
    std::uniform_real_distribution<> dist(0.5, 2.0);
    
    p.Q.resize(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        p.Q[i * n + i] = dist(rng);
        for (int j = i + 1; j < std::min(i + nu, n); ++j) {
            double val = 0.1 * dist(rng);
            p.Q[i * n + j] = val;
            p.Q[j * n + i] = val;
        }
    }
    
    p.g.resize(n);
    std::uniform_real_distribution<> g_dist(-5.0, 5.0);
    for (int i = 0; i < n; ++i) {
        p.g[i] = g_dist(rng);
    }
    
    // Box constraints
    p.A.resize(m * n, 0.0);
    p.b.resize(m);
    for (int i = 0; i < n; ++i) {
        p.A[i * n + i] = -1.0;
        p.b[i] = 1.0;
        p.A[(n + i) * n + i] = 1.0;
        p.b[n + i] = 1.0;
    }
    
    return p;
}

geoqp::Problem makeGeneralMPCProblem(int horizon) {
    // Create MPC problem with general (non-diagonal) constraints
    std::mt19937 rng(42);
    const int nu = 2;
    const int n = nu * horizon;
    const int m = horizon + n; // Coupling constraints + box constraints
    
    geoqp::Problem p;
    p.n = n;
    p.m = m;
    
    std::uniform_real_distribution<> dist(0.5, 2.0);
    
    // Block-diagonal Q (typical for MPC)
    p.Q.resize(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        p.Q[i * n + i] = dist(rng);
        for (int j = i + 1; j < std::min(i + nu, n); ++j) {
            double val = 0.1 * dist(rng);
            p.Q[i * n + j] = val;
            p.Q[j * n + i] = val;
        }
    }
    
    p.g.resize(n);
    std::uniform_real_distribution<> g_dist(-5.0, 5.0);
    for (int i = 0; i < n; ++i) {
        p.g[i] = g_dist(rng);
    }
    
    p.A.resize(m * n, 0.0);
    p.b.resize(m);
    
    // Coupling constraints: u_{2i} + u_{2i+1} + u_{2i+2} <= bound
    std::uniform_real_distribution<> coupling_dist(0.3, 0.7);
    for (int i = 0; i < horizon; ++i) {
        int idx1 = 2 * i;
        int idx2 = 2 * i + 1;
        p.A[i * n + idx1] = coupling_dist(rng);
        p.A[i * n + idx2] = coupling_dist(rng);
        if (idx2 + 1 < n) {
            p.A[i * n + idx2 + 1] = coupling_dist(rng);
        }
        p.b[i] = 1.5;
    }
    
    // Simple box constraints: -1 <= u_i <= 1
    for (int i = 0; i < n; ++i) {
        p.A[(horizon + i) * n + i] = 1.0;
        p.b[horizon + i] = 1.0;
    }
    
    return p;
}

int main() {
    std::cout << "=== GeoQP Timing Test ===\n\n";
    
    std::vector<int> horizons = {10, 25, 50};
    
    std::cout << "\n### BOX-CONSTRAINED MPC (diagonal constraints, fast path) ###\n";
    for (int h : horizons) {
        geoqp::Problem prob = makeMPCProblem(h);
        
        geoqp::Config config;
        config.tolerance = 1e-8;
        config.max_iterations = 1000;
        config.verbosity = 2;
        config.timing_verbosity = 1;
        
        std::cout << "\n--- Horizon " << h << " (n=" << 2*h << ") ---\n";
        
        geoqp::Solver solver(config);
        auto t_start = std::chrono::high_resolution_clock::now();
        geoqp::Solution sol = solver.solve(prob);
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
        
        std::cout << "Active constraints: " << sol.active_set.size() << "/" << prob.m << "\n";
        std::cout << "**TOTAL SOLVE TIME: " << elapsed.count() << " ms**\n";
    }
    
    std::cout << "\n\n### GENERAL MPC (coupling constraints, uses caching) ###\n";
    for (int h : horizons) {
        geoqp::Problem prob = makeGeneralMPCProblem(h);
        
        geoqp::Config config;
        config.tolerance = 1e-8;
        config.max_iterations = 1000;
        config.verbosity = 2;
        config.timing_verbosity = 1;
        config.use_caching = true; // Enable caching
        
        std::cout << "\n--- Horizon " << h << " (n=" << 2*h << ", m=" << prob.m << ") ---\n";
        
        geoqp::Solver solver(config);
        auto t_start = std::chrono::high_resolution_clock::now();
        geoqp::Solution sol = solver.solve(prob);
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
        
        std::cout << "Active constraints: " << sol.active_set.size() << "/" << prob.m << "\n";
        std::cout << "**TOTAL WITH CACHING: " << elapsed.count() << " ms**\n";
    }
    
    std::cout << "\n\n### GENERAL MPC (caching DISABLED for comparison) ###\n";
    for (int h : horizons) {
        geoqp::Problem prob = makeGeneralMPCProblem(h);
        
        geoqp::Config config;
        config.tolerance = 1e-8;
        config.max_iterations = 1000;
        config.verbosity = 2;
        config.timing_verbosity = 1;
        config.use_caching = false; // Disable caching
        
        std::cout << "\n--- Horizon " << h << " (n=" << 2*h << ", m=" << prob.m << ") ---\n";
        
        geoqp::Solver solver(config);
        auto t_start = std::chrono::high_resolution_clock::now();
        geoqp::Solution sol = solver.solve(prob);
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
        
        std::cout << "Active constraints: " << sol.active_set.size() << "/" << prob.m << "\n";
        std::cout << "**TOTAL WITHOUT CACHING: " << elapsed.count() << " ms**\n";
    }
    
    return 0;
}
