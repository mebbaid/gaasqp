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

int main() {
    std::cout << "=== GeoQP Timing Test ===\n\n";
    
    std::vector<int> horizons = {10, 25, 50, 75, 100};
    
    for (int h : horizons) {
        geoqp::Problem prob = makeMPCProblem(h);
        
        geoqp::Config config;
        config.tolerance = 1e-8;
        config.max_iterations = 1000;
        config.verbosity = 2;  // Show timing breakdown
        
        std::cout << "\n--- Horizon " << h << " (n=" << 2*h << ") ---\n";
        
        geoqp::Solver solver(config);
        
        // Time the solve
        auto t_start = std::chrono::high_resolution_clock::now();
        geoqp::Solution sol = solver.solve(prob);
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
        
        std::cout << "Status: " << (sol.status == geoqp::Status::SUCCESS ? "SUCCESS" : "FAILED") << "\n";
        std::cout << "Active constraints: " << sol.active_set.size() << "/" << prob.m << "\n";
        std::cout << "**TOTAL SOLVE TIME: " << elapsed.count() << " ms**\n";
    }
    
    return 0;
}
