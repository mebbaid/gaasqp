/**
 * Debug test for GeoQP Initializer
 */
#include <geoqp/geoqp.hpp>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    // Simple 2D problem with box constraints [-1, 1]^2
    // x_bar outside the box
    
    int n = 2;
    int m = 4;  // 4 constraints: x1 <= 1, x2 <= 1, -x1 <= 1, -x2 <= 1
    
    geoqp::Problem problem;
    problem.n = n;
    problem.m = m;
    
    // Q = I (identity)
    problem.Q = {1.0, 0.0, 0.0, 1.0};
    
    // g such that x_bar = -Q^{-1}g = [2, 2] (outside box)
    problem.g = {-2.0, -2.0};
    
    // Box constraints [-1, 1]^2
    // x1 <= 1:   [1, 0] x <= 1
    // x2 <= 1:   [0, 1] x <= 1
    // -x1 <= 1:  [-1, 0] x <= 1
    // -x2 <= 1:  [0, -1] x <= 1
    problem.A = {
        1.0, 0.0,   // x1 <= 1
        0.0, 1.0,   // x2 <= 1
        -1.0, 0.0,  // -x1 <= 1
        0.0, -1.0   // -x2 <= 1
    };
    problem.b = {1.0, 1.0, 1.0, 1.0};
    
    std::cout << "Simple 2D test:\n";
    std::cout << "  Q = I, g = [-2, -2]\n";
    std::cout << "  x_bar = -Q^{-1}g = [2, 2] (outside box)\n";
    std::cout << "  Box: [-1, 1]^2\n";
    std::cout << "  Expected optimal: [1, 1] (corner closest to x_bar)\n\n";
    
    geoqp::Config config;
    config.verbosity = 2;  // Debug output
    geoqp::Solver solver(config);
    
    auto sol = solver.solve(problem);
    
    std::cout << "\nResult:\n";
    std::cout << "  Status: " << (sol.status == geoqp::Status::SUCCESS ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "  x = [" << sol.x[0] << ", " << sol.x[1] << "]\n";
    std::cout << "  Objective: " << sol.objective_value << "\n";
    std::cout << "  Active constraints: " << sol.num_active_constraints << "\n";
    std::cout << "  Phase 1 iters: " << sol.phase1_iterations << "\n";
    std::cout << "  Phase 2 iters: " << sol.phase2_iterations << "\n";
    std::cout << "  Vertices explored: " << sol.vertices_explored << "\n";
    
    // Expected: x = [1, 1], obj = 0.5*2 - 4 = -3
    double expected_obj = 0.5 * (1*1 + 1*1) + (-2)*1 + (-2)*1;
    std::cout << "\n  Expected x = [1, 1], obj = " << expected_obj << "\n";
    
    if (std::abs(sol.x[0] - 1.0) > 0.01 || std::abs(sol.x[1] - 1.0) > 0.01) {
        std::cout << "\n  ERROR: Solution is wrong!\n";
        return 1;
    }
    
    std::cout << "\n  CORRECT!\n";
    return 0;
}
