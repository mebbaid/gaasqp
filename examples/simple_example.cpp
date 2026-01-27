#include <geoqp/geoqp.hpp>
#include <iostream>
#include <iomanip>

void print_vector(const std::string& name, const std::vector<double>& v) {
    std::cout << name << ": [";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

int main() {
    std::cout << "=== GeoQP Example: 2D Quadratic Program ===\n\n";
    
    // Problem: minimize (1/2)(x^2 + y^2) - 3x - 2y
    //          subject to x >= 0, y >= 0, x + y <= 2
    //
    // Unconstrained minimum: x_bar = [3, 2] (infeasible)
    // Expected solution: x* = [1.5, 0.5] on face x + y = 2
    
    geoqp::Problem problem;
    problem.n = 2;  // 2 variables
    problem.m = 3;  // 3 constraints
    
    // Objective: (1/2) x^T Q x + g^T x
    // Q = [1 0; 0 1] (identity)
    problem.Q = {1.0, 0.0,
                 0.0, 1.0};
    
    // g = [-3, -2]
    problem.g = {-3.0, -2.0};
    
    // Constraints: A x <= b
    // -x <= 0  (x >= 0)
    // -y <= 0  (y >= 0)
    // x + y <= 2
    problem.A = {-1.0,  0.0,   // -x <= 0
                  0.0, -1.0,   // -y <= 0
                  1.0,  1.0};  // x + y <= 2
    
    problem.b = {0.0, 0.0, 2.0};
    
    std::cout << "Problem setup:\n";
    std::cout << "  Variables: " << problem.n << "\n";
    std::cout << "  Constraints: " << problem.m << "\n";
    std::cout << "  Objective: (1/2)(x^2 + y^2) - 3x - 2y\n";
    std::cout << "  Constraints: x >= 0, y >= 0, x + y <= 2\n\n";
    
    // Configure solver
    geoqp::Config config;
    config.verbosity = 1;  // Show progress
    config.tolerance = 1e-8;
    
    // Create solver and solve
    geoqp::Solver solver(config);
    
    std::cout << "Solving...\n\n";
    geoqp::Solution solution = solver.solve(problem);
    
    // Display results
    std::cout << "\n=== Results ===\n";
    
    if (solution.status == geoqp::Status::SUCCESS) {
        std::cout << "Status: SUCCESS\n";
        print_vector("Solution x*", solution.x);
        std::cout << "Objective value: " << std::fixed << std::setprecision(6) 
                  << solution.objective_value << "\n";
        
        std::cout << "Active constraints: ";
        if (solution.active_set.empty()) {
            std::cout << "none (interior solution)\n";
        } else {
            std::cout << "[";
            for (size_t i = 0; i < solution.active_set.size(); ++i) {
                std::cout << solution.active_set[i];
                if (i < solution.active_set.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        
        print_vector("Dual multipliers", solution.lambda);
        
        std::cout << "\nPerformance:\n";
        std::cout << "  Phase I iterations: " << solution.phase1_iterations << "\n";
        std::cout << "  Total iterations: " << solution.iterations << "\n";
        std::cout << "  Solve time: " << std::scientific << solution.solve_time 
                  << " seconds\n";
        
        // Verify solution
        std::cout << "\nVerification:\n";
        double x = solution.x[0];
        double y = solution.x[1];
        std::cout << "  x = " << std::fixed << x << ", y = " << y << "\n";
        std::cout << "  x + y = " << (x + y) << " (should be ≤ 2)\n";
        std::cout << "  Feasible: " << (x >= -1e-6 && y >= -1e-6 && x + y <= 2.0 + 1e-6 ? "YES" : "NO") << "\n";
        
        // Expected solution: [1.5, 0.5]
        double error = std::sqrt((x - 1.5) * (x - 1.5) + (y - 0.5) * (y - 0.5));
        std::cout << "  Distance from expected [1.5, 0.5]: " << std::scientific << error << "\n";
        
    } else {
        std::cout << "Status: ";
        switch (solution.status) {
            case geoqp::Status::UNBOUNDED:
                std::cout << "UNBOUNDED\n";
                break;
            case geoqp::Status::INFEASIBLE:
                std::cout << "INFEASIBLE\n";
                break;
            case geoqp::Status::MAX_ITER:
                std::cout << "MAX_ITERATIONS\n";
                break;
            case geoqp::Status::NUMERICAL_ERROR:
                std::cout << "NUMERICAL_ERROR\n";
                break;
            case geoqp::Status::INVALID_INPUT:
                std::cout << "INVALID_INPUT\n";
                break;
            default:
                std::cout << "UNKNOWN\n";
        }
    }
    
    return solution.status == geoqp::Status::SUCCESS ? 0 : 1;
}
