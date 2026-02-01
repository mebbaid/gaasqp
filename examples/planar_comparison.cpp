/**
 * @file planar_comparison.cpp
 * @brief 2D visualization comparing GeoQP, OSQP, and qpOASES iterate trajectories
 * 
 * This example shows a simple 2D QP where the unconstrained minimum is outside
 * a polytope. We compare how each solver approaches the solution and output
 * the iterates for visualization.
 */

#include <geoqp/geoqp.hpp>
#include <qpOASES.hpp>
#include <QpSolversEigen/QpSolversEigen.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

// Store iterate information
struct Iterate {
    double x, y;
    std::string description;
};

/**
 * Simple 2D QP:
 *   min (1/2) x^T Q x + g^T x
 *   s.t. A x <= b
 * 
 * We create a pentagon-like polytope and place x_bar outside it.
 */
int main() {
    std::cout << "================================================================\n";
    std::cout << "  2D Planar QP: Comparing Solver Iterate Trajectories\n";
    std::cout << "================================================================\n\n";
    
    // Problem dimensions
    const int n = 2;
    
    // Hessian Q = I (identity) - circular level sets
    // This ensures Euclidean distance correlates with objective value
    std::vector<double> Q = {1.0, 0.0, 
                              0.0, 1.0};
    
    // Linear term: g such that x_bar = -Q^{-1}g = (3, 3) - outside the polytope
    // g = -Q * x_bar = -[3, 3]
    std::vector<double> g = {-3.0, -3.0};
    
    // Unconstrained minimum
    double x_bar_x = 3.0, x_bar_y = 3.0;
    
    std::cout << "Problem Setup:\n";
    std::cout << "  Q = I (identity) - circular level sets\n";
    std::cout << "  Unconstrained minimum x_bar = (" << x_bar_x << ", " << x_bar_y << ")\n\n";
    
    // Create a polytope (irregular pentagon)
    // Constraints: A x <= b
    //   x <= 2        (right wall)
    //   y <= 2        (top wall)
    //   -x <= 1       (left wall at x = -1)
    //   -y <= 1       (bottom wall at y = -1)
    //   x + y <= 3    (diagonal cut at top-right)
    //   -x - y <= 0   (diagonal cut at bottom-left, passes through origin)
    
    const int m = 6;
    std::vector<double> A = {
         1.0,  0.0,   // x <= 2
         0.0,  1.0,   // y <= 2
        -1.0,  0.0,   // -x <= 1  (x >= -1)
         0.0, -1.0,   // -y <= 1  (y >= -1)
         1.0,  1.0,   // x + y <= 3
        -1.0, -1.0    // -x - y <= 0  (x + y >= 0)
    };
    std::vector<double> b = {2.0, 2.0, 1.0, 1.0, 3.0, 0.0};
    
    std::cout << "Polytope (6 constraints):\n";
    std::cout << "  x <= 2, y <= 2, x >= -1, y >= -1\n";
    std::cout << "  x + y <= 3 (diagonal)\n";
    std::cout << "  x + y >= 0 (diagonal)\n\n";
    
    // The optimal solution should be on the diagonal x + y = 3
    // where the ellipse is tangent to this line
    
    std::cout << "========================================\n";
    std::cout << "  GeoQP Solver (Geometric Active Set)\n";
    std::cout << "========================================\n\n";
    
    std::vector<Iterate> geoqp_iterates;
    
    // Add x_bar as starting reference
    geoqp_iterates.push_back({x_bar_x, x_bar_y, "x_bar (unconstrained min)"});
    
    {
        geoqp::Problem problem;
        problem.n = n;
        problem.m = m;
        problem.Q = Q;
        problem.g = g;
        problem.A = A;
        problem.b = b;
        
        geoqp::Config config;
        config.verbosity = 2;  // Show details
        
        geoqp::Solver solver(config);
        geoqp::Solution sol = solver.solve(problem);
        
        std::cout << "\nGeoQP Result:\n";
        std::cout << "  Solution: (" << sol.x[0] << ", " << sol.x[1] << ")\n";
        std::cout << "  Objective: " << sol.objective_value << "\n";
        std::cout << "  Active constraints: " << sol.num_active_constraints << "\n";
        std::cout << "  Initializer iterations: " << sol.phase1_iterations << "\n";
        std::cout << "  Face hops: " << sol.phase2_iterations << "\n";
        std::cout << "  Total time: " << (sol.phase1_time + sol.phase2_time + sol.phase3_time) * 1000 << " ms\n\n";
        
        geoqp_iterates.push_back({sol.x[0], sol.x[1], "GeoQP solution"});
    }
    
    std::cout << "========================================\n";
    std::cout << "  qpOASES Solver (Parametric Active Set)\n";
    std::cout << "========================================\n\n";
    
    std::vector<Iterate> qpoases_iterates;
    qpoases_iterates.push_back({x_bar_x, x_bar_y, "x_bar (unconstrained min)"});
    
    {
        // qpOASES uses row-major for matrices
        qpOASES::real_t H[4] = {1.0, 0.0, 0.0, 1.0};
        qpOASES::real_t g_qp[2] = {-3.0, -3.0};
        qpOASES::real_t A_qp[12] = {
             1.0,  0.0,
             0.0,  1.0,
            -1.0,  0.0,
             0.0, -1.0,
             1.0,  1.0,
            -1.0, -1.0
        };
        qpOASES::real_t ubA[6] = {2.0, 2.0, 1.0, 1.0, 3.0, 0.0};
        qpOASES::real_t lbA[6] = {-1e10, -1e10, -1e10, -1e10, -1e10, -1e10};
        
        qpOASES::QProblem qp(2, 6);
        qpOASES::Options options;
        options.printLevel = qpOASES::PL_MEDIUM;
        qp.setOptions(options);
        
        qpOASES::int_t nWSR = 100;
        qp.init(H, g_qp, A_qp, nullptr, nullptr, lbA, ubA, nWSR);
        
        qpOASES::real_t x_opt[2];
        qp.getPrimalSolution(x_opt);
        
        double obj = qp.getObjVal();
        int iterations = static_cast<int>(nWSR);
        
        std::cout << "\nqpOASES Result:\n";
        std::cout << "  Solution: (" << x_opt[0] << ", " << x_opt[1] << ")\n";
        std::cout << "  Objective: " << obj << "\n";
        std::cout << "  Working set recalculations: " << iterations << "\n\n";
        
        qpoases_iterates.push_back({x_opt[0], x_opt[1], "qpOASES solution"});
    }
    
    std::cout << "========================================\n";
    std::cout << "  OSQP Solver (ADMM)\n";
    std::cout << "========================================\n\n";
    
    std::vector<Iterate> osqp_iterates;
    osqp_iterates.push_back({x_bar_x, x_bar_y, "x_bar (unconstrained min)"});
    
    {
        // Setup OSQP with QpSolversEigen
        Eigen::SparseMatrix<double> H_sparse(2, 2);
        H_sparse.insert(0, 0) = 1.0;
        H_sparse.insert(1, 1) = 1.0;
        H_sparse.makeCompressed();
        
        Eigen::VectorXd g_vec(2);
        g_vec << -3.0, -3.0;
        
        Eigen::SparseMatrix<double> A_sparse(6, 2);
        A_sparse.insert(0, 0) = 1.0;
        A_sparse.insert(1, 1) = 1.0;
        A_sparse.insert(2, 0) = -1.0;
        A_sparse.insert(3, 1) = -1.0;
        A_sparse.insert(4, 0) = 1.0; A_sparse.insert(4, 1) = 1.0;
        A_sparse.insert(5, 0) = -1.0; A_sparse.insert(5, 1) = -1.0;
        A_sparse.makeCompressed();
        
        Eigen::VectorXd lowerBound(6);
        lowerBound << -1e10, -1e10, -1e10, -1e10, -1e10, -1e10;
        
        Eigen::VectorXd upperBound(6);
        upperBound << 2.0, 2.0, 1.0, 1.0, 3.0, 0.0;
        
        QpSolversEigen::Solver solver;
        solver.instantiateSolver("osqp");
        solver.setBooleanParameter("verbose", true);
        
        solver.data()->setNumberOfVariables(2);
        solver.data()->setNumberOfInequalityConstraints(6);
        solver.data()->setHessianMatrix(H_sparse);
        solver.data()->setGradient(g_vec);
        solver.data()->setInequalityConstraintsMatrix(A_sparse);
        solver.data()->setLowerBound(lowerBound);
        solver.data()->setUpperBound(upperBound);
        
        solver.initSolver();
        solver.solveProblem();
        
        Eigen::VectorXd x_opt = solver.getSolution();
        
        std::cout << "\nOSQP Result:\n";
        std::cout << "  Solution: (" << x_opt[0] << ", " << x_opt[1] << ")\n";
        double obj = solver.getObjValue();
        std::cout << "  Objective: " << obj << "\n\n";
        
        osqp_iterates.push_back({x_opt[0], x_opt[1], "OSQP solution"});
    }
    
    // Summary comparison
    std::cout << "\n================================================================\n";
    std::cout << "  Summary: Iterate Trajectories\n";
    std::cout << "================================================================\n\n";
    
    std::cout << "Starting point (unconstrained minimum):\n";
    std::cout << "  x_bar = (3.0, 3.0) - OUTSIDE polytope\n\n";
    
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "GeoQP iterates (geometric face hopping):\n";
    for (size_t i = 0; i < geoqp_iterates.size(); ++i) {
        std::cout << "  [" << i << "] (" << geoqp_iterates[i].x << ", " 
                  << geoqp_iterates[i].y << ") - " << geoqp_iterates[i].description << "\n";
    }
    
    std::cout << "\nqpOASES iterates (parametric active set):\n";
    for (size_t i = 0; i < qpoases_iterates.size(); ++i) {
        std::cout << "  [" << i << "] (" << qpoases_iterates[i].x << ", " 
                  << qpoases_iterates[i].y << ") - " << qpoases_iterates[i].description << "\n";
    }
    
    std::cout << "\nOSQP iterates (ADMM - many internal iterations):\n";
    for (size_t i = 0; i < osqp_iterates.size(); ++i) {
        std::cout << "  [" << i << "] (" << osqp_iterates[i].x << ", " 
                  << osqp_iterates[i].y << ") - " << osqp_iterates[i].description << "\n";
    }
    
    std::cout << "\n================================================================\n";
    std::cout << "  Key Insight\n";
    std::cout << "================================================================\n\n";
    
    std::cout << "GeoQP's approach:\n";
    std::cout << "  1. Start at x_bar (unconstrained minimum)\n";
    std::cout << "  2. Ray trace toward feasible region -> find blocking face\n";
    std::cout << "  3. Hop to neighboring faces until optimal face found\n";
    std::cout << "  4. Project x_bar onto optimal face -> solution\n\n";
    
    std::cout << "This is fundamentally different from:\n";
    std::cout << "  - qpOASES: Starts from a feasible point, moves along constraint surfaces\n";
    std::cout << "  - OSQP: Uses ADMM with many small steps through the interior\n\n";
    
    std::cout << "For problems where x_bar is outside but the optimal face is 'visible',\n";
    std::cout << "GeoQP can find the solution with very few face hops.\n\n";
    
    // Output data for plotting
    std::cout << "================================================================\n";
    std::cout << "  Data for Plotting (Python/Matplotlib)\n";
    std::cout << "================================================================\n\n";
    
    std::cout << "# Polytope vertices (approximate for visualization)\n";
    std::cout << "polytope_x = [-1, 2, 2, 1, -1, 0, -1]\n";
    std::cout << "polytope_y = [1, 1, 2, 2, -1, -1, 1]\n\n";
    
    std::cout << "# Unconstrained minimum\n";
    std::cout << "x_bar = (3.0, 3.0)\n\n";
    
    std::cout << "# Solutions (should all be approximately the same)\n";
    std::cout << "geoqp_sol = (" << geoqp_iterates.back().x << ", " << geoqp_iterates.back().y << ")\n";
    std::cout << "qpoases_sol = (" << qpoases_iterates.back().x << ", " << qpoases_iterates.back().y << ")\n";
    std::cout << "osqp_sol = (" << osqp_iterates.back().x << ", " << osqp_iterates.back().y << ")\n";
    
    return 0;
}
