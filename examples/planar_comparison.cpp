/**
 * @file planar_octagon.cpp
 * @brief 2D Benchmark: GeoQP vs OSQP vs qpOASES on an Octagon
 *
 * Geometry: Regular Octagon created by intersecting a Box with a Diamond.
 * Unconstrained Minimum: Placed outside to test active-set identification.
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

// Helper to print nice output
struct Iterate
{
    double x, y;
    std::string description;
};

int main()
{
    std::cout << "================================================================\n";
    std::cout << "  2D Octagon QP: Comparing Solver Trajectories\n";
    std::cout << "================================================================\n\n";

    // ---------------------------------------------------------
    // 1. UNIFIED PROBLEM DEFINITION (The "Ground Truth")
    // ---------------------------------------------------------
    const int n = 2;
    const int m = 8; // Octagon = 8 inequalities

    // Objective: minimize ||x - x_bar||^2
    // x_bar = (4.0, 1.0) -> Target is to the right, slightly up
    double x_bar_x = 4.0;
    double x_bar_y = 1.0;

    // Q = I, g = -x_bar
    std::vector<double> Q_shared = {1.0, 0.0, 0.0, 1.0};
    std::vector<double> g_shared = {-x_bar_x, -x_bar_y};

    // Octagon Constraints (Intersection of Box and Diamond)
    // 1. Right:  x <= 2
    // 2. Top:    y <= 2
    // 3. Left:  -x <= 2  (x >= -2)
    // 4. Bottom:-y <= 2  (y >= -2)
    // 5. TR Diag: x + y <= 3
    // 6. TL Diag:-x + y <= 3
    // 7. BL Diag:-x - y <= 3
    // 8. BR Diag: x - y <= 3

    std::vector<double> A_shared = {
        1.0, 0.0,   // x <= 2
        0.0, 1.0,   // y <= 2
        -1.0, 0.0,  // x >= -2
        0.0, -1.0,  // y >= -2
        1.0, 1.0,   // x + y <= 3
        -1.0, 1.0,  // -x + y <= 3
        -1.0, -1.0, // -x - y <= 3
        1.0, -1.0   // x - y <= 3
    };

    std::vector<double> b_shared = {
        2.0, 2.0, 2.0, 2.0, // Box bounds
        3.0, 3.0, 3.0, 3.0  // Diagonal cuts
    };

    std::cout << "Problem Setup (Octagon):\n";
    std::cout << "  Target x_bar = (" << x_bar_x << ", " << x_bar_y << ")\n";
    std::cout << "  Constraints: Box[-2, 2] intersected with |x|+|y|<=3\n";

    // Geometric intuition:
    // Solution should be the vertex (2, 1).
    std::cout << "  Expected Solution: (2.0, 1.0) [Intersection of x=2 and x+y=3]\n\n";

    // ---------------------------------------------------------
    // 2. GEOQP SOLVER
    // ---------------------------------------------------------
    std::cout << "========================================\n";
    std::cout << "  GeoQP Solver\n";
    std::cout << "========================================\n";

    std::vector<Iterate> geoqp_iterates;
    geoqp_iterates.push_back({x_bar_x, x_bar_y, "Start"});

    // Track GeoQP iteration stats
    int geoqp_total_iters = -1;
    int geoqp_phase1_iters = -1;
    int geoqp_phase2_iters = -1;
    int geoqp_vertices_explored = -1;

    {
        geoqp::Problem problem;
        problem.n = n;
        problem.m = m;
        problem.Q = Q_shared;
        problem.g = g_shared;
        problem.A = A_shared;
        problem.b = b_shared;

        geoqp::Config config;
        config.verbosity = 3;  // Maximum verbosity
        config.use_caching =  false;

        geoqp::Solver solver(config);
        geoqp::Solution sol = solver.solve(problem);

        if (sol.status == geoqp::Status::SUCCESS && !sol.x.empty())
        {
            std::cout << "  Solution: (" << sol.x[0] << ", " << sol.x[1] << ")\n";
            std::cout << "  Active Set Size: " << sol.num_active_constraints << "\n";
            std::cout << "  GeoQP iterations: total=" << sol.iterations
                      << " phase1=" << sol.phase1_iterations
                      << " phase2=" << sol.phase2_iterations
                      << " vertices_explored=" << sol.vertices_explored << "\n";
            // store stats for final summary
            geoqp_total_iters = sol.iterations;
            geoqp_phase1_iters = sol.phase1_iterations;
            geoqp_phase2_iters = sol.phase2_iterations;
            geoqp_vertices_explored = sol.vertices_explored;
            geoqp_iterates.push_back({sol.x[0], sol.x[1], "GeoQP"});
        }
        else
        {
            std::cout << "  FAILED\n";
        }
    }

    // ---------------------------------------------------------
    // 3. QPOASES SOLVER
    // ---------------------------------------------------------
    std::cout << "\n========================================\n";
    std::cout << "  qpOASES Solver\n";
    std::cout << "========================================\n";

    std::vector<Iterate> qpoases_iterates;
    qpoases_iterates.push_back({x_bar_x, x_bar_y, "Start"});

    int qpoases_iters = -1; // configured/work limit (nWSR)
    {
        std::vector<double> lbA(m, -1.0e10); // One-sided inequality

        qpOASES::QProblem qp(n, m);
        qpOASES::Options options;
        options.printLevel = qpOASES::PL_LOW;
        qp.setOptions(options);

        int nWSR = 100;
        // Using same A_shared pointer
        qp.init(Q_shared.data(), g_shared.data(),
                A_shared.data(),
                nullptr, nullptr,
                lbA.data(), b_shared.data(),
                nWSR);

        std::vector<double> x_opt(n);
        qp.getPrimalSolution(x_opt.data());

    // qpOASES updates nWSR on return with the number of iterations performed
    qpoases_iters = nWSR;

    std::cout << "  Solution: (" << x_opt[0] << ", " << x_opt[1] << ")\n";
    std::cout << "  qpOASES iterations=" << nWSR << "\n";
        qpoases_iterates.push_back({x_opt[0], x_opt[1], "qpOASES"});
    }

    // ---------------------------------------------------------
    // 4. OSQP SOLVER
    // ---------------------------------------------------------
    std::cout << "\n========================================\n";
    std::cout << "  OSQP Solver\n";
    std::cout << "========================================\n";

    std::vector<Iterate> osqp_iterates;
    osqp_iterates.push_back({x_bar_x, x_bar_y, "Start"});

    {
        // Convert shared vectors to Eigen Sparse
        Eigen::SparseMatrix<double> H_sp(n, n);
        H_sp.setIdentity();
        Eigen::VectorXd g_vec(n);
        g_vec << g_shared[0], g_shared[1];

        Eigen::SparseMatrix<double> A_sp(m, n);
        std::vector<Eigen::Triplet<double>> triplets;
        for (int i = 0; i < m; ++i)
        {
            if (A_shared[i * 2] != 0)
                triplets.push_back({i, 0, A_shared[i * 2]});
            if (A_shared[i * 2 + 1] != 0)
                triplets.push_back({i, 1, A_shared[i * 2 + 1]});
        }
        A_sp.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::VectorXd lb_vec(m);
        lb_vec.setConstant(-1e10);
        Eigen::VectorXd ub_vec(m);
        for (int i = 0; i < m; ++i)
            ub_vec(i) = b_shared[i];

        QpSolversEigen::Solver solver;
        solver.instantiateSolver("osqp");
        solver.setBooleanParameter("verbose", false);

        solver.data()->setNumberOfVariables(n);
        solver.data()->setNumberOfInequalityConstraints(m);
        solver.data()->setHessianMatrix(H_sp);
        solver.data()->setGradient(g_vec);
        solver.data()->setInequalityConstraintsMatrix(A_sp);
        solver.data()->setLowerBound(lb_vec);
        solver.data()->setUpperBound(ub_vec);

        solver.initSolver();
        solver.solveProblem();

        Eigen::VectorXd x_opt = solver.getSolution();
        std::cout << "  Solution: (" << x_opt[0] << ", " << x_opt[1] << ")\n";
        osqp_iterates.push_back({x_opt[0], x_opt[1], "OSQP"});
    }

    // ---------------------------------------------------------
    // FINAL RESULTS
    // ---------------------------------------------------------
    std::cout << "\n================================================================\n";
    std::cout << "  Summary: Final Coordinates\n";
    std::cout << "================================================================\n";

    std::cout << "GeoQP:   (" << geoqp_iterates.back().x << ", " << geoqp_iterates.back().y << ")\n";
    std::cout << "qpOASES: (" << qpoases_iterates.back().x << ", " << qpoases_iterates.back().y << ")\n";
    std::cout << "OSQP:    (" << osqp_iterates.back().x << ", " << osqp_iterates.back().y << ")\n";
    std::cout << "\n================================================================\n";
    std::cout << "  Summary: Iterations\n";
    std::cout << "================================================================\n";

    // GeoQP detailed breakdown
    if (geoqp_total_iters >= 0)
    {
        std::cout << "GeoQP:   total=" << geoqp_total_iters
                  << " phase1=" << geoqp_phase1_iters
                  << " phase2=" << geoqp_phase2_iters
                  << " vertices_explored=" << geoqp_vertices_explored << "\n";
    }
    else
    {
        std::cout << "GeoQP:   (no iteration info)\n";
    }

    // qpOASES: report configured/work-limit nWSR
    if (qpoases_iters >= 0)
        std::cout << "qpOASES: iterations=" << qpoases_iters << "\n";
    else
        std::cout << "qpOASES: (no iteration info)\n";

    // OSQP: iteration info not exposed via this wrapper reliably
    std::cout << "OSQP:    (iterations: unknown)\n";

    return 0;
}