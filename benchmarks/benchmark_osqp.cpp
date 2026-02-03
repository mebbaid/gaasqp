/**
 * @file benchmark_osqp.cpp
 * @brief Benchmark comparing GeoQP vs OSQP with detailed diagnostics
 */

#include <geoqp/geoqp.hpp>
#include <QpSolversEigen/QpSolversEigen.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>

using Clock = std::chrono::high_resolution_clock;

Eigen::MatrixXd randomPD(int n, std::mt19937& rng, double condition = 10.0)
{
    std::normal_distribution<> dist(0.0, 1.0);
    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A(i, j) = dist(rng);
    Eigen::MatrixXd H = A.transpose() * A;
    H += Eigen::MatrixXd::Identity(n, n) * (H.trace() / (n * condition));
    return H;
}

void generateBoxConstraints(int n, Eigen::MatrixXd& A, Eigen::VectorXd& lb, 
                            Eigen::VectorXd& ub, double bound = 2.0)
{
    A = Eigen::MatrixXd::Identity(n, n);
    lb = -bound * Eigen::VectorXd::Ones(n);
    ub = bound * Eigen::VectorXd::Ones(n);
}

struct GeoQPResult {
    bool success;
    double solve_time_ms;
    Eigen::VectorXd solution;
    double obj_value;
    // Diagnostics
    int phase1_iters;
    int phase2_iters;
    int vertices_explored;
    int num_active;
    double phase1_time_ms;
    double phase2_time_ms;
    double phase3_time_ms;
};

struct OSQPResult {
    bool success;
    double solve_time_ms;
    Eigen::VectorXd solution;
    double obj_value;
};

GeoQPResult solveGeoQP(const Eigen::MatrixXd& H, const Eigen::VectorXd& g,
                       const Eigen::MatrixXd& A, const Eigen::VectorXd& lb,
                       const Eigen::VectorXd& ub,
                       geoqp::FaceHoppingStrategy strategy = geoqp::FaceHoppingStrategy::STANDARD)
{
    GeoQPResult result;
    int n = H.rows();
    int m = A.rows();
    
    geoqp::Problem problem;
    problem.n = n;
    problem.m = 2 * m;
    
    problem.Q.resize(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            problem.Q[i * n + j] = H(i, j);
    
    problem.g.resize(n);
    for (int i = 0; i < n; ++i)
        problem.g[i] = g(i);
    
    problem.A.resize(2 * m * n);
    problem.b.resize(2 * m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            problem.A[i * n + j] = A(i, j);
            problem.A[(m + i) * n + j] = -A(i, j);
        }
        problem.b[i] = ub(i);
        problem.b[m + i] = -lb(i);
    }
    
    geoqp::Config config;
    config.face_hopping_strategy = strategy;
    geoqp::Solver solver(config);
    auto start = Clock::now();
    geoqp::Solution sol = solver.solve(problem);
    auto end = Clock::now();
    
    result.success = (sol.status == geoqp::Status::SUCCESS);
    result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.solution.resize(n);
    for (int i = 0; i < n; ++i)
        result.solution(i) = sol.x[i];
    result.obj_value = sol.objective_value;
    
    // Diagnostics
    result.phase1_iters = sol.phase1_iterations;
    result.phase2_iters = sol.phase2_iterations;
    result.vertices_explored = sol.vertices_explored;
    result.num_active = sol.num_active_constraints;
    result.phase1_time_ms = sol.phase1_time * 1000.0;
    result.phase2_time_ms = sol.phase2_time * 1000.0;
    result.phase3_time_ms = sol.phase3_time * 1000.0;
    
    return result;
}

OSQPResult solveOSQP(const Eigen::MatrixXd& H, const Eigen::VectorXd& g,
                     const Eigen::MatrixXd& A, const Eigen::VectorXd& lb,
                     const Eigen::VectorXd& ub)
{
    OSQPResult result;
    int n = H.rows();
    int m = A.rows();
    
    Eigen::MatrixXd H_upper = H.triangularView<Eigen::Upper>();
    Eigen::SparseMatrix<double> H_sparse = H_upper.sparseView();
    Eigen::SparseMatrix<double> A_sparse = A.sparseView();
    
    Eigen::VectorXd g_copy = g;
    Eigen::VectorXd lb_copy = lb;
    Eigen::VectorXd ub_copy = ub;
    
    QpSolversEigen::Solver solver;
    bool ok = solver.instantiateSolver("osqp");
    if (!ok) {
        result.success = false;
        result.solve_time_ms = 0;
        result.solution = Eigen::VectorXd::Zero(n);
        result.obj_value = std::numeric_limits<double>::infinity();
        return result;
    }
    
    solver.setBooleanParameter("verbose", false);
    solver.data()->setNumberOfVariables(n);
    solver.data()->setNumberOfInequalityConstraints(m);
    ok = solver.data()->setHessianMatrix(H_sparse);
    ok = ok && solver.data()->setGradient(g_copy);
    ok = ok && solver.data()->setInequalityConstraintsMatrix(A_sparse);
    ok = ok && solver.data()->setLowerBound(lb_copy);
    ok = ok && solver.data()->setUpperBound(ub_copy);
    
    if (!ok) {
        result.success = false;
        result.solve_time_ms = 0;
        result.solution = Eigen::VectorXd::Zero(n);
        result.obj_value = std::numeric_limits<double>::infinity();
        return result;
    }
    
    auto start = Clock::now();
    ok = solver.initSolver();
    if (ok) {
        auto status = solver.solveProblem();
        ok = (status == QpSolversEigen::ErrorExitFlag::NoError);
    }
    auto end = Clock::now();
    
    result.success = ok;
    result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (ok) {
        Eigen::VectorXd sol = solver.getSolution();
        result.solution = sol;
        result.obj_value = solver.getObjValue();
    } else {
        result.solution = Eigen::VectorXd::Zero(n);
        result.obj_value = std::numeric_limits<double>::infinity();
    }
    return result;
}

void generateMPCProblem(int horizon, int n_states, int n_inputs,
                        Eigen::MatrixXd& H, Eigen::VectorXd& g,
                        Eigen::MatrixXd& A, Eigen::VectorXd& lb, Eigen::VectorXd& ub)
{
    int n = n_inputs * horizon;
    Eigen::MatrixXd Su = Eigen::MatrixXd::Zero(n_states * horizon, n);
    for (int k = 0; k < horizon; ++k) {
        for (int j = 0; j <= k; ++j) {
            double decay = std::pow(0.9, k - j);
            for (int i = 0; i < n_states; ++i)
                for (int l = 0; l < n_inputs; ++l)
                    if (i % n_inputs == l)
                        Su(k * n_states + i, j * n_inputs + l) = decay * 0.05;
        }
    }
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(n_states * horizon, n_states * horizon);
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(n, n);
    H = Su.transpose() * Q * Su + R;
    H = 0.5 * (H + H.transpose());
    g = Eigen::VectorXd::Random(n);
    A = Eigen::MatrixXd::Identity(n, n);
    lb = -Eigen::VectorXd::Ones(n);
    ub = Eigen::VectorXd::Ones(n);
}

int main()
{
    std::cout << "========================================\n";
    std::cout << "  GeoQP vs OSQP Benchmark (with diagnostics)\n";
    std::cout << "========================================\n\n";
    
    std::mt19937 rng(42);
    
    // Debug test: Simple MPC problem
    std::cout << "Debug: Single MPC problem (horizon=5)\n";
    std::cout << std::string(60, '-') << "\n";
    {
        Eigen::MatrixXd H, A;
        Eigen::VectorXd g, lb, ub;
        generateMPCProblem(5, 4, 2, H, g, A, lb, ub);
        
        int n = H.rows();
        
        // Check if x_bar = -H^{-1}*g is feasible
        Eigen::VectorXd x_bar = H.ldlt().solve(-g);
        std::cout << "x_bar range: [" << x_bar.minCoeff() << ", " << x_bar.maxCoeff() << "]\n";
        std::cout << "Box: [" << lb.minCoeff() << ", " << ub.maxCoeff() << "]\n";
        
        bool x_bar_feasible = true;
        for (int i = 0; i < n; ++i) {
            if (x_bar(i) < lb(i) - 1e-8 || x_bar(i) > ub(i) + 1e-8) {
                x_bar_feasible = false;
                break;
            }
        }
        std::cout << "x_bar feasible: " << (x_bar_feasible ? "YES" : "NO") << "\n";
        
        geoqp::Config config;
        config.verbosity = 0;  // Disable debug output for benchmark
        
        geoqp::Problem problem;
        problem.n = n;
        problem.m = 2 * n;
        problem.Q.resize(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                problem.Q[i * n + j] = H(i, j);
        problem.g.resize(n);
        for (int i = 0; i < n; ++i)
            problem.g[i] = g(i);
        problem.A.resize(2 * n * n);
        problem.b.resize(2 * n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                problem.A[i * n + j] = A(i, j);
                problem.A[(n + i) * n + j] = -A(i, j);
            }
            problem.b[i] = ub(i);
            problem.b[n + i] = -lb(i);
        }
        
        geoqp::Solver solver(config);
        auto sol = solver.solve(problem);
        
        auto or_ = solveOSQP(H, g, A, lb, ub);
        
        std::cout << "GeoQP solution range: [";
        double gmin = sol.x[0], gmax = sol.x[0];
        for (int i = 1; i < n; ++i) {
            gmin = std::min(gmin, sol.x[i]);
            gmax = std::max(gmax, sol.x[i]);
        }
        std::cout << gmin << ", " << gmax << "]\n";
        std::cout << "OSQP solution range:  [" << or_.solution.minCoeff() << ", " << or_.solution.maxCoeff() << "]\n";
        std::cout << "GeoQP obj: " << sol.objective_value << ", OSQP obj: " << or_.obj_value << "\n";
        std::cout << "Active constraints: " << sol.num_active_constraints << "\n";
        std::cout << "P1 iters: " << sol.phase1_iterations << ", P2 iters: " << sol.phase2_iterations << "\n";
        std::cout << "Vertices explored: " << sol.vertices_explored << "\n\n";
    }
    
    // Test 1: Random QP with varying sizes
    std::cout << "Test 1: Random QP with box constraints\n";
    std::cout << std::string(120, '-') << "\n";
    std::cout << std::setw(6) << "n" 
              << std::setw(10) << "GeoQP" 
              << std::setw(10) << "OSQP"
              << std::setw(10) << "Speedup"
              << std::setw(8) << "Active"
              << std::setw(8) << "P1 it"
              << std::setw(8) << "P2 it"
              << std::setw(8) << "Verts"
              << std::setw(10) << "P1 ms"
              << std::setw(10) << "P2 ms"
              << std::setw(10) << "P3 ms"
              << std::setw(12) << "Obj Diff"
              << "\n";
    std::cout << std::string(120, '-') << "\n";
    
    std::vector<int> sizes = {5, 10, 20, 30, 50, 75, 100};
    for (int n : sizes) {
        const int trials = 5;
        double geoqp_total = 0, osqp_total = 0, obj_diff_total = 0;
        int active_total = 0, p1_total = 0, p2_total = 0, verts_total = 0;
        double p1_time = 0, p2_time = 0, p3_time = 0;
        bool all_success = true;
        
        for (int t = 0; t < trials; ++t) {
            Eigen::MatrixXd H = randomPD(n, rng);
            Eigen::VectorXd g = Eigen::VectorXd::Random(n);
            
            // Scale gradient to make unconstrained minimum more likely to be infeasible
            // With probability ~0.1, keep small gradient (feasible x_bar)
            // With probability ~0.9, use much larger gradient to push x_bar outside box [-2, 2]
            std::bernoulli_distribution make_infeasible(0.9);
            if (make_infeasible(rng)) {
                // Random(n) gives values in [-1,1], typical H condition ~10
                // So x_bar ~ H^{-1}g has magnitude ~ ||g||/lambda_min ~ ||g||/0.1
                // To push outside box of size 2, need ||g|| >> 0.2
                // Scale by 5-15x to reliably push outside
                std::uniform_real_distribution<> scale_dist(15.0, 25.0);
                double scale = scale_dist(rng);
                g *= scale;
            } else {
                // Keep gradient small to ensure x_bar is inside box
                g *= 0.3;
            }
            
            Eigen::MatrixXd A;
            Eigen::VectorXd lb, ub;
            generateBoxConstraints(n, A, lb, ub);
            
            auto gr = solveGeoQP(H, g, A, lb, ub);
            auto or_ = solveOSQP(H, g, A, lb, ub);
            
            geoqp_total += gr.solve_time_ms;
            osqp_total += or_.solve_time_ms;
            obj_diff_total += std::abs(gr.obj_value - or_.obj_value);
            active_total += gr.num_active;
            p1_total += gr.phase1_iters;
            p2_total += gr.phase2_iters;
            verts_total += gr.vertices_explored;
            p1_time += gr.phase1_time_ms;
            p2_time += gr.phase2_time_ms;
            p3_time += gr.phase3_time_ms;
            all_success = all_success && gr.success && or_.success;
        }
        
        std::cout << std::setw(6) << n
                  << std::setw(10) << std::fixed << std::setprecision(3) << (geoqp_total/trials)
                  << std::setw(10) << (osqp_total/trials)
                  << std::setw(10) << std::setprecision(2) << (osqp_total/geoqp_total)
                  << std::setw(8) << (active_total/trials)
                  << std::setw(8) << (p1_total/trials)
                  << std::setw(8) << (p2_total/trials)
                  << std::setw(8) << (verts_total/trials)
                  << std::setw(10) << std::setprecision(3) << (p1_time/trials)
                  << std::setw(10) << (p2_time/trials)
                  << std::setw(10) << (p3_time/trials)
                  << std::setw(12) << std::scientific << std::setprecision(1) << (obj_diff_total/trials)
                  << (all_success ? "" : " FAIL")
                  << "\n";
    }
    
    // Test 2: MPC problems
    std::cout << "\nTest 2: MPC-like problems (tight constraints)\n";
    std::cout << std::string(130, '-') << "\n";
    std::cout << std::setw(8) << "Horizon" 
              << std::setw(6) << "n"
              << std::setw(10) << "GeoQP" 
              << std::setw(10) << "OSQP"
              << std::setw(10) << "Speedup"
              << std::setw(8) << "Active"
              << std::setw(8) << "P1 it"
              << std::setw(8) << "P2 it"
              << std::setw(8) << "Verts"
              << std::setw(10) << "P1 ms"
              << std::setw(10) << "P2 ms"
              << std::setw(10) << "P3 ms"
              << std::setw(12) << "Obj Diff"
              << "\n";
    std::cout << std::string(130, '-') << "\n";
    
    std::vector<int> horizons = {5, 10, 20, 30, 50};
    for (int horizon : horizons) {
        Eigen::MatrixXd H, A;
        Eigen::VectorXd g, lb, ub;
        generateMPCProblem(horizon, 4, 2, H, g, A, lb, ub);
        
        const int trials = 5;
        double geoqp_total = 0, osqp_total = 0, obj_diff_total = 0;
        int active_total = 0, p1_total = 0, p2_total = 0, verts_total = 0;
        double p1_time = 0, p2_time = 0, p3_time = 0;
        bool all_success = true;
        
        for (int t = 0; t < trials; ++t) {
            auto gr = solveGeoQP(H, g, A, lb, ub);
            auto or_ = solveOSQP(H, g, A, lb, ub);
            
            geoqp_total += gr.solve_time_ms;
            osqp_total += or_.solve_time_ms;
            obj_diff_total += std::abs(gr.obj_value - or_.obj_value);
            active_total += gr.num_active;
            p1_total += gr.phase1_iters;
            p2_total += gr.phase2_iters;
            verts_total += gr.vertices_explored;
            p1_time += gr.phase1_time_ms;
            p2_time += gr.phase2_time_ms;
            p3_time += gr.phase3_time_ms;
            all_success = all_success && gr.success && or_.success;
        }
        
        std::cout << std::setw(8) << horizon
                  << std::setw(6) << (2*horizon)
                  << std::setw(10) << std::fixed << std::setprecision(3) << (geoqp_total/trials)
                  << std::setw(10) << (osqp_total/trials)
                  << std::setw(10) << std::setprecision(2) << (osqp_total/geoqp_total)
                  << std::setw(8) << (active_total/trials)
                  << std::setw(8) << (p1_total/trials)
                  << std::setw(8) << (p2_total/trials)
                  << std::setw(8) << (verts_total/trials)
                  << std::setw(10) << std::setprecision(3) << (p1_time/trials)
                  << std::setw(10) << (p2_time/trials)
                  << std::setw(10) << (p3_time/trials)
                  << std::setw(12) << std::scientific << std::setprecision(1) << (obj_diff_total/trials)
                  << (all_success ? "" : " FAIL")
                  << "\n";
    }
    
    // Test 3: Compare where x_bar is feasible (best case for GeoQP)
    std::cout << "\nTest 3: Easy problems (x_bar often feasible)\n";
    std::cout << std::string(80, '-') << "\n";
    {
        const int trials = 20;
        int feasible_count = 0;
        double geoqp_total = 0, osqp_total = 0;
        
        for (int t = 0; t < trials; ++t) {
            int n = 20;
            Eigen::MatrixXd H = randomPD(n, rng);
            Eigen::VectorXd g = 0.1 * Eigen::VectorXd::Random(n);  // Small gradient
            Eigen::MatrixXd A;
            Eigen::VectorXd lb, ub;
            generateBoxConstraints(n, A, lb, ub, 10.0);  // Large box
            
            auto gr = solveGeoQP(H, g, A, lb, ub);
            auto or_ = solveOSQP(H, g, A, lb, ub);
            
            geoqp_total += gr.solve_time_ms;
            osqp_total += or_.solve_time_ms;
            if (gr.num_active == 0) feasible_count++;
        }
        
        std::cout << "  x_bar feasible in " << feasible_count << "/" << trials << " problems\n";
        std::cout << "  GeoQP avg: " << std::fixed << std::setprecision(3) << (geoqp_total/trials) << " ms\n";
        std::cout << "  OSQP avg:  " << (osqp_total/trials) << " ms\n";
        std::cout << "  Speedup: " << std::setprecision(1) << (osqp_total/geoqp_total) << "x\n";
    }
    
    // Test 4: MPC with diagonal Q (structured, realistic for decoupled state costs)
    std::cout << "\nTest 4: MPC with diagonal Q (structured)\n";
    std::cout << std::string(130, '-') << "\n";
    std::cout << std::setw(8) << "Horizon" 
              << std::setw(6) << "n"
              << std::setw(10) << "GeoQP" 
              << std::setw(10) << "OSQP"
              << std::setw(10) << "Speedup"
              << std::setw(8) << "Active"
              << std::setw(8) << "P1 it"
              << std::setw(8) << "P2 it"
              << std::setw(8) << "Verts"
              << std::setw(10) << "P1 ms"
              << std::setw(10) << "P2 ms"
              << std::setw(10) << "P3 ms"
              << std::setw(12) << "Obj Diff"
              << "\n";
    std::cout << std::string(130, '-') << "\n";
    
    for (int horizon : horizons) {
        const int n = 2 * horizon;  // 2 inputs per time step
        
        // Build diagonal Q: simple quadratic cost on each control input
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
        std::uniform_real_distribution<> weight_dist(0.5, 2.0);
        for (int i = 0; i < n; ++i) {
            H(i, i) = weight_dist(rng);
        }
        
        Eigen::VectorXd g = Eigen::VectorXd::Random(n) * 5.0;  // Moderate gradient
        Eigen::MatrixXd A;
        Eigen::VectorXd lb, ub;
        generateBoxConstraints(n, A, lb, ub, 1.0);  // Tight box constraints
        
        const int trials = 5;
        double geoqp_total = 0, osqp_total = 0, obj_diff_total = 0;
        int active_total = 0, p1_total = 0, p2_total = 0, verts_total = 0;
        double p1_time = 0, p2_time = 0, p3_time = 0;
        bool all_success = true;
        
        for (int t = 0; t < trials; ++t) {
            auto gr = solveGeoQP(H, g, A, lb, ub);
            auto or_ = solveOSQP(H, g, A, lb, ub);
            
            geoqp_total += gr.solve_time_ms;
            osqp_total += or_.solve_time_ms;
            obj_diff_total += std::abs(gr.obj_value - or_.obj_value);
            active_total += gr.num_active;
            p1_total += gr.phase1_iters;
            p2_total += gr.phase2_iters;
            verts_total += gr.vertices_explored;
            p1_time += gr.phase1_time_ms;
            p2_time += gr.phase2_time_ms;
            p3_time += gr.phase3_time_ms;
            all_success = all_success && gr.success && or_.success;
        }
        
        std::cout << std::setw(8) << horizon
                  << std::setw(6) << n
                  << std::setw(10) << std::fixed << std::setprecision(3) << (geoqp_total/trials)
                  << std::setw(10) << (osqp_total/trials)
                  << std::setw(10) << std::setprecision(2) << (osqp_total/geoqp_total)
                  << std::setw(8) << (active_total/trials)
                  << std::setw(8) << (p1_total/trials)
                  << std::setw(8) << (p2_total/trials)
                  << std::setw(8) << (verts_total/trials)
                  << std::setw(10) << std::setprecision(3) << (p1_time/trials)
                  << std::setw(10) << (p2_time/trials)
                  << std::setw(10) << (p3_time/trials)
                  << std::setw(12) << std::scientific << std::setprecision(1) << (obj_diff_total/trials)
                  << (all_success ? "" : " FAIL")
                  << "\n";
    }
    
    // Test 5: Face Hopping Strategy Comparison on MPC problems
    std::cout << "\n\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     Face Hopping Strategy Comparison (MPC Problems)           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    std::vector<std::pair<geoqp::FaceHoppingStrategy, std::string>> strategies = {
        {geoqp::FaceHoppingStrategy::STANDARD, "Standard"},
        {geoqp::FaceHoppingStrategy::BATCH, "Batch"},
        {geoqp::FaceHoppingStrategy::SENSITIVITY, "Sensitivity"},
        {geoqp::FaceHoppingStrategy::NEIGHBORS, "Neighbors"}
    };
    
    std::vector<int> strategy_horizons = {10, 20, 50};
    for (int horizon : strategy_horizons) {
        std::cout << "\n=== MPC Problem: Horizon = " << horizon << " (n = " << 2*horizon << ") ===\n";
        
        Eigen::MatrixXd H, A;
        Eigen::VectorXd g, lb, ub;
        generateMPCProblem(horizon, 4, 2, H, g, A, lb, ub);
        
        std::cout << std::string(120, '=') << "\n";
        std::cout << std::setw(15) << "Strategy" 
                  << std::setw(12) << "GeoQP (ms)"
                  << std::setw(12) << "OSQP (ms)"
                  << std::setw(10) << "Speedup"
                  << std::setw(10) << "Phase2 ms"
                  << std::setw(8) << "P2 Iter"
                  << std::setw(8) << "Active"
                  << std::setw(15) << "Objective"
                  << "\n";
        std::cout << std::string(120, '-') << "\n";
        
        // Solve with OSQP once (reference)
        const int trials = 3;
        double osqp_time = 0;
        double osqp_obj = 0;
        for (int t = 0; t < trials; ++t) {
            auto or_ = solveOSQP(H, g, A, lb, ub);
            osqp_time += or_.solve_time_ms;
            osqp_obj += or_.obj_value;
        }
        osqp_time /= trials;
        osqp_obj /= trials;
        
        // Test each GeoQP strategy
        for (const auto& [strategy, name] : strategies) {
            double geoqp_time = 0, phase2_time = 0, obj_value = 0;
            int phase2_iters = 0, active_cons = 0;
            
            for (int t = 0; t < trials; ++t) {
                auto gr = solveGeoQP(H, g, A, lb, ub, strategy);
                geoqp_time += gr.solve_time_ms;
                phase2_time += gr.phase2_time_ms;
                phase2_iters += gr.phase2_iters;
                active_cons += gr.num_active;
                obj_value += gr.obj_value;
            }
            
            geoqp_time /= trials;
            phase2_time /= trials;
            obj_value /= trials;
            phase2_iters /= trials;
            active_cons /= trials;
            
            double speedup = osqp_time / geoqp_time;
            
            std::cout << std::setw(15) << name
                      << std::setw(12) << std::fixed << std::setprecision(3) << geoqp_time
                      << std::setw(12) << osqp_time
                      << std::setw(10) << std::setprecision(2) << speedup
                      << std::setw(10) << std::setprecision(3) << phase2_time
                      << std::setw(8) << phase2_iters
                      << std::setw(8) << active_cons
                      << std::setw(15) << std::scientific << std::setprecision(6) << obj_value
                      << "\n";
        }
        std::cout << std::string(120, '=') << "\n";
        
        // Show Phase 2 speedups relative to Standard
        std::cout << "\nPhase II Time Comparison (vs Standard):\n";
        double standard_phase2 = 0;
        for (int t = 0; t < trials; ++t) {
            auto gr = solveGeoQP(H, g, A, lb, ub, geoqp::FaceHoppingStrategy::STANDARD);
            standard_phase2 += gr.phase2_time_ms;
        }
        standard_phase2 /= trials;
        
        for (size_t i = 1; i < strategies.size(); ++i) {
            double strat_phase2 = 0;
            for (int t = 0; t < trials; ++t) {
                auto gr = solveGeoQP(H, g, A, lb, ub, strategies[i].first);
                strat_phase2 += gr.phase2_time_ms;
            }
            strat_phase2 /= trials;
            
            double improvement = standard_phase2 / strat_phase2;
            std::cout << "  " << std::setw(15) << std::left << strategies[i].second << ": " 
                      << std::fixed << std::setprecision(2) << improvement << "x";
            if (improvement > 1.05) {
                std::cout << " ✓ FASTER";
            } else if (improvement < 0.95) {
                std::cout << " ✗ SLOWER";
            } else {
                std::cout << " ~ SAME";
            }
            std::cout << "\n";
        }
    }
    
    std::cout << "\nBenchmark complete.\n";
    return 0;
}
