/**
 * @file benchmark_osqp.cpp
 * @brief Benchmark comparing GeoQP vs OSQP on various QP problems
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

struct SolveResult {
    bool success;
    double solve_time_ms;
    Eigen::VectorXd solution;
    double obj_value;
};

SolveResult solveGeoQP(const Eigen::MatrixXd& H, const Eigen::VectorXd& g,
                       const Eigen::MatrixXd& A, const Eigen::VectorXd& lb,
                       const Eigen::VectorXd& ub)
{
    SolveResult result;
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
    
    geoqp::Solver solver;
    auto start = Clock::now();
    geoqp::Solution sol = solver.solve(problem);
    auto end = Clock::now();
    
    result.success = (sol.status == geoqp::Status::SUCCESS);
    result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.solution.resize(n);
    for (int i = 0; i < n; ++i)
        result.solution(i) = sol.x[i];
    result.obj_value = sol.objective_value;
    return result;
}

SolveResult solveOSQP(const Eigen::MatrixXd& H, const Eigen::VectorXd& g,
                      const Eigen::MatrixXd& A, const Eigen::VectorXd& lb,
                      const Eigen::VectorXd& ub)
{
    SolveResult result;
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

struct BenchmarkResult {
    int n_vars;
    double geoqp_time_ms;
    double osqp_time_ms;
    double speedup;
    double obj_diff;
    bool both_success;
};

BenchmarkResult runBenchmark(int n, std::mt19937& rng)
{
    BenchmarkResult result;
    result.n_vars = n;
    
    Eigen::MatrixXd H = randomPD(n, rng);
    Eigen::VectorXd g = Eigen::VectorXd::Random(n);
    Eigen::MatrixXd A;
    Eigen::VectorXd lb, ub;
    generateBoxConstraints(n, A, lb, ub);
    
    auto geoqp_result = solveGeoQP(H, g, A, lb, ub);
    auto osqp_result = solveOSQP(H, g, A, lb, ub);
    
    result.both_success = geoqp_result.success && osqp_result.success;
    result.geoqp_time_ms = geoqp_result.solve_time_ms;
    result.osqp_time_ms = osqp_result.solve_time_ms;
    result.speedup = osqp_result.solve_time_ms / geoqp_result.solve_time_ms;
    result.obj_diff = result.both_success ? 
        std::abs(geoqp_result.obj_value - osqp_result.obj_value) : 
        std::numeric_limits<double>::infinity();
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
    std::cout << "  GeoQP vs OSQP Benchmark\n";
    std::cout << "========================================\n\n";
    
    std::mt19937 rng(42);
    
    // Sanity check
    {
        std::cout << "Sanity check: simple 2-variable QP\n";
        Eigen::MatrixXd H(2, 2);
        H << 2, 0, 0, 2;
        Eigen::VectorXd g(2);
        g << -2, -2;
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(2, 2);
        Eigen::VectorXd lb(2), ub(2);
        lb << -10, -10;
        ub << 10, 10;
        
        auto geoqp_result = solveGeoQP(H, g, A, lb, ub);
        auto osqp_result = solveOSQP(H, g, A, lb, ub);
        
        std::cout << "  GeoQP: x=[" << geoqp_result.solution.transpose() << "], obj=" << geoqp_result.obj_value << "\n";
        std::cout << "  OSQP:  x=[" << osqp_result.solution.transpose() << "], obj=" << osqp_result.obj_value << "\n\n";
    }
    
    // Test 1: Varying sizes
    std::cout << "Test 1: Random QP with varying sizes\n";
    std::cout << std::setw(8) << "n" << std::setw(14) << "GeoQP (ms)" << std::setw(14) << "OSQP (ms)" 
              << std::setw(12) << "Speedup" << std::setw(14) << "Obj Diff" << std::setw(10) << "Status\n";
    std::cout << std::string(70, '-') << "\n";
    
    std::vector<int> sizes = {5, 10, 20, 30, 50, 75, 100};
    for (int n : sizes) {
        const int trials = 10;
        double geoqp_total = 0, osqp_total = 0, obj_diff_total = 0;
        bool all_success = true;
        for (int t = 0; t < trials; ++t) {
            auto r = runBenchmark(n, rng);
            geoqp_total += r.geoqp_time_ms;
            osqp_total += r.osqp_time_ms;
            obj_diff_total += r.obj_diff;
            all_success = all_success && r.both_success;
        }
        std::cout << std::setw(8) << n
                  << std::setw(14) << std::fixed << std::setprecision(4) << (geoqp_total/trials)
                  << std::setw(14) << (osqp_total/trials)
                  << std::setw(12) << std::setprecision(2) << (osqp_total/geoqp_total)
                  << std::setw(14) << std::scientific << std::setprecision(2) << (obj_diff_total/trials)
                  << std::setw(10) << (all_success ? "OK" : "FAIL") << "\n";
    }
    
    // Test 2: MPC problems
    std::cout << "\nTest 2: MPC-like problems\n";
    std::cout << std::setw(10) << "Horizon" << std::setw(8) << "n" << std::setw(14) << "GeoQP (ms)" 
              << std::setw(14) << "OSQP (ms)" << std::setw(12) << "Speedup" << std::setw(10) << "Status\n";
    std::cout << std::string(70, '-') << "\n";
    
    std::vector<int> horizons = {5, 10, 20, 30, 50};
    for (int horizon : horizons) {
        Eigen::MatrixXd H, A;
        Eigen::VectorXd g, lb, ub;
        generateMPCProblem(horizon, 4, 2, H, g, A, lb, ub);
        
        const int trials = 10;
        double geoqp_total = 0, osqp_total = 0;
        bool all_success = true;
        for (int t = 0; t < trials; ++t) {
            auto gr = solveGeoQP(H, g, A, lb, ub);
            auto or_ = solveOSQP(H, g, A, lb, ub);
            geoqp_total += gr.solve_time_ms;
            osqp_total += or_.solve_time_ms;
            all_success = all_success && gr.success && or_.success;
        }
        std::cout << std::setw(10) << horizon << std::setw(8) << (2*horizon)
                  << std::setw(14) << std::fixed << std::setprecision(4) << (geoqp_total/trials)
                  << std::setw(14) << (osqp_total/trials)
                  << std::setw(12) << std::setprecision(2) << (osqp_total/geoqp_total)
                  << std::setw(10) << (all_success ? "OK" : "FAIL") << "\n";
    }
    
    // Test 3: Repeated solves
    std::cout << "\nTest 3: Repeated solves (100 iterations, n=20)\n";
    {
        Eigen::MatrixXd H, A;
        Eigen::VectorXd g, lb, ub;
        generateMPCProblem(10, 4, 2, H, g, A, lb, ub);
        const int iterations = 100;
        
        auto start1 = Clock::now();
        for (int i = 0; i < iterations; ++i) {
            g = Eigen::VectorXd::Random(H.rows());
            solveGeoQP(H, g, A, lb, ub);
        }
        auto end1 = Clock::now();
        double geoqp_ms = std::chrono::duration<double, std::milli>(end1-start1).count();
        
        auto start2 = Clock::now();
        for (int i = 0; i < iterations; ++i) {
            g = Eigen::VectorXd::Random(H.rows());
            solveOSQP(H, g, A, lb, ub);
        }
        auto end2 = Clock::now();
        double osqp_ms = std::chrono::duration<double, std::milli>(end2-start2).count();
        
        std::cout << "  GeoQP: " << std::fixed << std::setprecision(2) << geoqp_ms << " ms total, "
                  << (geoqp_ms/iterations) << " ms/solve\n";
        std::cout << "  OSQP:  " << osqp_ms << " ms total, " << (osqp_ms/iterations) << " ms/solve\n";
        std::cout << "  Speedup: " << (osqp_ms/geoqp_ms) << "x\n";
    }
    
    std::cout << "\nBenchmark complete.\n";
    return 0;
}
