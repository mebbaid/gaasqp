/**
 * @file benchmark.cpp
 * @brief Unified Benchmark comparing GeoQP vs OSQP vs qpOASES
 *
 * Code written with the help of GitHub Copilot
 */

#include <geoqp/geoqp.hpp>
#include <qpOASES.hpp>
#include <QpSolversEigen/QpSolversEigen.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

using Clock = std::chrono::high_resolution_clock;

struct QPProblem
{
    int n;              // variables
    int m;              // inequality constraints (rows of A)
    Eigen::MatrixXd H;  // n x n Hessian
    Eigen::VectorXd g;  // n gradient
    Eigen::MatrixXd A;  // m x n constraint matrix
    Eigen::VectorXd lb; // m lower bounds
    Eigen::VectorXd ub; // m upper bounds
    std::string name;
};

Eigen::MatrixXd randomPD(int n, std::mt19937 &rng, double condition = 10.0)
{
    std::normal_distribution<> dist(0.0, 1.0);
    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A(i, j) = dist(rng);

    Eigen::MatrixXd H = A.transpose() * A;
    double trace = H.trace();
    if (trace < 1e-12)
        trace = 1.0;
    H += Eigen::MatrixXd::Identity(n, n) * (trace / (n * condition));
    return H;
}

// Problem 1: Simple box-constrained QP
QPProblem makeBoxQP(int n, std::mt19937 &rng, double bound = 1.0)
{
    QPProblem p;
    p.n = n;
    p.m = n; // Just Identity matrix for box constraints in OSQP/qpOASES format
    p.name = "Box QP (n=" + std::to_string(n) + ")";

    // Use a fixed seed for the matrix to ensure repeatability for same n
    std::mt19937 matrix_rng(12345 + n);
    p.H = randomPD(n, matrix_rng);

    std::normal_distribution<> dist(0.0, 1.0);
    p.g.resize(n);
    for (int i = 0; i < n; ++i)
        p.g(i) = dist(rng) * 5.0; // Push solution to boundary

    // Constraints: lb <= I*x <= ub
    p.A = Eigen::MatrixXd::Identity(n, n);
    p.lb = -bound * Eigen::VectorXd::Ones(n);
    p.ub = bound * Eigen::VectorXd::Ones(n);

    return p;
}

// Problem 2: Polytope-constrained QP (random inequality constraints)
QPProblem makePolytopeQP(int n, int m, std::mt19937 &rng)
{
    QPProblem p;
    p.n = n;
    p.m = m;
    p.name = "Polytope QP (n=" + std::to_string(n) + ", m=" + std::to_string(m) + ")";

    p.H = randomPD(n, rng);

    std::normal_distribution<> dist(0.0, 1.0);
    std::uniform_real_distribution<> udist(0.5, 2.0);

    p.g.resize(n);
    for (int i = 0; i < n; ++i)
        p.g(i) = dist(rng);

    p.A.resize(m, n);
    p.lb.resize(m);
    p.ub.resize(m);

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            p.A(i, j) = dist(rng);
        }
        p.lb(i) = -1e20; // One-sided
        p.ub(i) = udist(rng);
    }

    return p;
}

// Problem 3: Realistic MPC-like QP with state propagation
QPProblem makeMPCQP(int horizon, std::mt19937 &rng)
{
    const int n_states = 4;           // Number of states
    const int n_inputs = 2;           // Number of control inputs
    const int n = n_inputs * horizon; // Total decision variables

    QPProblem p;
    p.n = n;
    p.m = n; // box constraints: -1 <= u <= 1
    p.name = "MPC QP (horizon=" + std::to_string(horizon) + ", n=" + std::to_string(n) + ")";

    // Build state propagation matrix Su: x = Su * u
    Eigen::MatrixXd Su = Eigen::MatrixXd::Zero(n_states * horizon, n);
    for (int k = 0; k < horizon; ++k)
    {
        for (int j = 0; j <= k; ++j)
        {
            double decay = std::pow(0.9, k - j);
            for (int i = 0; i < n_states; ++i)
            {
                for (int l = 0; l < n_inputs; ++l)
                {
                    if (i % n_inputs == l)
                    {
                        Su(k * n_states + i, j * n_inputs + l) = decay * 0.05;
                    }
                }
            }
        }
    }

    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(n_states * horizon, n_states * horizon);
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(n, n);

    Eigen::MatrixXd H = Su.transpose() * Q * Su + R;
    p.H = 0.5 * (H + H.transpose());

    std::normal_distribution<> dist(0.0, 1.0);
    p.g.resize(n);
    for (int i = 0; i < n; ++i)
        p.g(i) = dist(rng);

    p.A = Eigen::MatrixXd::Identity(n, n);
    p.lb = -Eigen::VectorXd::Ones(n);
    p.ub = Eigen::VectorXd::Ones(n);

    return p;
}

// Problem 4: Ill-conditioned Box QP
QPProblem makeConditionedBoxQP(int n, std::mt19937 &rng, double condition = 1e6)
{
    QPProblem p = makeBoxQP(n, rng);
    p.name = "Ill-conditioned Box QP (n=" + std::to_string(n) + ", cond=" + std::to_string(condition) + ")";

    // Scale Hessian
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(p.H);
    Eigen::VectorXd D = es.eigenvalues();
    Eigen::MatrixXd V = es.eigenvectors();

    // Force condition number spread
    double min_eig = D.minCoeff();
    for (int i = 0; i < n; ++i)
    {
        double factor = std::pow(condition, static_cast<double>(i) / (n - 1));
        D(i) = min_eig * factor;
    }

    p.H = V * D.asDiagonal() * V.transpose();
    return p;
}

// Problem 5: Diagonal Q MPC (Structured)
QPProblem makeDiagonalMPCQP(int horizon, std::mt19937 &rng)
{
    const int n = 2 * horizon;
    QPProblem p;
    p.n = n;
    p.m = n;
    p.name = "Diagonal MPC (horizon=" + std::to_string(horizon) + ")";

    p.H = Eigen::MatrixXd::Zero(n, n);
    std::uniform_real_distribution<> weight_dist(0.5, 2.0);
    for (int i = 0; i < n; ++i)
    {
        p.H(i, i) = weight_dist(rng);
    }

    std::normal_distribution<> dist(0.0, 1.0);
    p.g.resize(n);
    for (int i = 0; i < n; ++i)
        p.g(i) = dist(rng) * 5.0;

    p.A = Eigen::MatrixXd::Identity(n, n);
    p.lb = -Eigen::VectorXd::Ones(n);
    p.ub = Eigen::VectorXd::Ones(n);

    return p;
}

struct SolverResult
{
    bool success = false;
    double solve_time_ms = 0.0;
    Eigen::VectorXd solution;
    double obj_value = 0.0;
    int iterations = 0;
    int active_constraints = 0;
    double phase1_time_ms = 0.0;
    double phase2_time_ms = 0.0;
    double phase3_time_ms = 0.0;
};

// GeoQP Wrapper
class GeoQPSolverWrapper
{
public:
    geoqp::Solver solver;
    std::vector<double> Q_flat;
    std::vector<double> A_flat;
    std::vector<double> g_vec;
    std::vector<double> b_vec;
    int current_n = -1;
    int current_m = -1;

    SolverResult solve(const QPProblem &prob, geoqp::FaceHoppingStrategy strategy = geoqp::FaceHoppingStrategy::STANDARD)
    {
        SolverResult result;
        int n = prob.n;
        // GeoQP handles bounds as explicit constraints Ax <= b
        // Each row of A has an upper and lower bound in the problem definition
        // We need to convert lb <= Ax <= ub into:
        // Ax <= ub
        // -Ax <= -lb

        // Count actual constraints (finite bounds)
        int n_constraints = 0;
        for (int i = 0; i < prob.m; ++i)
        {
            if (prob.ub(i) < 1e19)
                n_constraints++;
            if (prob.lb(i) > -1e19)
                n_constraints++;
        }

        bool structure_changed = (current_n != n || current_m != n_constraints);

        if (structure_changed)
        {
            Q_flat.resize(n * n);
            A_flat.resize(n_constraints * n);
            current_n = n;
            current_m = n_constraints;
        }

        // Always update Q (Hessian changes every trial)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Q_flat[i * n + j] = prob.H(i, j);

        g_vec.resize(n);
        for (int i = 0; i < n; ++i)
            g_vec[i] = prob.g(i);

        b_vec.resize(n_constraints);

        int row = 0;
        for (int i = 0; i < prob.m; ++i)
        {
            // Upper bound
            if (prob.ub(i) < 1e19)
            {
                // If structure changed OR we suspect A changes (random polytope?), we must update A
                // For safety, let's always update A for now. Benchmark overhead is negligible compared to solve time for large N?
                // Actually copying A is O(mn), verify cost. For n=100 m=200, it's 20000 doubles. Fast.
                // Optimally we'd track if A changed. But Problem struct doesn't say.

                for (int j = 0; j < n; ++j)
                    A_flat[row * n + j] = prob.A(i, j);

                b_vec[row] = prob.ub(i);
                row++;
            }
            // Lower bound
            if (prob.lb(i) > -1e19)
            {
                for (int j = 0; j < n; ++j)
                    A_flat[row * n + j] = -prob.A(i, j);

                b_vec[row] = -prob.lb(i);
                row++;
            }
        }

        geoqp::Config config;
        config.verbosity = 0;
        config.tolerance = 1e-9;             // Balance between precision and robustness
        config.constraint_tolerance = 1e-9;  // Keep same as tolerance for now
        config.face_hopping_strategy = strategy;
        // use_caching defaults to false (has numerical issues)
        solver.set_config(config);

        auto start = Clock::now();

        if (structure_changed)
        {
            // Full solve/setup
            geoqp::Problem gp;
            gp.n = n;
            gp.m = n_constraints;
            gp.Q = Q_flat;
            gp.A = A_flat;
            gp.g = g_vec;
            gp.b = b_vec;

            geoqp::Solution sol = solver.solve(gp); // Calls setup internally

            auto end = Clock::now();
            result.success = (sol.status == geoqp::Status::SUCCESS);
            result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (result.success && !sol.x.empty())
            {
                result.solution = Eigen::Map<const Eigen::VectorXd>(sol.x.data(), n);
            }
            else
            {
                result.solution = Eigen::VectorXd::Zero(n);
            }
            result.obj_value = sol.objective_value;
            result.iterations = sol.phase2_iterations;
            result.active_constraints = sol.num_active_constraints;
            result.phase1_time_ms = sol.phase1_time * 1000.0;
            result.phase2_time_ms = sol.phase2_time * 1000.0;
            result.phase3_time_ms = sol.phase3_time * 1000.0;
        }
        else
        {
            // Use hotstart if structure matches (assuming Q, A const)
            // Note: For random QPs in loop, structure often changes anyway unless we are careful
            // For now, we assume simple solve unless we specifically engineer benchmarks to reuse structure
            // But let's check if H changed?
            // For this benchmark file, we usually generate new H each trial.
            // So we primarily use the standard solve path but with flattened vectors.

            // Setup explicitly if needed, but solver.solve() handles it.
            // Let's stick to full solve call for correctness in random loops.
            geoqp::Problem gp;
            gp.n = n;
            gp.m = n_constraints;
            gp.Q = Q_flat;
            gp.A = A_flat;
            gp.g = g_vec;
            gp.b = b_vec;
            geoqp::Solution sol = solver.solve(gp);

            auto end = Clock::now();
            result.success = (sol.status == geoqp::Status::SUCCESS);
            result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (result.success && !sol.x.empty())
            {
                result.solution = Eigen::Map<const Eigen::VectorXd>(sol.x.data(), n);
            }
            else
            {
                result.solution = Eigen::VectorXd::Zero(n);
            }
            result.obj_value = sol.objective_value;
            result.iterations = sol.phase2_iterations;
            result.active_constraints = sol.num_active_constraints;
            result.phase1_time_ms = sol.phase1_time * 1000.0;
            result.phase2_time_ms = sol.phase2_time * 1000.0;
            result.phase3_time_ms = sol.phase3_time * 1000.0;
        }

        return result;
    }

    // For benching hotstart specifically
    SolverResult solve_hotstart(const QPProblem &prob)
    {
        // Assumes setup already called
        SolverResult result;
        int n = prob.n;
        g_vec.resize(n);
        for (int i = 0; i < n; ++i)
            g_vec[i] = prob.g(i);

        // Update bounds only
        int n_constraints = 0;
        for (int i = 0; i < prob.m; ++i)
        {
            if (prob.ub(i) < 1e19)
            {
                b_vec[n_constraints++] = prob.ub(i);
            }
            if (prob.lb(i) > -1e19)
            {
                b_vec[n_constraints++] = -prob.lb(i);
            }
        }

        auto start = Clock::now();
        geoqp::Solution sol = solver.solve_hotstart(g_vec.data(), b_vec.data());
        auto end = Clock::now();

        result.success = (sol.status == geoqp::Status::SUCCESS);
        result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (result.success && !sol.x.empty())
        {
            result.solution = Eigen::Map<const Eigen::VectorXd>(sol.x.data(), n);
        }
        else
        {
            result.solution = Eigen::VectorXd::Zero(n);
        }
        result.obj_value = sol.objective_value;
        result.iterations = sol.phase2_iterations;
        result.active_constraints = sol.num_active_constraints;
        result.phase1_time_ms = sol.phase1_time * 1000.0;
        result.phase2_time_ms = sol.phase2_time * 1000.0;
        result.phase3_time_ms = sol.phase3_time * 1000.0;

        return result;
    }
};

// OSQP Wrapper (matching benchmark_osqp.cpp exactly)
SolverResult solveOSQP(const Eigen::MatrixXd &H, const Eigen::VectorXd &g,
                       const Eigen::MatrixXd &A, const Eigen::VectorXd &lb,
                       const Eigen::VectorXd &ub)
{
    SolverResult result;
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
    if (!ok)
    {
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

    if (!ok)
    {
        result.success = false;
        result.solve_time_ms = 0;
        result.solution = Eigen::VectorXd::Zero(n);
        result.obj_value = std::numeric_limits<double>::infinity();
        return result;
    }

    auto start = Clock::now();
    ok = solver.initSolver();
    if (ok)
    {
        auto status = solver.solveProblem();
        ok = (status == QpSolversEigen::ErrorExitFlag::NoError);
    }
    auto end = Clock::now();

    result.success = ok;
    result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (ok)
    {
        Eigen::VectorXd sol = solver.getSolution();
        result.solution = sol;
        result.obj_value = solver.getObjValue();
    }
    else
    {
        result.solution = Eigen::VectorXd::Zero(n);
        result.obj_value = std::numeric_limits<double>::infinity();
    }
    return result;
}

// qpOASES Wrapper
SolverResult solveQPOASES(const Eigen::MatrixXd &H, const Eigen::VectorXd &g,
                          const Eigen::MatrixXd &A, const Eigen::VectorXd &lb,
                          const Eigen::VectorXd &ub)
{
    SolverResult result;
    int n = H.rows();
    int m = A.rows();

    // Copy to vectors
    std::vector<double> H_vec(n * n), A_vec(m * n), g_vec(n), lb_vec(m), ub_vec(m);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            H_vec[i * n + j] = H(i, j);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A_vec[i * n + j] = A(i, j);

    for (int i = 0; i < n; ++i)
        g_vec[i] = g(i);
    for (int i = 0; i < m; ++i)
    {
        lb_vec[i] = lb(i);
        ub_vec[i] = ub(i);
    }

    qpOASES::QProblem qp(n, m, qpOASES::HST_POSDEF);
    qpOASES::Options options;
    options.setToMPC();
    options.printLevel = qpOASES::PL_NONE;
    qp.setOptions(options);

    int nWSR = 1000;

    auto start = Clock::now();
    qpOASES::returnValue status = qp.init(
        H_vec.data(), g_vec.data(), A_vec.data(),
        nullptr, nullptr,             // simple bounds on x
        lb_vec.data(), ub_vec.data(), // constraints bounds
        nWSR);
    auto end = Clock::now();

    result.success = (status == qpOASES::SUCCESSFUL_RETURN);
    result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.iterations = nWSR;

    if (result.success)
    {
        std::vector<double> sol_vec(n);
        qp.getPrimalSolution(sol_vec.data());
        result.solution = Eigen::Map<const Eigen::VectorXd>(sol_vec.data(), n);
        result.obj_value = qp.getObjVal();
        result.active_constraints = qp.getNAC();
    }
    else
    {
        result.solution = Eigen::VectorXd::Zero(n);
    }

    return result;
}


void runBenchmarkSuit(const std::string &suiteName,
                      const std::vector<int> &sizes,
                      std::function<QPProblem(int, std::mt19937 &)> problemGen)
{

    std::cout << "\n>>> SUITE: " << suiteName << "\n";
    std::cout << std::string(105, '-') << "\n";
    std::cout << std::setw(6) << "Size"
              << std::setw(10) << "GeoQP(ms)"
              << std::setw(10) << "OSQP(ms)"
              << std::setw(11) << "qpOASES(ms)"
              << std::setw(9) << "Geo Vs O"
              << std::setw(9) << "Geo Vs Q"
              << std::setw(10) << "ObjDiff"
              << std::setw(8) << "Active"
              << std::setw(15) << "Result"
              << "\n";
    std::cout << std::string(105, '-') << "\n";

    std::mt19937 rng(42);
    int trials = 5;

    for (int n : sizes)
    {
        double t_geo = 0, t_osqp = 0, t_qpoases = 0;
        double diff_obj = 0;
        int active = 0;
        bool all_opt = true;

        GeoQPSolverWrapper geoWrapper;

        for (int t = 0; t < trials; ++t)
        {
            QPProblem p = problemGen(n, rng);

            auto res_geo = geoWrapper.solve(p);
            auto res_osqp = solveOSQP(p.H, p.g, p.A, p.lb, p.ub);
            auto res_qpoases = solveQPOASES(p.H, p.g, p.A, p.lb, p.ub);

            if (!res_geo.success || !res_osqp.success || !res_qpoases.success)
            {
                all_opt = false;
            }

            t_geo += res_geo.solve_time_ms;
            t_osqp += res_osqp.solve_time_ms;
            t_qpoases += res_qpoases.solve_time_ms;

            // Compare GeoQP vs qpOASES for objective
            if (res_geo.success && res_qpoases.success)
            {
                double obj_diff_trial = std::abs(res_geo.obj_value - res_qpoases.obj_value);
                diff_obj += obj_diff_trial;
            }

            active += res_geo.active_constraints;
        }

        std::cout << std::setw(6) << n
                  << std::setw(10) << std::fixed << std::setprecision(3) << (t_geo / trials)
                  << std::setw(10) << (t_osqp / trials)
                  << std::setw(11) << (t_qpoases / trials)
                  << std::setw(9) << std::setprecision(2) << (t_osqp / t_geo) << "x"
                  << std::setw(9) << (t_qpoases / t_geo) << "x"
                  << std::setw(10) << std::scientific << std::setprecision(1) << (diff_obj / trials)
                  << std::setw(8) << (int)(active / trials)
                  << std::setw(15) << (all_opt ? "PASS" : "FAIL")
                  << "\n";
    }
}

int main()
{
    std::cout << "=======================================================\n";
    std::cout << "  UNIFIED QP SOLVER BENCHMARK: GeoQP vs OSQP vs qpOASES\n";
    std::cout << "=======================================================\n";

    // 1. Box QP Suit
    runBenchmarkSuit("Random Box QP", {10, 30, 50, 75, 100}, [](int n, std::mt19937 &rng)
                     { return makeBoxQP(n, rng); });

    // 2. Polytope QP Suit
    runBenchmarkSuit("Polytope QP (Constraints m=2n)", {10, 30, 50, 75}, [](int n, std::mt19937 &rng)
                     { return makePolytopeQP(n, 2 * n, rng); });

    // 3. MPC Suit
    runBenchmarkSuit("MPC Problems (State=4, Input=2)", {5, 10, 20, 30, 50}, [](int horizon, std::mt19937 &rng)
                     { return makeMPCQP(horizon, rng); });

    // 4. Diagonal MPC
    runBenchmarkSuit("MPC Diagonal Q (Fast Path)", {5, 10, 20, 30, 50}, [](int horizon, std::mt19937 &rng)
                     { return makeDiagonalMPCQP(horizon, rng); });

    std::cout << "\nBenchmark Complete.\n";
    return 0;
}
