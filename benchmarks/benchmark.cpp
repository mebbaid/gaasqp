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
    Eigen::VectorXd lbA; // m lower bounds for constraints Ax
    Eigen::VectorXd ubA; // m upper bounds for constraints Ax
    Eigen::VectorXd lbX; // n lower bounds for variables x (simple bounds)
    Eigen::VectorXd ubX; // n upper bounds for variables x (simple bounds)
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
    p.m = 0; // No general constraints, only simple bounds
    p.name = "Box QP (n=" + std::to_string(n) + ")";

    // Use a fixed seed for the matrix to ensure repeatability for same n
    std::mt19937 matrix_rng(12345 + n);
    p.H = randomPD(n, matrix_rng);

    std::normal_distribution<> dist(0.0, 1.0);
    p.g.resize(n);
    for (int i = 0; i < n; ++i)
        p.g(i) = dist(rng) * 5.0; // Push solution to boundary

    // Simple bounds on variables: -bound <= x <= bound
    p.lbX = -bound * Eigen::VectorXd::Ones(n);
    p.ubX = bound * Eigen::VectorXd::Ones(n);
    
    // No general constraints
    p.A = Eigen::MatrixXd(0, n);
    p.lbA = Eigen::VectorXd(0);
    p.ubA = Eigen::VectorXd(0);

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
    p.lbA.resize(m);
    p.ubA.resize(m);

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            p.A(i, j) = dist(rng);
        }
        p.lbA(i) = -1e20; // One-sided
        p.ubA(i) = udist(rng);
    }
    
    // No simple bounds
    p.lbX = -1e20 * Eigen::VectorXd::Ones(n);
    p.ubX = 1e20 * Eigen::VectorXd::Ones(n);

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
    p.m = 0; // No general constraints, only input bounds
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

    // Simple bounds on inputs: -1 <= u <= 1
    p.lbX = -Eigen::VectorXd::Ones(n);
    p.ubX = Eigen::VectorXd::Ones(n);
    
    // No general constraints
    p.A = Eigen::MatrixXd(0, n);
    p.lbA = Eigen::VectorXd(0);
    p.ubA = Eigen::VectorXd(0);

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
    p.m = 0; // Only simple bounds
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

    // Simple bounds
    p.lbX = -Eigen::VectorXd::Ones(n);
    p.ubX = Eigen::VectorXd::Ones(n);
    
    // No general constraints
    p.A = Eigen::MatrixXd(0, n);
    p.lbA = Eigen::VectorXd(0);
    p.ubA = Eigen::VectorXd(0);

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
        int m = prob.m;
        
        // GeoQP handles bounds as explicit constraints Ax <= b
        // We need to convert:
        // - Simple bounds: lbX <= x <= ubX  ->  -I*x <= -lbX and I*x <= ubX
        // - General constraints: lbA <= Ax <= ubA  ->  -A*x <= -lbA and A*x <= ubA

        // Count actual constraints (finite bounds)
        int n_constraints = 0;
        
        // Count simple bound constraints
        for (int i = 0; i < n; ++i)
        {
            if (prob.lbX(i) > -1e19)
                n_constraints++; // -x_i <= -lbX_i
            if (prob.ubX(i) < 1e19)
                n_constraints++; // x_i <= ubX_i
        }
        
        // Count general constraints
        for (int i = 0; i < m; ++i)
        {
            if (prob.lbA(i) > -1e19)
                n_constraints++; // -A_i*x <= -lbA_i
            if (prob.ubA(i) < 1e19)
                n_constraints++; // A_i*x <= ubA_i
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
        
        // Add simple bound constraints
        for (int i = 0; i < n; ++i)
        {
            // Lower bound: -x_i <= -lbX_i
            if (prob.lbX(i) > -1e19)
            {
                for (int j = 0; j < n; ++j)
                    A_flat[row * n + j] = (i == j) ? -1.0 : 0.0;
                b_vec[row] = -prob.lbX(i);
                row++;
            }
            // Upper bound: x_i <= ubX_i
            if (prob.ubX(i) < 1e19)
            {
                for (int j = 0; j < n; ++j)
                    A_flat[row * n + j] = (i == j) ? 1.0 : 0.0;
                b_vec[row] = prob.ubX(i);
                row++;
            }
        }
        
        // Add general constraints
        for (int i = 0; i < m; ++i)
        {
            // Lower bound: -A_i*x <= -lbA_i
            if (prob.lbA(i) > -1e19)
            {
                for (int j = 0; j < n; ++j)
                    A_flat[row * n + j] = -prob.A(i, j);
                b_vec[row] = -prob.lbA(i);
                row++;
            }
            // Upper bound: A_i*x <= ubA_i
            if (prob.ubA(i) < 1e19)
            {
                for (int j = 0; j < n; ++j)
                    A_flat[row * n + j] = prob.A(i, j);
                b_vec[row] = prob.ubA(i);
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

    SolverResult solve_hotstart(const QPProblem &prob)
    {
        // Assumes setup already called
        SolverResult result;
        int n = prob.n;
        int m = prob.m;
        
        g_vec.resize(n);
        for (int i = 0; i < n; ++i)
            g_vec[i] = prob.g(i);

        // Update bounds only
        int n_constraints = 0;
        
        // Simple bounds
        for (int i = 0; i < n; ++i)
        {
            if (prob.lbX(i) > -1e19)
            {
                b_vec[n_constraints++] = -prob.lbX(i);
            }
            if (prob.ubX(i) < 1e19)
            {
                b_vec[n_constraints++] = prob.ubX(i);
            }
        }
        
        // General constraints
        for (int i = 0; i < m; ++i)
        {
            if (prob.lbA(i) > -1e19)
            {
                b_vec[n_constraints++] = -prob.lbA(i);
            }
            if (prob.ubA(i) < 1e19)
            {
                b_vec[n_constraints++] = prob.ubA(i);
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


SolverResult solveOSQP(const QPProblem &prob)
{
    SolverResult result;
    int n = prob.n;
    int m = prob.m;

    Eigen::MatrixXd H_upper = prob.H.triangularView<Eigen::Upper>();
    Eigen::SparseMatrix<double> H_sparse = H_upper.sparseView();

    Eigen::VectorXd g_copy = prob.g;

    // Count total constraints (variable bounds + general constraints)
    int total_constraints = 0;
    for (int i = 0; i < n; ++i)
    {
        if (prob.lbX(i) > -1e19)
            total_constraints++; // -x_i <= -lbX_i
        if (prob.ubX(i) < 1e19)
            total_constraints++; // x_i <= ubX_i
    }
    for (int i = 0; i < m; ++i)
    {
        if (prob.lbA(i) > -1e19)
            total_constraints++; // -A_i*x <= -lbA_i
        if (prob.ubA(i) < 1e19)
            total_constraints++; // A_i*x <= ubA_i
    }

    // Build combined constraint matrix
    Eigen::MatrixXd A_combined(total_constraints, n);
    Eigen::VectorXd lb_combined(total_constraints);
    Eigen::VectorXd ub_combined(total_constraints);

    int row = 0;
    
    // Add variable bounds as constraints
    for (int i = 0; i < n; ++i)
    {
        if (prob.lbX(i) > -1e19)
        {
            A_combined.row(row).setZero();
            A_combined(row, i) = 1.0;
            lb_combined(row) = prob.lbX(i);
            ub_combined(row) = 1e20;
            row++;
        }
        if (prob.ubX(i) < 1e19)
        {
            A_combined.row(row).setZero();
            A_combined(row, i) = 1.0;
            lb_combined(row) = -1e20;
            ub_combined(row) = prob.ubX(i);
            row++;
        }
    }
    
    // Add general constraints
    for (int i = 0; i < m; ++i)
    {
        if (prob.lbA(i) > -1e19)
        {
            A_combined.row(row) = prob.A.row(i);
            lb_combined(row) = prob.lbA(i);
            ub_combined(row) = 1e20;
            row++;
        }
        if (prob.ubA(i) < 1e19)
        {
            A_combined.row(row) = prob.A.row(i);
            lb_combined(row) = -1e20;
            ub_combined(row) = prob.ubA(i);
            row++;
        }
    }

    Eigen::SparseMatrix<double> A_sparse = A_combined.sparseView();

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
    solver.data()->setNumberOfInequalityConstraints(total_constraints);
    ok = solver.data()->setHessianMatrix(H_sparse);
    ok = ok && solver.data()->setGradient(g_copy);
    ok = ok && solver.data()->setInequalityConstraintsMatrix(A_sparse);
    ok = ok && solver.data()->setLowerBound(lb_combined);
    ok = ok && solver.data()->setUpperBound(ub_combined);

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

SolverResult solveQPOASES(const QPProblem &prob)
{
    SolverResult result;
    int n = prob.n;
    int m = prob.m;

    // Copy to vectors
    std::vector<double> H_vec(n * n), g_vec(n);
    std::vector<double> A_vec(m * n), lbA_vec(m), ubA_vec(m);
    std::vector<double> lbX_vec(n), ubX_vec(n);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            H_vec[i * n + j] = prob.H(i, j);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A_vec[i * n + j] = prob.A(i, j);

    for (int i = 0; i < n; ++i)
    {
        g_vec[i] = prob.g(i);
        lbX_vec[i] = prob.lbX(i);
        ubX_vec[i] = prob.ubX(i);
    }
    
    for (int i = 0; i < m; ++i)
    {
        lbA_vec[i] = prob.lbA(i);
        ubA_vec[i] = prob.ubA(i);
    }

    qpOASES::QProblem qp(n, m, qpOASES::HST_POSDEF);
    qpOASES::Options options;
    options.setToMPC();
    options.printLevel = qpOASES::PL_NONE;
    qp.setOptions(options);

    int nWSR = 1000;

    auto start = Clock::now();
    qpOASES::returnValue status = qp.init(
        H_vec.data(), g_vec.data(), 
        m > 0 ? A_vec.data() : nullptr,
        lbX_vec.data(), ubX_vec.data(),      // simple bounds on x
        m > 0 ? lbA_vec.data() : nullptr, 
        m > 0 ? ubA_vec.data() : nullptr,    // constraint bounds
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


// ─────────────────────────────────────────────────────────────────────────────
// Realistic MPC hotstart benchmark
//
// Models the real-world MPC use-case: Q and A (constraint matrix) are fixed
// across all time steps, only the linear term g (reference-dependent cost)
// and constraint bounds b change.
//
// Protocol:
//   1. Draw a single Q and A once ("structure").
//   2. GeoQP: call solver.setup() once, then solve_hotstart(g, b) per trial.
//   3. OSQP:  call initSolver() once (pre-factor KKT), then updateGradient()
//             + solveProblem() per trial.
//   4. qpOASES: first init(), then hotstart() per trial.
//
// Columns:
//   Size | GeoQP(ms) | OSQP(ms) | qpOASES(ms) | Geo/OSQP | Geo/qpOASES | Result
// ─────────────────────────────────────────────────────────────────────────────
void runHotstartBenchmark(const std::string &suiteName,
                          const std::vector<int> &sizes,
                          std::function<QPProblem(int, std::mt19937 &)> problemGen)
{
    std::cout << "\n>>> HOTSTART SUITE: " << suiteName << "\n";
    std::cout << "    (fixed Q+A structure, only g changes per trial)\n";
    std::cout << std::string(121, '-') << "\n";
    std::cout << std::setw(6)  << "Size"
              << std::setw(12) << "GeoQP(ms)"
              << std::setw(10) << "OSQP(ms)"
              << std::setw(13) << "qpOASES(ms)"
              << std::setw(10) << "Geo/OSQP"
              << std::setw(11) << "Geo/qpOAS"
              << std::setw(10) << "GeoHops"
              << std::setw(10) << "QpOIters"
              << std::setw(10) << "ObjDiff"
              << std::setw(8)  << "Active"
              << std::setw(12) << "Result"
              << "\n";
    std::cout << std::string(121, '-') << "\n";

    std::mt19937 rng(42);
    const int trials = 20;

    for (int n_size : sizes)
    {
        // ── 1. Generate structure (fixed Q, A, bounds) ─────────────────
        QPProblem structure = problemGen(n_size, rng);
        int n = structure.n;
        int m = structure.m;

        // Count constraints (same across all trials)
        int n_constraints = 0;
        for (int i = 0; i < n; ++i)
        {
            if (structure.lbX(i) > -1e19) n_constraints++;
            if (structure.ubX(i) < 1e19)  n_constraints++;
        }
        for (int i = 0; i < m; ++i)
        {
            if (structure.lbA(i) > -1e19) n_constraints++;
            if (structure.ubA(i) < 1e19)  n_constraints++;
        }

        // Build flat Q and A (fixed)
        std::vector<double> Q_flat(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Q_flat[i * n + j] = structure.H(i, j);

        std::vector<double> A_flat(n_constraints * n, 0.0);
        std::vector<double> b_fixed(n_constraints);
        {
            int row = 0;
            for (int i = 0; i < n; ++i)
            {
                if (structure.lbX(i) > -1e19)
                {
                    A_flat[row * n + i] = -1.0;
                    b_fixed[row++] = -structure.lbX(i);
                }
                if (structure.ubX(i) < 1e19)
                {
                    A_flat[row * n + i] = 1.0;
                    b_fixed[row++] = structure.ubX(i);
                }
            }
            for (int i = 0; i < m; ++i)
            {
                if (structure.lbA(i) > -1e19)
                {
                    for (int j = 0; j < n; ++j)
                        A_flat[row * n + j] = -structure.A(i, j);
                    b_fixed[row++] = -structure.lbA(i);
                }
                if (structure.ubA(i) < 1e19)
                {
                    for (int j = 0; j < n; ++j)
                        A_flat[row * n + j] = structure.A(i, j);
                    b_fixed[row++] = structure.ubA(i);
                }
            }
        }

        // ── 2. GeoQP: setup once ───────────────────────────────────────
        geoqp::Config geo_cfg;
        geo_cfg.verbosity = 0;
        geo_cfg.tolerance = 1e-9;
        geo_cfg.constraint_tolerance = 1e-9;
        geoqp::Solver geo_solver(geo_cfg);
        geo_solver.setup(n, n_constraints, Q_flat.data(), A_flat.data());

        // ── 3. OSQP: build once, warm-update per trial ─────────────────
        Eigen::MatrixXd H_upper = structure.H.triangularView<Eigen::Upper>();
        Eigen::SparseMatrix<double> H_sparse = H_upper.sparseView();

        Eigen::MatrixXd A_combined(n_constraints, n);
        Eigen::VectorXd lb_combined(n_constraints), ub_combined(n_constraints);
        {
            int row = 0;
            for (int i = 0; i < n; ++i)
            {
                if (structure.lbX(i) > -1e19)
                {
                    A_combined.row(row).setZero();
                    A_combined(row, i) = 1.0;
                    lb_combined(row) = structure.lbX(i);
                    ub_combined(row) = 1e20;
                    row++;
                }
                if (structure.ubX(i) < 1e19)
                {
                    A_combined.row(row).setZero();
                    A_combined(row, i) = 1.0;
                    lb_combined(row) = -1e20;
                    ub_combined(row) = structure.ubX(i);
                    row++;
                }
            }
            for (int i = 0; i < m; ++i)
            {
                if (structure.lbA(i) > -1e19)
                {
                    A_combined.row(row) = structure.A.row(i);
                    lb_combined(row) = structure.lbA(i);
                    ub_combined(row) = 1e20;
                    row++;
                }
                if (structure.ubA(i) < 1e19)
                {
                    A_combined.row(row) = structure.A.row(i);
                    lb_combined(row) = -1e20;
                    ub_combined(row) = structure.ubA(i);
                    row++;
                }
            }
        }
        Eigen::SparseMatrix<double> A_sparse = A_combined.sparseView();
        Eigen::VectorXd g0 = structure.g;

        QpSolversEigen::Solver osqp_solver;
        bool osqp_ok = osqp_solver.instantiateSolver("osqp");
        if (osqp_ok)
        {
            osqp_solver.setBooleanParameter("verbose", false);
            osqp_solver.data()->setNumberOfVariables(n);
            osqp_solver.data()->setNumberOfInequalityConstraints(n_constraints);
            osqp_ok = osqp_solver.data()->setHessianMatrix(H_sparse);
            osqp_ok = osqp_ok && osqp_solver.data()->setGradient(g0);
            osqp_ok = osqp_ok && osqp_solver.data()->setInequalityConstraintsMatrix(A_sparse);
            osqp_ok = osqp_ok && osqp_solver.data()->setLowerBound(lb_combined);
            osqp_ok = osqp_ok && osqp_solver.data()->setUpperBound(ub_combined);
            osqp_ok = osqp_ok && osqp_solver.initSolver(); // pre-factor KKT
        }

        // ── 4. qpOASES: init once with structure ───────────────────────
        std::vector<double> H_vec(n * n), g0_vec(n);
        std::vector<double> A_qp(m * n), lbA_qp(std::max(m,1)), ubA_qp(std::max(m,1));
        std::vector<double> lbX_qp(n), ubX_qp(n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                H_vec[i * n + j] = structure.H(i, j);
        for (int i = 0; i < n; ++i) { g0_vec[i] = structure.g(i); lbX_qp[i] = structure.lbX(i); ubX_qp[i] = structure.ubX(i); }
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j) A_qp[i * n + j] = structure.A(i, j);
            lbA_qp[i] = structure.lbA(i); ubA_qp[i] = structure.ubA(i);
        }
        qpOASES::QProblem qpoases_qp(n, m, qpOASES::HST_POSDEF);
        qpOASES::Options qpoases_opts;
        qpoases_opts.setToMPC();
        qpoases_opts.printLevel = qpOASES::PL_NONE;
        qpoases_qp.setOptions(qpoases_opts);
        int nWSR_init = 1000;
        qpOASES::returnValue qpoases_status = qpoases_qp.init(
            H_vec.data(), g0_vec.data(),
            m > 0 ? A_qp.data() : nullptr,
            lbX_qp.data(), ubX_qp.data(),
            m > 0 ? lbA_qp.data() : nullptr,
            m > 0 ? ubA_qp.data() : nullptr,
            nWSR_init);
        bool qpoases_ok = (qpoases_status == qpOASES::SUCCESSFUL_RETURN);

        // ── 5. Trial loop: only g varies ───────────────────────────────
        double t_geo = 0, t_osqp = 0, t_qpoases = 0;
        double diff_obj = 0;
        int active_total = 0;
        int geo_hops_total = 0;
        int qpoases_iters_total = 0;
        bool all_ok = true;

        std::vector<double> g_flat(n);
        std::normal_distribution<> g_dist(0.0, 5.0);

        for (int t = 0; t < trials; ++t)
        {
            // New random g only
            for (int i = 0; i < n; ++i) g_flat[i] = g_dist(rng);

            // GeoQP hotstart
            double geo_obj = 0;
            {
                auto t0 = Clock::now();
                geoqp::Solution sol = geo_solver.solve_hotstart(g_flat.data(), b_fixed.data());
                auto t1 = Clock::now();
                t_geo += std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (sol.status != geoqp::Status::SUCCESS) all_ok = false;
                active_total += sol.num_active_constraints;
                geo_hops_total += sol.phase2_iterations;
                geo_obj = sol.objective_value;
            }

            // OSQP: update gradient only
            double t_osqp_trial = 0;
            double osqp_obj = 0;
            if (osqp_ok)
            {
                Eigen::VectorXd g_eig = Eigen::Map<const Eigen::VectorXd>(g_flat.data(), n);
                auto t0 = Clock::now();
                bool upd = osqp_solver.updateGradient(g_eig);
                if (upd)
                {
                    auto status = osqp_solver.solveProblem();
                    upd = (status == QpSolversEigen::ErrorExitFlag::NoError);
                }
                auto t1 = Clock::now();
                t_osqp_trial = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (!upd) all_ok = false;
                osqp_obj = upd ? osqp_solver.getObjValue() : 0;
            }
            t_osqp += t_osqp_trial;

            // qpOASES hotstart
            double t_qpoases_trial = 0;
            double qpoases_obj = 0;
            if (qpoases_ok)
            {
                std::vector<double> g_qp(g_flat);
                int nWSR_hs = 1000;
                auto t0 = Clock::now();
                qpOASES::returnValue hs_ret = qpoases_qp.hotstart(
                    g_qp.data(), lbX_qp.data(), ubX_qp.data(),
                    m > 0 ? lbA_qp.data() : nullptr,
                    m > 0 ? ubA_qp.data() : nullptr,
                    nWSR_hs);
                auto t1 = Clock::now();
                t_qpoases_trial = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (hs_ret != qpOASES::SUCCESSFUL_RETURN) all_ok = false;
                qpoases_obj = qpoases_qp.getObjVal();
                qpoases_iters_total += nWSR_hs;
            }
            t_qpoases += t_qpoases_trial;

            // Objective diff: GeoQP vs qpOASES
            if (qpoases_ok)
                diff_obj += std::abs(geo_obj - qpoases_obj);
        }

        // Divide by trials
        double inv = 1.0 / trials;
        std::cout << std::setw(6)  << n_size
                  << std::setw(12) << std::fixed << std::setprecision(3) << (t_geo / trials)
                  << std::setw(10) << (t_osqp / trials)
                  << std::setw(13) << (t_qpoases / trials)
                  << std::setw(9)  << std::setprecision(2) << (t_osqp / t_geo) << "x"
                  << std::setw(10) << (t_qpoases / t_geo) << "x"
                  << std::setw(10) << std::fixed << std::setprecision(1) << (double(geo_hops_total) / trials)
                  << std::setw(10) << std::setprecision(1) << (double(qpoases_iters_total) / trials)
                  << std::setw(10) << std::scientific << std::setprecision(1) << (diff_obj * inv)
                  << std::setw(8)  << (int)(active_total / trials)
                  << std::setw(12) << (all_ok ? "PASS" : "FAIL")
                  << "\n";
    }
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
            auto res_osqp = solveOSQP(p);
            auto res_qpoases = solveQPOASES(p);

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

// ─────────────────────────────────────────────────────────────────────────────
// Theory verification suite (box / diagonal-Q problems only)
//
// Uses the native lb/ub API so that:
//   - expand_bounds_into_constraints fires (not the manual A/b expansion)
//   - sol.x_phase1 (v*) and sol.vertices_explored are meaningful
//   - The box fast-path (Proposition 3) is exercised
//
// Per Lemma 2 (math.tex §2): v* = argmax_{v ∈ V_good} ‖v − x̄‖_Q²
//   For diagonal Q the fast-path achieves the exact maximum.
//
// Per Theorem 1 (math.tex §2): Phase II hops ≤ # faces adjacent to F₀.
//   For a pure-box polytope with m = 2n bound constraints and k active at v*,
//   the upper bound is 2*(2n − k).
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<double> computeXbarDiag(const Eigen::MatrixXd &H,
                                            const Eigen::VectorXd &g)
{
    int n = (int)g.size();
    std::vector<double> xbar(n);
    for (int j = 0; j < n; ++j)
        xbar[j] = (H(j, j) > 1e-15) ? -g(j) / H(j, j) : 0.0;
    return xbar;
}

void runBoxTheorySuite(const std::string &suiteName,
                       const std::vector<int> &sizes,
                       std::function<QPProblem(int, std::mt19937 &)> problemGen,
                       bool diagonal_q = false)
{
    std::cout << "\n>>> THEORY VERIFICATION: " << suiteName << "\n";
    std::cout << "    (Prop 3: Verts=1 | Lemma 2: v*@bound=100% | Theorem 1: Ph2≤AdjBound)\n";

    const int W = 121;
    std::cout << std::string(W, '-') << "\n";
    std::cout << std::setw(6)  << "n"
              << std::setw(8)  << "Verts"
              << std::setw(12) << "v*@bound%"
              << std::setw(10) << "Ph2hops"
              << std::setw(11) << "AdjBound"
              << std::setw(10) << "T1(ms)"
              << std::setw(10) << "T2(ms)"
              << std::setw(10) << "T3(ms)"
              << std::setw(10) << "Ttot(ms)"
              << std::setw(11) << "Prop3"
              << std::setw(9)  << "Thm1"
              << std::setw(9)  << "Solved"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    std::mt19937 rng(42);
    const int trials = 10;

    geoqp::Config cfg;
    cfg.verbosity          = 0;
    cfg.tolerance          = 1e-9;
    cfg.constraint_tolerance = 1e-9;
    cfg.check_optimality   = true;
    geoqp::Solver solver(cfg);

    for (int n : sizes)
    {
        double sum_verts    = 0;
        double sum_at_bound = 0;   // fraction of coordinates at correct bound
        double sum_ph2      = 0;
        double sum_adj      = 0;
        double sum_t1 = 0, sum_t2 = 0, sum_t3 = 0, sum_tot = 0;
        bool   prop3_ok     = true;
        bool   thm1_ok      = true;
        bool   all_solved   = true;

        int constrained_trials = 0;
        for (int t = 0; t < trials; ++t)
        {
            QPProblem p = problemGen(n, rng);

            geoqp::Problem gp;
            gp.n = n;
            gp.m = 0;
            gp.Q.resize(n * n);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    gp.Q[i * n + j] = p.H(i, j);
            gp.g.resize(n);
            for (int i = 0; i < n; ++i) gp.g[i] = p.g(i);
            gp.lb.resize(n);
            gp.ub.resize(n);
            for (int i = 0; i < n; ++i)
            {
                gp.lb[i] = p.lbX(i);
                gp.ub[i] = p.ubX(i);
            }

            auto start = Clock::now();
            geoqp::Solution sol = solver.solve(gp);
            auto end   = Clock::now();
            double t_ms = std::chrono::duration<double, std::milli>(end - start).count();

            if (sol.status != geoqp::Status::SUCCESS)
            {
                all_solved = false;
                continue;
            }
            
            bool xbar_outside = (sol.phase1_iterations > 0 || sol.vertices_explored > 0);
            if (!xbar_outside)
            {
                sum_tot += t_ms; 
                continue;
            }
            ++constrained_trials;

            // ── Proposition 3: fast-path visits exactly 1 vertex ─────────
            sum_verts += sol.vertices_explored;
            if (sol.vertices_explored != 1) prop3_ok = false;

            // ── Lemma 2: v* sits at the correct bound per coordinate ──────
            if (!sol.x_phase1.empty())
            {
                auto xbar = computeXbarDiag(p.H, p.g);
                int correct = 0;
                for (int j = 0; j < n; ++j)
                {
                    double lb_j = gp.lb[j], ub_j = gp.ub[j];
                    double expected = (std::abs(xbar[j] - lb_j) > std::abs(xbar[j] - ub_j))
                                      ? lb_j : ub_j;
                    if (std::abs(sol.x_phase1[j] - expected) < 1e-6) ++correct;
                }
                sum_at_bound += 100.0 * correct / n;
            }

            // ── Theorem 1: Phase II hops ≤ adjacency bound ───────────────
            int k   = (int)sol.active_set.size();
            int m_exp = 2 * n;
            int adj = 2 * (m_exp - k);   // 2*(2n − k)
            sum_ph2 += sol.phase2_iterations;
            sum_adj += adj;
            if (sol.phase2_iterations > adj) thm1_ok = false;

            sum_t1  += sol.phase1_time * 1000.0;
            sum_t2  += sol.phase2_time * 1000.0;
            sum_t3  += sol.phase3_time * 1000.0;
            sum_tot += t_ms;
        }

        double inv = 1.0 / trials;
        double inv_c = (constrained_trials > 0) ? 1.0 / constrained_trials : 0.0;
        std::cout << std::setw(6)  << n
                  << std::setw(8)  << std::fixed << std::setprecision(1) << (sum_verts    * inv_c)
                  << std::setw(11) << std::setprecision(1) << (sum_at_bound * inv_c) << "%"
                  << std::setw(10) << std::setprecision(1) << (sum_ph2      * inv_c)
                  << std::setw(11) << std::setprecision(1) << (sum_adj      * inv_c)
                  << std::setw(10) << std::setprecision(3) << (sum_t1  * inv_c)
                  << std::setw(10) << std::setprecision(3) << (sum_t2  * inv_c)
                  << std::setw(10) << std::setprecision(3) << (sum_t3  * inv_c)
                  << std::setw(10) << std::setprecision(3) << (sum_tot * inv)
                  << std::setw(11) << (constrained_trials == 0 ? "n/a" : (prop3_ok  ? "OK" : "FAIL"))
                  << std::setw(9)  << (constrained_trials == 0 ? "n/a" : (thm1_ok   ? "OK" : "FAIL"))
                  << std::setw(9)  << (all_solved ? "OK" : "FAIL")
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

    // 5. Hotstart: Diagonal MPC, fixed Q+A, only g varies (realistic MPC scenario)
    runHotstartBenchmark("MPC Diagonal Q – Hotstart (fixed Q, varying g)", {5, 10, 20, 30, 50},
                         [](int horizon, std::mt19937 &rng) { return makeDiagonalMPCQP(horizon, rng); });

    // ── Theory verification tables ────────────────────────────────────────
    std::cout << "\n\n";
    std::cout << "=======================================================\n";
    std::cout << "  THEORY VERIFICATION  (Prop 3 / Lemma 2 / Theorem 1)\n";
    std::cout << "=======================================================\n";

    // Box QP: dense Q  — Prop 3 (fast-path) + Theorem 1 (Ph2 hops ≤ adj bound)
    // Lemma 2 column is informational (heuristic for dense Q; exact for diag Q)
    runBoxTheorySuite("Random Box QP (dense Q)",
                      {10, 30, 50, 75, 100},
                      [](int n, std::mt19937 &rng) { return makeBoxQP(n, rng); });

    // Diagonal MPC: Prop 3 + Lemma 2 (exact, Euclidean == Q-weighted) + Theorem 1
    runBoxTheorySuite("MPC Diagonal Q (exact Lemma 2)",
                      {5, 10, 20, 30, 50},
                      [](int horizon, std::mt19937 &rng) { return makeDiagonalMPCQP(horizon, rng); },
                      /*diagonal_q=*/true);

    std::cout << "\nBenchmark Complete.\n";
    return 0;
}
