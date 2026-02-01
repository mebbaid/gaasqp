/**
 * @file benchmark_qpoases.cpp
 * @brief Comprehensive benchmark comparing GeoQP vs qpOASES (active set method)
 * 
 * This benchmark compares two geometrically-inspired approaches:
 * - GeoQP: Our novel geometric active set method
 * - qpOASES: Classic parametric active set method
 * 
 * We show solution trajectories and verify correctness across problem types.
 */

#include <geoqp/geoqp.hpp>
#include <qpOASES.hpp>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;

//=============================================================================
// Problem Generators
//=============================================================================

struct QPProblem {
    int n;  // variables
    int m;  // inequality constraints
    std::vector<double> H;  // n x n Hessian (row-major)
    std::vector<double> g;  // n gradient
    std::vector<double> A;  // m x n constraint matrix (row-major)
    std::vector<double> lb; // m lower bounds
    std::vector<double> ub; // m upper bounds
    std::string name;
};

// Generate random positive definite matrix
std::vector<double> randomPD(int n, std::mt19937& rng, double condition = 10.0) {
    std::normal_distribution<> dist(0.0, 1.0);
    std::vector<double> A(n * n), H(n * n);
    
    for (int i = 0; i < n * n; ++i) A[i] = dist(rng);
    
    // H = A^T * A + regularization
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[k * n + i] * A[k * n + j];
            }
            H[i * n + j] = sum;
        }
    }
    
    // Add regularization
    double trace = 0.0;
    for (int i = 0; i < n; ++i) trace += H[i * n + i];
    double reg = trace / (n * condition);
    for (int i = 0; i < n; ++i) H[i * n + i] += reg;
    
    return H;
}

// Problem 1: Simple box-constrained QP
QPProblem makeBoxQP(int n, std::mt19937& rng, double bound = 1.0) {
    QPProblem p;
    p.n = n;
    p.m = 2 * n;  // upper and lower bounds
    p.name = "Box QP (n=" + std::to_string(n) + ")";
    
    p.H = randomPD(n, rng);
    
    std::normal_distribution<> dist(0.0, 1.0);
    p.g.resize(n);
    for (int i = 0; i < n; ++i) p.g[i] = dist(rng) * 5.0;  // Push solution to boundary
    
    // Box constraints: -bound <= x <= bound
    // Represented as: I*x <= bound and -I*x <= bound
    p.A.resize(2 * n * n, 0.0);
    p.lb.resize(2 * n);
    p.ub.resize(2 * n);
    
    for (int i = 0; i < n; ++i) {
        p.A[i * n + i] = 1.0;           // x_i <= bound
        p.A[(n + i) * n + i] = -1.0;    // -x_i <= bound (i.e., x_i >= -bound)
        p.lb[i] = -1e20;
        p.ub[i] = bound;
        p.lb[n + i] = -1e20;
        p.ub[n + i] = bound;
    }
    
    return p;
}

// Problem 2: Polytope-constrained QP (random inequality constraints)
QPProblem makePolytopeQP(int n, int m, std::mt19937& rng) {
    QPProblem p;
    p.n = n;
    p.m = m;
    p.name = "Polytope QP (n=" + std::to_string(n) + ", m=" + std::to_string(m) + ")";
    
    p.H = randomPD(n, rng);
    
    std::normal_distribution<> dist(0.0, 1.0);
    std::uniform_real_distribution<> udist(0.5, 2.0);
    
    p.g.resize(n);
    for (int i = 0; i < n; ++i) p.g[i] = dist(rng);
    
    p.A.resize(m * n);
    p.lb.resize(m);
    p.ub.resize(m);
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            p.A[i * n + j] = dist(rng);
        }
        p.lb[i] = -1e20;
        p.ub[i] = udist(rng);
    }
    
    return p;
}

// Problem 3: MPC-like QP (structured with box constraints on controls)
QPProblem makeMPCQP(int horizon, std::mt19937& rng) {
    int n = 2 * horizon;  // 2 control inputs per timestep
    QPProblem p;
    p.n = n;
    p.m = 2 * n;  // box constraints
    p.name = "MPC QP (horizon=" + std::to_string(horizon) + ", n=" + std::to_string(n) + ")";
    
    // Block-diagonal Hessian (typical for MPC)
    p.H.resize(n * n, 0.0);
    for (int t = 0; t < horizon; ++t) {
        // 2x2 block for each timestep
        int idx = 2 * t;
        p.H[idx * n + idx] = 1.0 + 0.1 * t;         // Increasing weight
        p.H[(idx+1) * n + (idx+1)] = 1.0 + 0.1 * t;
        p.H[idx * n + (idx+1)] = 0.1;               // Small coupling
        p.H[(idx+1) * n + idx] = 0.1;
    }
    
    // Gradient pushes solution outside bounds
    std::normal_distribution<> dist(0.0, 1.0);
    p.g.resize(n);
    for (int i = 0; i < n; ++i) p.g[i] = dist(rng) * 3.0;
    
    // Box constraints: -1 <= u <= 1
    p.A.resize(2 * n * n, 0.0);
    p.lb.resize(2 * n);
    p.ub.resize(2 * n);
    
    for (int i = 0; i < n; ++i) {
        p.A[i * n + i] = 1.0;
        p.A[(n + i) * n + i] = -1.0;
        p.lb[i] = -1e20;
        p.ub[i] = 1.0;
        p.lb[n + i] = -1e20;
        p.ub[n + i] = 1.0;
    }
    
    return p;
}

// Problem 4: Least squares with bound constraints
QPProblem makeLeastSquaresQP(int n, int m_data, std::mt19937& rng, double bound = 2.0) {
    QPProblem p;
    p.n = n;
    p.m = 2 * n;  // box constraints
    p.name = "Least Squares (n=" + std::to_string(n) + ", data=" + std::to_string(m_data) + ")";
    
    std::normal_distribution<> dist(0.0, 1.0);
    
    // H = C^T * C where C is m_data x n
    std::vector<double> C(m_data * n);
    for (int i = 0; i < m_data * n; ++i) C[i] = dist(rng);
    
    p.H.resize(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < m_data; ++k) {
                p.H[i * n + j] += C[k * n + i] * C[k * n + j];
            }
        }
    }
    // Regularization
    for (int i = 0; i < n; ++i) p.H[i * n + i] += 0.01;
    
    // g = -C^T * y for random target y
    std::vector<double> y(m_data);
    for (int i = 0; i < m_data; ++i) y[i] = dist(rng);
    
    p.g.resize(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < m_data; ++k) {
            p.g[i] -= C[k * n + i] * y[k];
        }
    }
    
    // Box constraints
    p.A.resize(2 * n * n, 0.0);
    p.lb.resize(2 * n);
    p.ub.resize(2 * n);
    
    for (int i = 0; i < n; ++i) {
        p.A[i * n + i] = 1.0;
        p.A[(n + i) * n + i] = -1.0;
        p.lb[i] = -1e20;
        p.ub[i] = bound;
        p.lb[n + i] = -1e20;
        p.ub[n + i] = bound;
    }
    
    return p;
}

//=============================================================================
// Solver Wrappers
//=============================================================================

struct SolverResult {
    bool success;
    double solve_time_ms;
    std::vector<double> solution;
    double obj_value;
    int iterations;
    int active_constraints;
};

// Solve with GeoQP
SolverResult solveGeoQP(const QPProblem& prob, int verbosity = 0) {
    SolverResult result;
    
    geoqp::Problem gp;
    gp.n = prob.n;
    gp.m = prob.m;
    gp.Q = prob.H;
    gp.g = prob.g;
    
    // Convert to Ax <= b form
    gp.A.resize(prob.m * prob.n);
    gp.b.resize(prob.m);
    for (int i = 0; i < prob.m; ++i) {
        for (int j = 0; j < prob.n; ++j) {
            gp.A[i * prob.n + j] = prob.A[i * prob.n + j];
        }
        gp.b[i] = prob.ub[i];
    }
    
    geoqp::Config config;
    config.verbosity = verbosity;
    
    geoqp::Solver solver(config);
    
    auto start = Clock::now();
    geoqp::Solution sol = solver.solve(gp);
    auto end = Clock::now();
    
    result.success = (sol.status == geoqp::Status::SUCCESS);
    result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.solution = sol.x;
    result.obj_value = sol.objective_value;
    result.iterations = sol.phase1_iterations + sol.phase2_iterations;
    result.active_constraints = sol.num_active_constraints;
    
    // Verify constraint feasibility
    if (result.success && !result.solution.empty()) {
        double max_violation = 0.0;
        for (int i = 0; i < prob.m; ++i) {
            double ax = 0.0;
            for (int j = 0; j < prob.n; ++j) {
                ax += prob.A[i * prob.n + j] * result.solution[j];
            }
            double violation = ax - prob.ub[i];
            if (violation > max_violation) {
                max_violation = violation;
            }
        }
        if (max_violation > 1e-6) {
            std::cerr << "WARNING: GeoQP solution violates constraints by " << max_violation << "\n";
        }
    }
    
    return result;
}

// Solve with qpOASES
SolverResult solveQPOASES(const QPProblem& prob) {
    SolverResult result;
    
    int n = prob.n;
    int m = prob.m;
    
    // qpOASES uses column-major, but with row-major input we need to transpose H
    // Actually qpOASES can use row-major with ROW_MAJOR_ORDERING option
    
    qpOASES::QProblem qp(n, m, qpOASES::HST_POSDEF);
    
    qpOASES::Options options;
    options.setToMPC();  // Optimized for MPC-like problems
    options.printLevel = qpOASES::PL_NONE;  // Set after setToMPC to override
    qp.setOptions(options);
    
    int nWSR = 1000;  // Max working set recalculations
    
    auto start = Clock::now();
    
    qpOASES::returnValue status = qp.init(
        prob.H.data(),
        prob.g.data(),
        prob.A.data(),
        nullptr,      // lb on x (none)
        nullptr,      // ub on x (none)
        prob.lb.data(),
        prob.ub.data(),
        nWSR
    );
    
    auto end = Clock::now();
    
    result.success = (status == qpOASES::SUCCESSFUL_RETURN);
    result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.iterations = nWSR;
    
    if (result.success) {
        result.solution.resize(n);
        qp.getPrimalSolution(result.solution.data());
        result.obj_value = qp.getObjVal();
        result.active_constraints = qp.getNAC();
    }
    
    return result;
}

//=============================================================================
// Utility Functions
//=============================================================================

double computeObjective(const std::vector<double>& x, const std::vector<double>& H,
                        const std::vector<double>& g, int n) {
    double obj = 0.0;
    for (int i = 0; i < n; ++i) {
        obj += g[i] * x[i];
        for (int j = 0; j < n; ++j) {
            obj += 0.5 * x[i] * H[i * n + j] * x[j];
        }
    }
    return obj;
}

double solutionDiff(const std::vector<double>& x1, const std::vector<double>& x2) {
    if (x1.size() != x2.size()) return -1.0;
    double diff = 0.0;
    for (size_t i = 0; i < x1.size(); ++i) {
        diff += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }
    return std::sqrt(diff);
}

void printSolutionRange(const std::vector<double>& x, const std::string& name) {
    if (x.empty()) {
        std::cout << name << ": empty\n";
        return;
    }
    double minv = *std::min_element(x.begin(), x.end());
    double maxv = *std::max_element(x.begin(), x.end());
    std::cout << name << " range: [" << std::fixed << std::setprecision(4) 
              << minv << ", " << maxv << "]\n";
}

//=============================================================================
// Benchmark Functions
//=============================================================================

void runSingleProblemDetailed(const QPProblem& prob) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Problem: " << prob.name << "\n";
    std::cout << "Variables: " << prob.n << ", Constraints: " << prob.m << "\n";
    std::cout << std::string(70, '-') << "\n";
    
    // Solve with both solvers
    auto geoqp_result = solveGeoQP(prob, 0);
    auto qpoases_result = solveQPOASES(prob);
    
    // Print results
    std::cout << "\n" << std::setw(20) << "Metric" 
              << std::setw(20) << "GeoQP" 
              << std::setw(20) << "qpOASES" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    std::cout << std::setw(20) << "Status" 
              << std::setw(20) << (geoqp_result.success ? "SUCCESS" : "FAILED")
              << std::setw(20) << (qpoases_result.success ? "SUCCESS" : "FAILED") << "\n";
    
    std::cout << std::setw(20) << "Time (ms)" 
              << std::setw(20) << std::fixed << std::setprecision(4) << geoqp_result.solve_time_ms
              << std::setw(20) << qpoases_result.solve_time_ms << "\n";
    
    std::cout << std::setw(20) << "Objective" 
              << std::setw(20) << std::setprecision(6) << geoqp_result.obj_value
              << std::setw(20) << qpoases_result.obj_value << "\n";
    
    std::cout << std::setw(20) << "Iterations" 
              << std::setw(20) << geoqp_result.iterations
              << std::setw(20) << qpoases_result.iterations << "\n";
    
    std::cout << std::setw(20) << "Active Constraints" 
              << std::setw(20) << geoqp_result.active_constraints
              << std::setw(20) << qpoases_result.active_constraints << "\n";
    
    if (geoqp_result.success && qpoases_result.success) {
        double sol_diff = solutionDiff(geoqp_result.solution, qpoases_result.solution);
        double obj_diff = std::abs(geoqp_result.obj_value - qpoases_result.obj_value);
        
        std::cout << std::setw(20) << "Solution ||diff||" 
                  << std::setw(20) << std::scientific << std::setprecision(2) << sol_diff
                  << std::setw(20) << "-" << "\n";
        
        std::cout << std::setw(20) << "Objective diff" 
                  << std::setw(20) << obj_diff
                  << std::setw(20) << "-" << "\n";
        
        double speedup = qpoases_result.solve_time_ms / geoqp_result.solve_time_ms;
        std::cout << std::setw(20) << "Speedup (GeoQP)" 
                  << std::setw(20) << std::fixed << std::setprecision(2) << speedup << "x"
                  << std::setw(20) << "-" << "\n";
    }
    
    // Print solution samples
    std::cout << "\nSolution samples (first 5 components):\n";
    int show = std::min(5, prob.n);
    std::cout << "  GeoQP:   [";
    for (int i = 0; i < show; ++i) {
        std::cout << std::fixed << std::setprecision(4) << geoqp_result.solution[i];
        if (i < show - 1) std::cout << ", ";
    }
    std::cout << (prob.n > 5 ? ", ..." : "") << "]\n";
    
    std::cout << "  qpOASES: [";
    for (int i = 0; i < show; ++i) {
        std::cout << std::fixed << std::setprecision(4) << qpoases_result.solution[i];
        if (i < show - 1) std::cout << ", ";
    }
    std::cout << (prob.n > 5 ? ", ..." : "") << "]\n";
    
    printSolutionRange(geoqp_result.solution, "  GeoQP");
    printSolutionRange(qpoases_result.solution, "  qpOASES");
}

void runBenchmarkSuite(int seed = 42) {
    std::mt19937 rng(seed);
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  GeoQP vs qpOASES Comprehensive Benchmark\n";
    std::cout << "  (Geometric Active Set vs Parametric Active Set)\n";
    std::cout << std::string(80, '=') << "\n";
    
    //=========================================================================
    // Test 1: Box-Constrained Problems (varying size)
    //=========================================================================
    std::cout << "\n\n### Test 1: Box-Constrained QP ###\n";
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::setw(8) << "n" 
              << std::setw(12) << "GeoQP(ms)"
              << std::setw(12) << "qpOASES(ms)"
              << std::setw(10) << "Speedup"
              << std::setw(10) << "Active"
              << std::setw(15) << "Obj Diff"
              << std::setw(15) << "Sol Diff" << "\n";
    std::cout << std::string(100, '-') << "\n";
    
    for (int n : {5, 10, 20, 30, 50, 75, 100}) {
        auto prob = makeBoxQP(n, rng);
        auto g = solveGeoQP(prob);
        auto q = solveQPOASES(prob);
        
        double speedup = q.solve_time_ms / g.solve_time_ms;
        double obj_diff = std::abs(g.obj_value - q.obj_value);
        double sol_diff = solutionDiff(g.solution, q.solution);
        
        std::cout << std::setw(8) << n
                  << std::setw(12) << std::fixed << std::setprecision(3) << g.solve_time_ms
                  << std::setw(12) << q.solve_time_ms
                  << std::setw(9) << std::setprecision(2) << speedup << "x"
                  << std::setw(10) << g.active_constraints
                  << std::setw(15) << std::scientific << std::setprecision(2) << obj_diff
                  << std::setw(15) << sol_diff << "\n";
    }
    
    //=========================================================================
    // Test 2: MPC-Like Problems
    //=========================================================================
    std::cout << "\n\n### Test 2: MPC-Like Problems ###\n";
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::setw(8) << "Horizon"
              << std::setw(6) << "n"
              << std::setw(12) << "GeoQP(ms)"
              << std::setw(12) << "qpOASES(ms)"
              << std::setw(10) << "Speedup"
              << std::setw(10) << "Active"
              << std::setw(15) << "Obj Diff" << "\n";
    std::cout << std::string(100, '-') << "\n";
    
    for (int horizon : {5, 10, 20, 30, 50}) {
        auto prob = makeMPCQP(horizon, rng);
        auto g = solveGeoQP(prob);
        auto q = solveQPOASES(prob);
        
        double speedup = q.solve_time_ms / g.solve_time_ms;
        double obj_diff = std::abs(g.obj_value - q.obj_value);
        
        std::cout << std::setw(8) << horizon
                  << std::setw(6) << prob.n
                  << std::setw(12) << std::fixed << std::setprecision(3) << g.solve_time_ms
                  << std::setw(12) << q.solve_time_ms
                  << std::setw(9) << std::setprecision(2) << speedup << "x"
                  << std::setw(10) << g.active_constraints
                  << std::setw(15) << std::scientific << std::setprecision(2) << obj_diff << "\n";
    }
    
    //=========================================================================
    // Test 3: Least Squares with Bounds
    //=========================================================================
    std::cout << "\n\n### Test 3: Bounded Least Squares ###\n";
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::setw(8) << "n"
              << std::setw(8) << "data"
              << std::setw(12) << "GeoQP(ms)"
              << std::setw(12) << "qpOASES(ms)"
              << std::setw(10) << "Speedup"
              << std::setw(10) << "Active"
              << std::setw(15) << "Obj Diff" << "\n";
    std::cout << std::string(100, '-') << "\n";
    
    for (int n : {10, 20, 30, 50}) {
        int m_data = 3 * n;
        auto prob = makeLeastSquaresQP(n, m_data, rng);
        auto g = solveGeoQP(prob);
        auto q = solveQPOASES(prob);
        
        double speedup = q.solve_time_ms / g.solve_time_ms;
        double obj_diff = std::abs(g.obj_value - q.obj_value);
        
        std::cout << std::setw(8) << n
                  << std::setw(8) << m_data
                  << std::setw(12) << std::fixed << std::setprecision(3) << g.solve_time_ms
                  << std::setw(12) << q.solve_time_ms
                  << std::setw(9) << std::setprecision(2) << speedup << "x"
                  << std::setw(10) << g.active_constraints
                  << std::setw(15) << std::scientific << std::setprecision(2) << obj_diff << "\n";
    }
    
    //=========================================================================
    // Detailed Examples
    //=========================================================================
    std::cout << "\n\n### Detailed Problem Analysis ###\n";
    
    // Small problem for detailed analysis
    auto small_box = makeBoxQP(5, rng);
    runSingleProblemDetailed(small_box);
    
    auto mpc_small = makeMPCQP(5, rng);
    runSingleProblemDetailed(mpc_small);
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Benchmark complete.\n";
}

int main() {
    runBenchmarkSuite(42);
    return 0;
}
