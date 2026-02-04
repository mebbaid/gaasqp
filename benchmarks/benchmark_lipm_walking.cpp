/**
 * @file benchmark_lipm_walking.cpp
 * @brief Benchmark LIPM Walking Controller MPC Problem
 * 
 * Based on the LIPM walking controller from Stéphane Caron:
 * https://github.com/stephane-caron/lipm_walking_controller/
 * 
 * Tests GeoQP (with different face hopping strategies) vs OSQP vs qpOASES
 * on a realistic humanoid walking MPC problem.
 * 
 * Code written with the help of Claude Sonnet 
 * 
 */

#include <geoqp/geoqp.hpp>
#include <QpSolversEigen/QpSolversEigen.hpp>
#include <qpOASES.hpp>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include <fstream>

using Clock = std::chrono::high_resolution_clock;

const double MAX_ZMP_DIST = 100.0;  // [m]

struct Parameters {
    double com_height = 0.84;              // [m]
    double dsp_duration = 0.1;             // [s] (double support phase)
    double foot_size = 0.065;              // [m]
    double gravity = 9.81;                 // [m/s²]
    double init_support_foot_pos = 0.09;   // [m]
    int nb_timesteps = 16;
    double sampling_period = 0.1;          // [s]
    double ssp_duration = 0.7;             // [s] (single support phase)
    std::vector<double> strides = {-0.18, 0.18};  // [m]
    
    double omega() const { return std::sqrt(gravity / com_height); }
    
    std::vector<double> dcm_from_state() const {
        return {1.0, 1.0 / omega(), 0.0};
    }
    
    std::vector<double> zmp_from_state() const {
        double w = omega();
        return {1.0, 0.0, -1.0 / (w * w)};
    }
};

struct MPCProblem {
    int n_states = 3;  // position, velocity, acceleration
    int n_inputs = 1;  // jerk
    int nb_timesteps;
    
    std::vector<double> A;  // State transition matrix (3x3)
    std::vector<double> B;  // Input matrix (3x1)
    std::vector<double> Q;  // State cost (3x3)
    double R;               // Input cost (scalar)
    
    std::vector<double> initial_state;
    std::vector<double> goal_state;
    std::vector<std::vector<double>> ineq_vector;  // Time-varying bounds
    
    std::vector<double> zmp_from_state;
};

MPCProblem buildMPCProblem(const Parameters& params) {
    MPCProblem prob;
    prob.nb_timesteps = params.nb_timesteps;
    
    double T = params.sampling_period;
    
    // State transition: x_{k+1} = A * x_k + B * u_k
    prob.A = {
        1.0, T, T*T/2.0,
        0.0, 1.0, T,
        0.0, 0.0, 1.0
    };
    
    prob.B = {T*T*T/6.0, T*T/2.0, T};
    
    // Terminal cost weight
    prob.Q = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    
    // Stage input cost (increased for better numerical conditioning)
    prob.R = 1e-2;
    
    prob.zmp_from_state = params.zmp_from_state();
    
    return prob;
}

// Convert MPC problem to dense QP: min 0.5 x'Hx + g'x s.t. Ax <= b
struct DenseQP {
    int n;  // number of decision variables
    int m;  // number of constraints
    std::vector<double> H;  // Hessian (n x n)
    std::vector<double> g;  // gradient (n)
    std::vector<double> A;  // constraints (m x n)
    std::vector<double> b;  // bounds (m)
    std::vector<double> lb;  // lower bounds on variables (n)
    std::vector<double> ub;  // upper bounds on variables (n)
};

DenseQP buildDenseQP(const MPCProblem& mpc) {
    DenseQP qp;
    int N = mpc.nb_timesteps;
    int nx = mpc.n_states;
    int nu = mpc.n_inputs;
    
    qp.n = N * nu;  // Decision variables: u_0, u_1, ..., u_{N-1}
    qp.m = 2 * N;   // Constraints: 2 per timestep (ZMP bounds)
    
    // Build Hessian: H = R * I (input regularization)
    qp.H.resize(qp.n * qp.n, 0.0);
    for (int i = 0; i < qp.n; ++i) {
        qp.H[i * qp.n + i] = mpc.R;
    }
    
    // Gradient (will be set after computing A_powers)
    qp.g.resize(qp.n, 0.0);
    
    // Variable bounds (reasonable jerk limits: ±10 m/s³)
    qp.lb.resize(qp.n, -10.0);
    qp.ub.resize(qp.n, 10.0);
    
    // Constraints: ZMP bounds at each timestep + variable bounds
    qp.m = 2 * N + 2 * qp.n;  // 2 ZMP constraints per timestep + 2 bounds per variable
    qp.A.resize(qp.m * qp.n, 0.0);
    qp.b.resize(qp.m);
    
    // Helper: matrix-vector multiply (3x3 * 3x1)
    auto matvec = [&](const std::vector<double>& mat, const std::vector<double>& vec) {
        std::vector<double> result(nx, 0.0);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nx; ++j) {
                result[i] += mat[i * nx + j] * vec[j];
            }
        }
        return result;
    };
    
    // Helper: matrix-matrix multiply (3x3 * 3x3)
    auto matmul = [&](const std::vector<double>& A_mat, const std::vector<double>& B_mat) {
        std::vector<double> result(nx * nx, 0.0);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nx; ++j) {
                for (int k = 0; k < nx; ++k) {
                    result[i * nx + j] += A_mat[i * nx + k] * B_mat[k * nx + j];
                }
            }
        }
        return result;
    };
    
    // Precompute powers of A: A^0, A^1, A^2, ..., A^N
    std::vector<std::vector<double>> A_powers(N + 1);
    A_powers[0].resize(nx * nx, 0.0);
    for (int i = 0; i < nx; ++i) {
        A_powers[0][i * nx + i] = 1.0;  // A^0 = I
    }
    A_powers[1] = mpc.A;  // A^1 = A
    for (int k = 2; k <= N; ++k) {
        A_powers[k] = matmul(mpc.A, A_powers[k-1]);  // A^k = A * A^{k-1}
    }
    
    // Now compute gradient: drive terminal state towards goal
    // Terminal state x_N = A^N * x_0 + sum_{j=0}^{N-1} A^{N-1-j} * B * u_j
    // Cost: (x_N - x_goal)' Q (x_N - x_goal)
    // Gradient component: g_j = Q * (A^N * x_0 - x_goal)' * A^{N-1-j} * B
    std::vector<double> x_N = matvec(A_powers[N], mpc.initial_state);
    double pos_error = x_N[0] - mpc.goal_state[0];
    
    for (int j = 0; j < N; ++j) {
        // Influence of u_j on terminal position
        std::vector<double> A_pow_B = matvec(A_powers[N - 1 - j], mpc.B);
        double influence = A_pow_B[0];  // Position component
        
        qp.g[j] = pos_error * influence * mpc.Q[0];  // Q[0] = position weight
    }
    
    // Build constraints for each timestep
    // At time k, state is: x_k = A^k * x_0 + sum_{j=0}^{k-1} A^{k-1-j} * B * u_j
    // ZMP constraint: zmp' * x_k <= bound
    for (int k = 0; k < N; ++k) {
        // Compute zmp' * A^k * x_0 (constant term)
        std::vector<double> A_k_x0 = matvec(A_powers[k], mpc.initial_state);
        double zmp_offset = 0.0;
        for (int i = 0; i < nx; ++i) {
            zmp_offset += mpc.zmp_from_state[i] * A_k_x0[i];
        }
        
        // For each input u_j (j = 0, ..., k-1), compute contribution to ZMP at time k
        for (int j = 0; j < k; ++j) {
            // State contribution: A^{k-1-j} * B * u_j
            std::vector<double> A_pow_B = matvec(A_powers[k - 1 - j], mpc.B);
            
            // ZMP contribution: zmp' * A^{k-1-j} * B
            double zmp_coef = 0.0;
            for (int i = 0; i < nx; ++i) {
                zmp_coef += mpc.zmp_from_state[i] * A_pow_B[i];
            }
            
            // Add to constraint matrix
            qp.A[(2*k) * qp.n + j] = zmp_coef;       // Upper bound
            qp.A[(2*k+1) * qp.n + j] = -zmp_coef;    // Lower bound
        }
        
        // Also account for input at time k (which affects state at time k+1, but we use k here)
        // Actually, u_k affects x_{k+1}, so for ZMP at time k, we only use u_0...u_{k-1}
        // The above loop is correct.
        
        // Set bounds
        // ineq_vector[k] = {upper_bound, -lower_bound} (note: second element is negated)
        qp.b[2*k] = mpc.ineq_vector[k][0] - zmp_offset;      // Upper: zmp <= upper_bound
        qp.b[2*k+1] = mpc.ineq_vector[k][1] + zmp_offset;    // Lower: -zmp <= -lower_bound (using negated form directly)
    }
    
    // Add variable bounds as linear constraints
    for (int i = 0; i < qp.n; ++i) {
        // Upper bound: u_i <= ub
        qp.A[(2*N + 2*i) * qp.n + i] = 1.0;
        qp.b[2*N + 2*i] = qp.ub[i];
        
        // Lower bound: -u_i <= -lb
        qp.A[(2*N + 2*i + 1) * qp.n + i] = -1.0;
        qp.b[2*N + 2*i + 1] = -qp.lb[i];
    }
    
    return qp;
}

struct BenchmarkResult {
    std::string solver_name;
    bool success;
    double solve_time_ms;
    double objective;
    int iterations;
    int active_constraints;
};

// Zero-copy warm-start version of GeoQP solver (NEW - uses setup/hotstart API)
BenchmarkResult solveWithGeoQPWarmStart(const DenseQP& qp, geoqp::Solver& solver, 
                                         const std::string& name) {
    BenchmarkResult result;
    result.solver_name = name;
    
    // Use zero-copy API - only pass pointers to changing data (g, b)
    auto start = Clock::now();
    geoqp::Solution sol = solver.solve_hotstart(qp.g.data(), qp.b.data());
    auto end = Clock::now();
    
    result.success = (sol.status == geoqp::Status::SUCCESS);
    result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.objective = sol.objective_value;
    result.iterations = sol.phase2_iterations;
    result.active_constraints = sol.num_active_constraints;
    
    return result;
}

// Warm-start version of OSQP solver  
BenchmarkResult solveWithOSQPWarmStart(const DenseQP& qp, QpSolversEigen::Solver& solver) {
    BenchmarkResult result;
    result.solver_name = "OSQP";
    
    int n = qp.n;
    int m = qp.m;
    
    // Update only the vectors (not the structure)
    Eigen::VectorXd g_eigen(n);
    Eigen::VectorXd b_eigen(m);
    
    for (int i = 0; i < n; ++i) {
        g_eigen(i) = qp.g[i];
    }
    
    for (int i = 0; i < m; ++i) {
        b_eigen(i) = qp.b[i];
    }
    
    Eigen::VectorXd lb = -1e20 * Eigen::VectorXd::Ones(m);
    
    // Update gradient and bounds (warm start)
    solver.updateGradient(g_eigen);
    solver.updateBounds(lb, b_eigen);
    
    auto start = Clock::now();
    auto status = solver.solveProblem();  // Warm-started solve
    auto end = Clock::now();
    
    result.success = (status == QpSolversEigen::ErrorExitFlag::NoError);
    result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (result.success) {
        result.objective = solver.getObjValue();
        result.iterations = -1;  // N/A
        result.active_constraints = 0;
    }
    
    return result;
}

// Warm-start version of qpOASES solver
BenchmarkResult solveWithQpOASESWarmStart(const DenseQP& qp, qpOASES::QProblem& qp_problem, 
                                           bool first_call) {
    BenchmarkResult result;
    result.solver_name = "qpOASES";
    
    int n = qp.n;
    int m = qp.m;
    
    qpOASES::real_t* H_data = new qpOASES::real_t[n * n];
    qpOASES::real_t* g_data = new qpOASES::real_t[n];
    qpOASES::real_t* A_data = new qpOASES::real_t[m * n];
    qpOASES::real_t* lbA = new qpOASES::real_t[m];
    qpOASES::real_t* ubA = new qpOASES::real_t[m];
    
    for (int i = 0; i < n * n; ++i) H_data[i] = qp.H[i];
    for (int i = 0; i < n; ++i) g_data[i] = qp.g[i];
    for (int i = 0; i < m * n; ++i) A_data[i] = qp.A[i];
    for (int i = 0; i < m; ++i) {
        lbA[i] = -1e20;
        ubA[i] = qp.b[i];
    }
    
    int nWSR = 1000;
    qpOASES::returnValue ret;
    
    auto start = Clock::now();
    if (first_call) {
        // Cold start for first iteration
        ret = qp_problem.init(H_data, g_data, A_data, nullptr, nullptr, lbA, ubA, nWSR);
    } else {
        // Warm start for subsequent iterations
        ret = qp_problem.hotstart(g_data, nullptr, nullptr, lbA, ubA, nWSR);
    }
    auto end = Clock::now();
    
    result.success = (ret == qpOASES::SUCCESSFUL_RETURN);
    result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.iterations = nWSR;
    
    if (result.success) {
        result.objective = qp_problem.getObjVal();
        result.active_constraints = qp_problem.getNAC();
    }
    
    delete[] H_data;
    delete[] g_data;
    delete[] A_data;
    delete[] lbA;
    delete[] ubA;
    
    return result;
}

// Original cold-start solvers (kept for reference/other benchmarks)
BenchmarkResult solveWithGeoQP(const DenseQP& qp, geoqp::FaceHoppingStrategy strategy, 
                                const std::string& name) {
    BenchmarkResult result;
    result.solver_name = name;
    
    geoqp::Problem prob;
    prob.n = qp.n;
    prob.m = qp.m;
    prob.Q = qp.H;
    prob.g = qp.g;
    prob.A = qp.A;
    prob.b = qp.b;
    
    geoqp::Config config;
    config.face_hopping_strategy = strategy;
    config.use_caching = true;
    config.tolerance = 1e-6;        // Looser tolerance for better numerical stability
    
    geoqp::Solver solver(config);
    
    auto start = Clock::now();
    geoqp::Solution sol = solver.solve(prob);
    auto end = Clock::now();
    
    result.success = (sol.status == geoqp::Status::SUCCESS);
    result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.objective = sol.objective_value;
    result.iterations = sol.phase2_iterations;
    result.active_constraints = sol.num_active_constraints;
    
    return result;
}

BenchmarkResult solveWithOSQP(const DenseQP& qp) {
    BenchmarkResult result;
    result.solver_name = "OSQP";
    
    int n = qp.n;
    int m = qp.m;
    
    Eigen::MatrixXd H_eigen(n, n);
    Eigen::VectorXd g_eigen(n);
    Eigen::MatrixXd A_eigen(m, n);
    Eigen::VectorXd b_eigen(m);
    
    for (int i = 0; i < n; ++i) {
        g_eigen(i) = qp.g[i];
        for (int j = 0; j < n; ++j) {
            H_eigen(i, j) = qp.H[i * n + j];
        }
    }
    
    for (int i = 0; i < m; ++i) {
        b_eigen(i) = qp.b[i];
        for (int j = 0; j < n; ++j) {
            A_eigen(i, j) = qp.A[i * n + j];
        }
    }
    
    Eigen::MatrixXd H_upper = H_eigen.triangularView<Eigen::Upper>();
    Eigen::SparseMatrix<double> H_sparse = H_upper.sparseView();
    Eigen::SparseMatrix<double> A_sparse = A_eigen.sparseView();
    
    QpSolversEigen::Solver solver;
    solver.instantiateSolver("osqp");
    solver.setBooleanParameter("verbose", false);
    // Note: QpSolversEigen doesn't expose all OSQP parameters
    // OSQP should converge quickly on well-conditioned MPC problems
    solver.data()->setNumberOfVariables(n);
    solver.data()->setNumberOfInequalityConstraints(m);
    solver.data()->setHessianMatrix(H_sparse);
    solver.data()->setGradient(g_eigen);
    solver.data()->setInequalityConstraintsMatrix(A_sparse);
    
    Eigen::VectorXd lb = -1e20 * Eigen::VectorXd::Ones(m);
    solver.data()->setLowerBound(lb);
    solver.data()->setUpperBound(b_eigen);
    
    auto start = Clock::now();
    solver.initSolver();
    auto status = solver.solveProblem();
    auto end = Clock::now();
    
    result.success = (status == QpSolversEigen::ErrorExitFlag::NoError);
    result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (result.success) {
        result.objective = solver.getObjValue();
        // Note: QpSolversEigen wrapper doesn't expose OSQP iteration count
        // OSQP typically takes 10-50 iterations for MPC problems
        result.iterations = -1;  // Not available through this interface
        result.active_constraints = 0;  // Not available through this interface
    }
    
    return result;
}

BenchmarkResult solveWithQpOASES(const DenseQP& qp) {
    BenchmarkResult result;
    result.solver_name = "qpOASES";
    
    int n = qp.n;
    int m = qp.m;
    
    // qpOASES uses row-major format
    qpOASES::real_t* H_data = new qpOASES::real_t[n * n];
    qpOASES::real_t* g_data = new qpOASES::real_t[n];
    qpOASES::real_t* A_data = new qpOASES::real_t[m * n];
    qpOASES::real_t* lbA = new qpOASES::real_t[m];
    qpOASES::real_t* ubA = new qpOASES::real_t[m];
    
    for (int i = 0; i < n * n; ++i) H_data[i] = qp.H[i];
    for (int i = 0; i < n; ++i) g_data[i] = qp.g[i];
    for (int i = 0; i < m * n; ++i) A_data[i] = qp.A[i];
    for (int i = 0; i < m; ++i) {
        lbA[i] = -1e20;
        ubA[i] = qp.b[i];
    }
    
    qpOASES::QProblem qp_problem(n, m, qpOASES::HST_POSDEF);
    qpOASES::Options options;
    options.setToMPC();  // Optimized for MPC-like problems
    options.printLevel = qpOASES::PL_NONE;  // Override default from setToMPC
    qp_problem.setOptions(options);
    
    int nWSR = 1000;  // Max working set recalculations
    
    auto start = Clock::now();
    qpOASES::returnValue ret = qp_problem.init(
        H_data, g_data, A_data, nullptr, nullptr, lbA, ubA, nWSR
    );
    auto end = Clock::now();
    
    result.success = (ret == qpOASES::SUCCESSFUL_RETURN);
    result.solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.iterations = nWSR;
    
    if (result.success) {
        result.objective = qp_problem.getObjVal();
        result.active_constraints = qp_problem.getNAC();  // Number of active constraints
    }
    
    delete[] H_data;
    delete[] g_data;
    delete[] A_data;
    delete[] lbA;
    delete[] ubA;
    
    return result;
}

class PhaseI {
public:
    PhaseI(const Parameters& params) : params_(params) {
        int nb_dsp_steps = static_cast<int>(std::round(params.dsp_duration / params.sampling_period));
        int nb_ssp_steps = static_cast<int>(std::round(params.ssp_duration / params.sampling_period));
        
        index_ = 5;  // Initial index in middle of first SSP
        nb_dsp_steps_ = nb_dsp_steps;
        nb_ssp_steps_ = nb_ssp_steps;
        stride_index_ = 0;
    }
    
    void advance() {
        index_++;
        if (index_ >= nb_dsp_steps_ + nb_ssp_steps_) {
            index_ = 0;
        }
    }
    
    void advance_stride() {
        stride_index_ = (stride_index_ + 1) % params_.strides.size();
    }
    
    double get_next_foot_pos(double foot_pos) const {
        return foot_pos + params_.strides[stride_index_];
    }
    
    double get_last_foot_pos(double foot_pos) const {
        int upcoming_stride = (stride_index_ + 1) % params_.strides.size();
        return get_next_foot_pos(foot_pos) + params_.strides[upcoming_stride];
    }
    
    void updateIneqVector(MPCProblem& mpc, double cur_foot_pos) {
        int offset = index_;
        int nb_init_dsp = std::max(0, nb_dsp_steps_ - offset);
        offset = std::max(0, offset - nb_dsp_steps_);
        int nb_init_ssp = std::max(0, nb_ssp_steps_ - offset);
        offset = std::max(0, offset - nb_ssp_steps_);
        
        int remaining = params_.nb_timesteps - nb_init_dsp - nb_init_ssp;
        int nb_next_dsp = std::min(nb_dsp_steps_, remaining);
        remaining = std::max(0, remaining - nb_dsp_steps_);
        int nb_next_ssp = std::min(nb_ssp_steps_, remaining);
        remaining = std::max(0, remaining - nb_ssp_steps_);
        int nb_last_dsp = std::min(nb_dsp_steps_, remaining);
        remaining = std::max(0, remaining - nb_ssp_steps_);
        int nb_last_ssp = std::min(nb_ssp_steps_, remaining);
        
        double next_foot_pos = get_next_foot_pos(cur_foot_pos);
        double last_foot_pos = get_last_foot_pos(cur_foot_pos);
        
        double cur_max = cur_foot_pos + 0.5 * params_.foot_size;
        double cur_min = cur_foot_pos - 0.5 * params_.foot_size;
        double next_max = next_foot_pos + 0.5 * params_.foot_size;
        double next_min = next_foot_pos - 0.5 * params_.foot_size;
        double last_max = last_foot_pos + 0.5 * params_.foot_size;
        double last_min = last_foot_pos - 0.5 * params_.foot_size;
        
        mpc.ineq_vector.clear();
        for (int i = 0; i < nb_init_dsp; ++i) 
            mpc.ineq_vector.push_back({MAX_ZMP_DIST, MAX_ZMP_DIST});
        for (int i = 0; i < nb_init_ssp; ++i) 
            mpc.ineq_vector.push_back({cur_max, -cur_min});
        for (int i = 0; i < nb_next_dsp; ++i) 
            mpc.ineq_vector.push_back({MAX_ZMP_DIST, MAX_ZMP_DIST});
        for (int i = 0; i < nb_next_ssp; ++i) 
            mpc.ineq_vector.push_back({next_max, -next_min});
        for (int i = 0; i < nb_last_dsp; ++i) 
            mpc.ineq_vector.push_back({MAX_ZMP_DIST, MAX_ZMP_DIST});
        for (int i = 0; i < nb_last_ssp; ++i) 
            mpc.ineq_vector.push_back({last_max, -last_min});
        
        // Ensure we have exactly nb_timesteps entries (pad with relaxed constraints if needed)
        while (mpc.ineq_vector.size() < static_cast<size_t>(params_.nb_timesteps)) {
            mpc.ineq_vector.push_back({MAX_ZMP_DIST, MAX_ZMP_DIST});
        }
        
        double goal_pos = nb_last_dsp > 0 ? last_foot_pos : next_foot_pos;
        mpc.goal_state = {goal_pos, 0.0, 0.0};
    }
    
    int index() const { return index_; }
    
private:
    const Parameters& params_;
    int index_;
    int nb_dsp_steps_;
    int nb_ssp_steps_;
    int stride_index_;
};

std::vector<double> integrate(const std::vector<double>& state, double jerk, double dt) {
    double p_0 = state[0], v_0 = state[1], a_0 = state[2];
    return {
        p_0 + dt * (v_0 + dt * (a_0 / 2.0 + dt * jerk / 6.0)),
        v_0 + dt * (a_0 + dt * (jerk / 2.0)),
        a_0 + dt * jerk
    };
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     LIPM Walking Controller MPC Benchmark                     ║\n";
    std::cout << "║     GeoQP (multiple strategies) vs OSQP vs qpOASES            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    Parameters params;
    std::cout << "Building MPC problem..." << std::endl;
    MPCProblem mpc_problem = buildMPCProblem(params);
    std::cout << "Creating phase stepper..." << std::endl;
    PhaseI phase(params);
    
    std::cout << "Initializing state..." << std::endl;
    double support_foot_pos = params.init_support_foot_pos;
    double init_accel = -params.omega() * params.omega() * support_foot_pos;
    double init_vel = 0.5 * params.omega() * support_foot_pos;
    std::vector<double> state = {0.0, init_vel, init_accel};
    
    std::cout << "Setup complete." << std::endl;
    
    int num_steps = 10;  // Number of MPC iterations to test
    
    std::cout << "Problem parameters:\n";
    std::cout << "  Horizon: " << params.nb_timesteps << " timesteps\n";
    std::cout << "  Sampling period: " << params.sampling_period << " s\n";
    std::cout << "  Decision variables: " << params.nb_timesteps << " (jerks)\n";
    std::cout << "  ZMP constraints: " << 2 * params.nb_timesteps << "\n";
    std::cout << "  Variable bound constraints: " << 2 * params.nb_timesteps << "\n";
    std::cout << "  Total constraints: " << 4 * params.nb_timesteps << "\n\n";
    
    std::vector<std::pair<geoqp::FaceHoppingStrategy, std::string>> geoqp_strategies = {
        {geoqp::FaceHoppingStrategy::STANDARD, "GeoQP-Standard"},
        {geoqp::FaceHoppingStrategy::BATCH, "GeoQP-Batch"},
        {geoqp::FaceHoppingStrategy::SENSITIVITY, "GeoQP-Sensitivity"},
        {geoqp::FaceHoppingStrategy::NEIGHBORS, "GeoQP-Neighbors"}
    };
        
    // Initialize GeoQP solvers (one per strategy) using unique_ptr since Solver is not copyable
    std::vector<std::unique_ptr<geoqp::Solver>> geoqp_solvers;
    for (const auto& [strategy, name] : geoqp_strategies) {
        geoqp::Config config;
        config.face_hopping_strategy = strategy;
        config.use_caching = true;
        config.tolerance = 1e-6;
        config.warm_start = true;  // Enable warm starting
        geoqp_solvers.push_back(std::make_unique<geoqp::Solver>(config));
    }
    
    // Initialize OSQP solver once with first problem structure
    mpc_problem.initial_state = state;
    phase.updateIneqVector(mpc_problem, support_foot_pos);
    DenseQP initial_qp = buildDenseQP(mpc_problem);
    
    // ZERO-COPY SETUP: Call setup() on each GeoQP solver to cache Q and A matrices
    for (auto& solver : geoqp_solvers) {
        solver->setup(initial_qp.n, initial_qp.m, initial_qp.H.data(), initial_qp.A.data());
    }
    
    QpSolversEigen::Solver osqp_solver;
    osqp_solver.instantiateSolver("osqp");
    osqp_solver.setBooleanParameter("verbose", false);
    osqp_solver.setBooleanParameter("warm_start", true);  // Enable warm starting
    
    int n = initial_qp.n;
    int m = initial_qp.m;
    
    Eigen::MatrixXd H_eigen(n, n);
    Eigen::MatrixXd A_eigen(m, n);
    Eigen::VectorXd g_eigen(n);
    Eigen::VectorXd b_eigen(m);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            H_eigen(i, j) = initial_qp.H[i * n + j];
        }
        g_eigen(i) = initial_qp.g[i];
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A_eigen(i, j) = initial_qp.A[i * n + j];
        }
        b_eigen(i) = initial_qp.b[i];
    }
    
    Eigen::MatrixXd H_upper = H_eigen.triangularView<Eigen::Upper>();
    Eigen::SparseMatrix<double> H_sparse = H_upper.sparseView();
    Eigen::SparseMatrix<double> A_sparse = A_eigen.sparseView();
    Eigen::VectorXd lb = -1e20 * Eigen::VectorXd::Ones(m);
    
    osqp_solver.data()->setNumberOfVariables(n);
    osqp_solver.data()->setNumberOfInequalityConstraints(m);
    osqp_solver.data()->setHessianMatrix(H_sparse);
    osqp_solver.data()->setGradient(g_eigen);
    osqp_solver.data()->setInequalityConstraintsMatrix(A_sparse);
    osqp_solver.data()->setLowerBound(lb);
    osqp_solver.data()->setUpperBound(b_eigen);
    osqp_solver.initSolver();  // Initial factorization
    
    // Initialize qpOASES solver once
    qpOASES::QProblem qpoases_solver(n, m, qpOASES::HST_POSDEF);
    qpOASES::Options qpoases_options;
    qpoases_options.setToMPC();
    qpoases_options.printLevel = qpOASES::PL_NONE;
    qpoases_solver.setOptions(qpoases_options);
    
    
    // Run benchmark over multiple MPC iterations
    std::vector<std::vector<BenchmarkResult>> all_results;
    
    for (int step = 0; step < num_steps; ++step) {
        
        mpc_problem.initial_state = state;
        phase.updateIneqVector(mpc_problem, support_foot_pos);
        
        DenseQP qp = buildDenseQP(mpc_problem);
        
        // Normalize constraints for qpOASES (SUGGESTION 2)
        for (int i = 0; i < qp.m; ++i) {
            double row_norm = 0.0;
            for (int j = 0; j < qp.n; ++j) {
                row_norm += qp.A[i * qp.n + j] * qp.A[i * qp.n + j];
            }
            row_norm = std::sqrt(row_norm);
            
            if (row_norm > 1e-9) {
                for (int j = 0; j < qp.n; ++j) {
                    qp.A[i * qp.n + j] /= row_norm;
                }
                qp.b[i] /= row_norm;
            }
        }
        
        std::vector<BenchmarkResult> step_results;
        
        // Test each GeoQP strategy (warm start)
        for (size_t i = 0; i < geoqp_strategies.size(); ++i) {
            step_results.push_back(solveWithGeoQPWarmStart(qp, *geoqp_solvers[i], geoqp_strategies[i].second));
        }
        
        // Test OSQP (warm start)
        step_results.push_back(solveWithOSQPWarmStart(qp, osqp_solver));
        
        // Test qpOASES (warm start)
        step_results.push_back(solveWithQpOASESWarmStart(qp, qpoases_solver, step == 0));
                
        all_results.push_back(step_results);
        
        // Integrate for next step (using first input from GeoQP-Standard)
        // In practice you'd use the actual solution, but for benchmarking we just need consistency
        state = integrate(state, 0.0, params.sampling_period);
        
        phase.advance();
        if (phase.index() == 0) {
            support_foot_pos = phase.get_next_foot_pos(support_foot_pos);
            phase.advance_stride();
        }
    }
    
    // Aggregate results
    std::cout << "\n";
    std::cout << std::string(120, '=') << "\n";
    std::cout << std::setw(20) << "Solver"
              << std::setw(15) << "Avg Time (ms)"
              << std::setw(12) << "Std Dev"
              << std::setw(12) << "Min (ms)"
              << std::setw(12) << "Max (ms)"
              << std::setw(12) << "Avg Iters"
              << std::setw(12) << "Avg Active"
              << std::setw(10) << "Success"
              << "\n";
    std::cout << std::string(120, '-') << "\n";
    
    int num_solvers = all_results[0].size();
    for (int s = 0; s < num_solvers; ++s) {
        double sum = 0.0, sum_sq = 0.0;
        double min_time = 1e10, max_time = 0.0;
        int total_iters = 0, total_active = 0, successes = 0;
        
        for (const auto& step_results : all_results) {
            double t = step_results[s].solve_time_ms;
            sum += t;
            sum_sq += t * t;
            min_time = std::min(min_time, t);
            max_time = std::max(max_time, t);
            total_iters += step_results[s].iterations;
            total_active += step_results[s].active_constraints;
            if (step_results[s].success) successes++;
        }
        
        double mean = sum / num_steps;
        double variance = (sum_sq / num_steps) - (mean * mean);
        double stddev = std::sqrt(std::max(0.0, variance));
        
        double avg_iters = (total_iters > 0 && all_results[0][s].iterations >= 0) 
                           ? static_cast<double>(total_iters) / num_steps 
                           : -1.0;
        
        std::cout << std::setw(20) << all_results[0][s].solver_name
                  << std::setw(15) << std::fixed << std::setprecision(3) << mean
                  << std::setw(12) << stddev
                  << std::setw(12) << min_time
                  << std::setw(12) << max_time;
        
        if (avg_iters >= 0) {
            std::cout << std::setw(12) << static_cast<int>(avg_iters);
        } else {
            std::cout << std::setw(12) << "N/A";
        }
        
        std::cout << std::setw(12) << (total_active / num_steps)
                  << std::setw(10) << successes << "/" << num_steps
                  << "\n";
    }
    std::cout << std::string(120, '=') << "\n";
    
    // Speedup analysis
    std::cout << "\nSpeedup Analysis (relative to OSQP):\n";
    double osqp_mean = 0.0;
    for (const auto& step_results : all_results) {
        for (const auto& res : step_results) {
            if (res.solver_name == "OSQP") {
                osqp_mean += res.solve_time_ms;
                break;
            }
        }
    }
    osqp_mean /= num_steps;
    
    for (int s = 0; s < num_solvers; ++s) {
        if (all_results[0][s].solver_name == "OSQP") continue;
        
        double sum = 0.0;
        for (const auto& step_results : all_results) {
            sum += step_results[s].solve_time_ms;
        }
        double mean = sum / num_steps;
        double speedup = osqp_mean / mean;
        
        std::cout << "  " << std::setw(20) << std::left << all_results[0][s].solver_name << ": "
                  << std::fixed << std::setprecision(2) << speedup << "x";
        if (speedup > 1.1) {
            std::cout << " ✓ FASTER";
        } else if (speedup < 0.9) {
            std::cout << " ✗ SLOWER";
        } else {
            std::cout << " ~ SAME";
        }
        std::cout << "\n";
    }
    

    return 0;
}
