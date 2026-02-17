/**
 * @brief Benchmark LIPM Walking Controller MPC Problem
 *
 * Mathematical Formulation:
 *   Min 0.5 * U' * H * U + g' * U
 *   s.t.
 *     lb <= U <= ub          (Box constraints on Jerk)
 *     lbA <= C * U <= ubA    (Polytopic constraints on ZMP)
 *
 * This benchmark compares:
 *   1. GeoQP (Dense, Active Set, Geometric-Initialization)
 *   2. OSQP (Sparse interface, ADMM, solving the Dense formulation)
 *   3. qpOASES (Dense, Active Set)
 *
 * Code written with the help of CoPilot
 */

#include <geoqp/geoqp.hpp>
#include <QpSolversEigen/QpSolversEigen.hpp>
#include <qpOASES.hpp>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <geoqp/internal.hpp>
#include <geoqp/timing.hpp>

using Clock = std::chrono::high_resolution_clock;


struct Parameters
{
    double com_height = 0.53;                    // [m]
    double dsp_duration = 0.1;                   // [s] double support
    double foot_size = 0.05;                    // [m] half-foot length
    double gravity = 9.81;                       // [m/s²]
    double init_x = 0.0;                        // [m] initial support foot x
    int N = 15;                                  // MPC horizon
    double dt = 0.1;                             // [s] sampling period
    double ssp_duration = 0.7;                   // [s] single support
    std::vector<double> strides = {-0.18, 0.18}; // [m]

    double omega() const { return std::sqrt(gravity / com_height); }
};


struct DenseQP
{
    int n = 0;             // decision variables (N)
    int m = 0;             // general constraints (2*N)
    std::vector<double> H; // n×n, row-major
    std::vector<double> g; // n

    // Jerk limits: lb <= x <= ub
    std::vector<double> lb;
    std::vector<double> ub;

    // ZMP limits: lbA <= C*x <= ubA
    std::vector<double> C; // m×n, row-major
    std::vector<double> lbA;
    std::vector<double> ubA;

    double const_term = 0.0; // For objective reporting
};

static std::vector<double> mv(const std::vector<double> &A, int rows, int cols,
                              const std::vector<double> &x)
{
    std::vector<double> y(rows, 0.0);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            y[i] += A[i * cols + j] * x[j];
    return y;
}

static std::vector<double> mm(const std::vector<double> &A, int ra, int ca,
                              const std::vector<double> &B, int cb)
{
    std::vector<double> C(ra * cb, 0.0);
    for (int i = 0; i < ra; ++i)
        for (int k = 0; k < ca; ++k)
            for (int j = 0; j < cb; ++j)
                C[i * cb + j] += A[i * ca + k] * B[k * cb + j];
    return C;
}


DenseQP buildQP(const Parameters &p,
                const std::vector<double> &x0,
                const std::vector<std::pair<double, double>> &zmp_bounds,
                const std::vector<double> &x_ref)
{
    const int nx = 3;
    const int N = p.N;
    const double T = p.dt;

    // x_{k+1} = Ad * x_k + Bd * u_k
    std::vector<double> Ad = {
        1.0, T, 0.5 * T * T,
        0.0, 1.0, T,
        0.0, 0.0, 1.0};
    std::vector<double> Bd = {T * T * T / 6.0, T * T / 2.0, T};

    // ZMP Output Matrix: z = Cz * x
    double w2 = p.omega() * p.omega();
    std::vector<double> Cz = {1.0, 0.0, -1.0 / w2};

    // Precompute Powers of Ad: Ap[k] = Ad^k
    std::vector<std::vector<double>> Ap(N + 1, std::vector<double>(nx * nx, 0.0));
    for (int i = 0; i < nx; ++i)
        Ap[0][i * nx + i] = 1.0;
    for (int k = 1; k <= N; ++k)
        Ap[k] = mm(Ad, nx, nx, Ap[k - 1], nx);

    // Phi block (row k, col j) = Ad^(k-j) * Bd  for j <= k
    std::vector<double> Phi(N * nx * N, 0.0);
    for (int k = 0; k < N; ++k)
    {
        for (int j = 0; j <= k; ++j)
        {
            auto col = mv(Ap[k - j], nx, nx, Bd);
            for (int i = 0; i < nx; ++i)
                Phi[(k * nx + i) * N + j] = col[i];
        }
    }

    // Free Response (N*nx)
    std::vector<double> fr(N * nx, 0.0);
    for (int k = 0; k < N; ++k)
    {
        auto blk = mv(Ap[k + 1], nx, nx, x0);
        for (int i = 0; i < nx; ++i)
            fr[k * nx + i] = blk[i];
    }

    // Tracking Error at u=0
    std::vector<double> e(N * nx);
    for (int i = 0; i < N * nx; ++i)
        e[i] = fr[i] - x_ref[i];

    // Cost Weights
    const double q_pos = 1.0;
    const double q_vel = 0.1;
    const double q_acc = 0.0;
    std::vector<double> Qd = {q_pos, q_vel, q_acc};
    const double R = 1e-2;

    // Helper: Qbar * Phi
    std::vector<double> QbarPhi(N * nx * N, 0.0);
    for (int r = 0; r < N * nx; ++r)
        for (int j = 0; j < N; ++j)
            QbarPhi[r * N + j] = Qd[r % nx] * Phi[r * N + j];

    DenseQP qp;
    qp.n = N;

    // Hessian H = Phi' * Qbar * Phi + R * I
    qp.H.assign(N * N, 0.0);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            double val = 0.0;
            for (int r = 0; r < N * nx; ++r)
            {
                val += Phi[r * N + i] * QbarPhi[r * N + j];
            }
            qp.H[i * N + j] = val;
        }
    }
    for (int i = 0; i < N; ++i)
    {
        qp.H[i * N + i] += R;
        qp.H[i * N + i] += 1e-8;
    }
    for (int i = 0; i < N; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            double s = 0.5 * (qp.H[i * N + j] + qp.H[j * N + i]);
            qp.H[i * N + j] = qp.H[j * N + i] = s;
        }
    }

    // Gradient g = Phi' * Qbar * e
    qp.g.assign(N, 0.0);
    for (int j = 0; j < N; ++j)
        for (int r = 0; r < N * nx; ++r)
            qp.g[j] += Phi[r * N + j] * Qd[r % nx] * e[r];

    for (int r = 0; r < N * nx; ++r)
        qp.const_term += 0.5 * Qd[r % nx] * e[r] * e[r];

    // Box Constraints (Input Jerk)
    const double u_max = 50.0; // m/s^3
    qp.lb.assign(N, -u_max);
    qp.ub.assign(N, u_max);

    // ZMP_{k+1} = Cz * (fr_k + Phi_k * U)
    // lb <= Cz*fr + Cz*Phi*U <= ub
    // lb - Cz*fr <= Cz*Phi*U <= ub - Cz*fr
    qp.m = N;
    qp.C.assign(N * N, 0.0);
    qp.lbA.resize(N);
    qp.ubA.resize(N);

    for (int k = 0; k < N; ++k)
    {
        for (int j = 0; j < N; ++j)
        {
            double val = 0.0;
            for (int i = 0; i < nx; ++i)
                val += Cz[i] * Phi[(k * nx + i) * N + j];
            qp.C[k * N + j] = val;
        }

        double z_fr = 0.0;
        for (int i = 0; i < nx; ++i)
            z_fr += Cz[i] * fr[k * nx + i];

        qp.lbA[k] = zmp_bounds[k].first - z_fr;
        qp.ubA[k] = zmp_bounds[k].second - z_fr;
    }

    return qp;
}


struct Phase
{
    int idx = 5;
    int stride_idx = 0;
    int nb_dsp;
    int nb_ssp;
    const Parameters *p;

    explicit Phase(const Parameters &par) : p(&par)
    {
        nb_dsp = (int)std::round(par.dsp_duration / par.dt);
        nb_ssp = (int)std::round(par.ssp_duration / par.dt);
    }

    void advance()
    {
        if (++idx >= nb_dsp + nb_ssp)
            idx = 0;
    }
    void advance_stride() { stride_idx = (stride_idx + 1) % (int)p->strides.size(); }

    double next_foot(double x) const { return x + p->strides[stride_idx]; }
    double last_foot(double x) const
    {
        int ni = (stride_idx + 1) % (int)p->strides.size();
        return next_foot(x) + p->strides[ni];
    }

    void fill(double sup_x,
              std::vector<std::pair<double, double>> &zmp_out,
              std::vector<double> &x_ref_out) const
    {
        const int N = p->N;
        const double h = 0.5 * p->foot_size;
        const double BIG = 100.0;

        int off = idx;
        int d0 = std::max(0, nb_dsp - off);
        off = std::max(0, off - nb_dsp);
        int s0 = std::max(0, nb_ssp - off);
        off = std::max(0, off - nb_ssp);

        int rem = N - d0 - s0;
        int d1 = std::min(nb_dsp, rem);
        rem = std::max(0, rem - d1);
        int s1 = std::min(nb_ssp, rem);
        rem = std::max(0, rem - s1);
        int d2 = std::min(nb_dsp, rem);
        rem = std::max(0, rem - d2);
        int s2 = std::min(nb_ssp, rem);

        double nxt = next_foot(sup_x);
        double lst = last_foot(sup_x);

        zmp_out.clear();
        auto push = [&](int n, double lo, double hi)
        { for (int i = 0; i < n; ++i) zmp_out.push_back({lo,hi}); };

        push(d0, -BIG, BIG);
        push(s0, sup_x - h, sup_x + h);
        push(d1, -BIG, BIG);
        push(s1, nxt - h, nxt + h);
        push(d2, -BIG, BIG);
        push(s2, lst - h, lst + h);
        while ((int)zmp_out.size() < N)
            zmp_out.push_back({-BIG, BIG});

        double goal = (d2 > 0) ? lst : nxt;
        x_ref_out.assign(N * 3, 0.0);
        for (int k = 0; k < N; ++k)
            x_ref_out[k * 3] = goal;
    }
};

static std::vector<double> integrate(const std::vector<double> &x, double jerk, double T)
{
    return {
        x[0] + T * x[1] + 0.5 * T * T * x[2] + T * T * T / 6.0 * jerk,
        x[1] + T * x[2] + 0.5 * T * T * jerk,
        x[2] + T * jerk};
}


struct Result
{
    std::string name;
    bool success = false;
    double time_ms = 0.0;
    double objective = 0.0;
    int iterations = -1;
    std::vector<double> solution;
};

static Result runGeoQP(const DenseQP &qp, geoqp::FaceHoppingStrategy strat, const std::string &name)
{
    geoqp::Problem prob;
    prob.n = qp.n;
    prob.m = qp.m;
    prob.Q = qp.H;
    prob.g = qp.g;
    prob.A = qp.C;

    prob.lb = qp.lb;
    prob.ub = qp.ub;

    // C x <= ubA
    // -C x <= -lbA
    int m_expanded = 2 * qp.m;
    prob.m = m_expanded;
    std::vector<double> A_exp(m_expanded * qp.n);
    std::vector<double> b_exp(m_expanded);

    for (int i = 0; i < qp.m; ++i)
    {
        for (int j = 0; j < qp.n; ++j)
        {
            A_exp[(2 * i) * qp.n + j] = qp.C[i * qp.n + j];      // +C
            A_exp[(2 * i + 1) * qp.n + j] = -qp.C[i * qp.n + j]; // -C
        }
        b_exp[2 * i] = qp.ubA[i];
        b_exp[2 * i + 1] = -qp.lbA[i];
    }
    prob.A = A_exp;
    prob.b = b_exp;

    geoqp::Config cfg;
    cfg.face_hopping_strategy = strat;
    // keep caller control of cfg.use_caching by passing via env var
    const char *env = std::getenv("GEOQP_USE_CACHING");
    cfg.use_caching = (env && std::string(env) == "1");
    cfg.warm_start = false;
    cfg.tolerance = 1e-7;

    geoqp::Solver solver(cfg);
    Result res;
    res.name = name;

    auto t0 = Clock::now();
    auto sol = solver.solve(prob);
    res.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    res.success = (sol.status == geoqp::Status::SUCCESS);
    res.objective = sol.objective_value + qp.const_term;
    res.iterations = sol.phase2_iterations;
    res.solution = sol.x;
    return res;
}

static Result runOSQP(const DenseQP &qp)
{
    Result res;
    res.name = "OSQP";
    int n = qp.n;
    int m = qp.m; // General constraints rows

    // Copy to Eigen
    Eigen::MatrixXd H_e(n, n);
    Eigen::VectorXd g_e(n);
    for (int i = 0; i < n; ++i)
    {
        g_e(i) = qp.g[i];
        for (int j = 0; j < n; ++j)
            H_e(i, j) = qp.H[i * n + j];
    }

    // OSQP Form: lb <= [I; C] * x <= ub
    int m_total = n + m;
    Eigen::VectorXd lb_all(m_total);
    Eigen::VectorXd ub_all(m_total);
    Eigen::MatrixXd A_all(m_total, n);

    // Box Constraints (Identity)
    A_all.topRows(n).setIdentity();
    for (int i = 0; i < n; ++i)
    {
        lb_all(i) = qp.lb[i];
        ub_all(i) = qp.ub[i];
    }

    // General Constraints (C)
    for (int i = 0; i < m; ++i)
    {
        lb_all(n + i) = qp.lbA[i];
        ub_all(n + i) = qp.ubA[i];
        for (int j = 0; j < n; ++j)
            A_all(n + i, j) = qp.C[i * n + j];
    }

    // Extract triangular view to dense first, then convert to sparse.
    Eigen::MatrixXd H_upper = H_e.triangularView<Eigen::Upper>();
    Eigen::SparseMatrix<double> H_s = H_upper.sparseView();
    Eigen::SparseMatrix<double> A_s = A_all.sparseView();

    QpSolversEigen::Solver solver;
    if (!solver.instantiateSolver("osqp"))
        return res;

    solver.setBooleanParameter("verbose", false);
    solver.data()->setNumberOfVariables(n);
    solver.data()->setNumberOfInequalityConstraints(m_total);
    solver.data()->setHessianMatrix(H_s);
    solver.data()->setGradient(g_e);
    solver.data()->setInequalityConstraintsMatrix(A_s);
    solver.data()->setLowerBound(lb_all);
    solver.data()->setUpperBound(ub_all);

    auto t0 = Clock::now();
    if (solver.initSolver())
    {
        if (solver.solveProblem() == QpSolversEigen::ErrorExitFlag::NoError)
        {
            res.success = true;
            res.objective = solver.getObjValue() + qp.const_term;
            auto sv = solver.getSolution();
            res.solution.assign(sv.data(), sv.data() + n);
        }
    }
    res.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return res;
}

static Result runQpOASES(const DenseQP &qp)
{
    int n = qp.n;
    int m = qp.m;


    std::vector<qpOASES::real_t> H_d(qp.H.begin(), qp.H.end());
    std::vector<qpOASES::real_t> g_d(qp.g.begin(), qp.g.end());
    std::vector<qpOASES::real_t> C_d(qp.C.begin(), qp.C.end());

    std::vector<qpOASES::real_t> lb_d(qp.lb.begin(), qp.lb.end());
    std::vector<qpOASES::real_t> ub_d(qp.ub.begin(), qp.ub.end());

    std::vector<qpOASES::real_t> lbA_d(qp.lbA.begin(), qp.lbA.end());
    std::vector<qpOASES::real_t> ubA_d(qp.ubA.begin(), qp.ubA.end());

    qpOASES::QProblem prob(n, m, qpOASES::HST_POSDEF);
    qpOASES::Options opts;
    opts.setToMPC();
    opts.printLevel = qpOASES::PL_NONE;
    prob.setOptions(opts);

    int nWSR = 1000;
    Result res;
    res.name = "qpOASES";

    auto t0 = Clock::now();
    auto ret = prob.init(H_d.data(), g_d.data(), C_d.data(),
                         lb_d.data(), ub_d.data(),
                         lbA_d.data(), ubA_d.data(), nWSR);
    res.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    res.success = (ret == qpOASES::SUCCESSFUL_RETURN);
    res.iterations = nWSR;

    if (res.success)
    {
        res.objective = prob.getObjVal() + qp.const_term;
        std::vector<qpOASES::real_t> sol(n);
        prob.getPrimalSolution(sol.data());
        res.solution.assign(sol.begin(), sol.end());
    }

    return res;
}


int main()
{
    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════╗\n"
              << "║  LIPM Walking Controller MPC Benchmark (v3 Corrected)        ║\n"
              << "║  GeoQP (4 strategies) vs OSQP vs qpOASES                     ║\n"
              << "║  Cold start with Box Constraints Separated                   ║\n"
              << "╚══════════════════════════════════════════════════════════════╝\n\n";

    Parameters p;

    double w = p.omega();
    std::vector<double> state = {
        0.0,
        0.5 * w * p.init_x,
        -w * w * p.init_x};
    double sup_x = p.init_x;
    Phase phase(p);

    const int num_steps = 10;
    std::cout << "Parameters:\n"
              << "  Horizon N       : " << p.N << "\n"
              << "  Sampling        : " << p.dt << " s\n"
              << "  Constraints     : " << p.N << " (2-sided ZMP) + " << p.N << " (Box Jerk)\n\n";

    // Optional: run one verbose solver step for debugging face hops when env var is set
    const char *vstep = std::getenv("GEOQP_VERBOSE_STEP");
    if (vstep && std::string(vstep) == "1")
    {
        std::vector<std::pair<double, double>> zmp_bounds;
        std::vector<double> x_ref;
        phase.fill(sup_x, zmp_bounds, x_ref);
        DenseQP qp = buildQP(p, state, zmp_bounds, x_ref);

        // Expand A and b as in runGeoQP
        int m_expanded = 2 * qp.m;
        std::vector<double> A_exp(m_expanded * qp.n);
        std::vector<double> b_exp(m_expanded);
        for (int i = 0; i < qp.m; ++i)
        {
            for (int j = 0; j < qp.n; ++j)
            {
                A_exp[(2 * i) * qp.n + j] = qp.C[i * qp.n + j];
                A_exp[(2 * i + 1) * qp.n + j] = -qp.C[i * qp.n + j];
            }
            b_exp[2 * i] = qp.ubA[i];
            b_exp[2 * i + 1] = -qp.lbA[i];
        }

        geoqp::Problem prob;
        prob.n = qp.n;
        prob.m = m_expanded;
        prob.Q = qp.H;
        prob.g = qp.g;
        prob.A = A_exp;
        prob.b = b_exp;
        prob.lb = qp.lb;
        prob.ub = qp.ub;

        geoqp::Config cfg;
        cfg.face_hopping_strategy = geoqp::FaceHoppingStrategy::STANDARD;
        cfg.use_caching = false;
        cfg.warm_start = false;
        cfg.tolerance = 1e-7;
        cfg.verbosity = 3;        // enable detailed face-hopper logs
        cfg.timing_verbosity = 1; // print per-hop timing summary

        geoqp::Solver solver(cfg);
        std::cout << "\n=== Verbose single-step solve (STANDARD) ===\n";
        auto t0 = Clock::now();
        auto sol = solver.solve(prob);
        double tms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        std::cout << "Verbose solve time: " << tms << " ms, phase2 iters: " << sol.phase2_iterations << "\n";
        return 0;
    }

    std::vector<std::pair<geoqp::FaceHoppingStrategy, std::string>> geoqp_strats = {
        {geoqp::FaceHoppingStrategy::STANDARD, "GeoQP-Standard"},
        {geoqp::FaceHoppingStrategy::BATCH, "GeoQP-Batch"},
        {geoqp::FaceHoppingStrategy::SENSITIVITY, "GeoQP-Sensitivity"},
        {geoqp::FaceHoppingStrategy::NEIGHBORS, "GeoQP-Neighbors"}};

    int n_geo = (int)geoqp_strats.size();
    int idx_osqp = n_geo;
    int idx_qpoases = n_geo + 1;
    int total_solvers = n_geo + 2;

    auto run_pass = [&](bool use_caching, std::vector<std::vector<Result>> &all_results) {
    if (use_caching) setenv("GEOQP_USE_CACHING", "1", 1);
    else unsetenv("GEOQP_USE_CACHING");

    // reset Schur instrumentation counters
    geoqp::internal::schur_stats_reset();
        // Use local copies so we don't mutate the outer simulation state
        auto state_local = state;
        auto phase_local = phase;
        double sup_x_local = sup_x;

        all_results.assign(num_steps, std::vector<Result>());

        // Build a template Q/A_exp using the initial state so we can setup solvers once
        std::vector<std::pair<double, double>> zmp_bounds0;
        std::vector<double> x_ref0;
        phase_local.fill(sup_x_local, zmp_bounds0, x_ref0);
        DenseQP qp0 = buildQP(p, state_local, zmp_bounds0, x_ref0);

        int m_expanded = 2 * qp0.m;
        std::vector<double> A_exp_template(m_expanded * qp0.n);
        for (int i = 0; i < qp0.m; ++i)
        {
            for (int j = 0; j < qp0.n; ++j)
            {
                A_exp_template[(2 * i) * qp0.n + j] = qp0.C[i * qp0.n + j];
                A_exp_template[(2 * i + 1) * qp0.n + j] = -qp0.C[i * qp0.n + j];
            }
        }

        // Construct one Solver per GeoQP strategy, setup once with Q and A_exp_template, and reuse across steps
        std::vector<std::unique_ptr<geoqp::Solver>> geoqp_solvers;
        geoqp_solvers.reserve(geoqp_strats.size());
        for (auto &pstr : geoqp_strats) {
            geoqp::Config cfg;
            cfg.face_hopping_strategy = pstr.first;
            cfg.use_caching = use_caching;
            // Enable warm starting across steps so the initializer is not run every time
            cfg.warm_start = false;
            cfg.tolerance = 1e-7;
            auto solver = std::make_unique<geoqp::Solver>(cfg);
            // setup with template Q and expanded A (A_exp_template)
            solver->setup(qp0.n, m_expanded, qp0.H.data(), A_exp_template.data());
            geoqp_solvers.emplace_back(std::move(solver));
        }

        for (int step = 0; step < num_steps; ++step)
        {
            std::vector<std::pair<double, double>> zmp_bounds;
            std::vector<double> x_ref;
            phase_local.fill(sup_x_local, zmp_bounds, x_ref);

            DenseQP qp = buildQP(p, state_local, zmp_bounds, x_ref);

            auto &step_res = all_results[step];

            // Run GeoQP solvers using the pre-constructed solver instances and hotstart API
            for (size_t si = 0; si < geoqp_strats.size(); ++si)
            {
                const auto &name = geoqp_strats[si].second;
                auto &solver = *geoqp_solvers[si];

                // build expanded b for this step
                std::vector<double> b_exp(m_expanded);
                for (int i = 0; i < qp.m; ++i)
                {
                    b_exp[2 * i] = qp.ubA[i];
                    b_exp[2 * i + 1] = -qp.lbA[i];
                }

                auto t0 = Clock::now();
                auto sol = solver.solve_hotstart(qp.g.data(), b_exp.data());
                Result r;
                r.name = name;
                r.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                r.success = (sol.status == geoqp::Status::SUCCESS);
                r.objective = sol.objective_value + qp.const_term;
                r.iterations = sol.phase2_iterations;
                r.solution = sol.x;
                step_res.push_back(r);
            }

            step_res.push_back(runOSQP(qp));

            step_res.push_back(runQpOASES(qp));

            double jerk = 0.0;
            if (step_res[idx_qpoases].success && !step_res[idx_qpoases].solution.empty())
            {
                jerk = step_res[idx_qpoases].solution[0];
            }
            else
            {
                std::cerr << "Warning: Step " << step << " failed to find solution.\n";
            }

            state_local = integrate(state_local, jerk, p.dt);
            phase_local.advance();
            if (phase_local.idx == 0)
            {
                sup_x_local = phase_local.next_foot(sup_x_local);
                phase_local.advance_stride();
            }
        }
    };


    std::cout << "\n"
              << std::string(120, '=') << "\n";
    std::cout << std::left << std::setw(20) << "Solver"
              << std::right << std::setw(12) << "Avg(ms)"
              << std::setw(10) << "Std"
              << std::setw(10) << "Min"
              << std::setw(10) << "Max"
              << std::setw(10) << "Iters"
              << std::setw(10) << "OK"
              << std::setw(14) << "AvgObj"
              << std::setw(16) << "Diff(vs qpOA)"
              << "\n"
              << std::string(120, '-') << "\n";

    // Run two passes: caching disabled then enabled

    // Run two passes: caching disabled then enabled
    std::vector<std::vector<Result>> all_results_no_cache, all_results_cache;
    run_pass(false, all_results_no_cache);
    run_pass(true, all_results_cache);

    auto print_table = [&](const std::vector<std::vector<Result>> &all_results, const std::string &title) {
        std::cout << "\n" << title << "\n";
        std::vector<double> ref_obj(num_steps, 0.0);
        int idx_qpoases_local = idx_qpoases;
        for (int i = 0; i < num_steps; ++i)
            if (all_results[i][idx_qpoases_local].success)
                ref_obj[i] = all_results[i][idx_qpoases_local].objective;

    for (int s = 0; s < total_solvers; ++s)
    {
    double sum_t = 0, sum_sq = 0, min_t = 1e9, max_t = 0;
        double sum_obj = 0, sum_diff = 0;
        int sum_it = 0, count_ok = 0;

            for (int i = 0; i < num_steps; ++i)
        {
            const auto &r = all_results[i][s];
            double t = r.time_ms;
            sum_t += t;
            sum_sq += t * t;
            if (t < min_t)
                min_t = t;
            if (t > max_t)
                max_t = t;

            if (r.iterations >= 0)
                sum_it += r.iterations;

            if (r.success)
            {
                count_ok++;
                sum_obj += r.objective;
                sum_diff += std::abs(r.objective - ref_obj[i]);
            }
        }

        double mean = sum_t / num_steps;
        double std = std::sqrt(std::max(0.0, sum_sq / num_steps - mean * mean));
        double avg_it = (sum_it > 0) ? (double)sum_it / num_steps : -1.0;
        double avg_obj = (count_ok > 0) ? sum_obj / count_ok : 0.0;
        double avg_diff = (count_ok > 0) ? sum_diff / count_ok : 0.0;

        std::cout << std::left << std::setw(20) << all_results[0][s].name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << mean
                  << std::setw(10) << std
                  << std::setw(10) << min_t
                  << std::setw(10) << max_t;

        if (avg_it >= 0)
            std::cout << std::setw(10) << (int)avg_it;
        else
            std::cout << std::setw(10) << "N/A";

        std::cout << std::setw(7) << count_ok << "/" << num_steps
                  << std::setw(14) << avg_obj
                  << std::setw(16) << std::scientific << std::setprecision(2) << avg_diff
                  << "\n";
    }
        std::cout << std::string(120, '=') << "\n";
    };

    print_table(all_results_no_cache, "Results (caching = OFF)");
    // Schur stats printing removed to reduce console noise.

    print_table(all_results_cache, "Results (caching = ON)");
    // Schur stats printing removed to reduce console noise.

    // Lightweight counters and detailed diagnostics suppressed by default.

    return 0;
}