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
                const std::vector<double> &x_ref,
                bool enforce_jerk = true)
{
    // Rebuild condensed QP using Sx/Su construction (as in examples/mpc_example.cpp)
    const int nx = 3;
    const int N = p.N;
    const double T = p.dt;

    // Continuous discrete dynamics as in previous implementation
    std::vector<double> Ad = {
        1.0, T, 0.5 * T * T,
        0.0, 1.0, T,
        0.0, 0.0, 1.0};
    std::vector<double> Bd = {T * T * T / 6.0, T * T / 2.0, T};

    // ZMP Output Matrix: z = Cz * x
    double w2 = p.omega() * p.omega();
    std::vector<double> Cz = {1.0, 0.0, -1.0 / w2};

    // Build Sx (NX*N x NX) and Su (NX*N x N) where control is scalar per step
    std::vector<double> Sx(nx * N * nx, 0.0);
    std::vector<double> Su(nx * N * N, 0.0);

    // Precompute powers of Ad: Apowers[k] = Ad^k
    std::vector<std::vector<double>> Apowers(N + 1, std::vector<double>(nx * nx, 0.0));
    for (int i = 0; i < nx; ++i) Apowers[0][i * nx + i] = 1.0;
    for (int k = 1; k <= N; ++k) Apowers[k] = mm(Apowers[k - 1], nx, nx, Ad, nx);

    // Fill Sx and Su analogous to mpc_example.cpp
    for (int k = 0; k < N; ++k)
    {
        // Sx block = Ad^{k+1}
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < nx; ++j)
                Sx[(k * nx + i) * nx + j] = Apowers[k + 1][i * nx + j];

        // Su block: columns j<=k contain A^{k-j} * Bd
        for (int j = 0; j <= k; ++j)
        {
            auto AkjBd = mv(Apowers[k - j], nx, nx, Bd);
            for (int i = 0; i < nx; ++i)
                Su[(k * nx + i) * N + j] = AkjBd[i];
        }
    }

    // Free response and e = Sx * x0 - x_ref
    std::vector<double> fr(nx * N, 0.0);
    for (int k = 0; k < N; ++k)
        for (int i = 0; i < nx; ++i)
            fr[k * nx + i] = 0.0;
    // compute fr = Sx * x0
    for (int row = 0; row < nx * N; ++row)
    {
        double sum = 0.0;
        for (int j = 0; j < nx; ++j)
            sum += Sx[row * nx + j] * x0[j];
        fr[row] = sum;
    }

    std::vector<double> e(nx * N, 0.0);
    for (int i = 0; i < nx * N; ++i) e[i] = fr[i] - x_ref[i];

    // Cost weights (keep previous choices)
    std::vector<double> Qd = {1.0, 1.0, 1.0};
    const double R = 1e-2;

    // Compute Qbar * Su by scaling rows
    std::vector<double> QSu(nx * N * N, 0.0);
    for (int r = 0; r < N * nx; ++r)
    {
        double q = Qd[r % nx];
        for (int j = 0; j < N; ++j)
            QSu[r * N + j] = q * Su[r * N + j];
    }

    DenseQP qp;
    qp.n = N;

    // H = Su^T * Qbar * Su + R * I
    qp.H.assign(N * N, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            double s = 0.0;
            for (int r = 0; r < N * nx; ++r)
                s += Su[r * N + i] * QSu[r * N + j];
            qp.H[i * N + j] = s;
        }
    for (int i = 0; i < N; ++i)
    {
        qp.H[i * N + i] += R;
        qp.H[i * N + i] += 1e-6;
    }
    // symmetrize
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
        {
            double s = 0.5 * (qp.H[i * N + j] + qp.H[j * N + i]);
            qp.H[i * N + j] = qp.H[j * N + i] = s;
        }

    // g = Su^T * Qbar * e
    qp.g.assign(N, 0.0);
    // compute Qbar*e
    std::vector<double> Qe(N * nx, 0.0);
    for (int r = 0; r < N * nx; ++r) Qe[r] = Qd[r % nx] * e[r];
    for (int i = 0; i < N; ++i)
    {
        double sum = 0.0;
        for (int r = 0; r < N * nx; ++r)
            sum += Su[r * N + i] * Qe[r];
        qp.g[i] = sum;
    }

    // constant term
    for (int r = 0; r < N * nx; ++r)
        qp.const_term += 0.5 * Qd[r % nx] * e[r] * e[r];

    // Box Constraints (Input Jerk)
    if (enforce_jerk)
    {
        const double u_max = 50.0; // m/s^3
        qp.lb.assign(N, -u_max);
        qp.ub.assign(N, u_max);
    }
    else
    {
        qp.lb.clear(); qp.ub.clear();
    }

    // ZMP constraints: C_k = Cz * Su_block_k  -> qp.C is m x n where m=N, n=N
    qp.m = N;
    qp.C.assign(N * N, 0.0);
    qp.lbA.resize(N);
    qp.ubA.resize(N);

    for (int k = 0; k < N; ++k)
    {
        // row k of C: Cz * Su rows (k*nx .. k*nx+nx-1)
        for (int j = 0; j < N; ++j)
        {
            double val = 0.0;
            for (int i = 0; i < nx; ++i)
                val += Cz[i] * Su[(k * nx + i) * N + j];
            qp.C[k * N + j] = val;
        }

        double z_fr = 0.0;
        for (int i = 0; i < nx; ++i)
            z_fr += Cz[i] * fr[k * nx + i];

        qp.lbA[k] = zmp_bounds[k].first - z_fr;
        qp.ubA[k] = zmp_bounds[k].second - z_fr;
        if (qp.lbA[k] > qp.ubA[k])
        {
            std::cerr << "[WARN] buildQP: lbA[" << k << "] > ubA[" << k << "] (" << qp.lbA[k] << " > " << qp.ubA[k] << "). Swapping.\n";
            std::swap(qp.lbA[k], qp.ubA[k]);
        }
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
    int facet_hops = -1;
    int avg_pivot = -1;
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

    // Only assign variable bounds if provided (avoid huge artificial bounds)
    if (!qp.lb.empty() && !qp.ub.empty()) {
        prob.lb = qp.lb;
        prob.ub = qp.ub;
    } else {
        prob.lb.clear(); prob.ub.clear();
    }

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
    cfg.constraint_tolerance = 1e-6; // relax constraint feasibility tolerance for robustness

    // Validate problem before solving to catch malformed inputs
    if (!geoqp::validate_problem(prob))
    {
        std::cerr << "[ERROR] runGeoQP: invalid problem for solver (n=" << prob.n << " m=" << prob.m << ")\n";
        Result fail; fail.name = name; fail.success = false; return fail;
    }

    geoqp::Solver solver(cfg);
    Result res;
    res.name = name;

    auto t0 = Clock::now();
    auto sol = solver.solve(prob);
    res.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    res.success = (sol.status == geoqp::Status::SUCCESS);
    res.objective = sol.objective_value + qp.const_term;
    // record total iterations (phase1+phase2) rather than only phase2
    res.iterations = sol.iterations;
    res.facet_hops = sol.phase2_facet_hops;
    // report phase2 pivot count as AvgPivot for compatibility with reference plots
    res.avg_pivot = sol.phase2_iterations;
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

    // Determine whether box (identity) constraints are present
    bool has_box = (int)qp.lb.size() == n && (int)qp.ub.size() == n;

    // OSQP Form: lb <= [I; C] * x <= ub  (if box present)
    int m_total = (has_box ? n + m : m);
    Eigen::VectorXd lb_all(m_total);
    Eigen::VectorXd ub_all(m_total);
    Eigen::MatrixXd A_all(m_total, n);

    // Box Constraints (Identity) if present
    if (has_box)
    {
        A_all.topRows(n).setIdentity();
        for (int i = 0; i < n; ++i)
        {
            lb_all(i) = qp.lb[i];
            ub_all(i) = qp.ub[i];
        }
    }

    // General Constraints (C)
    for (int i = 0; i < m; ++i)
    {
        int row = (has_box ? n + i : i);
        lb_all(row) = qp.lbA[i];
        ub_all(row) = qp.ubA[i];
        for (int j = 0; j < n; ++j)
            A_all(row, j) = qp.C[i * n + j];
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
    // QpSolversEigen/OSQP wrapper does not expose phase2-like hops; leave as -1
    res.facet_hops = -1;
    return res;
}

static Result runQpOASES(const DenseQP &qp)
{
    int n = qp.n;
    int m = qp.m;


    std::vector<qpOASES::real_t> H_d(qp.H.begin(), qp.H.end());
    std::vector<qpOASES::real_t> g_d(qp.g.begin(), qp.g.end());
    std::vector<qpOASES::real_t> C_d(qp.C.begin(), qp.C.end());

    std::vector<qpOASES::real_t> lb_d;
    std::vector<qpOASES::real_t> ub_d;
    if ((int)qp.lb.size() == n && (int)qp.ub.size() == n) {
        lb_d.assign(qp.lb.begin(), qp.lb.end());
        ub_d.assign(qp.ub.begin(), qp.ub.end());
    } else {
        // No box bounds provided (unbounded variables). Provide large finite bounds
        // qpOASES expects arrays of length n; choose large but finite limits.
        const qpOASES::real_t BIG = static_cast<qpOASES::real_t>(1e20);
        lb_d.assign(n, -BIG);
        ub_d.assign(n, BIG);
    }

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
    res.facet_hops = 0;

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

    // We will run test cases varying the MPC horizon from 2..120 (inclusive), step 2.
    const char *disable_jerk_env = std::getenv("GEOQP_DISABLE_JERK_BOUNDS");
    bool enforce_jerk = !(disable_jerk_env && std::string(disable_jerk_env) == "1");

    std::cout << "Parameters:\n"
              << "  Base Sampling   : " << p.dt << " s\n";
    if (enforce_jerk)
        std::cout << "  Jerk bounds     : ENABLED\n\n";
    else
        std::cout << "  Jerk bounds     : DISABLED (only ZMP constraints active)\n\n";

    // Echo debug env var for Phase II diagnostics
    const char *dbg_phase2_env = std::getenv("GEOQP_DBG_PHASE2");
    if (dbg_phase2_env)
        std::cout << "[INFO] GEOQP_DBG_PHASE2=" << dbg_phase2_env << "\n";
    else
        std::cout << "[INFO] GEOQP_DBG_PHASE2 not set\n";

    // Optional: run one verbose solver step for debugging face hops when env var is set
    const char *vstep = std::getenv("GEOQP_VERBOSE_STEP");
    if (vstep && std::string(vstep) == "1")
    {
        std::vector<std::pair<double, double>> zmp_bounds;
        std::vector<double> x_ref;
        phase.fill(sup_x, zmp_bounds, x_ref);
        DenseQP qp = buildQP(p, state, zmp_bounds, x_ref, enforce_jerk);

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

    // We'll test horizons h = 5,10,...,120
    std::vector<int> horizons;
    for (int h = 5; h <= 120; h += 5)
        horizons.push_back(h);

    // reset Schur instrumentation counters once
    geoqp::internal::schur_stats_reset();

    // Print a compact header for CASE-style parsing by the plotting script
    std::cout << "\nResults (horizons)\n";
    std::cout << "CASE format: n=<horizon> QP_n=<n> QP_m=<m> Geo_us=<ms> qpO_us=<ms> OSQP_us=<ms> Geo_it=<it> qpO_it=<it> OSQP_it=<it> Geo_hops=<hops> AvgPivot=<pivots> AvgObjGeo=<obj> Diff_vs_qpO=<diff> OK_geo=<ok> OK_qpO=<ok> OK_osqp=<ok>\n";

    for (int h : horizons)
    {
        // use local Parameters copy with updated horizon
        Parameters pc = p;
        pc.N = h;

        // create a Phase object bound to the local Parameters so zmp_bounds size matches pc.N
        Phase phase_local(pc);

        // prepare initial state and zmp/xref for this horizon
        std::vector<std::pair<double, double>> zmp_bounds;
        std::vector<double> x_ref;
        phase_local.fill(sup_x, zmp_bounds, x_ref);

        DenseQP qp = buildQP(pc, state, zmp_bounds, x_ref, enforce_jerk);

        // optional per-case debug via env var GEOQP_DEBUG_CASE=<h>
        const char *dbg_case_env = std::getenv("GEOQP_DEBUG_CASE");
        bool dbg_case = (dbg_case_env && std::atoi(dbg_case_env) == h);

        // Solve once per solver for this case (caching always OFF)
        // GeoQP: pick STANDARD strategy as representative
        if (dbg_case) {
            std::cerr << "[DEBUG] CASE h=" << h << " QP_n=" << qp.n << " QP_m=" << qp.m << "\n";
            std::cerr << "[DEBUG] H diag (first 8): ";
            for (int i = 0; i < std::min(8, qp.n); ++i) std::cerr << qp.H[i * qp.n + i] << " ";
            std::cerr << "\n[DEBUG] lb (first 8): ";
            for (int i = 0; i < std::min(8, (int)qp.lb.size()); ++i) std::cerr << qp.lb[i] << " ";
            std::cerr << "\n[DEBUG] ub (first 8): ";
            for (int i = 0; i < std::min(8, (int)qp.ub.size()); ++i) std::cerr << qp.ub[i] << " ";
            std::cerr << "\n[DEBUG] lbA (first 8): ";
            for (int i = 0; i < std::min(8, (int)qp.lbA.size()); ++i) std::cerr << qp.lbA[i] << " ";
            std::cerr << "\n[DEBUG] ubA (first 8): ";
            for (int i = 0; i < std::min(8, (int)qp.ubA.size()); ++i) std::cerr << qp.ubA[i] << " ";
            std::cerr << "\n[DEBUG] C row0 (first 12): ";
            for (int j = 0; j < std::min(12, qp.n); ++j) std::cerr << qp.C[0 * qp.n + j] << " ";
            std::cerr << "\n";
        }
        auto geo_res = runGeoQP(qp, geoqp::FaceHoppingStrategy::STANDARD, "GeoQP");
        auto osqp_res = runOSQP(qp);
        auto qpo_res = runQpOASES(qp);

        // compute diffs vs qpOASES objective when available
        double diff_geo = 0.0;
        if (qpo_res.success && geo_res.success)
            diff_geo = std::abs(geo_res.objective - qpo_res.objective);

        int ok_geo = geo_res.success ? 1 : 0;
        int ok_qpo = qpo_res.success ? 1 : 0;
        int ok_osqp = osqp_res.success ? 1 : 0;

        // Print a single CASE line per horizon with key=value tokens for easy parsing
        std::cout << "CASE "
                  << "n=" << h << " "
                  << "QP_n=" << qp.n << " "
                  << "QP_m=" << qp.m << " "
                  << "Geo_us=" << geo_res.time_ms << " "
                  << "qpO_us=" << qpo_res.time_ms << " "
                  << "OSQP_us=" << osqp_res.time_ms << " "
                  << "Geo_it=" << geo_res.iterations << " "
                  << "qpO_it=" << qpo_res.iterations << " "
                  << "OSQP_it=" << -1 << " "
                  << "Geo_hops=" << geo_res.facet_hops << " "
                  << "AvgPivot=" << geo_res.avg_pivot << " "
                  << "AvgObjGeo=" << geo_res.objective << " "
                  << "Diff_vs_qpO=" << diff_geo << " "
                  << "OK_geo=" << ok_geo << " "
                  << "OK_qpO=" << ok_qpo << " "
                  << "OK_osqp=" << ok_osqp << "\n";
    }


    std::cout << "\nFinished horizon cases.\n";
    return 0;
}