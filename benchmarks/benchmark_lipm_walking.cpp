/**
 * @brief Benchmark: DCM (Divergent Component of Motion) MPC — 2D formulation
 *
 * Dynamics (per axis, decoupled):
 *   xi_{k+1} = alpha * xi_k + (1 - alpha) * u_k,   alpha = exp(omega * T)
 * where xi_k in R^2 is the 2D DCM and u_k in R^2 is the 2D ZMP (the control input).
 *
 * Cost (consistent with IIT DCMModelPredictiveController):
 *   sum_{k=0}^{N-1} ||xi_k - xi_ref_k||^2_Q  +  sum_{k=0}^{N-1} ||u_k - u_{k-1}||^2_R
 * The input-increment penalty uses the Theta matrix: Theta * U = [u_0 - u_prev; u_1 - u_0; ...]
 * After condensing xi via the dynamics the Hessian H is dense (2N x 2N).
 *
 * Constraints:
 *   lb_k <= u_k <= ub_k  (per-step 2D box on ZMP position, one box per support phase)
 * This is an axis-aligned box on U in R^{2N}: lb <= U <= ub.  No general A matrix is
 * required — the feasible set is exactly the box P = T(B)+c treated in the paper.
 *
 * Solvers compared:
 *   1. GeoQP  (Dense active-set, geometric initialisation)
 *   2. qpOASES (Dense active-set, MPC options)
 *   3. OSQP   (Sparse ADMM, dense problem passed via QpSolversEigen)
 *
 * Code written with the help of GitHub Copilot.
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
#include <cstdlib>

using Clock = std::chrono::high_resolution_clock;


// ─────────────────────────────────────────────────────────────────────────────
//  Parameters
// ─────────────────────────────────────────────────────────────────────────────
struct Parameters
{
    double com_height   = 0.53;   // [m]  CoM height (LIPM assumption)
    double gravity      = 9.81;   // [m/s²]
    double dt           = 0.01;   // [s]  sampling period T (100 Hz — keeps alpha close to 1)
    int    N            = 15;     // MPC horizon (number of steps)
    double foot_half_x  = 0.10;   // [m]  half-length of each foot support rectangle (x)
    double foot_half_y  = 0.05;   // [m]  half-width  of each foot support rectangle (y)
    double stride_x     = 0.18;   // [m]  forward step length per stride
    double stride_y     = 0.10;   // [m]  lateral   step width  per stride
    double dsp_duration = 0.05;   // [s]  double-support phase  (5 steps at 100 Hz)
    double ssp_duration = 0.35;   // [s]  single-support phase  (35 steps at 100 Hz)

    // Weight scales (can be overridden via env vars MPC_Q_SCALE / MPC_R_SCALE)
    double q_scale = 1.0;
    double r_scale = 1.0;

    double omega() const { return std::sqrt(gravity / com_height); }
    // alpha = exp(omega*T) ≈ 1.044 at 100 Hz.  alpha^{2N} ≈ 3e4 for N=120 → well-conditioned.
    // (At 10 Hz, dt=0.1, alpha≈1.54, alpha^{2*45}≈6e16 > 1/eps → Cholesky failure.)
    double alpha() const { return std::exp(omega() * dt); }
};

// ─────────────────────────────────────────────────────────────────────────────
//  QP data structure
//  Decision variables U = [u_0^x, u_0^y, u_1^x, u_1^y, ..., u_{N-1}^x, u_{N-1}^y]
//  Interleaved x/y so that per-step box on u_k maps to 2 consecutive lb/ub entries.
//  n = 2*N.  No general constraints (m=0): all constraints are box lb <= U <= ub.
// ─────────────────────────────────────────────────────────────────────────────
struct DenseQP
{
    int n = 0;              // 2*N
    int m = 0;              // 0 (no general A matrix — pure box)
    std::vector<double> H;  // n×n, row-major, dense
    std::vector<double> g;  // n
    std::vector<double> lb; // n  (per-component lower bound on U)
    std::vector<double> ub; // n  (per-component upper bound on U)
    // kept for compatibility with runGeoQP / runQpOASES / runOSQP which read C/lbA/ubA
    std::vector<double> C;
    std::vector<double> lbA;
    std::vector<double> ubA;
    double const_term = 0.0;
};


// ─────────────────────────────────────────────────────────────────────────────
//  DCM MPC — condensed QP builder
//
//  Dynamics (scalar, per axis independently):
//    xi_{k+1} = alpha * xi_k + (1 - alpha) * u_k
//  => over horizon N:
//    Xi = S_xi * xi_0 + S_u * U
//  where S_xi[k] = alpha^{k+1}  (scalar repeated 2 times per row for 2D)
//        S_u[k,j] = alpha^{k-j} * (1-alpha)  for j<=k, 0 otherwise
//
//  Cost:
//    J = ||Xi - Xi_ref||^2_{tilde_Q}  +  ||Theta*U - u_prev_stacked||^2_{tilde_R}
//  tilde_Q = diag(Q, Q, ..., Q)   (N+1 blocks of Q = q_scale * I_2)
//  tilde_R = diag(R, R, ..., R)   (N   blocks of R = r_scale * I_2)
//  Theta is the N x N (2D: 2N x 2N) first-difference matrix:
//    Theta = [ I; -I I; 0 -I I; ... ]  so Theta*U = [u_0-u_prev; u1-u0; ...]
//
//  After condensing Xi = S_xi*xi0 + S_u*U:
//    H = S_u^T * tilde_Q * S_u  +  Theta^T * tilde_R * Theta
//    g = S_u^T * tilde_Q * (S_xi*xi0 - Xi_ref)  -  Theta^T * tilde_R * [u_prev; 0; ...]
//
//  Constraints:  lb <= U <= ub   (axis-aligned box in R^{2N}, no A matrix)
//  Per step k: the ZMP u_k must lie in a 2D rectangle defined by the current
//  support polygon.  This is axis-aligned => exact box.
// ─────────────────────────────────────────────────────────────────────────────

// Per-step support box: [lx, ux] x [ly, uy]
struct SupportBox { double lx, ux, ly, uy; };

DenseQP buildQP(const Parameters &p,
                const std::vector<double>        &xi0,          // 2D DCM: [xi_x, xi_y]
                const std::vector<double>        &u_prev,       // previous ZMP: [u_x, u_y]
                const std::vector<SupportBox>    &support,      // N support boxes
                const std::vector<double>        &xi_ref)       // 2*N reference DCM trajectory
{
    const int  N    = p.N;
    const int  n    = 2 * N;           // total decision variables
    const double al = p.alpha();       // e^{omega*T}
    const double be = 1.0 - al;        // (1 - alpha) = B_d scalar
    const double Q  = p.q_scale;       // DCM tracking weight (per component)
    const double R  = p.r_scale;       // input-increment weight (per component)

    // ── Build scalar S_u (N x N lower triangular) ──────────────────────────
    // S_u[k,j] = alpha^{k-j} * beta  for j <= k
    std::vector<double> Su_s(N * N, 0.0);
    for (int k = 0; k < N; ++k)
    {
        double coeff = be;
        for (int j = k; j >= 0; --j)
        {
            Su_s[k * N + j] = coeff;
            coeff *= al;
        }
    }
    // Free DCM response: xi_free[k] = alpha^{k+1} * xi0_component
    // (handled per axis below when computing e = xi_free - xi_ref)

    // ── Hessian H (2N x 2N) assembled by 2x2 blocks ─────────────────────────
    // H = S_u^T * Q * S_u  +  Theta^T * R * Theta
    // Both x and y axes are decoupled and identical in structure.
    // We compute the scalar (N x N) Hessian and replicate for x and y.
    //
    // Term 1: (S_u^T * Q * S_u)[i,j] = Q * sum_k Su[k,i]*Su[k,j]
    std::vector<double> H_s(N * N, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            double s = 0.0;
            for (int k = std::max(i,j); k < N; ++k)
                s += Su_s[k * N + i] * Su_s[k * N + j];
            H_s[i * N + j] = Q * s;
        }

    // Term 2: Theta^T * R * Theta.
    // Theta (N x N): Theta[k,k]=1, Theta[k,k-1]=-1 for k>=1
    // (Theta^T * Theta)[i,j]:
    //   diagonal [i,i]: 1 + (i<N-1 ? 1 : 0)  => 2 for i<N-1, 1 for i=N-1
    //   off-diag [i,i+1] = [i+1,i] = -1  for i < N-1
    for (int i = 0; i < N; ++i)
    {
        H_s[i * N + i] += R * (i < N - 1 ? 2.0 : 1.0);
        if (i < N - 1)
        {
            H_s[i * N + (i + 1)] += -R;
            H_s[(i + 1) * N + i] += -R;
        }
    }
    // Regularise for numerical stability
    for (int i = 0; i < N; ++i)
        H_s[i * N + i] += 1e-8;

    // ── Expand to 2N x 2N (interleaved x/y: col 2j = x of step j, col 2j+1 = y) ──
    DenseQP qp;
    qp.n = n;
    qp.m = 0;
    qp.H.assign(n * n, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            double h = H_s[i * N + j];
            // x block
            qp.H[(2*i)   * n + (2*j)  ] = h;
            // y block
            qp.H[(2*i+1) * n + (2*j+1)] = h;
        }

    // ── Pre-compute free DCM response: xi_free[k] = alpha^{k+1} * xi0_a ────
    // (k = 0..N-1 corresponds to steps 1..N)
    std::vector<double> alpha_pow(N + 1);  // alpha_pow[k] = alpha^{k+1}
    alpha_pow[0] = al;
    for (int k = 1; k < N; ++k)
        alpha_pow[k] = alpha_pow[k - 1] * al;

    // ── Gradient g (2N) ─────────────────────────────────────────────────────
    // g = S_u^T * Q * e  +  Theta^T * R * b
    // e[k] = alpha^{k+1} * xi0_a - xi_ref[k*2 + axis]
    // b = [-u_prev; 0; ...; 0]  (only slot 0 is nonzero)
    // => (Theta^T * R * b)[0] = -R * u_prev_a,  all others 0
    qp.g.assign(n, 0.0);
    for (int axis = 0; axis < 2; ++axis)
    {
        const double xi0_a   = xi0[axis];
        const double uprev_a = u_prev[axis];

        for (int i = 0; i < N; ++i)
        {
            double s = 0.0;
            for (int k = i; k < N; ++k)
            {
                double e_k = alpha_pow[k] * xi0_a - xi_ref[k * 2 + axis];
                s += Su_s[k * N + i] * e_k;
            }
            qp.g[2 * i + axis] = Q * s;
        }
        // Boundary term from input-increment penalty (Theta * U evaluated at k=0)
        qp.g[axis] += -R * uprev_a;
    }

    // ── Constant term (offset for objective comparison) ──────────────────────
    for (int axis = 0; axis < 2; ++axis)
        for (int k = 0; k < N; ++k)
        {
            double e = alpha_pow[k] * xi0[axis] - xi_ref[k * 2 + axis];
            qp.const_term += 0.5 * Q * e * e;
        }
    qp.const_term += 0.5 * R * (u_prev[0]*u_prev[0] + u_prev[1]*u_prev[1]);

    // ── Box constraints: lb <= U <= ub (axis-aligned box in R^{2N}) ──────────
    qp.lb.resize(n);
    qp.ub.resize(n);
    for (int k = 0; k < N; ++k)
    {
        qp.lb[2 * k    ] = support[k].lx;
        qp.ub[2 * k    ] = support[k].ux;
        qp.lb[2 * k + 1] = support[k].ly;
        qp.ub[2 * k + 1] = support[k].uy;
    }

    return qp;
}


// ─────────────────────────────────────────────────────────────────────────────
//  DCM walking schedule
//
//  Generates per-step ZMP support boxes for the MPC preview horizon.
//  Foot placements alternate left/right by stride_y; stride_x is the forward
//  step length.  Within each foot the ZMP is constrained to a rectangle of
//  half-size foot_half_x x foot_half_y centred on the foot position cx, cy.
//
//  To keep a pure box (no general A), the support box is simply
//    [cx - foot_half_x, cx + foot_half_x] x [cy - foot_half_y, cy + foot_half_y]
//  for each preview step.  During double-support we use the union rectangle
//  (or simply the current support foot box — either stays a box).
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<SupportBox> makeSupportBoxes(const Parameters &p,
                                                 double cx0, double cy0,
                                                 int N)
{
    const double hx = p.foot_half_x;
    const double hy = p.foot_half_y;

    // Enumerate foot placements for the preview horizon.
    // We precompute enough steps by tracking which foot and its position.
    // foot sign: +1 = right, -1 = left (alternate)
    const int nb_dsp = std::max(1, (int)std::round(p.dsp_duration / p.dt));
    const int nb_ssp = std::max(1, (int)std::round(p.ssp_duration / p.dt));
    const int step_len = nb_dsp + nb_ssp;

    std::vector<SupportBox> boxes;
    boxes.reserve(N);

    double foot_x = cx0;
    double foot_y = cy0;
    int    foot_sign = 1;  // first step is right foot
    int    phase_idx = 0;  // within current step cycle

    for (int k = 0; k < N; ++k)
    {
        SupportBox box;
        if (phase_idx < nb_dsp)
        {
            // Double-support: allow full ZMP within current support foot box
            box = {foot_x - hx, foot_x + hx, foot_y - hy, foot_y + hy};
        }
        else
        {
            // Single-support: ZMP constrained to current support foot
            box = {foot_x - hx, foot_x + hx, foot_y - hy, foot_y + hy};
        }
        boxes.push_back(box);

        ++phase_idx;
        if (phase_idx >= step_len)
        {
            // Advance to next footstep
            phase_idx = 0;
            foot_x += p.stride_x;
            foot_y += foot_sign * p.stride_y;
            foot_sign = -foot_sign;
        }
    }
    return boxes;
}

// Build a DCM reference trajectory: drive towards a goal walking straight.
// xi_ref[k*2+0] = goal_x,  xi_ref[k*2+1] = goal_y  (constant goal for simplicity)
static std::vector<double> makeDCMRef(int N, double goal_x, double goal_y)
{
    std::vector<double> ref(2 * N);
    for (int k = 0; k < N; ++k)
    {
        ref[2 * k    ] = goal_x;
        ref[2 * k + 1] = goal_y;
    }
    return ref;
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
    prob.Q = qp.H;
    prob.g = qp.g;
    prob.lb = qp.lb;
    prob.ub = qp.ub;

    // m=0: no general A constraints — pure box problem
    prob.m = 0;
    prob.A.clear();
    prob.b.clear();

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
              << "║  DCM MPC Benchmark                                           ║\n"
              << "║  GeoQP vs qpOASES vs OSQP — pure box constraints            ║\n"
              << "╚══════════════════════════════════════════════════════════════╝\n\n";

    Parameters p;

    // Read optional MPC weight scales from environment
    const char *qenv = std::getenv("MPC_Q_SCALE");
    const char *renv = std::getenv("MPC_R_SCALE");
    if (qenv) p.q_scale = std::atof(qenv);
    if (renv) p.r_scale = std::atof(renv);

    // Initial DCM and previous ZMP (at rest, foot centred at origin)
    const double xi0_x = 0.0, xi0_y = 0.0;
    const double u_prev_x = 0.0, u_prev_y = 0.0;
    const double cx0 = 0.0, cy0 = 0.0;    // initial support foot position

    std::cout << "Parameters:\n"
              << "  com_height = " << p.com_height << " m\n"
              << "  omega      = " << p.omega()    << " rad/s\n"
              << "  alpha      = " << p.alpha()    << "\n"
              << "  dt         = " << p.dt         << " s\n"
              << "  foot_half  = " << p.foot_half_x << " x " << p.foot_half_y << " m\n"
              << "  stride     = " << p.stride_x    << " x " << p.stride_y    << " m\n"
              << "  Q_scale    = " << p.q_scale     << "  R_scale = " << p.r_scale << "\n\n";

    // We will test horizons N = 5, 10, ..., 120
    std::vector<int> horizons;
    for (int h = 5; h <= 50; h += 5)
        horizons.push_back(h);

    // reset Schur instrumentation counters once
    geoqp::internal::schur_stats_reset();

    const char *repeats_env = std::getenv("GEOQP_REPEATS");
    int repeats = repeats_env ? std::atoi(repeats_env) : 30;

    const char *caching_env = std::getenv("GEOQP_USE_CACHING");

    std::cout << "\nResults (horizons)\n";
    std::cout << "# repeats=" << repeats
              << " GEOQP_USE_CACHING=" << (caching_env ? caching_env : "(unset)") << "\n";
    std::cout << "CASE format: n=<horizon> QP_n=<n> QP_m=<m> run_idx=<i>"
                 " Geo_us=<ms> qpO_us=<ms> OSQP_us=<ms>"
                 " Geo_it=<it> qpO_it=<it> OSQP_it=<it>"
                 " Geo_hops=<hops> AvgPivot=<pivots>"
                 " AvgObjGeo=<obj> Diff_vs_qpO=<diff>"
                 " OK_geo=<ok> OK_qpO=<ok> OK_osqp=<ok>\n";

    for (int h : horizons)
    {
        Parameters pc = p;
        pc.N = h;

        const std::vector<double> xi0   = {xi0_x, xi0_y};
        const std::vector<double> u_prev = {u_prev_x, u_prev_y};

        auto support = makeSupportBoxes(pc, cx0, cy0, h);
        // Goal: walk to first foot landing ~N*stride_x ahead
        double goal_x = cx0 + pc.N * pc.stride_x * 0.5;
        auto xi_ref   = makeDCMRef(h, goal_x, cy0);

        DenseQP qp = buildQP(pc, xi0, u_prev, support, xi_ref);

        for (int run_idx = 0; run_idx < repeats; ++run_idx)
        {
            auto geo_res  = runGeoQP(qp, geoqp::FaceHoppingStrategy::STANDARD, "GeoQP");
            auto osqp_res = runOSQP(qp);
            auto qpo_res  = runQpOASES(qp);

            double diff_geo = 0.0;
            if (qpo_res.success && geo_res.success)
                diff_geo = std::abs(geo_res.objective - qpo_res.objective);

            int ok_geo  = geo_res.success  ? 1 : 0;
            int ok_qpo  = qpo_res.success  ? 1 : 0;
            int ok_osqp = osqp_res.success ? 1 : 0;

            std::cout << "CASE "
                      << "n=" << h << " "
                      << "QP_n=" << qp.n << " "
                      << "QP_m=" << qp.m << " "
                      << "run_idx=" << run_idx << " "
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
    }

    std::cout << "\nFinished horizon cases.\n";
    return 0;
}