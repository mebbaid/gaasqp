/**
 * @brief Benchmark: DCM (Divergent Component of Motion) MPC 
 *
 *  (see https://github.com/gbionics/walking-controllers/blob/master/src/SimplifiedModelControllers/src/DCMModelPredictiveController.cpp):
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

    double q_scale = 1.0;
    double r_scale = 1.0;

    double omega() const { return std::sqrt(gravity / com_height); }
    double alpha() const { return std::exp(omega() * dt); }
};

struct DenseQP
{
    int n = 0;              // 2*N
    int m = 0;              // 0 (no general A matrix — pure box)
    std::vector<double> H;  // n×n, row-major, dense
    std::vector<double> g;  // n
    std::vector<double> lb; // n  (per-component lower bound on U)
    std::vector<double> ub; // n  (per-component upper bound on U)
    std::vector<double> C;
    std::vector<double> lbA;
    std::vector<double> ubA;
    double const_term = 0.0;
};


// Per-step support box: [lx, ux] x [ly, uy]
struct SupportBox { double lx, ux, ly, uy; };

// ─────────────────────────────────────────────────────────────────────────────
//  Per-step support polygon
//
//  During single-support (SS) the ZMP constraint is an axis-aligned rectangle
//  — represented as lb/ub bounds, firing GeoQP's box fast-path.
//
//  During double-support (DS) the ZMP constraint is the convex hull of both
//  feet, computed via Andrews' monotone chain.  The result is a set of
//  halfplane constraints  A_k * u_k <= b_k  (typically 6 rows for two
//  offset rectangles), passed as general A/b rows in the QP.
//
//  is_box == true  →  use lb/ub (SS)
//  is_box == false →  use halfplanes A_rows/b_rows (DS)
// ─────────────────────────────────────────────────────────────────────────────
struct SupportPolygon
{
    bool is_box = true;
    // SS: axis-aligned bounds
    double lx = 0, ux = 0, ly = 0, uy = 0;
    // DS: halfplane rows  A_rows[i] = {a0, a1},  b_rows[i] = b_i
    //     constraint: a0*u_x + a1*u_y <= b_i
    std::vector<std::array<double, 2>> A_rows;
    std::vector<double> b_rows;
};

// Monotone-chain 2-D convex hull of a set of points.
static std::vector<std::array<double,2>> convexHull2D(std::vector<std::array<double,2>> pts)
{
    int n = static_cast<int>(pts.size());
    if (n < 3) return pts;
    std::sort(pts.begin(), pts.end(), [](const auto &a, const auto &b){
        return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
    });
    auto cross = [](const auto &o, const auto &a, const auto &b){
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]);
    };
    std::vector<std::array<double,2>> h(2*n);
    int k = 0;
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && cross(h[k-2], h[k-1], pts[i]) <= 0) --k;
        h[k++] = pts[i];
    }
    for (int i = n-2, t = k+1; i >= 0; --i) {
        while (k >= t && cross(h[k-2], h[k-1], pts[i]) <= 0) --k;
        h[k++] = pts[i];
    }
    h.resize(k-1);
    return h;
}

// Build halfplane constraints from a CCW convex hull polygon.
// Each edge (p0→p1) gives:  (p1.y-p0.y)*u_x + (p0.x-p1.x)*u_y <= p0.x*p1.y - p1.x*p0.y
static SupportPolygon polygonFromHull(const std::vector<std::array<double,2>> &hull)
{
    SupportPolygon sp;
    sp.is_box = false;
    int n = static_cast<int>(hull.size());
    for (int i = 0; i < n; ++i) {
        const auto &p0 = hull[i];
        const auto &p1 = hull[(i+1) % n];
        double a0 = p1[1]-p0[1];
        double a1 = p0[0]-p1[0];
        double b  = p0[0]*p1[1] - p1[0]*p0[1];
        // Normalise to unit L2 norm. 
        const double norm = std::sqrt(a0*a0 + a1*a1);
        if (norm > 1e-12) { a0 /= norm; a1 /= norm; b /= norm; }
        sp.A_rows.push_back({a0, a1});
        sp.b_rows.push_back(b);
    }
    return sp;
}

// Build the convex hull of two foot rectangles for a DS step.
static SupportPolygon makeDSPolygon(double lx, double rx,  
                                    double ly, double ry, 
                                    double hx, double hy)
{
    std::vector<std::array<double,2>> pts;
    pts.reserve(8);
    for (double fx : {lx, rx})
        for (double fy : {ly, ry})
            for (auto [dx, dy] : std::array<std::array<double,2>,4>{{{-hx,-hy},{-hx,hy},{hx,-hy},{hx,hy}}})
                pts.push_back({fx+dx, fy+dy});
    return polygonFromHull(convexHull2D(pts));
}

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
    // (S_u^T * Q * S_u)[i,j] = Q * sum_k Su[k,i]*Su[k,j]
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
    // Regulariser
    for (int i = 0; i < N; ++i)
        H_s[i * N + i] += 1e-8;

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


    std::vector<double> alpha_pow(N + 1); 
    alpha_pow[0] = al;
    for (int k = 1; k < N; ++k)
        alpha_pow[k] = alpha_pow[k - 1] * al;

    // Gradient g 
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

    //  Constant term 
    for (int axis = 0; axis < 2; ++axis)
        for (int k = 0; k < N; ++k)
        {
            double e = alpha_pow[k] * xi0[axis] - xi_ref[k * 2 + axis];
            qp.const_term += 0.5 * Q * e * e;
        }
    qp.const_term += 0.5 * R * (u_prev[0]*u_prev[0] + u_prev[1]*u_prev[1]);

    // Box constraints: lb <= U <= ub 
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

//  buildQP overload for mixed SS/DS support polygons (GEOQP_ENABLE_DS=1)
DenseQP buildQP(const Parameters                    &p,
                const std::vector<double>            &xi0,
                const std::vector<double>            &u_prev,
                const std::vector<SupportPolygon>    &support,
                const std::vector<double>            &xi_ref)
{
    const int N = p.N;
    const int n = 2 * N;

    int m_total = 0;
    for (const auto &sp : support)
        if (!sp.is_box)
            m_total += static_cast<int>(sp.b_rows.size());

    // Build temporarily with infinite boxes to get H, g, const_term
    std::vector<SupportBox> dummy(N, {-1e20, 1e20, -1e20, 1e20});
    DenseQP qp = buildQP(p, xi0, u_prev, dummy, xi_ref);

    qp.m = m_total;

    // Overwrite per-step bounds; 
    for (int k = 0; k < N; ++k)
    {
        const auto &sp = support[k];
        if (sp.is_box)
        {
            qp.lb[2*k    ] = sp.lx;  qp.ub[2*k    ] = sp.ux;
            qp.lb[2*k + 1] = sp.ly;  qp.ub[2*k + 1] = sp.uy;
        }
        else
        {
            qp.lb[2*k    ] = qp.lb[2*k + 1] = -1e31;
            qp.ub[2*k    ] = qp.ub[2*k + 1] =  1e31;
        }
    }

    if (m_total > 0)
    {
        qp.C.assign(m_total * n, 0.0);
        qp.lbA.assign(m_total, -1e20);
        qp.ubA.resize(m_total);

        int row = 0;
        for (int k = 0; k < N; ++k)
        {
            const auto &sp = support[k];
            if (sp.is_box) continue;
            for (int r = 0; r < (int)sp.b_rows.size(); ++r)
            {
                qp.C[row * n + 2*k    ] = sp.A_rows[r][0];
                qp.C[row * n + 2*k + 1] = sp.A_rows[r][1];
                qp.ubA[row] = sp.b_rows[r];
                ++row;
            }
        }
    }

    return qp;
}


//  DCM walking schedule — pure-box version (default, fast-path)
static std::vector<SupportBox> makeSupportBoxes(const Parameters &p,
                                                 double cx0, double cy0,
                                                 int N)
{
    const double hx = p.foot_half_x;
    const double hy = p.foot_half_y;

    const int nb_dsp = std::max(1, (int)std::round(p.dsp_duration / p.dt));
    const int nb_ssp = std::max(1, (int)std::round(p.ssp_duration / p.dt));
    const int step_len = nb_dsp + nb_ssp;

    std::vector<SupportBox> boxes;
    boxes.reserve(N);

    double foot_x = cx0, foot_y = cy0;
    int foot_sign = 1, phase_idx = 0;

    for (int k = 0; k < N; ++k)
    {
        boxes.push_back({foot_x - hx, foot_x + hx, foot_y - hy, foot_y + hy});
        if (++phase_idx >= step_len)
        {
            phase_idx = 0;
            foot_x += p.stride_x;
            foot_y += foot_sign * p.stride_y;
            foot_sign = -foot_sign;
        }
    }
    return boxes;
}

//  DCM walking schedule — SS/DS version (GEOQP_ENABLE_DS=1)
static std::vector<SupportPolygon> makeSupportPolygons(const Parameters &p,
                                                        double cx0, double cy0,
                                                        int N)
{
    const double hx = p.foot_half_x;
    const double hy = p.foot_half_y;

    const int nb_dsp = std::max(1, (int)std::round(p.dsp_duration / p.dt));
    const int nb_ssp = std::max(1, (int)std::round(p.ssp_duration / p.dt));
    const int step_len = nb_dsp + nb_ssp;

    std::vector<SupportPolygon> polys;
    polys.reserve(N);

    double foot_x = cx0, foot_y = cy0;
    double prev_foot_x = cx0, prev_foot_y = cy0;
    int foot_sign = 1, phase_idx = 0;

    for (int k = 0; k < N; ++k)
    {
        if (phase_idx < nb_dsp)
        {
            polys.push_back(makeDSPolygon(prev_foot_x, foot_x,
                                          prev_foot_y, foot_y, hx, hy));
        }
        else
        {
            SupportPolygon sp;
            sp.is_box = true;
            sp.lx = foot_x - hx; sp.ux = foot_x + hx;
            sp.ly = foot_y - hy; sp.uy = foot_y + hy;
            polys.push_back(sp);
        }
        if (++phase_idx >= step_len)
        {
            phase_idx = 0;
            prev_foot_x = foot_x; prev_foot_y = foot_y;
            foot_x += p.stride_x;
            foot_y += foot_sign * p.stride_y;
            foot_sign = -foot_sign;
        }
    }
    return polys;
}

// Build a DCM reference trajectory
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

    // Pass general A*x <= b rows (populated for DS steps; empty for pure-box mode)
    prob.m = qp.m;
    prob.A = qp.C;
    prob.b = qp.ubA;

    geoqp::Config cfg;
    cfg.face_hopping_strategy = strat;
    const char *env = std::getenv("GEOQP_USE_CACHING");
    cfg.use_caching = !(env && std::string(env) == "0");
    cfg.warm_start = false;
    cfg.tolerance = 1e-7;
    cfg.constraint_tolerance = 1e-6; 

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

    res.iterations = sol.iterations;
    res.facet_hops = sol.phase2_facet_hops;
    res.avg_pivot = sol.phase2_iterations;
    res.solution = sol.x;
    return res;
}

static Result runOSQP(const DenseQP &qp)
{
    Result res;
    res.name = "OSQP";
    int n = qp.n;
    int m = qp.m; 
    Eigen::MatrixXd H_e(n, n);
    Eigen::VectorXd g_e(n);
    for (int i = 0; i < n; ++i)
    {
        g_e(i) = qp.g[i];
        for (int j = 0; j < n; ++j)
            H_e(i, j) = qp.H[i * n + j];
    }

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
    solver.setRealNumberParameter("eps_abs", 1e-7);
    solver.setRealNumberParameter("eps_rel", 1e-7);
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

    const char *qenv = std::getenv("MPC_Q_SCALE");
    const char *renv = std::getenv("MPC_R_SCALE");
    if (qenv) p.q_scale = std::atof(qenv);
    if (renv) p.r_scale = std::atof(renv);

    const double xi0_x = 0.0, xi0_y = 0.0;
    const double u_prev_x = 0.0, u_prev_y = 0.0;
    const double cx0 = 0.0, cy0 = 0.0;    

    std::cout << "Parameters:\n"
              << "  com_height = " << p.com_height << " m\n"
              << "  omega      = " << p.omega()    << " rad/s\n"
              << "  alpha      = " << p.alpha()    << "\n"
              << "  dt         = " << p.dt         << " s\n"
              << "  foot_half  = " << p.foot_half_x << " x " << p.foot_half_y << " m\n"
              << "  stride     = " << p.stride_x    << " x " << p.stride_y    << " m\n"
              << "  Q_scale    = " << p.q_scale     << "  R_scale = " << p.r_scale << "\n\n";

    // We will test horizons N = 5, 10, ..., 60
    std::vector<int> horizons;
    for (int h = 5; h <= 60; h += 5)
        horizons.push_back(h);

    geoqp::internal::schur_stats_reset();

    const char *repeats_env = std::getenv("GEOQP_REPEATS");
    int repeats = repeats_env ? std::atoi(repeats_env) : 30;

    const char *caching_env = std::getenv("GEOQP_USE_CACHING");
    const bool caching_on = !(caching_env && std::string(caching_env) == "0");

    const char *ds_env = std::getenv("GEOQP_ENABLE_DS");
    const bool enable_ds = (ds_env && std::string(ds_env) == "1");

    std::cout << "\nResults (horizons)\n";
    std::cout << "# repeats=" << repeats
              << " GEOQP_USE_CACHING=" << (caching_on ? "1" : "0")
              << " GEOQP_ENABLE_DS=" << (enable_ds ? "1" : "0")
              << " constraints=" << (enable_ds ? "SS-box+DS-hull" : "pure-box")
              << "\n";
    std::cout << "CASE format: n=<horizon> QP_n=<n> QP_m=<m> mode=<mode> run_idx=<i>"
                 " Geo_ms=<ms> qpO_ms=<ms> OSQP_ms=<ms>"
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

        // Goal: walk to first foot landing ~N*stride_x ahead
        double goal_x = cx0 + pc.N * pc.stride_x * 0.5;
        auto xi_ref   = makeDCMRef(h, goal_x, cy0);

        DenseQP qp;
        if (enable_ds)
        {
            auto support = makeSupportPolygons(pc, cx0, cy0, h);
            qp = buildQP(pc, xi0, u_prev, support, xi_ref);
        }
        else
        {
            auto support = makeSupportBoxes(pc, cx0, cy0, h);
            qp = buildQP(pc, xi0, u_prev, support, xi_ref);
        }

        const char *mode_str = enable_ds ? "DS" : "box";

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
                      << "mode=" << mode_str << " "
                      << "run_idx=" << run_idx << " "
                      << "Geo_ms=" << geo_res.time_ms << " "
                      << "qpO_ms=" << qpo_res.time_ms << " "
                      << "OSQP_ms=" << osqp_res.time_ms << " "
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