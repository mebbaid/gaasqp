#include "solver_suite.hpp"

#include <qpOASES.hpp>

#include <Eigen/Sparse>

#include <algorithm>
#include <chrono>
#include <cmath>

#ifdef HAVE_OSQP
#include <osqp/osqp.h>
#endif
#ifdef HAVE_DAQP
extern "C" {
#include <api.h>
#include <constants.h>
}
#endif
#ifdef HAVE_PIQP
#include <piqp/piqp.hpp>
#endif
#ifdef HAVE_CLARABEL
#include <clarabel.hpp>
#endif

namespace bench
{
    namespace
    {
        using Clock = std::chrono::steady_clock;
        using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        double us_since(Clock::time_point t0)
        {
            return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
        }

        // Map "no bound" (|v| ≥ kInf) to a solver's infinity.
        Eigen::VectorXd with_inf(const Eigen::VectorXd &v, double inf)
        {
            Eigen::VectorXd out = v;
            for (int i = 0; i < out.size(); ++i)
            {
                if (out(i) >= kInf) out(i) = inf;
                else if (out(i) <= -kInf) out(i) = -inf;
            }
            return out;
        }

        void finish(Run &r, const QP &qp)
        {
            if (r.x.size() == qp.n())
            {
                r.obj = objective(qp, r.x);
                r.viol = max_violation(qp, r.x);
            }
        }

        // ── qpOASES ─────────────────────────────────────────────────────
        Run run_qpoases(const QP &qp, bool box_variant)
        {
            Run r;
            r.id = box_variant ? SolverId::qpOASES_B : SolverId::qpOASES;
            const int n = qp.n(), m = qp.m();
            const RowMat H = qp.H; // symmetric; row-major copy for the C API
            const RowMat A = qp.A;
            const Eigen::VectorXd lb = with_inf(qp.lb, qpOASES::INFTY);
            const Eigen::VectorXd ub = with_inf(qp.ub, qpOASES::INFTY);
            const Eigen::VectorXd lbA = with_inf(qp.lbA, qpOASES::INFTY);
            const Eigen::VectorXd ubA = with_inf(qp.ubA, qpOASES::INFTY);

            qpOASES::Options opts;
            opts.setToMPC();
            opts.printLevel = qpOASES::PL_NONE;
            qpOASES::int_t nWSR = 10 * (n + m) + 1000;
            Eigen::VectorXd x(n);

            if (box_variant)
            {
                qpOASES::QProblemB prob(n, qpOASES::HST_POSDEF);
                prob.setOptions(opts);
                const auto t0 = Clock::now();
                const auto ret = prob.init(H.data(), qp.g.data(), lb.data(), ub.data(), nWSR);
                r.time_us = us_since(t0);
                r.ok = (ret == qpOASES::SUCCESSFUL_RETURN);
                if (r.ok) prob.getPrimalSolution(x.data());
            }
            else
            {
                qpOASES::QProblem prob(n, m, qpOASES::HST_POSDEF);
                prob.setOptions(opts);
                const auto t0 = Clock::now();
                const auto ret = prob.init(H.data(), qp.g.data(), m > 0 ? A.data() : nullptr,
                                           lb.data(), ub.data(),
                                           m > 0 ? lbA.data() : nullptr,
                                           m > 0 ? ubA.data() : nullptr, nWSR);
                r.time_us = us_since(t0);
                r.ok = (ret == qpOASES::SUCCESSFUL_RETURN);
                if (r.ok) prob.getPrimalSolution(x.data());
            }
            r.iters = static_cast<int>(nWSR);
            if (r.ok) r.x = x;
            return r;
        }

#ifdef HAVE_OSQP
        // ── OSQP (ADMM + solution polishing), plain C API ───────────────
        // (OsqpEigen is avoided on purpose: it takes Eigen objects across a
        //  library boundary, which breaks when this code is built with
        //  -march=native and the prebuilt library is not.)
        struct Csc
        {
            std::vector<OSQPFloat> x;
            std::vector<OSQPInt> i, p;
            OSQPInt rows = 0, cols = 0;
        };

        Csc to_csc(const Eigen::MatrixXd &M)
        {
            Eigen::SparseMatrix<double> S = M.sparseView();
            S.makeCompressed();
            Csc c;
            c.rows = S.rows();
            c.cols = S.cols();
            c.x.assign(S.valuePtr(), S.valuePtr() + S.nonZeros());
            c.i.assign(S.innerIndexPtr(), S.innerIndexPtr() + S.nonZeros());
            c.p.assign(S.outerIndexPtr(), S.outerIndexPtr() + S.cols() + 1);
            return c;
        }

        Run run_osqp(const QP &qp, double tol)
        {
            Run r;
            r.id = SolverId::OSQP;
            const int n = qp.n(), m = qp.m();
            const double inf = OSQP_INFTY;
            // Constraints [I; A] x ∈ [l, u]
            Eigen::MatrixXd C(n + m, n);
            C.topRows(n).setIdentity();
            if (m > 0) C.bottomRows(m) = qp.A;
            Eigen::VectorXd l(n + m), u(n + m);
            l << with_inf(qp.lb, inf), with_inf(qp.lbA, inf);
            u << with_inf(qp.ub, inf), with_inf(qp.ubA, inf);
            Csc P = to_csc(qp.H.triangularView<Eigen::Upper>());
            Csc K = to_csc(C);
            Eigen::VectorXd q = qp.g;
            OSQPCscMatrix *Pm = OSQPCscMatrix_new(P.rows, P.cols, P.x.size(), P.x.data(), P.i.data(), P.p.data());
            OSQPCscMatrix *Km = OSQPCscMatrix_new(K.rows, K.cols, K.x.size(), K.x.data(), K.i.data(), K.p.data());
            OSQPSettings *settings = OSQPSettings_new();
            osqp_set_default_settings(settings);
            settings->verbose = 0;
            settings->warm_starting = 0;
            settings->polishing = 1;
            settings->eps_abs = tol;
            settings->eps_rel = tol;

            OSQPSolver *solver = nullptr;
            const auto t0 = Clock::now();
            const OSQPInt flag = osqp_setup(&solver, Pm, q.data(), Km, l.data(), u.data(),
                                            n + m, n, settings);
            if (flag == 0)
                osqp_solve(solver);
            r.time_us = us_since(t0);
            if (flag == 0)
            {
                r.ok = (solver->info->status_val == OSQP_SOLVED);
                r.iters = static_cast<int>(solver->info->iter);
                if (r.ok)
                    r.x = Eigen::Map<const Eigen::VectorXd>(solver->solution->x, n);
                osqp_cleanup(solver);
            }
            OSQPSettings_free(settings);
            OSQPCscMatrix_free(Km);
            OSQPCscMatrix_free(Pm);
            return r;
        }
#endif

#ifdef HAVE_DAQP
        // ── DAQP (dual active set) ──────────────────────────────────────
        Run run_daqp(const QP &qp, double tol)
        {
            Run r;
            r.id = SolverId::DAQP;
            const int n = qp.n(), m = qp.m();
            RowMat H = qp.H;
            RowMat A = qp.A;
            Eigen::VectorXd f = qp.g;
            Eigen::VectorXd bl(n + m), bu(n + m); // [simple bounds; general rows]
            bl << with_inf(qp.lb, DAQP_INF), with_inf(qp.lbA, DAQP_INF);
            bu << with_inf(qp.ub, DAQP_INF), with_inf(qp.ubA, DAQP_INF);
            Eigen::VectorXd x(n), lam(n + m);

            DAQPProblem prob{};
            prob.n = n;
            prob.m = n + m;
            prob.ms = n;
            prob.H = H.data();
            prob.f = f.data();
            prob.A = m > 0 ? A.data() : nullptr;
            prob.bupper = bu.data();
            prob.blower = bl.data();
            prob.sense = nullptr;

            DAQPSettings settings;
            daqp_default_settings(&settings);
            settings.primal_tol = tol;

            DAQPResult res{};
            res.x = x.data();
            res.lam = lam.data();

            const auto t0 = Clock::now();
            daqp_quadprog(&res, &prob, &settings);
            r.time_us = us_since(t0);
            r.ok = (res.exitflag == DAQP_EXIT_OPTIMAL);
            r.iters = res.iter;
            if (r.ok) r.x = x;
            return r;
        }
#endif

#ifdef HAVE_PIQP
        // ── PIQP (proximal interior point, dense KKT) ───────────────────
        Run run_piqp(const QP &qp, double tol)
        {
            Run r;
            r.id = SolverId::PIQP;
            const int m = qp.m();
            const Eigen::VectorXd xl = with_inf(qp.lb, PIQP_INF), xu = with_inf(qp.ub, PIQP_INF);
            const Eigen::VectorXd hl = with_inf(qp.lbA, PIQP_INF), hu = with_inf(qp.ubA, PIQP_INF);

            piqp::DenseSolver<double> s;
            s.settings().verbose = false;
            s.settings().eps_abs = tol;
            s.settings().eps_rel = tol;
            s.settings().eps_duality_gap_abs = tol;
            s.settings().eps_duality_gap_rel = tol;

            const auto t0 = Clock::now();
            if (m > 0)
                s.setup(qp.H, qp.g, piqp::nullopt, piqp::nullopt, qp.A, hl, hu, xl, xu);
            else
                s.setup(qp.H, qp.g, piqp::nullopt, piqp::nullopt, piqp::nullopt, piqp::nullopt,
                        piqp::nullopt, xl, xu);
            const piqp::Status st = s.solve();
            r.time_us = us_since(t0);
            r.ok = (st == piqp::Status::PIQP_SOLVED);
            r.iters = static_cast<int>(s.result().info.iter);
            if (r.ok) r.x = s.result().x;
            return r;
        }
#endif

#ifdef HAVE_CLARABEL
        // ── Clarabel (interior point; QP as  Kx + s = k,  s ≥ 0) ────────
        Run run_clarabel(const QP &qp, double tol)
        {
            Run r;
            r.id = SolverId::Clarabel;
            const int n = qp.n(), m = qp.m();
            // One row per finite bound:  ±row · x ≤ ±bound.
            std::vector<Eigen::Triplet<double>> trip;
            std::vector<double> rhs;
            auto add_row = [&](const Eigen::RowVectorXd &a, double sign, double bound) {
                const int row = static_cast<int>(rhs.size());
                for (int j = 0; j < n; ++j)
                    if (a(j) != 0.0) trip.emplace_back(row, j, sign * a(j));
                rhs.push_back(sign * bound);
            };
            for (int i = 0; i < n; ++i)
            {
                const Eigen::RowVectorXd e = Eigen::RowVectorXd::Unit(n, i);
                if (qp.ub(i) < kInf) add_row(e, +1.0, qp.ub(i));
                if (qp.lb(i) > -kInf) add_row(e, -1.0, qp.lb(i));
            }
            for (int i = 0; i < m; ++i)
            {
                if (qp.ubA(i) < kInf) add_row(qp.A.row(i), +1.0, qp.ubA(i));
                if (qp.lbA(i) > -kInf) add_row(qp.A.row(i), -1.0, qp.lbA(i));
            }
            const int rows = static_cast<int>(rhs.size());
            Eigen::SparseMatrix<double> K(rows, n);
            K.setFromTriplets(trip.begin(), trip.end());
            K.makeCompressed();
            Eigen::VectorXd k = Eigen::Map<Eigen::VectorXd>(rhs.data(), rows);
            const Eigen::MatrixXd Hu = qp.H.triangularView<Eigen::Upper>();
            Eigen::SparseMatrix<double> P = Hu.sparseView();
            P.makeCompressed();
            Eigen::VectorXd q = qp.g;
            std::vector<clarabel::SupportedConeT<double>> cones{clarabel::NonnegativeConeT<double>(rows)};

            auto settings = clarabel::DefaultSettings<double>::default_settings();
            settings.verbose = false;
            settings.tol_gap_abs = tol;
            settings.tol_gap_rel = tol;
            settings.tol_feas = tol;

            const auto t0 = Clock::now();
            clarabel::DefaultSolver<double> s(P, q, K, k, cones, settings);
            s.solve();
            r.time_us = us_since(t0);
            const auto sol = s.solution();
            r.ok = (sol.status == clarabel::SolverStatus::Solved);
            r.iters = static_cast<int>(sol.iterations);
            if (r.ok) r.x = Eigen::Map<const Eigen::VectorXd>(sol.x.data(), n);
            return r;
        }
#endif
    } // namespace

    std::string name(SolverId id)
    {
        switch (id)
        {
        case SolverId::qpOASES: return "qpOASES";
        case SolverId::qpOASES_B: return "qpOASES-B";
        case SolverId::OSQP: return "OSQP";
        case SolverId::DAQP: return "DAQP";
        case SolverId::PIQP: return "PIQP";
        case SolverId::Clarabel: return "Clarabel";
        }
        return "?";
    }

    bool available(SolverId id)
    {
        switch (id)
        {
        case SolverId::qpOASES:
        case SolverId::qpOASES_B: return true;
#ifdef HAVE_OSQP
        case SolverId::OSQP: return true;
#endif
#ifdef HAVE_DAQP
        case SolverId::DAQP: return true;
#endif
#ifdef HAVE_PIQP
        case SolverId::PIQP: return true;
#endif
#ifdef HAVE_CLARABEL
        case SolverId::Clarabel: return true;
#endif
        default: return false;
        }
    }

    bool applicable(SolverId id, const QP &qp)
    {
        if (!available(id)) return false;
        if (id == SolverId::qpOASES_B) return qp.box_only();
        return true;
    }

    std::vector<SolverId> all_solvers()
    {
        std::vector<SolverId> out;
        for (SolverId id : {SolverId::qpOASES, SolverId::qpOASES_B, SolverId::OSQP,
                            SolverId::DAQP, SolverId::PIQP, SolverId::Clarabel})
            if (available(id)) out.push_back(id);
        return out;
    }

    Run run(SolverId id, const QP &qp, double tol)
    {
        Run r;
        r.id = id;
        if (!applicable(id, qp))
            return r;
        switch (id)
        {
        case SolverId::qpOASES: r = run_qpoases(qp, false); break;
        case SolverId::qpOASES_B: r = run_qpoases(qp, true); break;
#ifdef HAVE_OSQP
        case SolverId::OSQP: r = run_osqp(qp, tol); break;
#endif
#ifdef HAVE_DAQP
        case SolverId::DAQP: r = run_daqp(qp, tol); break;
#endif
#ifdef HAVE_PIQP
        case SolverId::PIQP: r = run_piqp(qp, tol); break;
#endif
#ifdef HAVE_CLARABEL
        case SolverId::Clarabel: r = run_clarabel(qp, tol); break;
#endif
        default: break;
        }
        (void)tol;
        finish(r, qp);
        return r;
    }

    double objective(const QP &qp, const Eigen::VectorXd &x)
    {
        return 0.5 * x.dot(qp.H * x) + qp.g.dot(x);
    }

    double max_violation(const QP &qp, const Eigen::VectorXd &x)
    {
        double v = 0.0;
        for (int i = 0; i < qp.n(); ++i)
        {
            if (qp.ub(i) < kInf) v = std::max(v, x(i) - qp.ub(i));
            if (qp.lb(i) > -kInf) v = std::max(v, qp.lb(i) - x(i));
        }
        if (qp.m() > 0)
        {
            const Eigen::VectorXd Ax = qp.A * x;
            for (int i = 0; i < qp.m(); ++i)
            {
                if (qp.ubA(i) < kInf) v = std::max(v, Ax(i) - qp.ubA(i));
                if (qp.lbA(i) > -kInf) v = std::max(v, qp.lbA(i) - Ax(i));
            }
        }
        return v;
    }

} // namespace bench
