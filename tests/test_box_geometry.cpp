// Numerical checks of the visible-facet result that replaces the half-space
// theorem (old claim:  ηᵀT⁻¹(x★ − c₀) ≥ 0  with ηᵢ = sᵢ/hᵢ).
//
// Claim labels (A–F) follow the derivation:
//   A  good vertices ⇔ σᵢ = −1 whenever ξ̄ᵢ ≥ 1;  2^{n−k} of them;  −1 is good
//   B  x★ lies on a visible facet {ξᵢ = 1}, i ∈ V
//   C  Σ_{i∈V} ξ★ᵢ ≥ 2 − k;  that region = conv(∪ visible facets)
//   D  every point of every visible facet is optimal for some H
//   E  exit facet i₀ = argmax ξ̄ᵢ ∈ V, same from every good vertex
//   F  H-metric half-space through c₀ and ball around ½(c₀ + x̄)
//
// The reference optimum is computed independently of the ξ-machinery: brute-
// force enumeration of all 3ⁿ active-set patterns in y-space, accepted only if
// it satisfies the KKT conditions.

#include <gtest/gtest.h>
#include "geoqp/box_geometry.hpp"
#include "geoqp/geoqp.hpp"
#include "geoqp/internal.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace geoqp;
using namespace geoqp::geometry;
using Vec = std::vector<double>;

namespace
{
    // ── Eigen helpers ───────────────────────────────────────────────────
    Eigen::MatrixXd to_mat(const Vec &M, int n)
    {
        Eigen::MatrixXd A(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                A(i, j) = M[i * n + j];
        return A;
    }

    Vec from_mat(const Eigen::MatrixXd &A)
    {
        const int n = static_cast<int>(A.rows());
        Vec M(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                M[i * n + j] = A(i, j);
        return M;
    }

    Eigen::VectorXd to_vec(const Vec &v) { return Eigen::Map<const Eigen::VectorXd>(v.data(), v.size()); }
    Vec from_vec(const Eigen::VectorXd &v) { return Vec(v.data(), v.data() + v.size()); }

    Vec symmetrize(const Vec &H, int n)
    {
        Eigen::MatrixXd M = to_mat(H, n);
        return from_mat(0.5 * (M + M.transpose()));
    }

    // ── independent reference QP solver (y-space, 3ⁿ enumeration + KKT) ──
    struct RefSol
    {
        bool ok = false;
        Eigen::VectorXd x, y;
        Eigen::VectorXd ry; // ∇_y f at y★ (= multipliers up to sign on active bounds)
    };

    RefSol solve_reference(const AffineBox &box, const Vec &Hv, const Vec &gv)
    {
        const int n = box.n;
        const Eigen::MatrixXd T = to_mat(box.T, n);
        const Eigen::MatrixXd H = to_mat(Hv, n);
        const Eigen::VectorXd c = to_vec(box.c), g = to_vec(gv);
        const Eigen::VectorXd l = to_vec(box.lower), u = to_vec(box.upper);
        const Eigen::MatrixXd Hy = T.transpose() * H * T;
        const Eigen::VectorXd gy = T.transpose() * (H * c + g);

        const double scale = std::max({1.0, Hy.cwiseAbs().maxCoeff(), gy.cwiseAbs().maxCoeff(),
                                       l.cwiseAbs().maxCoeff(), u.cwiseAbs().maxCoeff()});
        const double tol = 1e-9 * scale;

        long total = 1;
        for (int i = 0; i < n; ++i)
            total *= 3;

        RefSol best;
        for (long code = 0; code < total; ++code)
        {
            std::vector<int> pat(n);
            long cc = code;
            for (int i = 0; i < n; ++i)
            {
                pat[i] = static_cast<int>(cc % 3); // 0 free, 1 lower, 2 upper
                cc /= 3;
            }
            Eigen::VectorXd y = Eigen::VectorXd::Zero(n);
            std::vector<int> F, X;
            for (int i = 0; i < n; ++i)
            {
                if (pat[i] == 0)
                    F.push_back(i);
                else
                {
                    X.push_back(i);
                    y(i) = (pat[i] == 1) ? l(i) : u(i);
                }
            }
            if (!F.empty())
            {
                const int nf = static_cast<int>(F.size());
                Eigen::MatrixXd HFF(nf, nf);
                Eigen::VectorXd rhs(nf);
                for (int a = 0; a < nf; ++a)
                {
                    rhs(a) = -gy(F[a]);
                    for (int b = 0; b < nf; ++b)
                        HFF(a, b) = Hy(F[a], F[b]);
                    for (int j : X)
                        rhs(a) -= Hy(F[a], j) * y(j);
                }
                const Eigen::VectorXd yF = HFF.llt().solve(rhs);
                for (int a = 0; a < nf; ++a)
                    y(F[a]) = yF(a);
            }
            bool ok = true;
            for (int i : F)
                if (y(i) < l(i) - tol || y(i) > u(i) + tol)
                    ok = false;
            if (!ok)
                continue;
            const Eigen::VectorXd r = Hy * y + gy;
            for (int i = 0; i < n && ok; ++i)
            {
                if (pat[i] == 0 && std::abs(r(i)) > tol) ok = false;
                if (pat[i] == 1 && r(i) < -tol) ok = false; // multiplier on −yᵢ ≤ −ℓᵢ is rᵢ
                if (pat[i] == 2 && r(i) > tol) ok = false;  // multiplier on  yᵢ ≤  uᵢ is −rᵢ
            }
            if (!ok)
                continue;
            best.ok = true;
            best.y = y;
            best.x = T * y + c;
            best.ry = r;
            return best;
        }
        return best;
    }

    // ── random instances ────────────────────────────────────────────────
    enum class TKind { Identity, Diagonal, Dense };
    enum class HKind { Dense, IllConditioned, Diagonal };

    Eigen::MatrixXd random_orthogonal(int n, std::mt19937 &rng)
    {
        std::normal_distribution<> N(0.0, 1.0);
        Eigen::MatrixXd R(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                R(i, j) = N(rng);
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(R);
        return qr.householderQ();
    }

    Eigen::MatrixXd random_T(int n, TKind kind, std::mt19937 &rng)
    {
        std::uniform_real_distribution<> U(0.5, 2.0);
        std::bernoulli_distribution coin(0.5);
        switch (kind)
        {
        case TKind::Identity:
            return Eigen::MatrixXd::Identity(n, n);
        case TKind::Diagonal:
        {
            Eigen::MatrixXd T = Eigen::MatrixXd::Zero(n, n);
            for (int i = 0; i < n; ++i)
                T(i, i) = U(rng) * (coin(rng) ? 1.0 : -1.0);
            return T;
        }
        case TKind::Dense:
        default:
        {
            Eigen::VectorXd d(n);
            for (int i = 0; i < n; ++i)
                d(i) = U(rng);
            return random_orthogonal(n, rng) * d.asDiagonal() * random_orthogonal(n, rng);
        }
        }
    }

    Eigen::MatrixXd random_H(int n, HKind kind, std::mt19937 &rng)
    {
        std::normal_distribution<> N(0.0, 1.0);
        switch (kind)
        {
        case HKind::Diagonal:
        {
            std::uniform_real_distribution<> U(0.5, 4.0);
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
            for (int i = 0; i < n; ++i)
                H(i, i) = U(rng);
            return H;
        }
        case HKind::IllConditioned:
        {
            std::uniform_real_distribution<> E(0.0, 4.0); // cond up to 1e4
            const double cond = std::pow(10.0, E(rng));
            Eigen::VectorXd ev(n);
            for (int i = 0; i < n; ++i)
                ev(i) = std::pow(cond, n > 1 ? double(i) / (n - 1) : 0.0);
            const Eigen::MatrixXd Q = random_orthogonal(n, rng);
            return Q * ev.asDiagonal() * Q.transpose();
        }
        case HKind::Dense:
        default:
        {
            Eigen::MatrixXd R(n, n);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    R(i, j) = N(rng);
            Eigen::MatrixXd H = R.transpose() * R;
            H += 0.1 * (H.trace() / n) * Eigen::MatrixXd::Identity(n, n);
            return H;
        }
        }
    }

    struct Instance
    {
        AffineBox box;
        Vec H, g, xbar;
    };

    // ȳ violates exactly k_target bounds (k_target ∈ [1,n]; 0 → random k).
    Instance random_instance(int n, TKind tk, HKind hk, std::mt19937 &rng, int k_target = 0)
    {
        std::normal_distribution<> N(0.0, 1.0);
        std::uniform_real_distribution<> W(0.3, 2.0), In(-0.95, 0.95), Out(0.05, 3.0);
        std::bernoulli_distribution coin(0.5);

        Instance I;
        I.box.n = n;
        const Eigen::MatrixXd T = random_T(n, tk, rng);
        I.box.T = from_mat(T);
        I.box.c.resize(n);
        I.box.lower.resize(n);
        I.box.upper.resize(n);
        for (int i = 0; i < n; ++i)
        {
            I.box.c[i] = 0.5 * N(rng);
            const double m = 0.5 * N(rng), w = W(rng);
            I.box.lower[i] = m - w;
            I.box.upper[i] = m + w;
        }

        int k = k_target;
        if (k <= 0)
            k = std::uniform_int_distribution<>(1, n)(rng);
        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i)
            idx[i] = i;
        std::shuffle(idx.begin(), idx.end(), rng);
        Eigen::VectorXd ybar(n);
        for (int i = 0; i < n; ++i)
        {
            const double mu = 0.5 * (I.box.lower[i] + I.box.upper[i]);
            const double h = 0.5 * (I.box.upper[i] - I.box.lower[i]);
            ybar(i) = mu + h * In(rng);
        }
        for (int a = 0; a < k; ++a)
        {
            const int i = idx[a];
            const double h = 0.5 * (I.box.upper[i] - I.box.lower[i]);
            ybar(i) = coin(rng) ? I.box.upper[i] + h * Out(rng) : I.box.lower[i] - h * Out(rng);
        }
        const Eigen::VectorXd xbar = T * ybar + to_vec(I.box.c);
        const Eigen::MatrixXd H = random_H(n, hk, rng);
        I.H = from_mat(H);
        I.xbar = from_vec(xbar);
        I.g = from_vec(-H * xbar);
        return I;
    }

    std::vector<std::vector<int>> all_sign_vectors(int n)
    {
        std::vector<std::vector<int>> out;
        for (int mask = 0; mask < (1 << n); ++mask)
        {
            std::vector<int> s(n);
            for (int i = 0; i < n; ++i)
                s[i] = ((mask >> i) & 1) ? 1 : -1;
            out.push_back(s);
        }
        return out;
    }

    // Definition 1 in x-space on the half-space representation:
    // a_rᵀ(x̄ − v) < 0 for every active row r at v.
    bool is_good_by_definition(const AffineBox &box, const Vec &xbar, const Vec &v)
    {
        Vec A, b;
        box.halfspace(A, b);
        const int n = box.n;
        for (int r = 0; r < 2 * n; ++r)
        {
            double av = 0.0, ad = 0.0;
            for (int j = 0; j < n; ++j)
            {
                av += A[r * n + j] * v[j];
                ad += A[r * n + j] * (xbar[j] - v[j]);
            }
            const double scale = std::max(1.0, std::abs(b[r]));
            if (std::abs(av - b[r]) <= 1e-9 * scale && !(ad < 0.0))
                return false;
        }
        return true;
    }

    long factorial(int k)
    {
        long f = 1;
        for (int i = 2; i <= k; ++i)
            f *= i;
        return f;
    }

    // The counterexample from the task, embedded in Rⁿ (n ≥ 3) as
    // H = blkdiag(H₃, I), x̄ = (2, 0, …, 0), box [−1,1]ⁿ.
    Instance counterexample(int n)
    {
        Instance I;
        I.box = AffineBox::axis_aligned(Vec(n, -1.0), Vec(n, 1.0));
        Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n);
        H.topLeftCorner(3, 3) << 3, -1, -1,
                                 -1, 2, -1,
                                 -1, -1, 2;
        Eigen::VectorXd xbar = Eigen::VectorXd::Zero(n);
        xbar(0) = 2.0;
        I.H = from_mat(H);
        I.xbar = from_vec(xbar);
        I.g = from_vec(-H * xbar);
        return I;
    }

    const std::vector<TKind> kAllT = {TKind::Identity, TKind::Diagonal, TKind::Dense};
    const std::vector<HKind> kAllH = {HKind::Dense, HKind::IllConditioned, HKind::Diagonal};
    constexpr double kFacetTol = 1e-7;
} // namespace

// ═════════════════════════════ old theorem ══════════════════════════════

TEST(OldHalfspaceTheorem, CounterexampleFromDerivation)
{
    const Instance I = counterexample(3);
    EXPECT_EQ(I.g, (Vec{-6.0, 2.0, 2.0}));

    const RefSol ref = solve_reference(I.box, I.H, I.g);
    ASSERT_TRUE(ref.ok);
    EXPECT_NEAR(ref.x(0), 1.0, 1e-12);
    EXPECT_NEAR(ref.x(1), -1.0, 1e-12);
    EXPECT_NEAR(ref.x(2), -1.0, 1e-12);

    // ∇f(x★) = (−1, 0, 0): multiplier 1 on x₁ ≤ 1, zero on the two active lower bounds.
    const Eigen::VectorXd grad = to_mat(I.H, 3) * ref.x + to_vec(I.g);
    EXPECT_NEAR(grad(0), -1.0, 1e-12);
    EXPECT_NEAR(grad(1), 0.0, 1e-12);
    EXPECT_NEAR(grad(2), 0.0, 1e-12);

    const BoxFrame f = make_frame(I.box, I.xbar);
    EXPECT_EQ(f.s, (Vec{1.0, 1.0, 1.0})); // ties ȳᵢ = μᵢ give sᵢ = +1
    EXPECT_EQ(f.violated, (std::vector<int>{0}));

    // Old claim fails: 1ᵀξ★ = −1.
    EXPECT_NEAR(old_halfspace_value(f, from_vec(ref.x)), -1.0, 1e-12);
    // New claims hold.
    EXPECT_EQ(visible_facets_containing(f, from_vec(ref.x), kFacetTol), (std::vector<int>{0}));
    EXPECT_NEAR(corrected_cut_margin(f, from_vec(ref.x)), 0.0, 1e-12);
    EXPECT_GE(h_halfspace_margin(f, I.H, from_vec(ref.x)), 0.0);

    // GaasQP itself solves it correctly.
    Problem p;
    p.n = 3;
    p.m = 0;
    p.Q = I.H;
    p.g = I.g;
    p.lb = I.box.lower;
    p.ub = I.box.upper;
    Config cfg;
    cfg.verbosity = -1;
    const Solution sol = Solver(cfg).solve(p);
    ASSERT_EQ(sol.status, Status::SUCCESS);
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(sol.x[i], ref.x(i), 1e-9);
}

TEST(OldHalfspaceTheorem, FailsInEveryDimensionAtLeastThree)
{
    for (int n = 3; n <= 8; ++n)
    {
        const Instance I = counterexample(n);
        const RefSol ref = solve_reference(I.box, I.H, I.g);
        ASSERT_TRUE(ref.ok) << "n=" << n;
        const BoxFrame f = make_frame(I.box, I.xbar);
        EXPECT_NEAR(old_halfspace_value(f, from_vec(ref.x)), -1.0, 1e-10) << "n=" << n;
        EXPECT_GE(corrected_cut_margin(f, from_vec(ref.x)), -1e-10) << "n=" << n;
    }
}

TEST(OldHalfspaceTheorem, HoldsInTwoDimensions)
{
    std::mt19937 rng(2024);
    int checked = 0;
    for (TKind tk : kAllT)
        for (HKind hk : kAllH)
            for (int t = 0; t < 300; ++t)
            {
                const Instance I = random_instance(2, tk, hk, rng);
                const RefSol ref = solve_reference(I.box, I.H, I.g);
                ASSERT_TRUE(ref.ok);
                const BoxFrame f = make_frame(I.box, I.xbar);
                EXPECT_GE(old_halfspace_value(f, from_vec(ref.x)), -1e-9);
                ++checked;
            }
    // Adversarial: the tightness Hessians realise every point of every visible facet.
    std::uniform_real_distribution<> U(-1.0, 1.0);
    for (int t = 0; t < 500; ++t)
    {
        const Instance I = random_instance(2, kAllT[t % 3], HKind::Dense, rng);
        const BoxFrame f = make_frame(I.box, I.xbar);
        for (int i : f.violated)
        {
            Vec p = {U(rng), U(rng)};
            p[i] = 1.0;
            const Vec H = symmetrize(tightness_hessian(f, p, i, 0.3), 2);
            const Eigen::VectorXd gg = -to_mat(H, 2) * to_vec(I.xbar);
            const RefSol ref = solve_reference(I.box, H, from_vec(gg));
            ASSERT_TRUE(ref.ok);
            EXPECT_GE(old_halfspace_value(f, from_vec(ref.x)), -1e-9);
            ++checked;
        }
    }
    RecordProperty("instances_checked", checked);
}

TEST(OldHalfspaceTheorem, RandomSearchViolationRate)
{
    // Informational only (the embedding test above already proves failure for
    // every n ≥ 3): how often does a random instance violate the old claim?
    std::mt19937 rng(7);
    for (int n = 3; n <= 6; ++n)
    {
        int viol = 0, total = 0;
        for (int t = 0; t < 300; ++t)
        {
            const Instance I = random_instance(n, kAllT[t % 3], kAllH[(t / 3) % 3], rng);
            const RefSol ref = solve_reference(I.box, I.H, I.g);
            ASSERT_TRUE(ref.ok);
            const BoxFrame f = make_frame(I.box, I.xbar);
            if (old_halfspace_value(f, from_vec(ref.x)) < -1e-9)
                ++viol;
            ++total;
        }
        std::cout << "[ info ] n=" << n << " old-theorem violations: " << viol << "/" << total << "\n";
    }
}

// ═════════════════════════════ A. good vertices ═════════════════════════

TEST(GoodVertices, SignRuleMatchesDefinitionAndCount)
{
    std::mt19937 rng(11);
    for (int n = 1; n <= 6; ++n)
        for (TKind tk : kAllT)
            for (int t = 0; t < 40; ++t)
            {
                const Instance I = random_instance(n, tk, HKind::Dense, rng);
                const BoxFrame f = make_frame(I.box, I.xbar);
                int good = 0;
                for (const auto &sigma : all_sign_vectors(n))
                {
                    const Vec v = vertex_from_sign(f, sigma);
                    const bool by_def = is_good_by_definition(I.box, I.xbar, v);
                    EXPECT_EQ(by_def, is_good_vertex_sign(f, sigma)) << "n=" << n;
                    good += by_def;
                }
                EXPECT_EQ(good, 1 << (n - f.k())) << "n=" << n << " k=" << f.k();
                EXPECT_TRUE(is_good_by_definition(I.box, I.xbar, antipodal_vertex(f)));
                EXPECT_TRUE(is_good_by_definition(I.box, I.xbar, lemma1_vertex(f)));
            }
}

TEST(GoodVertices, ExactBoundaryTieXiBarEqualsOne)
{
    // ȳ₀ = u₀ exactly (ξ̄₀ = 1, not violated), ȳ₁ > u₁ (violated), rest inside.
    const int n = 4;
    const AffineBox box = AffineBox::axis_aligned(Vec(n, -1.0), Vec(n, 1.0));
    const Vec xbar = {1.0, 2.5, 0.25, -0.5};
    const BoxFrame f = make_frame(box, xbar);
    EXPECT_EQ(f.violated, (std::vector<int>{1}));
    EXPECT_DOUBLE_EQ(f.xibar[0], 1.0);
    int good = 0;
    for (const auto &sigma : all_sign_vectors(n))
    {
        const bool by_def = is_good_by_definition(box, xbar, vertex_from_sign(f, sigma));
        EXPECT_EQ(by_def, is_good_vertex_sign(f, sigma));
        good += by_def;
    }
    EXPECT_EQ(good, 1 << (n - 2)); // two coordinates have ξ̄ᵢ ≥ 1, not 2^{n−k} = 2^{n−1}
}

TEST(GoodVertices, SolverFastPathReturnsAntipodalVertex)
{
    // Documents which good vertex the box fast path actually uses.
    std::mt19937 rng(13);
    int differs_from_lemma1 = 0;
    for (int n = 2; n <= 8; ++n)
        for (int t = 0; t < 30; ++t)
        {
            const Instance I = random_instance(n, TKind::Identity, HKind::Dense, rng);
            const BoxFrame f = make_frame(I.box, I.xbar);
            Vec A, b;
            I.box.halfspace(A, b);
            Config cfg;
            cfg.verbosity = -1;
            internal::Initializer init(I.H.data(), A.data(), b.data(), I.xbar, 2 * n, n, cfg);
            int it = 0, ve = 0;
            const Vec v = init.find_good_vertex(antipodal_vertex(f), it, ve);
            EXPECT_EQ(ve, 1);
            const Vec va = antipodal_vertex(f), vl = lemma1_vertex(f);
            for (int i = 0; i < n; ++i)
                EXPECT_NEAR(v[i], va[i], 1e-12);
            if (va != vl)
                ++differs_from_lemma1;
        }
    std::cout << "[ info ] instances where antipodal ≠ Lemma-1 vertex: " << differs_from_lemma1 << "\n";
}

// ═════════════════════════════ B. visible facet ═════════════════════════

TEST(VisibleFacetLemma, OptimumLiesOnAVisibleFacet)
{
    std::mt19937 rng(17);
    for (int n = 1; n <= 6; ++n)
        for (TKind tk : kAllT)
            for (HKind hk : kAllH)
                for (int t = 0; t < 40; ++t)
                {
                    const Instance I = random_instance(n, tk, hk, rng);
                    const RefSol ref = solve_reference(I.box, I.H, I.g);
                    ASSERT_TRUE(ref.ok);
                    const BoxFrame f = make_frame(I.box, I.xbar);
                    const auto vis = visible_facets_containing(f, from_vec(ref.x), kFacetTol);
                    EXPECT_FALSE(vis.empty()) << "n=" << n << " k=" << f.k();
                    // Same statement read directly in y: y★ᵢ sits on the bound ȳ crosses.
                    bool y_form = false;
                    for (int i : f.violated)
                    {
                        const double bound = f.ybar[i] > f.upper[i] ? f.upper[i] : f.lower[i];
                        y_form |= std::abs(ref.y(i) - bound) <= 1e-9 * std::max(1.0, std::abs(bound));
                    }
                    EXPECT_TRUE(y_form);
                }
}

// ═════════════════════════════ C. corrected cut ═════════════════════════

TEST(CorrectedCut, ContainsOptimum)
{
    std::mt19937 rng(19);
    for (int n = 1; n <= 6; ++n)
        for (TKind tk : kAllT)
            for (HKind hk : kAllH)
                for (int t = 0; t < 40; ++t)
                {
                    const Instance I = random_instance(n, tk, hk, rng);
                    const RefSol ref = solve_reference(I.box, I.H, I.g);
                    ASSERT_TRUE(ref.ok);
                    const BoxFrame f = make_frame(I.box, I.xbar);
                    const Vec x = from_vec(ref.x);
                    EXPECT_GE(corrected_cut_margin(f, x), -1e-9);
                    // η-form and ξ-form agree.
                    const Vec xi = to_xi(f, x);
                    double sv = 0.0;
                    for (int i : f.violated)
                        sv += xi[i];
                    EXPECT_NEAR(corrected_cut_value(f, x), sv, 1e-9);
                }
}

TEST(CorrectedCut, IsConvexHullOfVisibleFacets)
{
    // In ξ-coordinates with V = {0..k−1}.  P₊ = B ∩ {Σ_V ξ ≥ 2−k} equals
    // conv(∪ visible facets) iff (i) a box vertex satisfies the cut exactly when
    // it lies on a visible facet and (ii) the cut hyperplane creates no new
    // vertex, i.e. it never strictly separates the endpoints of a box edge.
    for (int n = 1; n <= 7; ++n)
        for (int k = 1; k <= n; ++k)
        {
            Vec xbar(n, 0.5);
            for (int i = 0; i < k; ++i)
                xbar[i] = 2.0;
            const BoxFrame f = make_frame(AffineBox::axis_aligned(Vec(n, -1.0), Vec(n, 1.0)), xbar);
            ASSERT_EQ(f.k(), k);
            const double thr = corrected_cut_threshold(f);
            const auto verts = all_sign_vectors(n);
            for (const auto &sigma : verts)
            {
                const Vec v(sigma.begin(), sigma.end());
                bool on_visible = false;
                for (int i = 0; i < k; ++i)
                    on_visible |= (sigma[i] == 1);
                EXPECT_EQ(corrected_cut_value(f, v) >= thr - 1e-12, on_visible) << n << "," << k;
                for (int j = 0; j < n; ++j)
                {
                    if (sigma[j] != -1)
                        continue;
                    Vec w = v;
                    w[j] = 1.0;
                    const double a = corrected_cut_value(f, v) - thr;
                    const double b = corrected_cut_value(f, w) - thr;
                    EXPECT_FALSE((a < -1e-12 && b > 1e-12) || (a > 1e-12 && b < -1e-12))
                        << "cut creates a new vertex on an edge, n=" << n << " k=" << k;
                }
            }
            if (k == 2)
                EXPECT_NEAR(corrected_cut_value(f, f.c0), thr, 1e-15); // passes through c₀
        }
}

TEST(CorrectedCut, RemovedVolumeFraction)
{
    // Removed fraction of B is 1/k! : the whole box for k = 1 (P₊ is a facet),
    // exactly half for k = 2, less than half for k ≥ 3.
    std::mt19937 rng(23);
    std::uniform_real_distribution<> U(-1.0, 1.0);
    const int n = 5, samples = 200000;
    for (int k = 1; k <= n; ++k)
    {
        Vec xbar(n, 0.5);
        for (int i = 0; i < k; ++i)
            xbar[i] = 2.0;
        const BoxFrame f = make_frame(AffineBox::axis_aligned(Vec(n, -1.0), Vec(n, 1.0)), xbar);
        int removed = 0;
        Vec x(n);
        for (int s = 0; s < samples; ++s)
        {
            for (double &xi : x)
                xi = U(rng);
            removed += corrected_cut_margin(f, x) < 0.0;
        }
        const double p = 1.0 / factorial(k);
        const double est = double(removed) / samples;
        const double sd = std::sqrt(std::max(p * (1 - p), 1e-12) / samples);
        EXPECT_NEAR(est, p, 5 * sd + 1e-12) << "k=" << k;
        if (k >= 3)
            EXPECT_LT(est, 0.5);
    }
}

// ═════════════════════════════ D. tightness ═════════════════════════════

TEST(Tightness, EveryVisibleFacetPointIsOptimalForSomeH)
{
    std::mt19937 rng(29);
    std::uniform_real_distribution<> U(-1.0, 1.0);
    std::uniform_int_distribution<> snap(0, 3); // 0: −1, 1: +1, else interior
    for (int n = 1; n <= 5; ++n)
        for (TKind tk : kAllT)
            for (double eps : {1e-3, 0.1, 1.0, 10.0})
                for (int t = 0; t < 25; ++t)
                {
                    const Instance I = random_instance(n, tk, HKind::Dense, rng);
                    const BoxFrame f = make_frame(I.box, I.xbar);
                    const int i = f.violated[std::uniform_int_distribution<>(0, f.k() - 1)(rng)];
                    Vec p(n);
                    for (int j = 0; j < n; ++j)
                    {
                        const int sn = snap(rng);
                        p[j] = sn == 0 ? -1.0 : sn == 1 ? 1.0 : U(rng);
                    }
                    p[i] = 1.0;

                    // H̃ a = eᵢ and H̃ ≻ 0
                    const Vec Ht = tightness_hessian_xi(f, p, i, eps);
                    const Eigen::MatrixXd Htm = to_mat(Ht, n);
                    Eigen::VectorXd a = to_vec(f.xibar) - to_vec(p);
                    EXPECT_GT(a(i), 0.0);
                    const Eigen::VectorXd Ha = Htm * a;
                    for (int j = 0; j < n; ++j)
                        EXPECT_NEAR(Ha(j), j == i ? 1.0 : 0.0, 1e-9);
                    EXPECT_GT(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(Htm).eigenvalues().minCoeff(), 0.0);

                    // Back in x: optimum is D⁻¹p + c₀.
                    const Vec H = symmetrize(tightness_hessian(f, p, i, eps), n);
                    const Eigen::VectorXd g = -to_mat(H, n) * to_vec(I.xbar);
                    const RefSol ref = solve_reference(I.box, H, from_vec(g));
                    ASSERT_TRUE(ref.ok) << "n=" << n << " eps=" << eps;
                    const Vec target = from_xi(f, p);
                    const double sc = 1.0 + to_vec(target).cwiseAbs().maxCoeff();
                    for (int j = 0; j < n; ++j)
                        EXPECT_NEAR(ref.x(j), target[j], 1e-7 * sc) << "n=" << n << " eps=" << eps;
                }
}

TEST(Tightness, OptimumCanLieOnTheFacetOppositeF0)
{
    // "The facet opposite F₀ is never needed" cannot mean x★ avoids it: with
    // ξ̄ = (3, 2) and p = (−1, 1), x★ sits on {ξ_{i₀} = −1}.  It is however also
    // on the visible facet {ξ₁ = 1}, which shares a ridge with F₀.
    const AffineBox box = AffineBox::axis_aligned(Vec(2, -1.0), Vec(2, 1.0));
    const Vec xbar = {3.0, 2.0};
    const BoxFrame f = make_frame(box, xbar);
    ASSERT_EQ(predicted_exit_wall(f).coord, 0);
    const Vec p = {-1.0, 1.0};
    const Vec H = symmetrize(tightness_hessian(f, p, 1, 1.0), 2);
    const Eigen::VectorXd g = -to_mat(H, 2) * to_vec(xbar);
    const RefSol ref = solve_reference(box, H, from_vec(g));
    ASSERT_TRUE(ref.ok);
    EXPECT_NEAR(ref.x(0), -1.0, 1e-12); // on the facet opposite F₀
    EXPECT_NEAR(ref.x(1), 1.0, 1e-12);
    EXPECT_EQ(visible_facets_containing(f, from_vec(ref.x), kFacetTol), (std::vector<int>{1}));
}

// ═════════════════════════════ E. exit facet ════════════════════════════

TEST(ExitFacet, AntipodalRayTimesAndArgmax)
{
    std::mt19937 rng(31);
    for (int n = 1; n <= 7; ++n)
        for (TKind tk : kAllT)
            for (int t = 0; t < 40; ++t)
            {
                const Instance I = random_instance(n, tk, HKind::Dense, rng);
                const BoxFrame f = make_frame(I.box, I.xbar);
                // Independent ray trace in y from ŷ = μ − s∘h.
                const Eigen::MatrixXd Tinv = to_mat(I.box.T, n).inverse();
                const Eigen::VectorXd yhat = Tinv * (to_vec(antipodal_vertex(f)) - to_vec(I.box.c));
                const Eigen::VectorXd ybar = Tinv * (to_vec(I.xbar) - to_vec(I.box.c));
                const Vec tp = antipodal_exit_times(f);
                int argmin_t = 0, argmax_xi = 0;
                for (int i = 0; i < n; ++i)
                {
                    const double d = ybar(i) - yhat(i);
                    ASSERT_NE(d, 0.0);
                    const double bound = d > 0 ? I.box.upper[i] : I.box.lower[i];
                    const double ti = (bound - yhat(i)) / d;
                    EXPECT_NEAR(ti, tp[i], 1e-9 * std::max(1.0, ti));
                    EXPECT_EQ(ti < 1.0, f.is_violated(i));
                    // Wall reached is the {ξᵢ = 1} wall.
                    EXPECT_EQ(d > 0, xi_plus_wall(f, i).upper);
                    if (tp[i] < tp[argmin_t]) argmin_t = i;
                    if (f.xibar[i] > f.xibar[argmax_xi]) argmax_xi = i;
                }
                EXPECT_EQ(argmin_t, argmax_xi);
                const Wall w = predicted_exit_wall(f);
                EXPECT_EQ(w.coord, argmax_xi);
                EXPECT_TRUE(f.is_violated(w.coord));
                const bool expect_upper = f.ybar[w.coord] > f.upper[w.coord];
                EXPECT_EQ(w.upper, expect_upper); // F₀ is the visible facet of i₀
            }
}

TEST(ExitFacet, SameWallFromEveryGoodVertexAndLemma1Vertex)
{
    std::mt19937 rng(37);
    for (int n = 1; n <= 6; ++n)
        for (TKind tk : kAllT)
            for (int t = 0; t < 30; ++t)
            {
                const Instance I = random_instance(n, tk, HKind::Dense, rng);
                const BoxFrame f = make_frame(I.box, I.xbar);
                const Wall w0 = predicted_exit_wall(f);
                std::vector<Vec> starts = {lemma1_vertex(f)};
                for (const auto &sigma : all_sign_vectors(n))
                    if (is_good_vertex_sign(f, sigma))
                        starts.push_back(vertex_from_sign(f, sigma));
                for (const Vec &v : starts)
                {
                    const auto walls = exit_walls_by_ray(f, v);
                    ASSERT_EQ(walls.size(), 1u);
                    EXPECT_EQ(walls[0].coord, w0.coord);
                    EXPECT_EQ(walls[0].upper, w0.upper);
                    EXPECT_LT(walls[0].t, 1.0);
                }
            }
}

TEST(ExitFacet, TieExitsThroughRidgeOfVisibleFacets)
{
    const int n = 4;
    const AffineBox box = AffineBox::axis_aligned(Vec(n, -1.0), Vec(n, 1.0));
    const Vec xbar = {3.0, 0.2, -3.0, 1.5}; // ξ̄ = (3, 0.2, 3, 1.5), tie on {0, 2}
    const BoxFrame f = make_frame(box, xbar);
    for (const auto &sigma : all_sign_vectors(n))
    {
        if (!is_good_vertex_sign(f, sigma))
            continue;
        const auto walls = exit_walls_by_ray(f, vertex_from_sign(f, sigma));
        ASSERT_EQ(walls.size(), 2u);
        EXPECT_EQ(walls[0].coord, 0);
        EXPECT_TRUE(walls[0].upper);
        EXPECT_EQ(walls[1].coord, 2);
        EXPECT_FALSE(walls[1].upper);
    }
}

TEST(ExitFacet, F0EqualOrAdjacentToAnOptimalFacet)
{
    std::mt19937 rng(41);
    for (int n = 1; n <= 6; ++n)
        for (TKind tk : kAllT)
            for (HKind hk : kAllH)
                for (int t = 0; t < 25; ++t)
                {
                    const Instance I = random_instance(n, tk, hk, rng);
                    const RefSol ref = solve_reference(I.box, I.H, I.g);
                    ASSERT_TRUE(ref.ok);
                    const BoxFrame f = make_frame(I.box, I.xbar);
                    const int i0 = predicted_exit_wall(f).coord;
                    const auto vis = visible_facets_containing(f, from_vec(ref.x), kFacetTol);
                    ASSERT_FALSE(vis.empty());
                    // Each F★ = {ξᵢ = 1} is F₀ or meets F₀ in the ridge {ξ_{i₀} = ξᵢ = 1},
                    // which is non-empty because i ≠ i₀ are different coordinates.
                    for (int i : vis)
                    {
                        Vec r(n, 0.0);
                        r[i0] = 1.0;
                        r[i] = 1.0;
                        const Vec xr = from_xi(f, r);
                        const auto on = visible_facets_containing(f, xr, 1e-9);
                        EXPECT_TRUE(std::count(on.begin(), on.end(), i0) == 1);
                        EXPECT_TRUE(std::count(on.begin(), on.end(), i) == 1);
                    }
                }
}

TEST(ExitFacet, SolverInitialFaceIsPredictedWall)
{
    // Axis-aligned box: GaasQP's FaceHopper seeds the working set with {F₀}.
    std::mt19937 rng(43);
    for (int n = 1; n <= 10; ++n)
        for (HKind hk : kAllH)
            for (int t = 0; t < 20; ++t)
            {
                const Instance I = random_instance(n, TKind::Identity, hk, rng);
                const BoxFrame f = make_frame(I.box, I.xbar);
                Vec A, b;
                I.box.halfspace(A, b); // rows [0,n): yᵢ ≤ uᵢ, rows [n,2n): −yᵢ ≤ −ℓᵢ
                Config cfg;
                cfg.verbosity = -1;
                internal::Initializer init(I.H.data(), A.data(), b.data(), I.xbar, 2 * n, n, cfg);
                int it = 0, ve = 0;
                const Vec v = init.find_good_vertex(antipodal_vertex(f), it, ve);
                internal::FaceHopper fh(I.H.data(), A.data(), b.data(), I.xbar, 2 * n, n, cfg);
                const auto face = fh.find_initial_face(v);
                ASSERT_EQ(face.size(), 1u);
                const Wall w = predicted_exit_wall(f);
                const int expected_row = w.upper ? w.coord : n + w.coord;
                EXPECT_EQ(face[0], expected_row) << "n=" << n;
            }
}

// ═════════════════════════════ F. H-metric half ═════════════════════════

TEST(HMetricHalf, ContainsOptimumAndBall)
{
    std::mt19937 rng(47);
    for (int n = 1; n <= 6; ++n)
        for (TKind tk : kAllT)
            for (HKind hk : kAllH)
                for (int t = 0; t < 40; ++t)
                {
                    const Instance I = random_instance(n, tk, hk, rng);
                    const RefSol ref = solve_reference(I.box, I.H, I.g);
                    ASSERT_TRUE(ref.ok);
                    const BoxFrame f = make_frame(I.box, I.xbar);
                    const Vec x = from_vec(ref.x);
                    const Eigen::MatrixXd H = to_mat(I.H, n);
                    const Eigen::VectorXd a = to_vec(I.xbar) - to_vec(f.c0);
                    const Eigen::VectorXd b = ref.x - to_vec(f.c0);
                    const double sc = 1e-9 * std::max(1.0, a.dot(H * a));
                    EXPECT_GE(h_halfspace_margin(f, I.H, x), -sc);
                    EXPECT_GT(a.dot(H * b), 0.0);   // strict: x★ ≠ c₀
                    EXPECT_GT(b.dot(H * b), 0.0);
                    EXPECT_GE(h_ball_margin(f, I.H, x), -1e-9 * std::max(1.0, std::sqrt(a.dot(H * a))));
                }
}

TEST(HMetricHalf, AntipodalSideCriterion)
{
    std::mt19937 rng(53);
    int dense_fail = 0, dense_total = 0;
    for (int n = 1; n <= 6; ++n)
        for (TKind tk : kAllT)
            for (int t = 0; t < 40; ++t)
            {
                Instance I = random_instance(n, tk, HKind::Dense, rng);
                BoxFrame f = make_frame(I.box, I.xbar);
                // ⟨x̄ − c₀, v̂ − c₀⟩_H = −1ᵀH̃ξ̄
                const double side = antipodal_side_value(f, I.H);
                const double crit = antipodal_side_criterion(f, I.H);
                EXPECT_NEAR(side, -crit, 1e-9 * std::max(1.0, std::abs(crit)));
                dense_fail += !(crit > 0.0);
                ++dense_total;

                // Diagonal H̃ ⇒ criterion > 0.
                std::uniform_real_distribution<> U(0.1, 5.0);
                Vec Ht(n * n, 0.0);
                for (int i = 0; i < n; ++i)
                    Ht[i * n + i] = U(rng);
                const Vec Hd = symmetrize(original_hessian(f, Ht), n);
                EXPECT_GT(antipodal_side_criterion(f, Hd), 0.0);
                EXPECT_LT(antipodal_side_value(f, Hd), 0.0);
            }
    std::cout << "[ info ] dense H: antipodal vertex NOT in opposite half in "
              << dense_fail << "/" << dense_total << " instances\n";

    // The quoted counterexample: ξ̄ = (2, 0), H̃ = [[1, −2], [−2, 5]].
    const AffineBox box = AffineBox::axis_aligned(Vec(2, -1.0), Vec(2, 1.0));
    const BoxFrame f = make_frame(box, Vec{2.0, 0.0});
    const Vec H = {1.0, -2.0, -2.0, 5.0};
    EXPECT_EQ(transformed_hessian(f, H), H); // D = I here
    EXPECT_DOUBLE_EQ(antipodal_side_criterion(f, H), -2.0);
    EXPECT_GT(antipodal_side_value(f, H), 0.0); // v̂ on the same side as x̄
    // x★ is still in the H-half.
    const Eigen::VectorXd g = -to_mat(H, 2) * to_vec(f.xbar);
    const RefSol ref = solve_reference(box, H, from_vec(g));
    ASSERT_TRUE(ref.ok);
    EXPECT_GE(h_halfspace_margin(f, H, from_vec(ref.x)), -1e-12);
}

// ═════════════════════════════ solver's antipodal frame ═════════════════

namespace
{
    // Box rows in a shuffled order, optionally with looser duplicates, so the
    // frame has to find the tightest row per bound.
    struct BoxRows
    {
        Vec A, b;
        std::vector<int> ub_row, lb_row; // tightest row per coordinate
    };

    BoxRows make_box_rows(const Vec &lo, const Vec &hi, bool duplicates, std::mt19937 &rng)
    {
        const int n = static_cast<int>(lo.size());
        struct R { int var; int sign; double b; bool tight; };
        std::vector<R> rows;
        for (int j = 0; j < n; ++j)
        {
            rows.push_back({j, +1, hi[j], true});
            rows.push_back({j, -1, -lo[j], true});
            if (duplicates)
            {
                rows.push_back({j, +1, hi[j] + 0.5, false});
                rows.push_back({j, -1, -lo[j] + 0.25, false});
            }
        }
        std::shuffle(rows.begin(), rows.end(), rng);
        BoxRows out;
        const int m = static_cast<int>(rows.size());
        out.A.assign(m * n, 0.0);
        out.b.resize(m);
        out.ub_row.assign(n, -1);
        out.lb_row.assign(n, -1);
        for (int r = 0; r < m; ++r)
        {
            out.A[r * n + rows[r].var] = rows[r].sign;
            out.b[r] = rows[r].b;
            if (rows[r].tight)
                (rows[r].sign > 0 ? out.ub_row : out.lb_row)[rows[r].var] = r;
        }
        return out;
    }
} // namespace

TEST(SolverAntipodalFrame, MatchesGeometryModule)
{
    std::mt19937 rng(59);
    for (int n = 1; n <= 9; ++n)
        for (bool dup : {false, true})
            for (int t = 0; t < 30; ++t)
            {
                const Instance I = random_instance(n, TKind::Identity, HKind::Dense, rng);
                const BoxFrame f = make_frame(I.box, I.xbar);
                // T = I but c ≠ 0: the x-space bounds are ℓ + c and u + c.
                Vec lo(n), hi(n);
                for (int i = 0; i < n; ++i)
                {
                    lo[i] = I.box.lower[i] + I.box.c[i];
                    hi[i] = I.box.upper[i] + I.box.c[i];
                }
                const BoxRows R = make_box_rows(lo, hi, dup, rng);
                const int m = static_cast<int>(R.b.size());
                Config cfg;
                cfg.verbosity = -1;
                internal::Initializer init(I.H.data(), R.A.data(), R.b.data(), I.xbar, m, n, cfg);
                const internal::AntipodalFrame &af = init.antipodal_frame();
                ASSERT_TRUE(af.valid);
                EXPECT_EQ(af.violated, f.violated);
                const Vec vref = antipodal_vertex(f); // via μ − s∘h and T: last-bit rounding
                for (int i = 0; i < n; ++i)
                {
                    EXPECT_NEAR(af.v_hat[i], vref[i], 1e-12 * (1 + std::abs(vref[i])));
                    // The solver's v̂ is the bound itself, bit for bit.
                    EXPECT_TRUE(af.v_hat[i] == lo[i] || af.v_hat[i] == hi[i]);
                    EXPECT_EQ(af.s[i], f.s[i]);
                    EXPECT_NEAR(af.zeta_bar[i], f.xibar[i] + 1.0, 1e-12 * (1 + f.xibar[i]));
                    EXPECT_GE(af.zeta_bar[i], 1.0);
                }
                const Wall w = predicted_exit_wall(f);
                EXPECT_EQ(af.i0, w.coord);
                EXPECT_EQ(af.exit_row, w.upper ? R.ub_row[w.coord] : R.lb_row[w.coord]);

                // Phase I returns v̂ and Phase II seeds {F₀}, by frame and by ray alike.
                int it = 0, ve = 0;
                const Vec v = init.find_good_vertex(af.v_hat, it, ve);
                EXPECT_EQ(v, af.v_hat);
                internal::FaceHopper by_ray(I.H.data(), R.A.data(), R.b.data(), I.xbar, m, n, cfg);
                internal::FaceHopper by_frame(I.H.data(), R.A.data(), R.b.data(), I.xbar, m, n, cfg);
                by_frame.set_antipodal_frame(&af);
                const auto face_ray = by_ray.find_initial_face(v);
                const auto face_frame = by_frame.find_initial_face(v);
                EXPECT_EQ(face_frame, (std::vector<int>{af.exit_row}));
                ASSERT_EQ(face_ray.size(), 1u);
                // Duplicated rows: the ray may stop at the tight row only.
                EXPECT_EQ(face_ray, face_frame) << "n=" << n << " dup=" << dup;
            }
}

TEST(SolverAntipodalFrame, TieAtMidpointPicksLowerBound)
{
    // x̄ᵢ = μᵢ ⇒ sᵢ = +1 ⇒ v̂ᵢ = ℓᵢ (ζ̄ᵢ = 1); previously the fast path chose uᵢ.
    const int n = 3;
    const Vec lo = {-1, 0, -2}, hi = {1, 4, 2};
    std::mt19937 rng(61);
    const BoxRows R = make_box_rows(lo, hi, false, rng);
    const Vec xbar = {0.0, 9.0, 0.0};
    const Vec H = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Config cfg;
    internal::Initializer init(H.data(), R.A.data(), R.b.data(), xbar, 2 * n, n, cfg);
    const auto &af = init.antipodal_frame();
    EXPECT_EQ(af.v_hat, (Vec{-1.0, 0.0, -2.0}));
    EXPECT_EQ(af.zeta_bar, (Vec{1.0, 4.5, 1.0}));
    EXPECT_EQ(af.violated, (std::vector<int>{1}));
    EXPECT_EQ(af.exit_row, R.ub_row[1]);
}

TEST(SolverAntipodalFrame, InvalidWithoutFullBox)
{
    // x₂ has no upper bound: no frame, no box fast path.
    const int n = 2;
    const Vec A = {1, 0, -1, 0, 0, -1};
    const Vec b = {1, 1, 1};
    const Vec H = {1, 0, 0, 1};
    Config cfg;
    internal::Initializer init(H.data(), A.data(), b.data(), Vec{3.0, 0.0}, 3, n, cfg);
    EXPECT_FALSE(init.antipodal_frame().valid);
}

// ═════════════════════════════ solver certificate ═══════════════════════

namespace
{
    Problem to_problem(const Instance &I)
    {
        Problem p;
        p.n = I.box.n;
        p.Q = I.H;
        p.g = I.g;
        I.box.halfspace(p.A, p.b);
        p.m = 2 * p.n;
        return p;
    }
} // namespace

TEST(SolverCertificate, AxisAlignedBoxes)
{
    std::mt19937 rng(67);
    Config cfg;
    cfg.verbosity = -1;
    int solved = 0;
    for (int n = 1; n <= 8; ++n)
        for (HKind hk : kAllH)
            for (int t = 0; t < 30; ++t)
            {
                const Instance I = random_instance(n, TKind::Identity, hk, rng);
                const Solution sol = Solver(cfg).solve(to_problem(I));
                ASSERT_EQ(sol.status, Status::SUCCESS);
                const RefSol ref = solve_reference(I.box, I.H, I.g);
                ASSERT_TRUE(ref.ok);
                for (int i = 0; i < n; ++i)
                    ASSERT_NEAR(sol.x[i], ref.x(i), 1e-6 * (1 + std::abs(ref.x(i))));
                EXPECT_GE(sol.violated_active_count, 1) << "n=" << n;
                ASSERT_FALSE(std::isnan(sol.visible_cut_margin));
                const BoxFrame f = make_frame(I.box, I.xbar);
                EXPECT_NEAR(sol.visible_cut_margin, corrected_cut_margin(f, sol.x), 1e-9);
                EXPECT_GE(sol.visible_cut_margin, -1e-9);
                ++solved;
            }
    RecordProperty("solved", solved);
}

TEST(SolverCertificate, DenseAffineBoxesUseGeneralLemma)
{
    // General path (no frame): the margin is not defined, the count still is.
    std::mt19937 rng(71);
    Config cfg;
    cfg.verbosity = -1;
    for (int n = 2; n <= 5; ++n)
        for (int t = 0; t < 15; ++t)
        {
            const Instance I = random_instance(n, TKind::Dense, HKind::Dense, rng);
            const Solution sol = Solver(cfg).solve(to_problem(I));
            ASSERT_EQ(sol.status, Status::SUCCESS);
            EXPECT_GE(sol.violated_active_count, 1) << "n=" << n;
            EXPECT_TRUE(std::isnan(sol.visible_cut_margin));
        }
}

TEST(SolverCertificate, CounterexampleAndFeasibleXbar)
{
    const Instance I = counterexample(3);
    Config cfg;
    cfg.verbosity = -1;
    const Solution sol = Solver(cfg).solve(to_problem(I));
    ASSERT_EQ(sol.status, Status::SUCCESS);
    EXPECT_EQ(sol.violated_active_count, 1); // x₁ ≤ 1, multiplier 1
    EXPECT_NEAR(sol.visible_cut_margin, 0.0, 1e-12);

    Instance J = I;
    J.xbar = {0.5, 0.0, 0.0};
    const Eigen::VectorXd g = -to_mat(J.H, 3) * to_vec(J.xbar);
    J.g = from_vec(g);
    const Solution s2 = Solver(cfg).solve(to_problem(J));
    ASSERT_EQ(s2.status, Status::SUCCESS);
    EXPECT_EQ(s2.violated_active_count, -1);
    EXPECT_TRUE(std::isnan(s2.visible_cut_margin));
}

TEST(SolverCertificate, Hotstart)
{
    std::mt19937 rng(73);
    const int n = 6;
    Config cfg;
    cfg.verbosity = -1;
    const Instance I0 = random_instance(n, TKind::Identity, HKind::Dense, rng);
    Problem p0 = to_problem(I0);
    Solver hot(cfg);
    hot.setup(n, p0.m, p0.Q.data(), p0.A.data());
    for (int t = 0; t < 30; ++t)
    {
        Instance I = random_instance(n, TKind::Identity, HKind::Dense, rng);
        I.H = I0.H; // same Q, new box and x̄
        const Eigen::VectorXd g = -to_mat(I.H, n) * to_vec(I.xbar);
        I.g = from_vec(g);
        Problem p = to_problem(I);
        const Solution hs = hot.solve_hotstart(p.g.data(), p.b.data());
        const Solution fresh = Solver(cfg).solve(p);
        ASSERT_EQ(hs.status, Status::SUCCESS);
        EXPECT_EQ(hs.violated_active_count, fresh.violated_active_count);
        EXPECT_NEAR(hs.visible_cut_margin, fresh.visible_cut_margin, 1e-9);
        EXPECT_GE(hs.violated_active_count, 1);
    }
}

// ═════════════════════════════ box kernel ═══════════════════════════════

// Solver::solve sends pure boxes through BoxKernel.  It must reproduce the
// row-based path (Initializer frame + FaceHopper plain loop) pivot for pivot.
TEST(BoxKernel, SamePivotsAndFaceAsRowBasedPath)
{
    std::mt19937 rng(79);
    int compared = 0;
    for (int n = 1; n <= 12; ++n)
        for (HKind hk : kAllH)
            for (bool dup : {false, true})
                for (int t = 0; t < 12; ++t)
                {
                    const Instance I = random_instance(n, TKind::Identity, hk, rng);
                    Vec lo(n), hi(n);
                    for (int i = 0; i < n; ++i)
                    {
                        lo[i] = I.box.lower[i] + I.box.c[i];
                        hi[i] = I.box.upper[i] + I.box.c[i];
                    }
                    const BoxRows R = make_box_rows(lo, hi, dup, rng); // shuffled rows
                    const int m = static_cast<int>(R.b.size());
                    Config cfg;
                    cfg.verbosity = -1;

                    // Row-based reference.
                    internal::Initializer init(I.H.data(), R.A.data(), R.b.data(), I.xbar, m, n, cfg);
                    int it1 = 0, ve = 0;
                    const Vec v = init.find_good_vertex(Vec(), it1, ve);
                    Vec chol(n * n);
                    internal::LinearAlgebra::cholesky(I.H.data(), chol.data(), n);
                    internal::FaceHopper fh(I.H.data(), R.A.data(), R.b.data(), I.xbar, m, n, cfg, chol, false);
                    fh.set_antipodal_frame(&init.antipodal_frame());
                    const auto f0 = fh.find_initial_face(v);
                    int piv = 0, hops = 0;
                    const auto face = fh.find_optimal_face(f0, piv, hops);

                    // Kernel, through the public API.
                    Problem p;
                    p.n = n;
                    p.m = m;
                    p.Q = I.H;
                    p.g = I.g;
                    p.A = R.A;
                    p.b = R.b;
                    const Solution sol = Solver(cfg).solve(p);
                    ASSERT_EQ(sol.status, Status::SUCCESS);
                    if (dup)
                    {
                        if (n > 8)
                            continue; // the reference enumerates 3ⁿ patterns
                        // The row-based path treats a looser duplicate row as a
                        // separate constraint and can activate both rows of one
                        // variable (a singular working set).  The kernel merges
                        // duplicates, so check it against the independent
                        // reference instead.
                        const RefSol ref = solve_reference(I.box, I.H, I.g);
                        ASSERT_TRUE(ref.ok);
                        for (int i = 0; i < n; ++i)
                            EXPECT_NEAR(sol.x[i], ref.x(i), 1e-7 * (1 + std::abs(ref.x(i))));
                        for (int r : sol.active_set)
                            EXPECT_TRUE(std::count(R.ub_row.begin(), R.ub_row.end(), r) +
                                        std::count(R.lb_row.begin(), R.lb_row.end(), r) == 1)
                                << "active row is not a tightest row";
                        continue;
                    }
                    ASSERT_TRUE(fh.has_last_solution()) << "n=" << n;
                    EXPECT_EQ(sol.phase2_iterations, piv) << "n=" << n;
                    EXPECT_EQ(sol.phase2_facet_hops, hops) << "n=" << n;
                    EXPECT_EQ(sol.active_set, face) << "n=" << n;
                    for (int i = 0; i < n; ++i)
                    {
                        EXPECT_NEAR(sol.x[i], fh.last_x()[i], 1e-12 * (1 + std::abs(sol.x[i])));
                        EXPECT_EQ(sol.x_phase1[i], v[i]);
                    }
                    ++compared;
                }
    EXPECT_GT(compared, 300);
}

TEST(BoxKernel, BoundsAndExplicitRowsAgree)
{
    // The same box given as lb/ub, as explicit rows, and half-and-half.
    std::mt19937 rng(83);
    Config cfg;
    cfg.verbosity = -1;
    for (int n = 2; n <= 9; ++n)
        for (int t = 0; t < 15; ++t)
        {
            const Instance I = random_instance(n, TKind::Identity, HKind::Dense, rng);
            Vec lo(n), hi(n);
            for (int i = 0; i < n; ++i)
            {
                lo[i] = I.box.lower[i] + I.box.c[i];
                hi[i] = I.box.upper[i] + I.box.c[i];
            }
            Problem bounds;
            bounds.n = n;
            bounds.m = 0;
            bounds.Q = I.H;
            bounds.g = I.g;
            bounds.lb = lo;
            bounds.ub = hi;
            const Solution a = Solver(cfg).solve(bounds);

            Problem rows = bounds; // upper bounds as rows, lower bounds as lb
            rows.ub.clear();
            rows.m = n;
            rows.A.assign(n * n, 0.0);
            rows.b = hi;
            for (int i = 0; i < n; ++i) rows.A[i * n + i] = 1.0;
            const Solution b = Solver(cfg).solve(rows);

            ASSERT_EQ(a.status, Status::SUCCESS);
            ASSERT_EQ(b.status, Status::SUCCESS);
            EXPECT_EQ(a.phase2_iterations, b.phase2_iterations);
            EXPECT_EQ(a.active_set.size(), b.active_set.size());
            for (int i = 0; i < n; ++i)
                EXPECT_NEAR(a.x[i], b.x[i], 1e-12 * (1 + std::abs(a.x[i])));
            ASSERT_EQ(b.lambda.size(), static_cast<size_t>(n)); // duals of the explicit rows only
            for (double l : b.lambda) EXPECT_GE(l, -1e-9);
            const RefSol ref = solve_reference(I.box, I.H, I.g);
            ASSERT_TRUE(ref.ok);
            for (int i = 0; i < n; ++i)
                EXPECT_NEAR(a.x[i], ref.x(i), 1e-7 * (1 + std::abs(ref.x(i))));
        }
}

TEST(BoxKernel, NotUsedWhenABoundIsMissingOrARowIsGeneral)
{
    // These must still solve correctly through the general path.
    Config cfg;
    cfg.verbosity = -1;
    Problem p;
    p.n = 2;
    p.m = 1;
    p.Q = {2, 0, 0, 2};
    p.g = {-8, -6};
    p.A = {1, 1};
    p.b = {2}; // x + y ≤ 2 : a general row
    p.lb = {0, 0};
    p.ub = {5, 5};
    const Solution s = Solver(cfg).solve(p);
    ASSERT_EQ(s.status, Status::SUCCESS);
    EXPECT_NEAR(s.x[0], 1.5, 1e-8);
    EXPECT_NEAR(s.x[1], 0.5, 1e-8);
    EXPECT_TRUE(std::isnan(s.visible_cut_margin)); // not a pure box
}

TEST(LazyQInverse, MatchesPrecomputedRows)
{
    std::mt19937 rng(89);
    for (int n : {2, 7, 20})
    {
        const Instance I = random_instance(n, TKind::Identity, HKind::Dense, rng);
        Vec A, b;
        I.box.halfspace(A, b);
        Vec chol(n * n);
        internal::LinearAlgebra::cholesky(I.H.data(), chol.data(), n);
        Config cfg;
        internal::FaceHopper full(I.H.data(), A.data(), b.data(), I.xbar, 2 * n, n, cfg, chol, true);
        internal::FaceHopper lazy(I.H.data(), A.data(), b.data(), I.xbar, 2 * n, n, cfg, chol, false);
        EXPECT_EQ(full.get_Q_inv().size(), static_cast<size_t>(n * n));
        EXPECT_TRUE(lazy.get_Q_inv().empty()); // never hand out a partial inverse
        for (int v = n - 1; v >= 0; v -= 2)   // any order, some rows never requested
        {
            const double *r1 = full.q_inv_row(v);
            const double *r2 = lazy.q_inv_row(v);
            for (int j = 0; j < n; ++j) EXPECT_EQ(r1[j], r2[j]);
            EXPECT_EQ(lazy.q_inv_row(v), r2); // stable pointer on repeat
        }
    }
}
