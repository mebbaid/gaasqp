#include <gtest/gtest.h>
#include "geoqp/geoqp.hpp"

using namespace geoqp;

TEST(IntegrationTest, Simple2DProblem) {
    // Problem: minimize (1/2)(x^2 + y^2) - 3x - 2y
    //          subject to x >= 0, y >= 0, x + y <= 2
    // Expected solution: x* = [1.5, 0.5]
    
    Problem problem;
    problem.n = 2;
    problem.m = 3;
    
    // Q = I, g = [-3, -2]
    problem.Q = {1.0, 0.0, 0.0, 1.0};
    problem.g = {-3.0, -2.0};
    
    // Constraints: -x <= 0, -y <= 0, x + y <= 2
    problem.A = {-1.0,  0.0,
                  0.0, -1.0,
                  1.0,  1.0};
    problem.b = {0.0, 0.0, 2.0};
    
    Config config;
    config.verbosity = 0;
    
    Solver solver(config);
    Solution solution = solver.solve(problem);
    
    EXPECT_EQ(solution.status, Status::SUCCESS);
    ASSERT_EQ(solution.x.size(), 2u);
    
    EXPECT_NEAR(solution.x[0], 1.5, 1e-4);
    EXPECT_NEAR(solution.x[1], 0.5, 1e-4);
    
    // Check objective value: (1/2)(1.5^2 + 0.5^2) - 3*1.5 - 2*0.5
    // = (1/2)(2.25 + 0.25) - 4.5 - 1.0 = 1.25 - 5.5 = -4.25
    EXPECT_NEAR(solution.objective_value, -4.25, 1e-4);
}

TEST(IntegrationTest, UnconstrainedMinimumFeasible) {
    // Problem where unconstrained minimum is feasible
    // minimize (1/2)(x^2 + y^2)
    // subject to x >= -5, y >= -5
    // Expected: x* = [0, 0]
    
    Problem problem;
    problem.n = 2;
    problem.m = 2;
    
    problem.Q = {1.0, 0.0, 0.0, 1.0};
    problem.g = {0.0, 0.0};
    
    problem.A = {-1.0,  0.0,
                  0.0, -1.0};
    problem.b = {5.0, 5.0};
    
    Config config;
    config.verbosity = 0;
    
    Solver solver(config);
    Solution solution = solver.solve(problem);
    
    EXPECT_EQ(solution.status, Status::SUCCESS);
    ASSERT_EQ(solution.x.size(), 2u);
    
    EXPECT_NEAR(solution.x[0], 0.0, 1e-6);
    EXPECT_NEAR(solution.x[1], 0.0, 1e-6);
    EXPECT_NEAR(solution.objective_value, 0.0, 1e-6);
}

TEST(IntegrationTest, BoxConstraints) {
    // minimize (1/2)(x^2 + y^2) - 2x - y
    // subject to 0 <= x <= 1, 0 <= y <= 1
    // Expected: x* = [1, 1] (corner solution)
    
    Problem problem;
    problem.n = 2;
    problem.m = 4;
    
    problem.Q = {1.0, 0.0, 0.0, 1.0};
    problem.g = {-2.0, -1.0};
    
    // x >= 0, y >= 0, x <= 1, y <= 1
    problem.A = {-1.0,  0.0,
                  0.0, -1.0,
                  1.0,  0.0,
                  0.0,  1.0};
    problem.b = {0.0, 0.0, 1.0, 1.0};
    
    Config config;
    config.verbosity = 0;
    
    Solver solver(config);
    Solution solution = solver.solve(problem);
    
    EXPECT_EQ(solution.status, Status::SUCCESS);
    ASSERT_EQ(solution.x.size(), 2u);
    
    EXPECT_NEAR(solution.x[0], 1.0, 1e-4);
    EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
}

TEST(IntegrationTest, SingleConstraintActive) {
    // minimize (1/2)x^2 - 5x
    // subject to x <= 3
    // Unconstrained min: x = 5
    // Expected: x* = 3
    
    Problem problem;
    problem.n = 1;
    problem.m = 1;
    
    problem.Q = {1.0};
    problem.g = {-5.0};
    problem.A = {1.0};
    problem.b = {3.0};
    
    Config config;
    config.verbosity = 0;
    
    Solver solver(config);
    Solution solution = solver.solve(problem);
    
    EXPECT_EQ(solution.status, Status::SUCCESS);
    ASSERT_EQ(solution.x.size(), 1u);
    
    EXPECT_NEAR(solution.x[0], 3.0, 1e-4);
    
    // Objective: (1/2)*9 - 5*3 = 4.5 - 15 = -10.5
    EXPECT_NEAR(solution.objective_value, -10.5, 1e-4);
}

TEST(IntegrationTest, ThreeDimensionalProblem) {
    // minimize (1/2)(x^2 + y^2 + z^2) - x - y - z
    // subject to x + y + z <= 2, x >= 0, y >= 0, z >= 0
    // Expected by symmetry: x* = y* = z* = 2/3
    
    Problem problem;
    problem.n = 3;
    problem.m = 4;
    
    problem.Q = {1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0};
    problem.g = {-1.0, -1.0, -1.0};
    
    problem.A = { 1.0,  1.0,  1.0,
                 -1.0,  0.0,  0.0,
                  0.0, -1.0,  0.0,
                  0.0,  0.0, -1.0};
    problem.b = {2.0, 0.0, 0.0, 0.0};
    
    Config config;
    config.verbosity = 0;
    
    Solver solver(config);
    Solution solution = solver.solve(problem);
    
    EXPECT_EQ(solution.status, Status::SUCCESS);
    ASSERT_EQ(solution.x.size(), 3u);
    
    EXPECT_NEAR(solution.x[0], 2.0/3.0, 1e-3);
    EXPECT_NEAR(solution.x[1], 2.0/3.0, 1e-3);
    EXPECT_NEAR(solution.x[2], 2.0/3.0, 1e-3);
}

// Hot start must agree with a fresh solve.  Regression: FaceHopper::update and
// Initializer::update used to keep A·x̄ and the box bounds from setup(), where
// they were computed from placeholder data, and returned SUCCESS with a wrong x.
TEST(IntegrationTest, HotstartMatchesFreshSolve)
{
    const int n = 6, m = 2 * n;
    unsigned seed = 12345u;
    auto rnd = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return (seed >> 8) / double(1u << 24) - 0.5; // uniform [−0.5, 0.5)
    };

    std::vector<double> Q(n * n, 0.0), R(n * n);
    for (auto &r : R) r = 2.0 * rnd();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                Q[i * n + j] += R[k * n + i] * R[k * n + j];
    for (int i = 0; i < n; ++i) Q[i * n + i] += 0.5;

    std::vector<double> A(m * n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        A[i * n + i] = 1.0;        // x_i ≤ u_i
        A[(n + i) * n + i] = -1.0; // −x_i ≤ −ℓ_i
    }

    Config cfg;
    cfg.verbosity = -1;
    Solver hot(cfg);
    hot.setup(n, m, Q.data(), A.data());

    for (int t = 0; t < 40; ++t)
    {
        std::vector<double> g(n), b(m);
        for (auto &v : g) v = 12.0 * rnd();
        for (int i = 0; i < n; ++i)
        {
            const double c = rnd(), w = 0.6 + rnd();
            b[i] = c + w;       // u_i
            b[n + i] = -(c - w); // −ℓ_i
        }

        Problem p;
        p.n = n;
        p.m = m;
        p.Q = Q;
        p.g = g;
        p.A = A;
        p.b = b;
        const Solution ref = Solver(cfg).solve(p);
        const Solution hs = hot.solve_hotstart(g.data(), b.data());
        ASSERT_EQ(ref.status, Status::SUCCESS);
        ASSERT_EQ(hs.status, Status::SUCCESS);
        for (int i = 0; i < n; ++i)
            EXPECT_NEAR(hs.x[i], ref.x[i], 1e-8) << "trial " << t << " i=" << i;
        EXPECT_NEAR(hs.objective_value, ref.objective_value, 1e-8) << "trial " << t;
    }
}
