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
