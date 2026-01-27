#include <gtest/gtest.h>
#include "geoqp/geoqp.hpp"

using namespace geoqp;

TEST(UtilitiesTest, ValidateProblemValid) {
    Problem problem;
    problem.n = 2;
    problem.m = 2;
    problem.Q = {1.0, 0.0, 0.0, 1.0};
    problem.g = {-1.0, -1.0};
    problem.A = {1.0, 0.0, 0.0, 1.0};
    problem.b = {1.0, 1.0};
    
    EXPECT_TRUE(validate_problem(problem));
}

TEST(UtilitiesTest, ValidateProblemInvalidDimensions) {
    Problem problem;
    problem.n = 0;
    problem.m = 2;
    
    EXPECT_FALSE(validate_problem(problem));
}

TEST(UtilitiesTest, ValidateProblemWrongQSize) {
    Problem problem;
    problem.n = 2;
    problem.m = 2;
    problem.Q = {1.0, 0.0, 0.0};  // Too small
    problem.g = {-1.0, -1.0};
    problem.A = {1.0, 0.0, 0.0, 1.0};
    problem.b = {1.0, 1.0};
    
    EXPECT_FALSE(validate_problem(problem));
}

TEST(UtilitiesTest, ComputeUnconstrainedMinimum) {
    // Q = I, g = [-3, -2]
    // x_bar = -Q^{-1} g = [3, 2]
    std::vector<double> Q = {1.0, 0.0, 0.0, 1.0};
    std::vector<double> g = {-3.0, -2.0};
    
    auto x_bar = compute_unconstrained_minimum(Q, g, 2);
    
    ASSERT_EQ(x_bar.size(), 2u);
    EXPECT_NEAR(x_bar[0], 3.0, 1e-10);
    EXPECT_NEAR(x_bar[1], 2.0, 1e-10);
}

TEST(UtilitiesTest, IsFeasibleTrue) {
    std::vector<double> x = {1.0, 1.0};
    std::vector<double> A = {1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    std::vector<double> b = {2.0, 2.0, 3.0};
    
    EXPECT_TRUE(is_feasible(x, A, b, 3, 2));
}

TEST(UtilitiesTest, IsFeasibleFalse) {
    std::vector<double> x = {3.0, 2.0};
    std::vector<double> A = {1.0, 1.0};
    std::vector<double> b = {4.0};
    
    EXPECT_FALSE(is_feasible(x, A, b, 1, 2));  // 3 + 2 = 5 > 4
}

TEST(UtilitiesTest, ComputeObjective) {
    // (1/2) x^T Q x + g^T x with Q = I, g = [-3, -2], x = [1, 1]
    // (1/2)(1 + 1) + (-3 - 2) = 1 - 5 = -4
    std::vector<double> x = {1.0, 1.0};
    std::vector<double> Q = {1.0, 0.0, 0.0, 1.0};
    std::vector<double> g = {-3.0, -2.0};
    
    double obj = compute_objective(x, Q, g, 2);
    
    EXPECT_NEAR(obj, -4.0, 1e-10);
}

TEST(UtilitiesTest, IsPositiveDefinite) {
    std::vector<double> Q_pd = {2.0, 0.0, 0.0, 2.0};
    EXPECT_TRUE(is_positive_definite(Q_pd, 2));
    
    std::vector<double> Q_not_pd = {-1.0, 0.0, 0.0, 1.0};
    EXPECT_FALSE(is_positive_definite(Q_not_pd, 2));
}
