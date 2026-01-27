#include <gtest/gtest.h>
#include "geoqp/internal.hpp"
#include <cmath>

using namespace geoqp::internal;

TEST(LinearAlgebraTest, MatVec) {
    // Test matrix-vector multiplication
    double A[] = {1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0};
    double x[] = {1.0, 2.0, 3.0};
    double y[2];
    
    LinearAlgebra::matvec(A, x, y, 2, 3);
    
    EXPECT_NEAR(y[0], 14.0, 1e-10);  // 1*1 + 2*2 + 3*3 = 14
    EXPECT_NEAR(y[1], 32.0, 1e-10);  // 4*1 + 5*2 + 6*3 = 32
}

TEST(LinearAlgebraTest, Dot) {
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};
    
    double result = LinearAlgebra::dot(x, y, 3);
    
    EXPECT_NEAR(result, 32.0, 1e-10);  // 1*4 + 2*5 + 3*6 = 32
}

TEST(LinearAlgebraTest, Norm) {
    double x[] = {3.0, 4.0};
    
    double result = LinearAlgebra::norm(x, 2);
    
    EXPECT_NEAR(result, 5.0, 1e-10);  // sqrt(9 + 16) = 5
}

TEST(LinearAlgebraTest, Cholesky) {
    // Test Cholesky decomposition of identity matrix
    double A[] = {1.0, 0.0,
                  0.0, 1.0};
    double L[4];
    
    LinearAlgebra::cholesky(A, L, 2);
    
    EXPECT_NEAR(L[0], 1.0, 1e-10);
    EXPECT_NEAR(L[1], 0.0, 1e-10);
    EXPECT_NEAR(L[2], 0.0, 1e-10);
    EXPECT_NEAR(L[3], 1.0, 1e-10);
}

TEST(LinearAlgebraTest, CholeskyDiagonal) {
    // Test Cholesky of diagonal matrix
    double A[] = {4.0, 0.0,
                  0.0, 9.0};
    double L[4];
    
    LinearAlgebra::cholesky(A, L, 2);
    
    EXPECT_NEAR(L[0], 2.0, 1e-10);
    EXPECT_NEAR(L[1], 0.0, 1e-10);
    EXPECT_NEAR(L[2], 0.0, 1e-10);
    EXPECT_NEAR(L[3], 3.0, 1e-10);
}

TEST(LinearAlgebraTest, SolvePDSystem) {
    // Solve Q x = b where Q = I, b = [3, 4]
    double Q[] = {1.0, 0.0,
                  0.0, 1.0};
    double b[] = {3.0, 4.0};
    double x[2];
    
    LinearAlgebra::solve_pd_system(Q, b, x, 2);
    
    EXPECT_NEAR(x[0], 3.0, 1e-10);
    EXPECT_NEAR(x[1], 4.0, 1e-10);
}

TEST(LinearAlgebraTest, SolvePDSystem2) {
    // Solve Q x = b where Q = [2 0; 0 3], b = [4, 6]
    double Q[] = {2.0, 0.0,
                  0.0, 3.0};
    double b[] = {4.0, 6.0};
    double x[2];
    
    LinearAlgebra::solve_pd_system(Q, b, x, 2);
    
    EXPECT_NEAR(x[0], 2.0, 1e-10);
    EXPECT_NEAR(x[1], 2.0, 1e-10);
}

TEST(LinearAlgebraTest, SolveLinearSystem) {
    // Solve A x = b where A = [2 1; 1 3], b = [5, 6]
    // Solution: x = [2, 1]
    double A[] = {2.0, 1.0,
                  1.0, 3.0};
    double b[] = {5.0, 6.0};
    double x[2];
    
    LinearAlgebra::solve_linear_system(A, b, x, 2);
    
    // Verify by checking residual: A*x - b should be small
    double res0 = A[0]*x[0] + A[1]*x[1] - b[0];
    double res1 = A[2]*x[0] + A[3]*x[1] - b[1];
    EXPECT_NEAR(res0, 0.0, 1e-8);
    EXPECT_NEAR(res1, 0.0, 1e-8);
}
