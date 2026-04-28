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

TEST(LinearAlgebraTest, ForwardSubstitution) {
    // Test forward substitution: solve L * x = b
    // L = [2  0  0]
    //     [1  3  0]
    //     [4  2  5]
    // b = [4, 7, 15]
    // Solution: x = [2, 5/3, 1]
    
    double L[] = {2.0, 0.0, 0.0,
                  1.0, 3.0, 0.0,
                  4.0, 2.0, 5.0};
    double b[] = {4.0, 7.0, 15.0};
    double x[3];
    
    LinearAlgebra::forward_sub(L, b, x, 3);
    
    // x[0] = 4 / 2 = 2
    // x[1] = (7 - 1*2) / 3 = 5/3
    // x[2] = (15 - 4*2 - 2*(5/3)) / 5 = (15 - 8 - 10/3) / 5 = (21/3 - 10/3) / 5 = 11/15
    EXPECT_NEAR(x[0], 2.0, 1e-10);
    EXPECT_NEAR(x[1], 5.0/3.0, 1e-10);
    EXPECT_NEAR(x[2], (15.0 - 8.0 - 2.0*5.0/3.0)/5.0, 1e-10);
    
    // Verify: L * x should equal b
    double check[3];
    LinearAlgebra::matvec(L, x, check, 3, 3);
    EXPECT_NEAR(check[0], b[0], 1e-8);
    EXPECT_NEAR(check[1], b[1], 1e-8);
    EXPECT_NEAR(check[2], b[2], 1e-8);
}

TEST(LinearAlgebraTest, BackwardSubstitution) {
    // Test backward substitution: solve L^T * x = b
    // L = [2  0  0]      L^T = [2  1  4]
    //     [1  3  0]            [0  3  2]
    //     [4  2  5]            [0  0  5]
    // b = [10, 8, 5]
    // Solution computed from bottom up:
    // x[2] = 5/5 = 1
    // x[1] = (8 - 2*1)/3 = 2
    // x[0] = (10 - 1*2 - 4*1)/2 = 2
    
    double L[] = {2.0, 0.0, 0.0,
                  1.0, 3.0, 0.0,
                  4.0, 2.0, 5.0};
    double b[] = {10.0, 8.0, 5.0};
    double x[3];
    
    LinearAlgebra::backward_sub(L, b, x, 3);
    
    EXPECT_NEAR(x[2], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 2.0, 1e-10);
    EXPECT_NEAR(x[0], 2.0, 1e-10);
    
    // Verify: L^T * x should equal b
    // Manually compute L^T * x
    double check0 = 2.0*x[0] + 1.0*x[1] + 4.0*x[2];
    double check1 = 0.0*x[0] + 3.0*x[1] + 2.0*x[2];
    double check2 = 0.0*x[0] + 0.0*x[1] + 5.0*x[2];
    
    EXPECT_NEAR(check0, b[0], 1e-8);
    EXPECT_NEAR(check1, b[1], 1e-8);
    EXPECT_NEAR(check2, b[2], 1e-8);
}

TEST(LinearAlgebraTest, SolveCholeskyDecomposed) {
    // Test that solve_cholesky = backward_sub(forward_sub())
    // Solve (L * L^T) * x = b
    // L = [2  0]
    //     [1  3]
    // L*L^T = [4  2]
    //         [2 10]
    // b = [8, 14]
    // Solution: x should satisfy [4 2; 2 10] * x = [8, 14]
    
    double L[] = {2.0, 0.0,
                  1.0, 3.0};
    double b[] = {8.0, 14.0};
    double x[2];
    
    LinearAlgebra::solve_cholesky(L, b, x, 2);
    
    // Verify by computing L*L^T * x
    double Q[] = {4.0, 2.0,
                  2.0, 10.0};
    double check[2];
    LinearAlgebra::matvec(Q, x, check, 2, 2);
    
    EXPECT_NEAR(check[0], b[0], 1e-8);
    EXPECT_NEAR(check[1], b[1], 1e-8);
}

