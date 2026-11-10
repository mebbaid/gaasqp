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


// ── factorisation fast paths ────────────────────────────────────────────────
// cholesky() picks a path by size and structure (diagonal shortcut, right-looking
// loop, Eigen for large n, reference loop as fallback).  All must satisfy
// L·Lᵀ = A on the lower triangle and leave the strict upper triangle untouched.
TEST(LinearAlgebraTest, CholeskyAllPathsReconstructA)
{
    unsigned seed = 7u;
    auto rnd = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return (seed >> 8) / double(1u << 24) - 0.5;
    };
    for (int n : {1, 2, 3, 15, 16, 17, 40, 80, 81, 150})
    {
        std::vector<double> R(n * n), A(n * n, 0.0);
        for (auto &r : R) r = rnd();
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
            {
                double s = (i == j) ? 0.05 * n : 0.0;
                for (int k = 0; k < n; ++k) s += R[k * n + i] * R[k * n + j];
                A[i * n + j] = s;
            }
        const double marker = -12345.0;
        std::vector<double> L(n * n, marker);
        LinearAlgebra::cholesky(A.data(), L.data(), n);
        for (int i = 0; i < n; ++i)
        {
            EXPECT_GT(L[i * n + i], 0.0) << "n=" << n;
            for (int j = i + 1; j < n; ++j)
                EXPECT_EQ(L[i * n + j], marker) << "upper triangle written, n=" << n;
            for (int j = 0; j <= i; ++j)
            {
                double s = 0.0;
                for (int k = 0; k <= j; ++k) s += L[i * n + k] * L[j * n + k];
                EXPECT_NEAR(s, A[i * n + j], 1e-11 * (1.0 + std::abs(A[i * n + j]))) << "n=" << n;
            }
        }
    }
}

TEST(LinearAlgebraTest, CholeskyDiagonalAndIndefinite)
{
    // Diagonal: exact square roots, zeros below the diagonal.
    for (int n : {4, 100})
    {
        std::vector<double> A(n * n, 0.0), L(n * n, 9.0);
        for (int i = 0; i < n; ++i) A[i * n + i] = 1.0 + i;
        LinearAlgebra::cholesky(A.data(), L.data(), n);
        for (int i = 0; i < n; ++i)
        {
            EXPECT_EQ(L[i * n + i], std::sqrt(1.0 + i));
            for (int j = 0; j < i; ++j) EXPECT_EQ(L[i * n + j], 0.0);
        }
    }
    // Indefinite matrices are rejected on every path.
    for (int n : {3, 40, 120})
    {
        std::vector<double> A(n * n, 0.0), L(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) A[i * n + j] = (i == j) ? 1.0 : 0.9;
        A[(n - 1) * n + (n - 1)] = -5.0;
        EXPECT_THROW(LinearAlgebra::cholesky(A.data(), L.data(), n), geoqp::NumericalException) << "n=" << n;
    }
}

TEST(LinearAlgebraTest, ForwardSubUnitIsBitIdenticalToForwardSub)
{
    unsigned seed = 11u;
    auto rnd = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return (seed >> 8) / double(1u << 24) - 0.5;
    };
    for (int n : {1, 2, 5, 8, 13, 37})
    {
        std::vector<double> L(n * n, 0.0);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < i; ++j) L[i * n + j] = rnd();
            L[i * n + i] = 1.0 + std::abs(rnd());
        }
        for (int v = 0; v < n; ++v)
        {
            std::vector<double> e(n, 0.0), ref(n), got(n);
            e[v] = 1.0;
            LinearAlgebra::forward_sub(L.data(), e.data(), ref.data(), n);
            LinearAlgebra::forward_sub_unit(L.data(), v, got.data(), n);
            for (int i = 0; i < n; ++i)
                EXPECT_EQ(got[i], ref[i]) << "n=" << n << " v=" << v << " i=" << i;
        }
    }
}

TEST(LinearAlgebraTest, InplaceSolveMatchesGeneralSolve)
{
    const int n = 6;
    std::vector<double> A = {4, 1, 0, 2, 0, 1,  1, 5, 1, 0, 1, 0,  0, 1, 6, 1, 0, 2,
                             2, 0, 1, 7, 1, 0,  0, 1, 0, 1, 8, 1,  1, 0, 2, 0, 1, 9};
    std::vector<double> b = {1, -2, 3, -4, 5, -6}, x1(n), x2(n);
    LinearAlgebra::solve_linear_system(A.data(), b.data(), x1.data(), n);
    std::vector<double> Ac = A, bc = b;
    ASSERT_TRUE(LinearAlgebra::solve_linear_system_inplace(Ac.data(), bc.data(), x2.data(), n));
    for (int i = 0; i < n; ++i) EXPECT_EQ(x1[i], x2[i]);
    std::vector<double> Z(4, 0.0), z(2, 1.0), xz(2);
    EXPECT_FALSE(LinearAlgebra::solve_linear_system_inplace(Z.data(), z.data(), xz.data(), 2));
}
