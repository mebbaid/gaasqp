#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
#include "geoqp/timing.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace geoqp
{
    namespace internal
    {

        void LinearAlgebra::matvec(const double *GEOQP_RESTRICT A, const double *GEOQP_RESTRICT x,
                                   double *GEOQP_RESTRICT y, int m, int n)
        {
            geoqp::timing::ScopedTimer _t_matvec(geoqp::timing::la_matvec_ns, geoqp::timing::la_matvec_count);
            geoqp::timing::la_matvec_elems.fetch_add(static_cast<uint64_t>(static_cast<uint64_t>(m) * static_cast<uint64_t>(n)), std::memory_order_relaxed);
            for (int i = 0; i < m; ++i)
            {
                const double *GEOQP_RESTRICT row = A + i * n;

                double sum0 = 0.0;
                double sum1 = 0.0;
                double sum2 = 0.0;
                double sum3 = 0.0;

                int j = 0;
                for (; j + 3 < n; j += 4)
                {
                    sum0 += row[j] * x[j];
                    sum1 += row[j + 1] * x[j + 1];
                    sum2 += row[j + 2] * x[j + 2];
                    sum3 += row[j + 3] * x[j + 3];
                }

                double sum = (sum0 + sum1) + (sum2 + sum3);

                for (; j < n; ++j)
                {
                    sum += row[j] * x[j];
                }

                y[i] = sum;
            }
        }

        double LinearAlgebra::dot(const double *GEOQP_RESTRICT x, const double *GEOQP_RESTRICT y, int n)
        {
            switch (n)
            {
            case 0:
                return 0.0;
            case 1:
                return x[0] * y[0];
            case 2:
                return x[0] * y[0] + x[1] * y[1];
            case 3:
                return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
            case 4:
                return x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
            case 5:
                return x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3] + x[4] * y[4];
            case 6:
                return x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3] + x[4] * y[4] + x[5] * y[5];
            case 7:
                return x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3] + x[4] * y[4] + x[5] * y[5] + x[6] * y[6];
            case 8:
                return x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3] + x[4] * y[4] + x[5] * y[5] + x[6] * y[6] + x[7] * y[7];
            case 9:
            {
                double r = 0.0;
                for (int i = 0; i < 9; ++i)
                    r += x[i] * y[i];
                return r;
            }
            case 10:
            {
                double r = 0.0;
                for (int i = 0; i < 10; ++i)
                    r += x[i] * y[i];
                return r;
            }
            case 11:
            {
                double r = 0.0;
                for (int i = 0; i < 11; ++i)
                    r += x[i] * y[i];
                return r;
            }
            case 12:
            {
                double r = 0.0;
                for (int i = 0; i < 12; ++i)
                    r += x[i] * y[i];
                return r;
            }
            case 13:
            {
                double r = 0.0;
                for (int i = 0; i < 13; ++i)
                    r += x[i] * y[i];
                return r;
            }
            case 14:
            {
                double r = 0.0;
                for (int i = 0; i < 14; ++i)
                    r += x[i] * y[i];
                return r;
            }
            case 15:
            {
                double r = 0.0;
                for (int i = 0; i < 15; ++i)
                    r += x[i] * y[i];
                return r;
            }
            case 16:
            {
                double r = 0.0;
                for (int i = 0; i < 16; ++i)
                    r += x[i] * y[i];
                return r;
            }
            default:
                break;
            }

            // Generic fallback: 4-way unrolled loop (no per-call timer here).
            double sum0 = 0.0;
            double sum1 = 0.0;
            double sum2 = 0.0;
            double sum3 = 0.0;

            int i = 0;
            for (; i + 3 < n; i += 4)
            {
                sum0 += x[i] * y[i];
                sum1 += x[i + 1] * y[i + 1];
                sum2 += x[i + 2] * y[i + 2];
                sum3 += x[i + 3] * y[i + 3];
            }

            double result = (sum0 + sum1) + (sum2 + sum3);

            for (; i < n; ++i)
            {
                result += x[i] * y[i];
            }

            return result;
        }

        double LinearAlgebra::norm(const double *GEOQP_RESTRICT x, int n)
        {
            return std::sqrt(dot(x, x, n));
        }

        void LinearAlgebra::cholesky(const double *GEOQP_RESTRICT A, double *GEOQP_RESTRICT L, int n)
        {
            geoqp::timing::ScopedTimer _t_chol(geoqp::timing::la_cholesky_ns, geoqp::timing::la_cholesky_count);
            // Zero only the lower-triangular portion that we write.
            // The old std::fill(L, L+n*n, 0) wrote the full n² elements (e.g. 20 KB for
            // n=50) but only the lower triangle is ever set; the upper stays unused.
            for (int i = 0; i < n; ++i)
                for (int j = 0; j <= i; ++j)
                    L[i * n + j] = 0.0;

            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j <= i; ++j)
                {
                    double sum0 = 0.0;
                    double sum1 = 0.0;
                    double sum2 = 0.0;
                    double sum3 = 0.0;

                    int k = 0;
                    for (; k + 3 < j; k += 4)
                    {
                        sum0 += L[i * n + k] * L[j * n + k];
                        sum1 += L[i * n + k + 1] * L[j * n + k + 1];
                        sum2 += L[i * n + k + 2] * L[j * n + k + 2];
                        sum3 += L[i * n + k + 3] * L[j * n + k + 3];
                    }

                    double sum = (sum0 + sum1) + (sum2 + sum3);

                    for (; k < j; ++k)
                    {
                        sum += L[i * n + k] * L[j * n + k];
                    }

                    if (i == j)
                    {
                        double val = A[i * n + i] - sum;
                        // Round near-zero to zero before sqrt to avoid sqrt(0⁻) from
                        // floating-point noise (same guard used in matrix_math.h CholeskyDec).
                        if (val < 0.0)
                        {
                            if (val < -1e-9)
                                throw NumericalException("Matrix is not positive definite");
                            val = 0.0;
                        }
                        L[i * n + j] = std::sqrt(val);
                    }
                    else
                    {
                        L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
                    }
                }
            }
        }

        void LinearAlgebra::forward_sub(const double *GEOQP_RESTRICT L, const double *GEOQP_RESTRICT b,
                                        double *GEOQP_RESTRICT x, int n)
        {
            geoqp::timing::ScopedTimer _t_for(geoqp::timing::la_forward_ns, geoqp::timing::la_forward_count);
            for (int i = 0; i < n; ++i)
            {
                const double *GEOQP_RESTRICT Li = L + i * n;

                double sum0 = 0.0;
                double sum1 = 0.0;
                double sum2 = 0.0;
                double sum3 = 0.0;

                int j = 0;
                for (; j + 3 < i; j += 4)
                {
                    sum0 += Li[j] * x[j];
                    sum1 += Li[j + 1] * x[j + 1];
                    sum2 += Li[j + 2] * x[j + 2];
                    sum3 += Li[j + 3] * x[j + 3];
                }

                double sum = (sum0 + sum1) + (sum2 + sum3);

                for (; j < i; ++j)
                {
                    sum += Li[j] * x[j];
                }

                const double Lii = Li[i];
                x[i] = (b[i] - sum) / Lii;
            }
        }

        void LinearAlgebra::backward_sub(const double *GEOQP_RESTRICT L, const double *GEOQP_RESTRICT b,
                                         double *GEOQP_RESTRICT x, int n)
        {
            geoqp::timing::ScopedTimer _t_back(geoqp::timing::la_backward_ns, geoqp::timing::la_backward_count);
            // L is lower-triangular, stored row-major.  We solve Lᵀ x = b.
            // Naive: x[i] = (b[i] - Σ_{j>i} L[j*n+i]*x[j]) / L[i*n+i]
            // The sum reads L column-by-column (stride n) — cache-hostile for large n.
            //
            // Equivalent scatter formulation: initialise x = b, then for each already-solved
            // x[j] (j from n-1 down to 0) propagate its contribution to all i < j before
            // dividing by the diagonal.  This accesses L row j (contiguous) each iteration.
            for (int i = 0; i < n; ++i)
                x[i] = b[i];

            for (int j = n - 1; j >= 0; --j)
            {
                const double *GEOQP_RESTRICT Lj = L + j * n;
                x[j] /= Lj[j];                  // diagonal divide (Lᵀ[j][j] == L[j][j])
                const double xj = x[j];

                // Subtract x[j] * Lᵀ[i][j] = x[j] * L[j][i] for all i < j.
                // L[j][i] is stored in row j, positions 0..j-1 — contiguous memory.
                int i = 0;
                for (; i + 3 < j; i += 4)
                {
                    x[i]     -= xj * Lj[i];
                    x[i + 1] -= xj * Lj[i + 1];
                    x[i + 2] -= xj * Lj[i + 2];
                    x[i + 3] -= xj * Lj[i + 3];
                }
                for (; i < j; ++i)
                    x[i] -= xj * Lj[i];
            }
        }

        void LinearAlgebra::solve_cholesky(const double *GEOQP_RESTRICT L, const double *GEOQP_RESTRICT b,
                                           double *GEOQP_RESTRICT x, int n)
        {
            geoqp::timing::ScopedTimer _t_schol(geoqp::timing::la_solve_cholesky_ns, geoqp::timing::la_solve_cholesky_count);
            static thread_local std::vector<double> tl_workspace;
            if (tl_workspace.size() < static_cast<size_t>(n))
                tl_workspace.resize(n);
            forward_sub(L, b, tl_workspace.data(), n);
            backward_sub(L, tl_workspace.data(), x, n);
        }

        void LinearAlgebra::solve_cholesky(const double *GEOQP_RESTRICT L, const double *GEOQP_RESTRICT b,
                                           double *GEOQP_RESTRICT x, double *GEOQP_RESTRICT workspace, int n)
        {
            forward_sub(L, b, workspace, n);
            backward_sub(L, workspace, x, n);
        }

        void LinearAlgebra::solve_pd_system(const double *GEOQP_RESTRICT Q, const double *GEOQP_RESTRICT b,
                                            double *GEOQP_RESTRICT x, int n)
        {
            static thread_local std::vector<double> tl_L;
            if (tl_L.size() < static_cast<size_t>(n) * static_cast<size_t>(n))
                tl_L.resize(n * n);
            cholesky(Q, tl_L.data(), n);
            solve_cholesky(tl_L.data(), b, x, n);
        }

        void LinearAlgebra::solve_linear_system(const double *A, const double *b, double *x, int n)
        {
            geoqp::timing::ScopedTimer _t_solve(geoqp::timing::solve_ns, geoqp::timing::solve_count);
            geoqp::timing::ScopedTimer _t_solve_full(geoqp::timing::la_cholesky_ns, geoqp::timing::la_cholesky_count);
            static thread_local std::vector<double> tl_A_copy;
            static thread_local std::vector<double> tl_b_copy;
            if (tl_A_copy.size() < static_cast<size_t>(n) * static_cast<size_t>(n))
                tl_A_copy.resize(n * n);
            if (tl_b_copy.size() < static_cast<size_t>(n))
                tl_b_copy.resize(n);
            std::memcpy(tl_A_copy.data(), A, n * n * sizeof(double));
            std::memcpy(tl_b_copy.data(), b, n * sizeof(double));
            double *GEOQP_RESTRICT A_copy = tl_A_copy.data();
            double *GEOQP_RESTRICT b_copy = tl_b_copy.data();

            for (int k = 0; k < n; ++k)
            {
                int pivot = k;
                double max_val = std::abs(A_copy[k * n + k]);
                for (int i = k + 1; i < n; ++i)
                {
                    double val = std::abs(A_copy[i * n + k]);
                    if (val > max_val)
                    {
                        max_val = val;
                        pivot = i;
                    }
                }

                if (max_val < 1e-14)
                {
                    throw NumericalException("Singular matrix in linear system");
                }

                if (pivot != k)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        std::swap(A_copy[k * n + j], A_copy[pivot * n + j]);
                    }
                    std::swap(b_copy[k], b_copy[pivot]);
                }

                for (int i = k + 1; i < n; ++i)
                {
                    double factor = A_copy[i * n + k] / A_copy[k * n + k];
                    for (int j = k; j < n; ++j)
                    {
                        A_copy[i * n + j] -= factor * A_copy[k * n + j];
                    }
                    b_copy[i] -= factor * b_copy[k];
                }
            }

            for (int i = n - 1; i >= 0; --i)
            {
                double sum = 0.0;
                for (int j = i + 1; j < n; ++j)
                {
                    sum += A_copy[i * n + j] * x[j];
                }
                x[i] = (b_copy[i] - sum) / A_copy[i * n + i];
            }
        }

        bool LinearAlgebra::cholesky_rank1_update(double *L, const double *v, int k)
        {
            geoqp::timing::ScopedTimer _t_rank1(geoqp::timing::la_rank1_ns, geoqp::timing::la_rank1_count);
            static thread_local std::vector<double> tl_w;
            if (tl_w.size() < static_cast<size_t>(k))
                tl_w.resize(k);
            std::memcpy(tl_w.data(), v, k * sizeof(double));

            for (int i = 0; i < k; ++i)
            {
                double Lii = L[i * k + i];
                double wi = tl_w[i];
                double r = std::sqrt(Lii * Lii + wi * wi);

                if (r < 1e-14)
                    return false;

                double c = r / Lii;
                double s = wi / Lii;
                L[i * k + i] = r;

                for (int j = i + 1; j < k; ++j)
                {
                    const double Lji     = L[j * k + i];
                    const double wj_old  = tl_w[j];
                    // Givens update: new_w[j] = c*w[j] - s*L[j,i],  new_L[j,i] = s*w[j] + c*L[j,i]
                    // (divided through to keep L normalised — standard rank-1 Cholesky update)
                    tl_w[j]      = wj_old - s * Lji;          // uses old wj and old Lji
                    L[j * k + i] = (Lji + s * wj_old) / c;   // uses old wj and old Lji
                }
            }
            return true;
        }

        bool LinearAlgebra::cholesky_extend(
            const double *L, int k,
            const double *s, double sigma,
            double *L_new)
        {
            geoqp::timing::ScopedTimer _t_extend(geoqp::timing::la_extend_ns, geoqp::timing::la_extend_count);
            // Given L (k x k) such that L*L^T = S,
            // extend to L' ((k+1) x (k+1)) for [S, s; s^T, sigma]
            //
            // The new Cholesky factor has form:
            // L' = [L,     0  ]
            //      [l^T, l_new]
            // where L * l = s  =>  l = L^{-1} s (forward solve)
            // and l_new = sqrt(sigma - ||l||^2)

            const int k1 = k + 1;

            for (int i = 0; i < k; ++i)
            {
                for (int j = 0; j <= i; ++j)
                {
                    L_new[i * k1 + j] = L[i * k + j];
                }
                L_new[i * k1 + k] = 0.0;
            }

            static thread_local std::vector<double> tl_l;
            if (tl_l.size() < static_cast<size_t>(k))
                tl_l.resize(k);
            for (int i = 0; i < k; ++i)
            {
                double sum = 0.0;
                for (int j = 0; j < i; ++j)
                {
                    sum += L[i * k + j] * tl_l[j];
                }
                tl_l[i] = (s[i] - sum) / L[i * k + i];
            }

            for (int j = 0; j < k; ++j)
            {
                L_new[k * k1 + j] = tl_l[j];
            }

            double l_norm_sq = 0.0;
            for (int i = 0; i < k; ++i)
            {
                l_norm_sq += tl_l[i] * tl_l[i];
            }

            double diag_sq = sigma - l_norm_sq;
            if (diag_sq < 1e-14)
            {
                return false;
            }

            L_new[k * k1 + k] = std::sqrt(diag_sq);
            return true;
        }

        bool LinearAlgebra::cholesky_shrink(
            const double *L, int k, int idx,
            double *L_new)
        {
            geoqp::timing::ScopedTimer _t_shrink(geoqp::timing::la_shrink_ns, geoqp::timing::la_shrink_count);
            if (k <= 1 || idx < 0 || idx >= k)
            {
                return false;
            }

            const int k1 = k - 1;

            if (idx == k1)
            {
                for (int i = 0; i < k1; ++i)
                {
                    for (int j = 0; j <= i; ++j)
                    {
                        L_new[i * k1 + j] = L[i * k + j];
                    }
                }
                return true;
            }

            static thread_local std::vector<double> tl_L_work;
            if (tl_L_work.size() < static_cast<size_t>(k) * static_cast<size_t>(k))
                tl_L_work.resize(k * k);
            for (int i = 0; i < k; ++i)
            {
                for (int j = 0; j <= i; ++j)
                {
                    tl_L_work[i * k + j] = L[i * k + j];
                }
            }

            for (int i = idx; i < k1; ++i)
            {
                double a = tl_L_work[i * k + i];
                double b = tl_L_work[(i + 1) * k + i];

                if (std::abs(b) < 1e-14)
                {
                    continue;
                }

                double r = std::sqrt(a * a + b * b);
                if (r < 1e-14)
                {
                    return false;
                }

                double c = a / r;
                double s = b / r;

                for (int j = 0; j <= i + 1 && j < k; ++j)
                {
                    double L_i_j = (j <= i) ? tl_L_work[i * k + j] : 0.0;
                    double L_ip1_j = (j <= i + 1) ? tl_L_work[(i + 1) * k + j] : 0.0;

                    double new_i = c * L_i_j + s * L_ip1_j;
                    double new_ip1 = -s * L_i_j + c * L_ip1_j;

                    if (j <= i)
                        tl_L_work[i * k + j] = new_i;
                    if (j <= i + 1)
                        tl_L_work[(i + 1) * k + j] = new_ip1;
                }

                for (int row = i + 2; row < k; ++row)
                {
                    double L_row_i = (i < row) ? tl_L_work[row * k + i] : 0.0;
                    double L_row_ip1 = (i + 1 < row) ? tl_L_work[row * k + i + 1] : 0.0;

                    if (i < row)
                        tl_L_work[row * k + i] = c * L_row_i + s * L_row_ip1;
                    if (i + 1 < row)
                        tl_L_work[row * k + i + 1] = -s * L_row_i + c * L_row_ip1;
                }
            }

            for (int i = 0; i < k1; ++i)
            {
                for (int j = 0; j <= i; ++j)
                {
                    L_new[i * k1 + j] = tl_L_work[i * k + j];
                }
            }

            return true;
        }

    } // namespace internal
} // namespace geoqp
