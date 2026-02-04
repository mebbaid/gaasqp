

#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
#include <cstring>

namespace geoqp
{
    namespace internal
    {

        bool ProjectionSolver::solve(
            const std::vector<int> &active_set,
            const double *GEOQP_RESTRICT Q,
            const double *GEOQP_RESTRICT A,
            const double *GEOQP_RESTRICT b,
            const std::vector<double> &x_bar,
            const std::vector<double> &Q_chol,
            int m,
            int n,
            std::vector<double> &x_star_out,
            std::vector<double> &lambda_out)
        {

            const int k = static_cast<int>(active_set.size());

            if (k == 0)
            {
                lambda_out.clear();
                x_star_out = x_bar;
                return true;
            }

            // Pre-allocate all needed arrays
            std::vector<double> A_active(k * n);
            std::vector<double> b_active(k);
            std::vector<double> residual(k);
            std::vector<double> Q_inv_AT(n * k); // Q^{-1} A^T (n x k matrix)
            std::vector<double> S(k * k);        // Schur complement

            const double *GEOQP_RESTRICT xbar_ptr = x_bar.data();
            double *GEOQP_RESTRICT A_active_ptr = A_active.data();
            double *GEOQP_RESTRICT b_active_ptr = b_active.data();

            // Build A_active and b_active
            for (int i = 0; i < k; ++i)
            {
                const int constraint_idx = active_set[i];
                b_active_ptr[i] = b[constraint_idx];
                const double *GEOQP_RESTRICT A_row = A + constraint_idx * n;
                std::memcpy(A_active_ptr + i * n, A_row, n * sizeof(double));
            }

            // Compute residual: A_active * x_bar - b_active
            LinearAlgebra::matvec(A_active_ptr, xbar_ptr, residual.data(), k, n);
            for (int i = 0; i < k; ++i)
            {
                residual[i] -= b_active_ptr[i];
            }

            const double *GEOQP_RESTRICT Q_chol_ptr = Q_chol.empty() ? nullptr : Q_chol.data();
            double *GEOQP_RESTRICT Q_inv_AT_ptr = Q_inv_AT.data();

            // Compute Q^{-1} A^T by solving Q * y = a_col for each constraint
            for (int j = 0; j < k; ++j)
            {
                // Get j-th row of A_active as a column vector
                std::vector<double> a_col(n);
                double *GEOQP_RESTRICT a_col_ptr = a_col.data();
                const double *GEOQP_RESTRICT A_active_row = A_active_ptr + j * n;

                for (int i = 0; i < n; ++i)
                {
                    a_col_ptr[i] = A_active_row[i];
                }

                // Solve Q * y = a_col for y using pre-computed Cholesky if available
                std::vector<double> y(n);

                if (Q_chol_ptr)
                {
                    LinearAlgebra::solve_cholesky(Q_chol_ptr, a_col_ptr, y.data(), n);
                }
                else
                {
                    // Fallback to general solve
                    std::vector<double> Q_copy(n * n);
                    std::memcpy(Q_copy.data(), Q, n * n * sizeof(double));

                    try
                    {
                        LinearAlgebra::solve_linear_system(Q_copy.data(), a_col_ptr, y.data(), n);
                    }
                    catch (...)
                    {
                        return false;
                    }
                }

                // Store as j-th column of Q^{-1} A^T
                const double *GEOQP_RESTRICT y_ptr = y.data();
                for (int i = 0; i < n; ++i)
                {
                    Q_inv_AT_ptr[i * k + j] = y_ptr[i];
                }
            }

            double *GEOQP_RESTRICT S_ptr = S.data();

            // Compute Schur complement: S = A_active * Q^{-1} A^T with loop unrolling
            for (int i = 0; i < k; ++i)
            {
                const double *GEOQP_RESTRICT A_row = A_active_ptr + i * n;

                for (int j = 0; j < k; ++j)
                {
                    double sum0 = 0.0;
                    double sum1 = 0.0;
                    double sum2 = 0.0;
                    double sum3 = 0.0;

                    int l = 0;
                    for (; l + 3 < n; l += 4)
                    {
                        sum0 += A_row[l] * Q_inv_AT_ptr[l * k + j];
                        sum1 += A_row[l + 1] * Q_inv_AT_ptr[(l + 1) * k + j];
                        sum2 += A_row[l + 2] * Q_inv_AT_ptr[(l + 2) * k + j];
                        sum3 += A_row[l + 3] * Q_inv_AT_ptr[(l + 3) * k + j];
                    }

                    double sum = (sum0 + sum1) + (sum2 + sum3);

                    for (; l < n; ++l)
                    {
                        sum += A_row[l] * Q_inv_AT_ptr[l * k + j];
                    }

                    S_ptr[i * k + j] = sum;
                }
            }

            // Solve S * lambda = residual
            lambda_out.resize(k);
            try
            {
                LinearAlgebra::solve_linear_system(S_ptr, residual.data(), lambda_out.data(), k);
            }
            catch (...)
            {
                return false;
            }

            // Compute x* = x_bar - Q^{-1} A^T * lambda with loop unrolling
            x_star_out.resize(n);
            double *GEOQP_RESTRICT xstar_ptr = x_star_out.data();
            const double *GEOQP_RESTRICT lambda_ptr = lambda_out.data();

            for (int i = 0; i < n; ++i)
            {
                double corr0 = 0.0;
                double corr1 = 0.0;
                double corr2 = 0.0;
                double corr3 = 0.0;

                int j = 0;
                for (; j + 3 < k; j += 4)
                {
                    corr0 += Q_inv_AT_ptr[i * k + j] * lambda_ptr[j];
                    corr1 += Q_inv_AT_ptr[i * k + j + 1] * lambda_ptr[j + 1];
                    corr2 += Q_inv_AT_ptr[i * k + j + 2] * lambda_ptr[j + 2];
                    corr3 += Q_inv_AT_ptr[i * k + j + 3] * lambda_ptr[j + 3];
                }

                double correction = (corr0 + corr1) + (corr2 + corr3);

                for (; j < k; ++j)
                {
                    correction += Q_inv_AT_ptr[i * k + j] * lambda_ptr[j];
                }

                xstar_ptr[i] = xbar_ptr[i] - correction;
            }

            return true;
        }

    } // namespace internal
} // namespace geoqp
