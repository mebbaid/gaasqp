#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
#include "geoqp/timing.hpp"
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
            std::vector<double> &lambda_out,
            const std::vector<double> &Q_inv,
            bool q_is_diagonal,
            const std::vector<double> &Q_diag)
        {

            const int k = static_cast<int>(active_set.size());
            geoqp::timing::ScopedTimer _t_proj(geoqp::timing::projection_ns, geoqp::timing::projection_count);

            if (k == 0)
            {
                lambda_out.clear();
                x_star_out = x_bar;
                return true;
            }

            // Reuse thread-local workspaces to avoid repeated allocations on hot path
            static thread_local std::vector<double> tl_A_active;
            static thread_local std::vector<double> tl_b_active;
            static thread_local std::vector<double> tl_residual;
            static thread_local std::vector<double> tl_Q_inv_AT; // Q^{-1} A^T (n x k matrix)
            static thread_local std::vector<double> tl_S;        // Schur complement
            static thread_local std::vector<double> tl_a_col;
            static thread_local std::vector<double> tl_y;
            static thread_local std::vector<double> tl_Q_copy;

            if (tl_A_active.size() < static_cast<size_t>(k) * static_cast<size_t>(n)) tl_A_active.resize(k * n);
            if (tl_b_active.size() < static_cast<size_t>(k)) tl_b_active.resize(k);
            if (tl_residual.size() < static_cast<size_t>(k)) tl_residual.resize(k);
            if (tl_Q_inv_AT.size() < static_cast<size_t>(n) * static_cast<size_t>(k)) tl_Q_inv_AT.resize(n * k);
            if (tl_S.size() < static_cast<size_t>(k) * static_cast<size_t>(k)) tl_S.resize(k * k);

            const double *GEOQP_RESTRICT xbar_ptr = x_bar.data();
            double *GEOQP_RESTRICT A_active_ptr = tl_A_active.data();
            double *GEOQP_RESTRICT b_active_ptr = tl_b_active.data();

            if (xbar_ptr == nullptr) {
                return false;
            }

            // Build A_active and b_active
            std::vector<int> box_indices(k, -1);
            std::vector<double> box_signs(k, 0.0);

            for (int i = 0; i < k; ++i)
            {
                const int constraint_idx = active_set[i];
                if (A == nullptr || b == nullptr || constraint_idx < 0 || constraint_idx >= m) {
                    return false;
                }
                b_active_ptr[i] = b[constraint_idx];
                const double *GEOQP_RESTRICT A_row = A + constraint_idx * n;
                std::memcpy(A_active_ptr + i * n, A_row, n * sizeof(double));

                // Identify if this is a box constraint
                int p = -1;
                double sign = 0.0;
                int non_zeros = 0;
                for (int l = 0; l < n; ++l) {
                    if (A_row[l] != 0.0) {
                        non_zeros++;
                        p = l;
                        sign = A_row[l];
                    }
                }
                if (non_zeros == 1) {
                    box_indices[i] = p;
                    box_signs[i] = sign;
                }
            }

            // Compute residual: A_active * x_bar - b_active
            LinearAlgebra::matvec(A_active_ptr, xbar_ptr, tl_residual.data(), k, n);
            for (int i = 0; i < k; ++i)
            {
                tl_residual[i] -= b_active_ptr[i];
            }

            const double *GEOQP_RESTRICT Q_chol_ptr = Q_chol.empty() ? nullptr : Q_chol.data();
            double *GEOQP_RESTRICT Q_inv_AT_ptr = tl_Q_inv_AT.data();

            // Compute Q^{-1} A^T by solving Q * y = a_col for each constraint
            const bool use_Q_inv = !Q_inv.empty() && (static_cast<int>(Q_inv.size()) >= n * n);
            const double* GEOQP_RESTRICT Q_inv_ptr = use_Q_inv ? Q_inv.data() : nullptr;

            for (int j = 0; j < k; ++j)
            {
                const double * GEOQP_RESTRICT A_active_row = A_active_ptr + j * n;
                const int p_j = box_indices[j];
                const double sign_j = box_signs[j];

                if (q_is_diagonal && !Q_diag.empty()) {
                    for (int i = 0; i < n; ++i) {
                        Q_inv_AT_ptr[i * k + j] = A_active_row[i] / Q_diag[i];
                    }
                    continue;
                }

                if (use_Q_inv) {
                    if (p_j != -1) {
                        // S construction will use Q_inv directly for box constraints
                        // We still need to populate Q_inv_AT for others or for the final x_star computation
                        const double* GEOQP_RESTRICT Q_inv_col = Q_inv_ptr + p_j * n;
                        for (int i = 0; i < n; ++i) {
                            Q_inv_AT_ptr[i * k + j] = Q_inv_col[i] * sign_j;
                        }
                    } else {
                        for (int i = 0; i < n; ++i) {
                            double val = 0.0;
                            const double* GEOQP_RESTRICT Qi = Q_inv_ptr + i * n;
                            int l = 0;
                            GEOQP_UNROLL_LOOP_4
                            for (; l + 3 < n; l += 4) {
                                val += Qi[l] * A_active_row[l];
                                val += Qi[l+1] * A_active_row[l+1];
                                val += Qi[l+2] * A_active_row[l+2];
                                val += Qi[l+3] * A_active_row[l+3];
                            }
                            for (; l < n; ++l) val += Qi[l] * A_active_row[l];
                            Q_inv_AT_ptr[i * k + j] = val;
                        }
                    }
                    continue;
                }

                // Get j-th row of A_active as a column vector (reuse workspace)
                if (tl_a_col.size() < static_cast<size_t>(n)) tl_a_col.resize(n);
                double *GEOQP_RESTRICT a_col_ptr = tl_a_col.data();
                for (int i = 0; i < n; ++i) a_col_ptr[i] = A_active_row[i];

                // Solve Q * y = a_col for y using pre-allocated workspace
                if (tl_y.size() < static_cast<size_t>(n)) tl_y.resize(n);
                double *GEOQP_RESTRICT y_ptr = tl_y.data();

                if (Q_chol_ptr)
                {
                    LinearAlgebra::solve_cholesky(Q_chol_ptr, a_col_ptr, y_ptr, n);
                }
                else
                {
                    // Fallback to general solve using a reusable copy buffer
                    if (tl_Q_copy.size() < static_cast<size_t>(n) * static_cast<size_t>(n)) tl_Q_copy.resize(n * n);
                    std::memcpy(tl_Q_copy.data(), Q, n * n * sizeof(double));

                    try
                    {
                        LinearAlgebra::solve_linear_system(tl_Q_copy.data(), a_col_ptr, y_ptr, n);
                    }
                    catch (...)
                    {
                        return false;
                    }
                }

                // Store as j-th column of Q^{-1} A^T
                for (int i = 0; i < n; ++i)
                {
                    Q_inv_AT_ptr[i * k + j] = y_ptr[i];
                }
            }

            double *GEOQP_RESTRICT S_ptr = tl_S.data();
            // zero only the used portion of S (k x k)
            for (int i = 0; i < k * k; ++i) S_ptr[i] = 0.0;

            {
                geoqp::timing::ScopedTimer _t_schur(geoqp::timing::schur_ns, geoqp::timing::schur_count);
                // Compute Schur complement: S = A_active * Q^{-1} A^T
            for (int i = 0; i < k; ++i)
            {
                const double *GEOQP_RESTRICT A_row = A_active_ptr + i * n;
                const int p_i = box_indices[i];
                const double sign_i = box_signs[i];

                for (int j = i; j < k; ++j) // Only compute upper triangle
                {
                    if (p_i != -1) {
                        S_ptr[i * k + j] = sign_i * Q_inv_AT_ptr[p_i * k + j];
                    } else {
                        const int p_j = box_indices[j];
                        if (p_j != -1) {
                             S_ptr[i * k + j] = box_signs[j] * Q_inv_AT_ptr[p_j * k + i];
                        } else {
                            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                            int l = 0;
                            GEOQP_UNROLL_LOOP_4
                            for (; l + 3 < n; l += 4) {
                                sum0 += A_row[l] * Q_inv_AT_ptr[l * k + j];
                                sum1 += A_row[l + 1] * Q_inv_AT_ptr[(l + 1) * k + j];
                                sum2 += A_row[l + 2] * Q_inv_AT_ptr[(l + 2) * k + j];
                                sum3 += A_row[l + 3] * Q_inv_AT_ptr[(l + 3) * k + j];
                            }
                            double sum = (sum0 + sum1) + (sum2 + sum3);
                            for (; l < n; ++l) sum += A_row[l] * Q_inv_AT_ptr[l * k + j];
                            S_ptr[i * k + j] = sum;
                        }
                    }
                        S_ptr[j * k + i] = S_ptr[i * k + j]; // Symmetry
                    }
                }
                }

                // Solve S * lambda = residual
            lambda_out.resize(k);
            try
            {
                LinearAlgebra::solve_linear_system(S_ptr, tl_residual.data(), lambda_out.data(), k);
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
