#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
#include <cmath>
#include <algorithm>

namespace geoqp
{
    namespace internal
    {
        namespace
        {
            // Computes ||Qx + g + A^T lambda||_2 over the active set.
            double stationarity_residual(
                const double *Q, const double *g, const double *A, int n,
                const double *x, const double *lambda,
                const std::vector<int> &active_set)
            {
                std::vector<double> stat(n);
                LinearAlgebra::matvec(Q, x, stat.data(), n, n);
                for (int j = 0; j < n; ++j) stat[j] += g[j];
                for (int idx : active_set)
                {
                    const double lam = lambda[idx];
                    const double *GEOQP_RESTRICT a_row = A + idx * n;
                    for (int j = 0; j < n; ++j) stat[j] += lam * a_row[j];
                }
                return std::sqrt(LinearAlgebra::dot(stat.data(), stat.data(), n));
            }
        } // anonymous namespace

        KKTChecker::KKTChecker(
            const double *Q, const double *g,
            const double *A, const double *b,
            int m, int n,
            double tol) : Q_(Q), g_(g), A_(A), b_(b), m_(m), n_(n), tol_(tol) {}

        bool KKTChecker::check(
            const std::vector<double> &x,
            const std::vector<double> &lambda,
            const std::vector<int> &active_set) const
        {
            const double *GEOQP_RESTRICT x_ptr = x.data();
            const double *GEOQP_RESTRICT lambda_ptr = lambda.data();

            for (size_t i = 0; i < active_set.size(); ++i)
            {
                if (lambda_ptr[active_set[i]] < -tol_)
                    return false;
            }

            for (int i = 0; i < m_; ++i)
            {
                const double *GEOQP_RESTRICT a_row = A_ + i * n_;
                double lhs = LinearAlgebra::dot(a_row, x_ptr, n_);
                if (lhs > b_[i] + tol_)
                    return false;
            }

            for (int idx : active_set)
            {
                const double *GEOQP_RESTRICT a_row = A_ + idx * n_;
                double lhs = LinearAlgebra::dot(a_row, x_ptr, n_);
                if (std::abs(lhs - b_[idx]) > tol_)
                    return false;
            }

            double residual  = stationarity_residual(Q_, g_, A_, n_, x_ptr, lambda_ptr, active_set);
            std::vector<double> grad(n_);
            LinearAlgebra::matvec(Q_, x_ptr, grad.data(), n_, n_);
            for (int j = 0; j < n_; ++j) grad[j] += g_[j];
            double grad_norm = LinearAlgebra::norm(grad.data(), n_);

            double relative_residual = residual / (1.0 + grad_norm);
            double scaled_tol = std::sqrt(tol_);
            return relative_residual <= scaled_tol;
        }

        double KKTChecker::compute_residual(
            const std::vector<double> &x,
            const std::vector<double> &lambda,
            const std::vector<int> &active_set) const
        {
            return stationarity_residual(Q_, g_, A_, n_, x.data(), lambda.data(), active_set);
        }

    } // namespace internal
} // namespace geoqp
