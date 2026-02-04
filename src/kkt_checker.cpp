#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
#include <cmath>
#include <algorithm>

namespace geoqp
{
    namespace internal
    {

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
                int constraint_idx = active_set[i];
                if (lambda_ptr[constraint_idx] < -tol_)
                {
                    return false;
                }
            }

            for (int i = 0; i < m_; ++i)
            {
                const double *GEOQP_RESTRICT a_row = A_ + i * n_;

                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                int j = 0;
                for (; j + 3 < n_; j += 4)
                {
                    sum0 += a_row[j] * x_ptr[j];
                    sum1 += a_row[j + 1] * x_ptr[j + 1];
                    sum2 += a_row[j + 2] * x_ptr[j + 2];
                    sum3 += a_row[j + 3] * x_ptr[j + 3];
                }
                double lhs = (sum0 + sum1) + (sum2 + sum3);
                for (; j < n_; ++j)
                {
                    lhs += a_row[j] * x_ptr[j];
                }

                if (lhs > b_[i] + tol_)
                {
                    return false;
                }
            }

            for (int idx : active_set)
            {
                const double *GEOQP_RESTRICT a_row = A_ + idx * n_;

                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                int j = 0;
                for (; j + 3 < n_; j += 4)
                {
                    sum0 += a_row[j] * x_ptr[j];
                    sum1 += a_row[j + 1] * x_ptr[j + 1];
                    sum2 += a_row[j + 2] * x_ptr[j + 2];
                    sum3 += a_row[j + 3] * x_ptr[j + 3];
                }
                double lhs = (sum0 + sum1) + (sum2 + sum3);
                for (; j < n_; ++j)
                {
                    lhs += a_row[j] * x_ptr[j];
                }

                if (std::abs(lhs - b_[idx]) > tol_)
                {
                    return false;
                }
            }

            std::vector<double> Qx(n_);
            double *GEOQP_RESTRICT Qx_ptr = Qx.data();

            for (int i = 0; i < n_; ++i)
            {
                const double *GEOQP_RESTRICT Q_row = Q_ + i * n_;

                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                int j = 0;
                for (; j + 3 < n_; j += 4)
                {
                    sum0 += Q_row[j] * x_ptr[j];
                    sum1 += Q_row[j + 1] * x_ptr[j + 1];
                    sum2 += Q_row[j + 2] * x_ptr[j + 2];
                    sum3 += Q_row[j + 3] * x_ptr[j + 3];
                }
                Qx_ptr[i] = (sum0 + sum1) + (sum2 + sum3);
                for (; j < n_; ++j)
                {
                    Qx_ptr[i] += Q_row[j] * x_ptr[j];
                }
            }

            std::vector<double> grad(n_);
            double *GEOQP_RESTRICT grad_ptr = grad.data();
            for (int j = 0; j < n_; ++j)
            {
                grad_ptr[j] = Qx_ptr[j] + g_[j];
            }

            std::vector<double> dual_contribution(n_, 0.0);
            double *GEOQP_RESTRICT dual_ptr = dual_contribution.data();

            for (size_t i = 0; i < active_set.size(); ++i)
            {
                int constraint_idx = active_set[i];
                const double *GEOQP_RESTRICT a_row = A_ + constraint_idx * n_;
                double lam = lambda_ptr[constraint_idx];

                int j = 0;
                for (; j + 3 < n_; j += 4)
                {
                    dual_ptr[j] += lam * a_row[j];
                    dual_ptr[j + 1] += lam * a_row[j + 1];
                    dual_ptr[j + 2] += lam * a_row[j + 2];
                    dual_ptr[j + 3] += lam * a_row[j + 3];
                }
                for (; j < n_; ++j)
                {
                    dual_ptr[j] += lam * a_row[j];
                }
            }

            double residual = 0.0;
            double grad_norm = 0.0;
            for (int j = 0; j < n_; ++j)
            {
                double diff = grad_ptr[j] + dual_ptr[j]; 
                residual += diff * diff;
                grad_norm += grad_ptr[j] * grad_ptr[j];
            }
            residual = std::sqrt(residual);
            grad_norm = std::sqrt(grad_norm);


            double relative_residual = residual / (1.0 + grad_norm);
            double scaled_tol = std::sqrt(tol_); 

            return relative_residual <= scaled_tol;
        }

        double KKTChecker::compute_residual(
            const std::vector<double> &x,
            const std::vector<double> &lambda,
            const std::vector<int> &active_set) const
        {
            const double *GEOQP_RESTRICT x_ptr = x.data();
            const double *GEOQP_RESTRICT lambda_ptr = lambda.data();

            std::vector<double> Qx(n_);
            double *GEOQP_RESTRICT Qx_ptr = Qx.data();

            for (int i = 0; i < n_; ++i)
            {
                const double *GEOQP_RESTRICT Q_row = Q_ + i * n_;

                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                int j = 0;
                for (; j + 3 < n_; j += 4)
                {
                    sum0 += Q_row[j] * x_ptr[j];
                    sum1 += Q_row[j + 1] * x_ptr[j + 1];
                    sum2 += Q_row[j + 2] * x_ptr[j + 2];
                    sum3 += Q_row[j + 3] * x_ptr[j + 3];
                }
                Qx_ptr[i] = (sum0 + sum1) + (sum2 + sum3);
                for (; j < n_; ++j)
                {
                    Qx_ptr[i] += Q_row[j] * x_ptr[j];
                }
            }

            std::vector<double> grad(n_);
            double *GEOQP_RESTRICT grad_ptr = grad.data();
            for (int j = 0; j < n_; ++j)
            {
                grad_ptr[j] = Qx_ptr[j] + g_[j];
            }

            std::vector<double> dual_contribution(n_, 0.0);
            double *GEOQP_RESTRICT dual_ptr = dual_contribution.data();

            for (size_t i = 0; i < active_set.size(); ++i)
            {
                int constraint_idx = active_set[i];
                const double *GEOQP_RESTRICT a_row = A_ + constraint_idx * n_;
                double lam = lambda_ptr[constraint_idx];

                int j = 0;
                for (; j + 3 < n_; j += 4)
                {
                    dual_ptr[j] += lam * a_row[j];
                    dual_ptr[j + 1] += lam * a_row[j + 1];
                    dual_ptr[j + 2] += lam * a_row[j + 2];
                    dual_ptr[j + 3] += lam * a_row[j + 3];
                }
                for (; j < n_; ++j)
                {
                    dual_ptr[j] += lam * a_row[j];
                }
            }

            double residual = 0.0;
            for (int j = 0; j < n_; ++j)
            {
                double diff = grad_ptr[j] + dual_ptr[j]; 
                residual += diff * diff;
            }

            return std::sqrt(residual);
        }

    } // namespace internal
} // namespace geoqp
