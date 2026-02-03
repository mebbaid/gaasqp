#include "geoqp/geoqp.hpp"
#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
#include <cmath>
#include <limits>

namespace geoqp
{

    bool validate_problem(const Problem &problem)
    {
        if (problem.n <= 0 || problem.m < 0)
        {
            return false;
        }

        if (problem.Q.size() != static_cast<size_t>(problem.n * problem.n))
        {
            return false;
        }
        if (problem.g.size() != static_cast<size_t>(problem.n))
        {
            return false;
        }
        if (problem.A.size() != static_cast<size_t>(problem.m * problem.n))
        {
            return false;
        }
        if (problem.b.size() != static_cast<size_t>(problem.m))
        {
            return false;
        }

        for (int i = 0; i < problem.n; ++i)
        {
            for (int j = i + 1; j < problem.n; ++j)
            {
                double diff = std::abs(problem.Q[i * problem.n + j] -
                                       problem.Q[j * problem.n + i]);
                if (diff > 1e-10)
                {
                    return false;
                }
            }
        }

        for (int i = 0; i < problem.n; ++i)
        {
            if (problem.Q[i * problem.n + i] <= 0.0)
            {
                return false;
            }
        }

        return true;
    }

    bool is_positive_definite(const std::vector<double> &Q, int n)
    {
        try
        {
            std::vector<double> L(n * n);
            internal::LinearAlgebra::cholesky(Q.data(), L.data(), n);
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    std::vector<double> compute_unconstrained_minimum(
        const std::vector<double> &Q,
        const std::vector<double> &g,
        int n)
    {
        std::vector<double> x_bar(n);
        std::vector<double> neg_g(n);

        const double* GEOQP_RESTRICT g_ptr = g.data();
        double* GEOQP_RESTRICT neg_g_ptr = neg_g.data();
        
        int i = 0;
        for (; i + 3 < n; i += 4)
        {
            neg_g_ptr[i] = -g_ptr[i];
            neg_g_ptr[i + 1] = -g_ptr[i + 1];
            neg_g_ptr[i + 2] = -g_ptr[i + 2];
            neg_g_ptr[i + 3] = -g_ptr[i + 3];
        }
        for (; i < n; ++i)
        {
            neg_g_ptr[i] = -g_ptr[i];
        }

        internal::LinearAlgebra::solve_pd_system(Q.data(), neg_g.data(), x_bar.data(), n);

        return x_bar;
    }

    bool is_feasible(
        const std::vector<double> &x,
        const std::vector<double> &A,
        const std::vector<double> &b,
        int m, int n,
        double tol)
    {
        const double* GEOQP_RESTRICT x_ptr = x.data();
        const double* GEOQP_RESTRICT A_ptr = A.data();
        const double* GEOQP_RESTRICT b_ptr = b.data();

        for (int i = 0; i < m; ++i)
        {
            const double* GEOQP_RESTRICT row = A_ptr + i * n;
            
            double sum0 = 0.0;
            double sum1 = 0.0;
            double sum2 = 0.0;
            double sum3 = 0.0;

            int j = 0;
            for (; j + 3 < n; j += 4)
            {
                sum0 += row[j] * x_ptr[j];
                sum1 += row[j + 1] * x_ptr[j + 1];
                sum2 += row[j + 2] * x_ptr[j + 2];
                sum3 += row[j + 3] * x_ptr[j + 3];
            }
            
            double lhs = (sum0 + sum1) + (sum2 + sum3);
            
            for (; j < n; ++j)
            {
                lhs += row[j] * x_ptr[j];
            }

            if (lhs > b_ptr[i] + tol)
            {
                return false;
            }
        }
        return true;
    }

    double compute_objective(
        const std::vector<double> &x,
        const std::vector<double> &Q,
        const std::vector<double> &g,
        int n)
    {

        std::vector<double> Qx(n);
        internal::LinearAlgebra::matvec(Q.data(), x.data(), Qx.data(), n, n);

        double quadratic = internal::LinearAlgebra::dot(x.data(), Qx.data(), n);

        double linear = internal::LinearAlgebra::dot(g.data(), x.data(), n);

        return 0.5 * quadratic + linear;
    }

} // namespace geoqp
