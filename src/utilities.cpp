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

        for (int i = 0; i < n; ++i) neg_g[i] = -g[i];

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
        for (int i = 0; i < m; ++i) {
            const double *row = A.data() + i * n;
            double lhs = internal::LinearAlgebra::dot(row, x.data(), n);
            if (lhs > b[i] + tol) return false;
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
