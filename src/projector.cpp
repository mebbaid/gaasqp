#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace geoqp {
namespace internal {

Projector::Projector(
    const double* Q, const double* A, const double* b,
    const std::vector<double>& x_bar,
    const std::vector<double>& Q_chol,
    int m, int n,
    const Config& config,
    const std::vector<double>* Q_inv,
    bool q_is_diagonal,
    const std::vector<double>* Q_diag,
    bool has_diagonal_constraints,
    const std::vector<int>* constraint_var,
    const std::vector<double>* constraint_sign
) : Q_(Q), A_(A), b_(b), x_bar_(x_bar), Q_chol_(Q_chol), Q_inv_(Q_inv), m_(m), n_(n), config_(config),
    q_is_diagonal_(q_is_diagonal), Q_diag_(Q_diag),
    has_diagonal_constraints_(has_diagonal_constraints),
    constraint_var_(constraint_var), constraint_sign_(constraint_sign) {}

std::vector<double> Projector::project_onto_face(
    const std::vector<int>& active_set,
    std::vector<double>& lambda_out
) {
    const int k = static_cast<int>(active_set.size());

    // O(k) closed-form for diagonal Q + box constraints
    if (has_diagonal_constraints_ && q_is_diagonal_ && Q_diag_ && constraint_var_ && constraint_sign_)
    {
        std::vector<double> x_star = x_bar_;
        lambda_out.resize(k);

        for (int i = 0; i < k; ++i)
        {
            const int idx  = active_set[i];
            const int v    = (*constraint_var_)[idx];
            const double s = (*constraint_sign_)[idx];
            const double b = b_[idx];
            lambda_out[i]  = (s * x_bar_[v] - b) * (*Q_diag_)[v];
            x_star[v]      = b / s;
        }
        return x_star;
    }

    std::vector<double> x_star;
    bool success = ProjectionSolver::solve(
        active_set, Q_, A_, b_, x_bar_, Q_chol_,
        m_, n_, x_star, lambda_out,
        Q_inv_ ? *Q_inv_ : std::vector<double>(),
        q_is_diagonal_,
        Q_diag_ ? *Q_diag_ : std::vector<double>()
    );

    if (!success) {
        throw NumericalException(
            "Failed to solve for Lagrange multipliers in Phase III: Singular matrix in linear system"
        );
    }

    return x_star;
}

} // namespace internal
} // namespace geoqp
