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
    const std::vector<double>* Q_inv
) : Q_(Q), A_(A), b_(b), x_bar_(x_bar), Q_chol_(Q_chol), Q_inv_(Q_inv), m_(m), n_(n), config_(config) {}

std::vector<double> Projector::project_onto_face(
    const std::vector<int>& active_set,
    std::vector<double>& lambda_out
) {
    std::vector<double> x_star;
    
    bool success = ProjectionSolver::solve(
        active_set, Q_, A_, b_, x_bar_, Q_chol_,
        m_, n_, x_star, lambda_out,
        Q_inv_ ? *Q_inv_ : std::vector<double>()
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
