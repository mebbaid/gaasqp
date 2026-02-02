#include "geoqp/internal.hpp"
#include <cstring>

namespace geoqp {
namespace internal {

//=============================================================================
// ConstraintCache Implementation
//=============================================================================

const std::vector<double>& ConstraintCache::get_z(
    int constraint_idx,
    const double* A,
    const double* L_Q
) {
    if (constraint_idx < 0 || constraint_idx >= m_) {
        throw std::out_of_range("Invalid constraint index");
    }
    
    auto& entry = cache_[constraint_idx];
    
    if (!entry.computed) {
        entry.z.resize(n_);
        const double* a_row = A + constraint_idx * n_;
        
        // Solve L_Q * z = a_row using forward substitution
        LinearAlgebra::forward_sub(L_Q, a_row, entry.z.data(), n_);
        entry.computed = true;
    }
    
    return entry.z;
}

//=============================================================================
// SchurComplementManager Implementation
//=============================================================================

bool SchurComplementManager::initialize(
    const std::vector<int>& face,
    const double* A,
    const double* L_Q,
    int n,
    ConstraintCache& cache
) {
    k_ = static_cast<int>(face.size());
    active_constraints_ = face;
    
    if (k_ == 0) {
        L_S_.clear();
        return true;
    }
    
    // Build full Schur complement S = Z^T * Z where Z has z_i vectors as columns
    std::vector<double> S(k_ * k_);
    
    for (int i = 0; i < k_; ++i) {
        const auto& z_i = cache.get_z(face[i], A, L_Q);
        
        // Diagonal element: z_i^T * z_i
        S[i * k_ + i] = LinearAlgebra::dot(z_i.data(), z_i.data(), n);
        
        // Off-diagonal (symmetric): z_i^T * z_j
        for (int j = i + 1; j < k_; ++j) {
            const auto& z_j = cache.get_z(face[j], A, L_Q);
            double val = LinearAlgebra::dot(z_i.data(), z_j.data(), n);
            S[i * k_ + j] = val;
            S[j * k_ + i] = val;
        }
    }
    
    // Compute Cholesky factorization of S
    L_S_.resize(k_ * k_);
    try {
        LinearAlgebra::cholesky(S.data(), L_S_.data(), k_);
        return true;
    } catch (...) {
        return false;
    }
}

bool SchurComplementManager::add_constraint(
    int constraint_idx,
    const double* A,
    const double* L_Q,
    int n,
    ConstraintCache& cache
) {
    const auto& z_new = cache.get_z(constraint_idx, A, L_Q);
    
    // Compute column s of Schur complement: s[i] = z_i^T * z_new
    std::vector<double> s(k_);
    for (int i = 0; i < k_; ++i) {
        const auto& z_i = cache.get_z(active_constraints_[i], A, L_Q);
        s[i] = LinearAlgebra::dot(z_i.data(), z_new.data(), n);
    }
    
    // Compute diagonal element: sigma = z_new^T * z_new
    double sigma = LinearAlgebra::dot(z_new.data(), z_new.data(), n);
    
    // Extend Cholesky factor from k×k to (k+1)×(k+1)
    std::vector<double> L_new((k_ + 1) * (k_ + 1));
    bool success = LinearAlgebra::cholesky_extend(
        L_S_.data(), k_,
        s.data(), sigma,
        L_new.data()
    );
    
    if (success) {
        L_S_ = std::move(L_new);
        active_constraints_.push_back(constraint_idx);
        k_++;
        return true;
    }
    
    return false;
}

bool SchurComplementManager::remove_constraint(int face_index) {
    if (face_index < 0 || face_index >= k_) {
        return false;
    }
    
    if (k_ == 1) {
        // Removing the last constraint
        L_S_.clear();
        active_constraints_.clear();
        k_ = 0;
        return true;
    }
    
    // Shrink Cholesky factor from k×k to (k-1)×(k-1)
    std::vector<double> L_new((k_ - 1) * (k_ - 1));
    bool success = LinearAlgebra::cholesky_shrink(
        L_S_.data(), k_, face_index,
        L_new.data()
    );
    
    if (success) {
        L_S_ = std::move(L_new);
        active_constraints_.erase(active_constraints_.begin() + face_index);
        k_--;
        return true;
    }
    
    return false;
}

bool SchurComplementManager::solve(
    const std::vector<double>& rhs,
    std::vector<double>& lambda
) const {
    if (k_ == 0) {
        lambda.clear();
        return true;
    }
    
    if (static_cast<int>(rhs.size()) != k_) {
        return false;
    }
    
    lambda.resize(k_);
    
    // Solve (L_S * L_S^T) * lambda = rhs using cached Cholesky factor
    // This is just: solve_cholesky(L_S, rhs, lambda)
    std::vector<double> y(k_);
    LinearAlgebra::forward_sub(L_S_.data(), rhs.data(), y.data(), k_);
    LinearAlgebra::backward_sub(L_S_.data(), y.data(), lambda.data(), k_);
    
    return true;
}

} // namespace internal
} // namespace geoqp
