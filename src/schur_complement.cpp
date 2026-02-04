#include "geoqp/internal.hpp"
#include "geoqp/compiler_hints.hpp"
#include <cstring>

namespace geoqp
{
    namespace internal
    {

        const std::vector<double> &ConstraintCache::get_z(
            int constraint_idx,
            const double *A,
            const double *L_Q)
        {
            if (constraint_idx < 0 || constraint_idx >= m_)
            {
                throw std::out_of_range("Invalid constraint index");
            }

            auto &entry = cache_[constraint_idx];

            if (!entry.computed)
            {
                entry.z.resize(n_);
                const double *a_row = A + constraint_idx * n_;

                LinearAlgebra::forward_sub(L_Q, a_row, entry.z.data(), n_);
                entry.computed = true;
            }

            return entry.z;
        }

        bool SchurComplementManager::initialize(
            const std::vector<int> &face,
            const double *GEOQP_RESTRICT A,
            const double *GEOQP_RESTRICT L_Q,
            int n,
            ConstraintCache &cache)
        {
            k_ = static_cast<int>(face.size());
            active_constraints_ = face;

            if (k_ == 0)
            {
                L_S_.clear();
                return true;
            }

            std::vector<double> S(k_ * k_);
            double *GEOQP_RESTRICT S_ptr = S.data();

            for (int i = 0; i < k_; ++i)
            {
                const auto &z_i = cache.get_z(face[i], A, L_Q);
                const double *GEOQP_RESTRICT z_i_ptr = z_i.data();

                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                int j = 0;
                for (; j + 3 < n; j += 4)
                {
                    sum0 += z_i_ptr[j] * z_i_ptr[j];
                    sum1 += z_i_ptr[j + 1] * z_i_ptr[j + 1];
                    sum2 += z_i_ptr[j + 2] * z_i_ptr[j + 2];
                    sum3 += z_i_ptr[j + 3] * z_i_ptr[j + 3];
                }
                double diag = (sum0 + sum1) + (sum2 + sum3);
                for (; j < n; ++j)
                {
                    diag += z_i_ptr[j] * z_i_ptr[j];
                }
                S_ptr[i * k_ + i] = diag;

                for (int ji = i + 1; ji < k_; ++ji)
                {
                    const auto &z_j = cache.get_z(face[ji], A, L_Q);
                    const double *GEOQP_RESTRICT z_j_ptr = z_j.data();

                    sum0 = 0.0;
                    sum1 = 0.0;
                    sum2 = 0.0;
                    sum3 = 0.0;
                    int l = 0;
                    for (; l + 3 < n; l += 4)
                    {
                        sum0 += z_i_ptr[l] * z_j_ptr[l];
                        sum1 += z_i_ptr[l + 1] * z_j_ptr[l + 1];
                        sum2 += z_i_ptr[l + 2] * z_j_ptr[l + 2];
                        sum3 += z_i_ptr[l + 3] * z_j_ptr[l + 3];
                    }
                    double val = (sum0 + sum1) + (sum2 + sum3);
                    for (; l < n; ++l)
                    {
                        val += z_i_ptr[l] * z_j_ptr[l];
                    }
                    S_ptr[i * k_ + ji] = val;
                    S_ptr[ji * k_ + i] = val;
                }
            }

            L_S_.resize(k_ * k_);
            try
            {
                LinearAlgebra::cholesky(S.data(), L_S_.data(), k_);
                return true;
            }
            catch (...)
            {
                return false;
            }
        }

        bool SchurComplementManager::add_constraint(
            int constraint_idx,
            const double *GEOQP_RESTRICT A,
            const double *GEOQP_RESTRICT L_Q,
            int n,
            ConstraintCache &cache)
        {
            const auto &z_new = cache.get_z(constraint_idx, A, L_Q);
            const double *GEOQP_RESTRICT z_new_ptr = z_new.data();

            std::vector<double> s(k_);
            double *GEOQP_RESTRICT s_ptr = s.data();

            for (int i = 0; i < k_; ++i)
            {
                const auto &z_i = cache.get_z(active_constraints_[i], A, L_Q);
                const double *GEOQP_RESTRICT z_i_ptr = z_i.data();

                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                int j = 0;
                for (; j + 3 < n; j += 4)
                {
                    sum0 += z_i_ptr[j] * z_new_ptr[j];
                    sum1 += z_i_ptr[j + 1] * z_new_ptr[j + 1];
                    sum2 += z_i_ptr[j + 2] * z_new_ptr[j + 2];
                    sum3 += z_i_ptr[j + 3] * z_new_ptr[j + 3];
                }
                s_ptr[i] = (sum0 + sum1) + (sum2 + sum3);
                for (; j < n; ++j)
                {
                    s_ptr[i] += z_i_ptr[j] * z_new_ptr[j];
                }
            }

            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
            int j = 0;
            for (; j + 3 < n; j += 4)
            {
                sum0 += z_new_ptr[j] * z_new_ptr[j];
                sum1 += z_new_ptr[j + 1] * z_new_ptr[j + 1];
                sum2 += z_new_ptr[j + 2] * z_new_ptr[j + 2];
                sum3 += z_new_ptr[j + 3] * z_new_ptr[j + 3];
            }
            double sigma = (sum0 + sum1) + (sum2 + sum3);
            for (; j < n; ++j)
            {
                sigma += z_new_ptr[j] * z_new_ptr[j];
            }

            std::vector<double> L_new((k_ + 1) * (k_ + 1));
            bool success = LinearAlgebra::cholesky_extend(
                L_S_.data(), k_,
                s.data(), sigma,
                L_new.data());

            if (success)
            {
                L_S_ = std::move(L_new);
                active_constraints_.push_back(constraint_idx);
                k_++;
                return true;
            }

            return false;
        }

        bool SchurComplementManager::remove_constraint(int face_index)
        {
            if (face_index < 0 || face_index >= k_)
            {
                return false;
            }

            if (k_ == 1)
            {
                L_S_.clear();
                active_constraints_.clear();
                k_ = 0;
                return true;
            }

            std::vector<double> L_new((k_ - 1) * (k_ - 1));
            bool success = LinearAlgebra::cholesky_shrink(
                L_S_.data(), k_, face_index,
                L_new.data());

            if (success)
            {
                L_S_ = std::move(L_new);
                active_constraints_.erase(active_constraints_.begin() + face_index);
                k_--;
                return true;
            }

            return false;
        }

        bool SchurComplementManager::solve(
            const std::vector<double> &rhs,
            std::vector<double> &lambda) const
        {
            if (k_ == 0)
            {
                lambda.clear();
                return true;
            }

            if (static_cast<int>(rhs.size()) != k_)
            {
                return false;
            }

            lambda.resize(k_);

            std::vector<double> y(k_);
            LinearAlgebra::forward_sub(L_S_.data(), rhs.data(), y.data(), k_);
            LinearAlgebra::backward_sub(L_S_.data(), y.data(), lambda.data(), k_);

            return true;
        }

    } // namespace internal
} // namespace geoqp
