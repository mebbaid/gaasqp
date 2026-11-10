#ifndef GEOQP_BOX_GEOMETRY_HPP
#define GEOQP_BOX_GEOMETRY_HPP

// Geometry of a strictly convex QP over an affine image of a box,
//
//     min ½ xᵀHx + gᵀx   s.t.  x ∈ P = { Ty + c : ℓ ≤ y ≤ u },
//
// relative to the exterior unconstrained minimiser x̄ = −H⁻¹g.
//
// This module computes the objects of the "visible-facet" result that
// replaces the (false) half-space theorem  ηᵀT⁻¹(x★ − c₀) ≥ 0  with η = s/h:
//
//   ξ-coordinates   ξᵢ = sᵢ(yᵢ − μᵢ)/hᵢ,   P ↦ [−1,1]ⁿ,  ξ̄ ≥ 0
//   violated set    V = { i : ξ̄ᵢ > 1 } = { i : ȳᵢ > uᵢ or ȳᵢ < ℓᵢ },  k = |V|
//   visible facet   { ξᵢ = 1 },  i ∈ V
//   corrected cut   Σ_{i∈V} ξᵢ ≥ 2 − k
//   exit facet      i₀ = argmax ξ̄ᵢ  (ray from any good vertex toward x̄)
//   H-metric half   ⟨x̄ − c₀, x − c₀⟩_H ≥ ‖x − c₀‖²_H
//
// All matrices are dense, row-major, n×n.  No Eigen dependency.

#include <limits>
#include <vector>

namespace geoqp
{
    namespace geometry
    {
        // P = { Ty + c : lower ≤ y ≤ upper }
        struct AffineBox
        {
            int n = 0;
            std::vector<double> T; // n×n row-major, nonsingular
            std::vector<double> c;
            std::vector<double> lower;
            std::vector<double> upper;

            // T = I, c = 0.
            static AffineBox axis_aligned(const std::vector<double> &lower,
                                          const std::vector<double> &upper);

            // Half-space representation (Remark 1):
            //   A = [T⁻¹; −T⁻¹],  b = [u + T⁻¹c; −ℓ − T⁻¹c]   (2n rows).
            // Row i < n is the upper wall of coordinate i, row n+i the lower wall.
            void halfspace(std::vector<double> &A, std::vector<double> &b) const;
        };

        // Everything that depends on (P, x̄) but not on H.
        struct BoxFrame
        {
            int n = 0;
            std::vector<double> T, Tinv, c;
            std::vector<double> lower, upper;
            std::vector<double> mu;   // ½(ℓ + u)
            std::vector<double> h;    // ½(u − ℓ)
            std::vector<double> c0;   // Tμ + c
            std::vector<double> xbar;
            std::vector<double> ybar; // T⁻¹(x̄ − c)
            std::vector<double> s;    // +1 if ȳᵢ ≥ μᵢ, else −1
            std::vector<double> xibar;
            std::vector<int> violated; // V, ascending

            int k() const { return static_cast<int>(violated.size()); }
            bool is_violated(int i) const;
            // ξ̄ᵢ ≥ 1, decided on ȳ against the bounds (no rounding through ξ).
            bool at_or_beyond_bound(int i) const;
        };

        BoxFrame make_frame(const AffineBox &box, const std::vector<double> &xbar);

        // ── coordinate maps ─────────────────────────────────────────────
        std::vector<double> to_y(const BoxFrame &f, const std::vector<double> &x);
        std::vector<double> to_xi(const BoxFrame &f, const std::vector<double> &x);
        std::vector<double> from_xi(const BoxFrame &f, const std::vector<double> &xi);
        std::vector<double> D_matrix(const BoxFrame &f);     // diag(s/h)·T⁻¹
        std::vector<double> D_inverse(const BoxFrame &f);    // T·diag(s·h)
        // H̃ = D⁻ᵀ H D⁻¹
        std::vector<double> transformed_hessian(const BoxFrame &f, const std::vector<double> &H);
        // H = Dᵀ H̃ D
        std::vector<double> original_hessian(const BoxFrame &f, const std::vector<double> &Ht);

        // ── A. good vertices ────────────────────────────────────────────
        // σ ∈ {±1}ⁿ in ξ-coordinates.  Good  ⇔  σᵢ = −1 whenever ξ̄ᵢ ≥ 1.
        bool is_good_vertex_sign(const BoxFrame &f, const std::vector<int> &sigma);
        std::vector<double> vertex_from_sign(const BoxFrame &f, const std::vector<int> &sigma);
        std::vector<double> antipodal_vertex(const BoxFrame &f); // ξ = −1, ŷ = μ − s∘h
        // Lemma 1 rule: ŷᵢ = ℓᵢ if ȳᵢ > ℓᵢ, else uᵢ.
        std::vector<double> lemma1_vertex(const BoxFrame &f);

        // ── E. exit facet ───────────────────────────────────────────────
        struct Wall
        {
            int coord = -1;
            bool upper = true; // y_coord = u_coord (true) or ℓ_coord (false)
            double t = std::numeric_limits<double>::infinity();
        };

        // Trace x(t) = v + t(x̄ − v) in y-space and return every wall reached at
        // the minimal t ≥ 0 (several on a ridge), ascending by coordinate.
        std::vector<Wall> exit_walls_by_ray(const BoxFrame &f, const std::vector<double> &v,
                                            double rel_tie_tol = 1e-12);
        // tᵢ = 2/(ξ̄ᵢ + 1): ray from the antipodal vertex reaches wall {ξᵢ = 1}.
        std::vector<double> antipodal_exit_times(const BoxFrame &f);
        // The visible wall {ξᵢ = 1} of coordinate i.
        Wall xi_plus_wall(const BoxFrame &f, int i);
        // i₀ = argmax ξ̄ᵢ (smallest index on ties), its {ξ = 1} wall.
        Wall predicted_exit_wall(const BoxFrame &f);

        // ── B/C. visible facets and corrected cut ───────────────────────
        // Indices i ∈ V with |ξᵢ(x) − 1| ≤ tol.
        std::vector<int> visible_facets_containing(const BoxFrame &f, const std::vector<double> &x,
                                                   double tol);
        // Old (false) claim:  Σᵢ ξᵢ = ηᵀT⁻¹(x − c₀) ≥ 0  with ηᵢ = sᵢ/hᵢ for all i.
        double old_halfspace_value(const BoxFrame &f, const std::vector<double> &x);
        // Corrected:  ηᵀT⁻¹(x − c₀) = Σ_{i∈V} ξᵢ,  ηᵢ = sᵢ/hᵢ on V and 0 off V.
        std::vector<double> corrected_eta(const BoxFrame &f);
        double corrected_cut_value(const BoxFrame &f, const std::vector<double> &x);
        double corrected_cut_threshold(const BoxFrame &f); // 2 − k
        double corrected_cut_margin(const BoxFrame &f, const std::vector<double> &x);

        // ── F. H-metric central half ────────────────────────────────────
        // ⟨x̄ − c₀, x − c₀⟩_H − ‖x − c₀‖²_H   (claimed ≥ 0 at x = x★)
        double h_halfspace_margin(const BoxFrame &f, const std::vector<double> &H,
                                  const std::vector<double> &x);
        // ½‖x̄ − c₀‖_H − ‖x − ½(c₀ + x̄)‖_H   (claimed ≥ 0 at x = x★)
        double h_ball_margin(const BoxFrame &f, const std::vector<double> &H,
                             const std::vector<double> &x);
        // ⟨x̄ − c₀, v̂ − c₀⟩_H for the antipodal vertex v̂ (< 0 ⇔ v̂ in the opposite half)
        double antipodal_side_value(const BoxFrame &f, const std::vector<double> &H);
        // 1ᵀ H̃ ξ̄   (claimed: v̂ in the opposite half ⇔ this is > 0)
        double antipodal_side_criterion(const BoxFrame &f, const std::vector<double> &H);

        // ── D. tightness ────────────────────────────────────────────────
        // For p ∈ [−1,1]ⁿ with pᵢ = 1, i ∈ V, and ε > 0, returns H (x-coordinates)
        // built from  H̃ = (1/aᵢ) eᵢeᵢᵀ + ε(I − aaᵀ/‖a‖²),  a = ξ̄ − p.
        // With g = −Hx̄ the optimum is x★ = D⁻¹p + c₀.
        std::vector<double> tightness_hessian_xi(const BoxFrame &f, const std::vector<double> &p,
                                                 int i, double eps);
        std::vector<double> tightness_hessian(const BoxFrame &f, const std::vector<double> &p,
                                              int i, double eps);

    } // namespace geometry
} // namespace geoqp

#endif // GEOQP_BOX_GEOMETRY_HPP
