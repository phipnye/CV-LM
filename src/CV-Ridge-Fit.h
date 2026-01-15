#pragma once

#include <RcppEigen.h>

#include <optional>

#include "Enums-enums.h"
#include "Stats-computations.h"
#include "Utils-Decompositions-utils.h"

namespace CV::Ridge {

template <Enums::AnalyticMethod CVMethod>
class Fit {
  // Eigen objects
  const Eigen::BDCSVD<Eigen::MatrixXd> udvT_;
  const Eigen::VectorXd coordShrinkFactors_;
  const std::optional<Eigen::VectorXd> diagHat_;

  // References
  const Eigen::Map<Eigen::VectorXd>& y_;

  // Scalars
  const Eigen::Index nrow_;

  // Flags
  const bool centered_;

 public:
  explicit Fit(const Eigen::Map<Eigen::VectorXd>& y,
               const Eigen::Map<Eigen::MatrixXd>& x, const double threshold,
               const double lambda, const bool centered)
      : udvT_{[&]() -> Eigen::BDCSVD<Eigen::MatrixXd> {
          constexpr bool checkSuccess{true};  // check success of SVD
          return Utils::Decompositions::svd<checkSuccess>(
              x, Eigen::ComputeThinU, threshold);
        }()},

        // Coordinate shrinkage factors = d_j^2 / (d_j^2 + lambda) [See ESL
        // p.66]
        coordShrinkFactors_{[&]() -> Eigen::VectorXd {
          const Eigen::VectorXd singularValsSq{
              Utils::Decompositions::getSingularVals(udvT_).array().square()};

          // Lambda should be strictly positive at this point so we do not need
          // to worry about zero division
          Eigen::VectorXd coordShrinkFactors{singularValsSq.array() /
                                             (singularValsSq.array() + lambda)};

          // Using explicit NRVO to prevent unexpected errors from returning
          // expression templates with dangling references
          return coordShrinkFactors;
        }()},

        // Diagonal of hat matrix
        diagHat_{[&]() -> std::optional<Eigen::VectorXd> {
          // We only need the diagonal entries of the projection matrix for
          // LOOCV
          if constexpr (CVMethod == Enums::AnalyticMethod::LOOCV) {
            // h_ii = sum_{j=1}^{p} (u_ij^2 * ((d_j^2) / (d_j^2 + lambda)))
            Eigen::VectorXd diagHat{
                (udvT_.matrixU().array().square().rowwise() *
                 coordShrinkFactors_.array().transpose())
                    .rowwise()
                    .sum()};

            // If the data was centered in R, we need to add 1/n to the diagonal
            // entries to capture the dropped intercept column (manually
            // verified in R this is the case regardless of whether the data is
            // narrow or wide)
            if (centered) {
              diagHat.array() += (1.0 / static_cast<double>(x.rows()));
            }

            return diagHat;
          } else {
            return std::nullopt;
          }
        }()},

        // Refereneces
        y_{y},

        // Scalars
        nrow_{x.rows()},

        // Flags
        centered_{centered} {}

  // Class should be immobile due to its intended use
  Fit(const Fit&) = delete;
  Fit& operator=(const Fit&) = delete;

  [[nodiscard]] double cv() const {
    // We should never have zero division here for lambda > 0
    if constexpr (CVMethod == Enums::AnalyticMethod::GCV) {
      return Stats::gcv(rss(), traceHat(), nrow_);
    } else {
      // This assert should also serve as making sure the diagHat_ member has a
      // value before "dereferencing"
      Enums::assertExpected<CVMethod, Enums::AnalyticMethod::LOOCV>();
      return Stats::loocv(residuals(), *diagHat_);
    }
  }

 private:
  // Sum of squared residuals
  [[nodiscard]] double rss() const {
    const Eigen::MatrixXd& u{udvT_.matrixU()};
    const Eigen::VectorXd uTy{u.transpose() * y_};
    /*
     * resid = y - X * beta
     *       = y - X * (X'X + LI)^-1 X'y
     *       = y - U D(D^2 + LI)^-1 DU'y
     *
     * rss = ||y - U D(D^2 + LI)^-1 DU'y||^2
     *     = ||y - USU'y||^2
     *     = [y - USU'y]'[y - USU'y]
     *     = [y' - (USU'y)'][y - USU'y]
     *     = y'y - y'USU'y - (USU'y)'y + (USU'y)'USU'y
     *     = ||y||^2 - y'USU'y - y'USU'y + y'US'U'USU'y
     *     = ||y||^2 - 2 * y'USU'y + y'U S^2 U'y
     *     = ||y||^2 - (U'y)' 2S U'y + (U'y)' S^2 U'y
     *     = ||y||^2 - sum_{j} (2 s_j - s_j^2) (U'y)_j^2
     */
    return y_.squaredNorm() - ((coordShrinkFactors_.array() *
                                (2.0 -  // NOLINT(*-avoid-magic-numbers)
                                 coordShrinkFactors_.array())) *
                               uTy.array().square())
                                  .sum();
  }

  // Trace of hat matrix
  [[nodiscard]] double traceHat() const {
    // If the data was centered in R, we need to add one to capture the dropped
    // intercept column
    const double correction{centered_ ? 1.0 : 0.0};

    // Trace(H) sum_{j=1}^{p} d_j^2 / (d_j^2 + lambda) [See ESL p.68]
    return correction + coordShrinkFactors_.sum();
  }

  // Ridge regression residuals
  [[nodiscard]] Eigen::VectorXd residuals() const {
    /*
     * [See ESL p.66]
     * resid = y - X * beta
     *       = y - X * (X'X + LI)^-1 X'y
     *       = y - U D(D^2 + LI)^-1 DU'y
     *       = y - sum_{j=1}^{p} u_j (d_j^2) / (d_j^2 + lambda) u_j'y
     */
    const Eigen::MatrixXd& u{udvT_.matrixU()};
    const Eigen::VectorXd uTy{u.transpose() * y_};

    // Using explicit NRVO to prevent unexpected errors from returning
    // expression templates with dangling references
    Eigen::VectorXd resid{
        y_ - (u * (coordShrinkFactors_.array() * uTy.array()).matrix())};
    return resid;
  }
};

}  // namespace CV::Ridge
