#pragma once

#include <RcppEigen.h>

#include <utility>

#include "ConstexprOptional.h"
#include "Enums.h"
#include "ResponseWrapper.h"
#include "Stats-computations.h"
#include "Utils-Decompositions.h"

namespace CV::Ridge {

template <Enums::AnalyticMethod Analytic, Enums::CenteringMethod Centering>
class Fit {
  // Static boolean flags for control-flow/methodology reasoning
  static constexpr bool meanCenter{Centering == Enums::CenteringMethod::Mean};
  static constexpr bool useLOOCV{Analytic == Enums::AnalyticMethod::LOOCV};

  // Eigen objects
  const Eigen::BDCSVD<Eigen::MatrixXd> udvT_;
  const Eigen::VectorXd coordShrinkFactors_;

  // Conditional object depending on whether we need to center the data
  ResponseWrapper<Centering> y_;

  // Scalars
  const Eigen::Index nrow_;

  // Optional members
  using OptionalVector = ConstexprOptional<useLOOCV, Eigen::VectorXd>;
  const OptionalVector diagHat_;  // only needed for LOOCV

  // Coordinate shrinkage factors = d_j^2 / (d_j^2 + lambda) [See ESL p.66]
  static Eigen::VectorXd constructCoordShrinkFactors(
      const Eigen::BDCSVD<Eigen::MatrixXd>& udvT, const double lambda) {
    const Eigen::VectorXd singularValsSq{
        Utils::Decompositions::getSingularVals(udvT).array().square()};

    // Lambda should be strictly positive at this point so we do not need
    // to worry about zero division
    return Eigen::VectorXd{singularValsSq.array() /
                           (singularValsSq.array() + lambda)};
  }

 public:
  explicit Fit(const Eigen::Map<Eigen::VectorXd>& y,
               const Eigen::Map<Eigen::MatrixXd>& x, const double threshold,
               const double lambda)
      // Either SVD of original or centered design matrix
      : udvT_{Utils::Decompositions::svd<Centering>(x, Eigen::ComputeThinU,
                                                    threshold)},
        coordShrinkFactors_{constructCoordShrinkFactors(udvT_, lambda)},

        // Either centered response vector or reference to original response
        // from R
        y_{y},

        // Scalars
        nrow_{x.rows()},

        // Diagonal of hat matrix
        diagHat_{[&]() -> OptionalVector {
          // We only need the diagonal entries of the hat matrix for LOOCV
          if constexpr (useLOOCV) {
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
            if constexpr (meanCenter) {
              diagHat.array() += (1.0 / static_cast<double>(udvT_.rows()));
            }

            return OptionalVector{std::move(diagHat)};
          } else {
            return OptionalVector::empty();
          }
        }()} {}

  // Class should be immobile due to its intended use
  Fit(const Fit&) = delete;
  Fit& operator=(const Fit&) = delete;

  [[nodiscard]] double cv() const {
    if constexpr (useLOOCV) {
      return Stats::loocv(residuals(), *diagHat_);
    } else {
      Enums::assertExpected<Analytic, Enums::AnalyticMethod::GCV>();
      return Stats::gcv(rss(), traceHat(), nrow_);
    }
  }

 private:
  // Trace of hat matrix
  [[nodiscard]] double traceHat() const {
    // If the data was centered in R, we need to add one to capture the dropped
    // intercept column
    constexpr double correction{meanCenter ? 1.0 : 0.0};

    // Trace(H) = sum_{j=1}^{p} d_j^2 / (d_j^2 + lambda) [See ESL p.68]
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
    const auto& u{udvT_.matrixU()};
    const auto& y{y_.get()};
    const Eigen::VectorXd uTy{u.transpose() * y};
    return Eigen::VectorXd{
        y - (u * (coordShrinkFactors_.array() * uTy.array()).matrix())};
  }

  // Sum of squared residuals
  [[nodiscard]] double rss() const {
    const auto& y{y_.get()};
    const Eigen::VectorXd uTy{udvT_.matrixU().transpose() * y};

    /*
     * rss = ||resid||^2
     *     = ||y - U D(D^2 + LI)^-1 DU'y||^2
     *     = ||y - USU'y||^2
     *     = [y - USU'y]'[y - USU'y]
     *     = [y' - (USU'y)'][y - USU'y]
     *     = y'y - y'USU'y - (USU'y)'y + (USU'y)'USU'y
     *     = ||y||^2 - y'USU'y - y'USU'y + y'US'U'USU'y
     *     = ||y||^2 - 2 * y'USU'y + y'U S^2 U'y
     *     = ||y||^2 - (U'y)' 2S U'y + (U'y)' S^2 U'y
     *     = ||y||^2 - sum_{j} (2 s_j - s_j^2) (U'y)_j^2
     */
    const double correctionTerm{
        ((coordShrinkFactors_.array() * (2.0 - coordShrinkFactors_.array())) *
         uTy.array().square())
            .sum()};
    return y.squaredNorm() - correctionTerm;
  }
};

}  // namespace CV::Ridge
