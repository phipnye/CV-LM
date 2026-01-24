#ifndef CV_LM_CV_DETERMINISTIC_OLS_FIT_H
#define CV_LM_CV_DETERMINISTIC_OLS_FIT_H

#include <RcppEigen.h>

#include <utility>

#include "ConstexprOptional.h"
#include "Enums.h"
#include "Stats.h"
#include "Utils-Data.h"
#include "Utils-Decompositions.h"

namespace CV::Deterministic::OLS {

template <Enums::AnalyticMethod Analytic, Enums::CenteringMethod Centering>
class Fit {
  // Static boolean flags for control-flow/methodology reasoning
  static constexpr bool meanCenter{Centering == Enums::CenteringMethod::Mean};
  static constexpr bool useLOOCV{Analytic == Enums::AnalyticMethod::LOOCV};

  // Eigen objects
  const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> qtz_;
  const Eigen::VectorXd qTy_;

  // Scalars
  const Eigen::Index nrow_;
  const Eigen::Index rank_;

  // Optional members
  using OptionalVector = ConstexprOptional<useLOOCV, Eigen::VectorXd>;
  const OptionalVector diagHat_;  // only needed for loocv

 public:
  // Main ctor
  explicit Fit(const Eigen::Map<Eigen::VectorXd>& y,
               const Eigen::Map<Eigen::MatrixXd>& x, const double threshold)
      // Compute complete orthogonal decomposition of X (XP = QTZ)
      : qtz_{Utils::Decompositions::cod<Centering>(x, threshold)},

        // Construct Q'y
        qTy_{qtz_.householderQ().transpose() *
             (meanCenter ? Utils::Data::centerResponse(y) : y)},

        // Scalars
        nrow_{x.rows()},
        rank_{qtz_.rank()},

        // Diagonal of hat matrix
        diagHat_{[&]() -> OptionalVector {
          // We only need the diagonal entries of the hat matrix for LOOCV
          if constexpr (useLOOCV) {
            // Leverage values: h_ii = [X(X'X)^-1 X']_ii
            // Using QR=QTZ, H = Q_1Q_1' so h_ii = sum_{j=1}^{rank} q_{ij}^2
            // (rowwise squared norm of thin Q)
            const Eigen::MatrixXd qThin{
                qtz_.householderQ() *
                Eigen::MatrixXd::Identity(x.rows(), qtz_.rank())};
            Eigen::VectorXd diagHat{qThin.rowwise().squaredNorm()};

            // If the data was centered, we need to add 1/n (diag(11')/n) to the
            // diagonal entries to capture the dropped intercept column
            if constexpr (meanCenter) {
              diagHat.array() += (1.0 / static_cast<double>(qtz_.rows()));
            }

            return OptionalVector::make(std::move(diagHat));
          } else {
            return OptionalVector::empty();
          }
        }()} {}

  // Public facing generic method for obtaining deterministic CV result
  [[nodiscard]] double cv() const {
    if constexpr (useLOOCV) {
      return Stats::loocv(residuals(), diagHat_.value());
    } else {
      Enums::assertExpected<Analytic, Enums::AnalyticMethod::GCV>();

      // If the data was centered we need to add one to capture the dropped
      // intercept column
      constexpr double correction{meanCenter ? 1.0 : 0.0};

      // trace(H) = rank(X)
      const double traceHat{static_cast<double>(rank_) + correction};
      return Stats::gcv(rss(), traceHat, nrow_);
    }
  }

 private:
  // Sum of squared residuals
  [[nodiscard]] double rss() const {
    // Calculate RSS (using the full n x n orthogonal matrix Q, we transform y
    // into Q'y and partition the squared norm of y into two components:
    // ||y||^2 = ||(Q'y).head(rank)||^2 + ||(Q'y).tail(n - rank)||^2
    // where the first term is the ESS and the second term is the RSS [see
    // "Matrix Computations" Golub p.263 4th ed.]
    return qTy_.tail(nrow_ - rank_).squaredNorm();
  }

  [[nodiscard]] Eigen::VectorXd residuals() const {
    // Zero out the components in the column space (the first 'rank' elements,
    // leaving only the components in the orthogonal complement) [see "Matrix
    // Computations" Golub p.263 4th ed.]
    Eigen::VectorXd resid{Eigen::VectorXd::Zero(nrow_)};
    const Eigen::Index tailSize{nrow_ - rank_};
    resid.tail(tailSize) = qTy_.tail(tailSize);

    // Transform back to original space: resid = Q * [0, Q'y.tail(n - rank)]'
    resid.applyOnTheLeft(qtz_.householderQ());
    return resid;
  }
};

}  // namespace CV::Deterministic::OLS

#endif  // CV_LM_CV_DETERMINISTIC_OLS_FIT_H
