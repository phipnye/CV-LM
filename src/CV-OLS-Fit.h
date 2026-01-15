#pragma once

#include <RcppEigen.h>

#include <optional>

#include "Enums-enums.h"
#include "Stats-computations.h"
#include "Utils-Decompositions-utils.h"

namespace CV::OLS {

template <Enums::AnalyticMethod CVMethod>
class Fit {
  // Eigen objects
  const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> qtz_;
  const Eigen::VectorXd qTy_;
  const std::optional<Eigen::VectorXd> diagHat_;  // not needed for GCV

  // Scalars
  const Eigen::Index nrow_;
  const Eigen::Index rank_;

 public:
  explicit Fit(const Eigen::Map<Eigen::VectorXd>& y,
               const Eigen::Map<Eigen::MatrixXd>& x, const double threshold)
      // Compute complete orthogonal decomposition of X (XP = QTZ)
      : qtz_{Utils::Decompositions::cod(x, threshold)},

        // Construct Q'y
        qTy_{qtz_.householderQ().transpose() * y},

        // Diagonal of hat matrix
        diagHat_{[&]() -> std::optional<Eigen::VectorXd> {
          // We only need the diagonal entries of the projection matrix for
          // LOOCV
          if constexpr (CVMethod == Enums::AnalyticMethod::LOOCV) {
            // Leverage values: h_ii = [X(X'X)^-1 X']_ii
            // Using QR, H = Q_1Q_1' so h_ii = sum_{j=1}^{rank} q_{ij}^2
            // (rowwise squared norm of thin Q)
            const Eigen::MatrixXd qThin{
                qtz_.householderQ() *
                Eigen::MatrixXd::Identity(x.rows(), qtz_.rank())};

            // Use NRVO to prevent against potential dangling references with
            // expression templates
            Eigen::VectorXd diagHat{qThin.rowwise().squaredNorm()};
            return diagHat;
          } else {
            return std::nullopt;
          }
        }()},

        // Scalars
        nrow_{x.rows()},
        rank_{qtz_.rank()} {}

  // Class should be immobile based on its intended use
  Fit(const Fit&) = delete;
  Fit& operator=(const Fit&) = delete;

  [[nodiscard]] double cv() const {
    if constexpr (CVMethod == Enums::AnalyticMethod::GCV) {
      const double traceHat{static_cast<double>(rank_)};  // trace(H) = rank(X)
      return Stats::gcv(rss(), traceHat, nrow_);
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
    // Calculate RSS (using the full n x n orthogonal matrix Q, we transform y
    // into Q'y and partition the squared norm of y into two components:
    // ||y||^2 = ||(Q'y).head(rank)||^2 + ||(Q'y).tail(n - rank)||^2
    // where the first term is the ESS the second term is the RSS [see "Matrix
    // Computations" Golub p.263 4th ed.]
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

}  // namespace CV::OLS
