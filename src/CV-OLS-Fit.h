#pragma once

#include <RcppEigen.h>

#include <optional>

namespace CV::OLS {

template <bool NeedHat>
class Fit {
  // Eigen objects
  const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod_;
  const Eigen::VectorXd qty_;
  const std::optional<Eigen::ArrayXd> diagH_;  // not need if !NeedHat

  // Scalars
  const Eigen::Index nrow_;
  const Eigen::Index rank_;

 public:
  explicit Fit(const Eigen::Map<Eigen::VectorXd>& y,
               const Eigen::Map<Eigen::MatrixXd>& x, const double threshold)
      // Compute complete orthogonal decomposition of x
      : cod_{[&]() {
          // Perform QR/Complete orthogonal decomposition XP = QTZ
          Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod{x};

          // Prescribe threshold to decomposition where singular values are
          // considered zero "A pivot will be considered nonzero if its absolute
          // value is strictly greater than |pivot|⩽threshold×|maxpivot| " -
          // this threshold only affects other methods like solve and rank, not
          // the decomposition itself
          cod.setThreshold(threshold);
          return cod;
        }()},

        // Construct Q'y
        qty_{cod_.householderQ().transpose() * y},

        // Diagonal of hat matrix
        diagH_{[&]() {
          if constexpr (NeedHat) {
            // Leverage values: h_ii = [X(X'X)^-1 X']_ii
            // Using QR, H = Q_1Q_1' so h_ii = sum_{j=1}^{rank} q_{ij}^2
            // (rowwise squared norm of thin Q)
            const Eigen::MatrixXd qThin{
                cod_.householderQ() *
                Eigen::MatrixXd::Identity(x.rows(), cod_.rank())};
            Eigen::ArrayXd diagH{qThin.rowwise().squaredNorm().array()};
            return diagH;
          } else {
            return std::nullopt;
          }
        }()},

        // Scalars
        nrow_{x.rows()},
        rank_{cod_.rank()} {}

  // Class should be immobile based on its intended use
  Fit(const Fit&) = delete;
  Fit& operator=(const Fit&) = delete;

  // GCV = MSE / (1 - trace(H)/n)^2
  [[nodiscard]] double gcv() const {
    // Thought was given to preventing zero-division but decided returning inf
    // is the most "mathematically" honest answer
    const double mrl{meanResidualLeverage()};
    return mse() / (mrl * mrl);
  }

  // LOOCV_error_i = e_i / (1 - h_ii))
  [[nodiscard]] double loocv() const {
    static_assert(NeedHat,
                  "LOOCV requires Fit template parameter NeedHat = true");
    return (residuals().array() / (1.0 - *diagH_)).square().mean();
  }

 private:
  // Sum of squared residuals
  [[nodiscard]] double rss() const {
    // Calculate RSS (using the full n x n orthogonal matrix Q, we transform y
    // into Q'y and partition the squared norm of y into two components:
    // ||y||^2 = ||(Q'y).head(rank)||^2 + ||(Q'y).tail(n - rank)||^2
    // where the first term is the ESS the second term is the RSS
    return qty_.tail(nrow_ - rank_).squaredNorm();
  }

  // Mean squared error
  [[nodiscard]] double mse() const {
    return rss() / static_cast<double>(nrow_);
  }

  // Mean residual leverage = (1 - trace(H)/n)
  [[nodiscard]] double meanResidualLeverage() const {
    return 1.0 - (static_cast<double>(rank_) /
                  static_cast<double>(nrow_));  // trace(H) = rank(X)
  }

  [[nodiscard]] Eigen::VectorXd residuals() const {
    // Zero out the components in the column space (the first 'rank' elements,
    // leaving only the components in the orthogonal complement)
    Eigen::VectorXd resid{qty_};
    resid.head(rank_).setZero();

    // Transform back to original space: resid = Q * [0, Q'y.tail(n - rank)]'
    resid.applyOnTheLeft(cod_.householderQ());
    return resid;
  }
};
}  // namespace CV::OLS
