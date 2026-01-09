#pragma once

#include <RcppEigen.h>

#include <type_traits>
#include <utility>

namespace CV::OLS {

template <bool NeedHat>
class Fit {
  const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod_;
  const Eigen::Index nrow_;
  const Eigen::Index rank_;
  const Eigen::VectorXd qty_;
  // Using std::conditional to avoid allocating memory for diagH_ if not needed
  const std::conditional_t<NeedHat, Eigen::ArrayXd, bool> diagH_;

 public:
  explicit Fit(Eigen::VectorXd y, const Eigen::MatrixXd& x,
               const double threshold)
      : cod_{[&]() {
          // Pre-allocate
          Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(x.rows(),
                                                                      x.cols());

          // Prescribe threshold to QR decomposition where singular values are
          // considered zero "A pivot will be considered nonzero if its absolute
          // value is strictly greater than |pivot|⩽threshold×|maxpivot| "
          cod.setThreshold(threshold);

          // Perform QR/Complete orthogonal decomposition XP = QTZ
          cod.compute(x);
          return cod;
        }()},
        nrow_{x.rows()},
        rank_{cod_.rank()},
        qty_{[&]() {
          // Construct Q'y
          Eigen::VectorXd qty{std::move(y)};
          qty.applyOnTheLeft(cod_.householderQ().transpose());
          return qty;
        }()},
        diagH_{[&]() {
          if constexpr (NeedHat) {
            // Leverage values: h_ii = [X(X'X)^-1 X']_ii
            // Using QR, H = Q_1Q_1' so h_ii = sum_{j=1}^{rank} q_{ij}^2
            // (rowwise squared norm of thin Q)
            Eigen::MatrixXd qThin{Eigen::MatrixXd::Identity(nrow_, rank_)};
            qThin.applyOnTheLeft(cod_.householderQ());
            Eigen::ArrayXd diagH{qThin.rowwise().squaredNorm().array()};
            return diagH;
          } else {
            return false;
          }
        }()} {}

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
    return (residuals().array() / (1.0 - diagH_)).square().mean();
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
