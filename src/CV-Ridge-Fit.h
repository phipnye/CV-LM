#pragma once

#include <RcppEigen.h>

#include <algorithm>
#include <optional>

namespace CV::Ridge {

template <bool NeedHat>
class Fit {
  // Eigen objects
  const Eigen::MatrixXd
      inv_;  // holds (X'X + lambda * I)^-1 if primal/narrow, or (XX'
             // + lambda * I)^-1 if dual/wide
  const Eigen::VectorXd resid_;
  const std::optional<Eigen::ArrayXd> diagH_;

  // Scalars
  const Eigen::Index nrow_;
  const Eigen::Index ncol_;
  const double lambda_;

  // Flags
  const bool centered_;

 public:
  explicit Fit(const Eigen::Map<Eigen::VectorXd>& y,
               const Eigen::Map<Eigen::MatrixXd>& x, const double lambda,
               const bool centered)
      :  // Decompose and invert appropriate matrix
        inv_{[&]() {
          // Create X'X + lambda * I or XX' + lambda * I
          const Eigen::Index nrow{x.rows()};
          const Eigen::Index ncol{x.cols()};
          const Eigen::Index dim{std::min(nrow, ncol)};
          Eigen::MatrixXd mat{Eigen::MatrixXd::Zero(dim, dim)};
          mat.diagonal().fill(lambda);

          if (nrow < ncol) {
            // Outer product matrix XX' + lambda * I
            mat.selfadjointView<Eigen::Lower>().rankUpdate(x);
          } else {
            // Regularized covariance matrix X'X + lambda * I
            mat.selfadjointView<Eigen::Lower>().rankUpdate(x.transpose());
          }

          // Eigen documentation states: "While the
          // Cholesky decomposition is particularly useful to solve selfadjoint
          // problems like D^*D x = b, for that purpose, we recommend the
          // Cholesky decomposition without square root which is more stable and
          // even faster."
          const Eigen::LDLT<Eigen::Ref<Eigen::MatrixXd>> ldlt{
              mat};  // use in-place decomposition

          // LDLT supports solving in-place
          Eigen::MatrixXd inv{Eigen::MatrixXd::Identity(dim, dim)};
          ldlt.solveInPlace(inv);  // returns true (no need to check)
          return inv;
        }()},

        /*
         * resid   = y - X * beta
         * primal: = y - X (X'X + lambda * I)^-1 * X'y
         * dual:   = y - X (X' * alpha) where beta = X' * alpha
         *         = y - X [X' (XX' + lambda * I)^-1 * y]
         *         = [I - XX'(XX' + lambda * I)^-1] * y
         *         = [I - XX'Z] * y
         *         = [Z^-1 Z - XX'Z] * y
         *         = [Z^-1 - XX'] * Zy
         *         = [(XX' + lambda * I) - XX'] * (XX' + lambda * I)^-1 y
         *         = [lambda * I] * (XX' + lambda * I)^-1 y
         *         = lambda * (XX' + lambda * I)^-1 y
         */
        resid_{[&]() {
          const Eigen::Index nrow{x.rows()};
          Eigen::VectorXd resid(nrow);

          if (nrow < x.cols()) {
            resid.noalias() = lambda * (inv_ * y);
          } else {
            resid.noalias() = y - (x * (inv_ * (x.transpose() * y)));
          }

          return resid;
        }()},

        // Diagonal of hat matrix
        diagH_{[&]() {
          if constexpr (NeedHat) {
            const Eigen::Index nrow{x.rows()};
            Eigen::ArrayXd diagH(nrow);

            if (nrow < x.cols()) {
              // diag(H) = diag(I - lambda * (XX' + lambda * I)^-1)
              diagH = 1.0 - (lambda * inv_.diagonal().array());
            } else {
              // h_ii = x_i' * (X'X + lambda * I)^-1 * x_i
              diagH = (x * inv_).cwiseProduct(x).rowwise().sum().array();
            }

            // If the data was centered in R, we need to add 1/n to the diagonal
            // entries to capture the dropped intercept column (manually
            // verified in R this is the case regardless of whether the data is
            // narrow or wide)
            if (centered) {
              diagH += (1.0 / static_cast<double>(nrow));
            }

            return diagH;
          } else {
            return std::nullopt;
          }
        }()},

        // Scalars
        nrow_{x.rows()},
        ncol_{x.cols()},
        lambda_{lambda},

        // Flags
        centered_{centered} {}

  // Class should be immobile due to its intended use
  Fit(const Fit&) = delete;
  Fit& operator=(const Fit&) = delete;

  // GCV = MSE / (1 - trace(H)/n)^2
  [[nodiscard]] double gcv() const {
    const double mrl{meanResidualLeverage()};
    return mse() / (mrl * mrl);
  }

  // LOOCV_error_i = e_i / (1 - h_ii))
  [[nodiscard]] double loocv() const {
    static_assert(NeedHat,
                  "LOOCV requires Fit template parameter NeedHat = true");
    return (resid_.array() / (1.0 - *diagH_)).square().mean();
  }

 private:
  // Sum of squared residuals
  [[nodiscard]] double rss() const { return resid_.squaredNorm(); }

  // Mean squared error
  [[nodiscard]] double mse() const {
    return rss() / static_cast<double>(nrow_);
  }

  // Mean residual leverage = (1 - trace(H)/n)
  [[nodiscard]] double meanResidualLeverage() const {
    return 1.0 - (traceH() / static_cast<double>(nrow_));
  }

  // trace of hat matrix
  [[nodiscard]] double traceH() const {
    // If the data was centered in R, we need to add one to capture the dropped
    // intercept column
    const double correction{centered_ ? 1.0 : 0.0};

    if (nrow_ < ncol_) {
      // Using trace(AB) = trace(BA) and XX' = (XX' + lambda * I) - lambda * I
      // trace(H) = n - lambda * trace((XX' + lambda * I)^-1)
      return static_cast<double>(nrow_) + correction - lambda_ * inv_.trace();
    }

    // Using trace(AB) = trace(BA) and X'X = (X'X + lambda * I) - lambda * I
    // trace(H) = p - lambda * trace((X'X + lambda * I)^-1)
    return static_cast<double>(ncol_) + correction - lambda_ * inv_.trace();
  }
};

}  // namespace CV::Ridge
