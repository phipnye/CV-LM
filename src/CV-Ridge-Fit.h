#pragma once

#include <RcppEigen.h>

#include <algorithm>
#include <type_traits>

namespace CV::Ridge {

template <bool NeedHat>
class Fit {
  // Scalars
  const Eigen::Index nrow_;
  const Eigen::Index ncol_;
  const double lambda_;

  // Eigen objects
  const Eigen::MatrixXd inv_;  // holds (X'X + lambda * I)^-1 if primal, or (XX'
                               // + lambda * I)^-1 if wide
  const Eigen::VectorXd resid_;
  const std::conditional_t<NeedHat, Eigen::ArrayXd, bool> diagH_;

  // Flags
  const bool centered_;

 public:
  explicit Fit(const Eigen::Map<Eigen::VectorXd>& y,
               const Eigen::Map<Eigen::MatrixXd>& x, const double lambda,
               const bool centered)
      : nrow_{x.rows()},
        ncol_{x.cols()},
        lambda_{lambda},

        // Decompose and invert appropriate matrix
        inv_{[&]() {
          // Create X'X + lambda * I or XX' + lambda * I
          const Eigen::Index dim{std::min(nrow_, ncol_)};
          Eigen::MatrixXd mat{Eigen::MatrixXd::Zero(dim, dim)};
          mat.diagonal().fill(lambda_);

          if (nrow_ < ncol_) {
            // Kernel matrix XX' + lambda * I
            mat.selfadjointView<Eigen::Lower>().rankUpdate(x);
          } else {
            // Regularized covariance matrix X'X + lambda * I
            mat.selfadjointView<Eigen::Lower>().rankUpdate(x.transpose());
          }

          // Eigen documentation states: "While the
          // Cholesky decomposition is particularly useful to solve selfadjoint
          // problems like D^*D x = b, for that purpose, we recommend the
          // Cholesky decomposition without square root which is more stable and
          // even faster." LDLT supports in-place decomposition
          const Eigen::LDLT<Eigen::Ref<Eigen::MatrixXd>> ldlt{mat};

          // LDLT supports solving in-place
          Eigen::MatrixXd inv{Eigen::MatrixXd::Identity(dim, dim)};
          ldlt.solveInPlace(inv);  // returns true (no need to check)
          return inv;
        }()},

        /*
        resid   = y - x * beta
        primal: = y - (X'X + lambda * I)^-1 * X'y
        dual:   = lambda * (XX' + lambda * I)^-1 y
         */
        resid_{[&]() {
          Eigen::VectorXd resid(nrow_);

          if (nrow_ < ncol_) {
            resid.noalias() = lambda_ * (inv_ * y);
          } else {
            resid = y;
            resid.noalias() = x * (inv_ * (x.transpose() * y));
          }

          return resid;
        }()},

        // Diagonal of hat matrix
        diagH_{[&]() {
          if constexpr (NeedHat) {
            Eigen::ArrayXd diagH(nrow_);

            if (nrow_ < ncol_) {
              // diag(H) = diag(I - lambda * (XX' + lambda * I)^-1)
              diagH = 1.0 - (lambda_ * inv_.diagonal().array());
            } else {
              // h_ii = x_i' * (X'X + lambda * I)^-1 * x_i
              diagH = (x * inv_).cwiseProduct(x).rowwise().sum().array();
            }

            // If the data was centered in R, we need to add 1/n to the diagonal
            // entries to capture the dropped intercept column (manually
            // verified in R this is the case regardless of whether the data is
            // narrow or wide)
            if (centered) {
              diagH += (1.0 / static_cast<double>(nrow_));
            }

            return diagH;
          } else {
            return false;
          }
        }()},
        centered_{centered} {}

  Fit(const Fit&) = delete;
  Fit(Fit&&) = default;
  Fit& operator=(const Fit&) = delete;
  Fit& operator=(Fit&&) = default;

  // GCV = MSE / (1 - trace(H)/n)^2
  [[nodiscard]] double gcv() const {
    const double mrl{meanResidualLeverage()};
    return mse() / (mrl * mrl);
  }

  // LOOCV_error_i = e_i / (1 - h_ii))
  [[nodiscard]] double loocv() const {
    static_assert(NeedHat,
                  "LOOCV requires Fit template parameter NeedHat = true");
    return (resid_.array() / (1.0 - diagH_)).square().mean();
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
    //
    // Note: It was manually verified in R that for a model with an intercept
    // and centered features, the trace of the full hat matrix:
    // H = X_full (X_full' X_full + lambda * I_p)^-1 X_full'
    // is equal to the shortcut trace of the centered features plus one:
    // trace(H) = [n - lambda * trace((X_c X_c' + lambda * I_n)^-1)] + 1.0
    // This holds regardless of whether the primal or dual shortcut is used
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
