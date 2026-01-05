#include "CV-Ridge-Fit.h"

#include <RcppEigen.h>

namespace CV::Ridge {

Fit::Fit(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
         const double lambda, const bool centered, const bool needHat)
    : nrow_{x.rows()},
      ncol_{x.cols()},
      lambda_{lambda},
      centered_{centered},

      // Decompose and invert (X'X + lambda*I)
      xtxLambdaInv_{[&]() {
        // Create regularized covariance matrix
        Eigen::MatrixXd xtxLambda{Eigen::MatrixXd::Zero(ncol_, ncol_)};
        xtxLambda.diagonal().fill(lambda_);
        xtxLambda.selfadjointView<Eigen::Lower>().rankUpdate(x.transpose());

        // Eigen documentation states: "While the
        // Cholesky decomposition is particularly useful to solve selfadjoint
        // problems like D^*D x = b, for that purpose, we recommend the Cholesky
        // decomposition without square root which is more stable and even
        // faster." LDLT supports in-place decomposition
        Eigen::LDLT<Eigen::Ref<Eigen::MatrixXd>> ldlt{xtxLambda};

        // LDLT supports solving in-place
        Eigen::MatrixXd xtxLambdaInv{Eigen::MatrixXd::Identity(ncol_, ncol_)};
        ldlt.solveInPlace(xtxLambdaInv);  // returns true (no need to check)
        return xtxLambdaInv;
      }()},

      // resid = y - x * beta = y - (X'X + lambda*I)^-1 * X'y
      resid_{y - x * (xtxLambdaInv_ * (x.transpose() * y))} {
  // h_ii = x_i' * (X'X + lambda * I)^-1 * x_i
  if (needHat) {
    diagH_ = (x * xtxLambdaInv_).cwiseProduct(x).rowwise().sum().array();

    // If the data was centered in R, we need to add 1/n to the diagonal entries
    // to capture the dropped intercept column
    if (centered) {
      diagH_ += (1.0 / nrow_);
    }
  }
}

// --- Public interface

// GCV = MSE / (1 - trace(H)/n)^2
double Fit::gcv() const {
  const double mrl{meanResidualLeverage()};
  return mse() / (mrl * mrl);
}

// LOOCV_error_i = e_i / (1 - h_ii))
double Fit::loocv() const {
  return (resid_.array() / (1.0 - diagH_)).square().mean();
}

// --- Internal math

// Sum of squared residuals
double Fit::rss() const { return resid_.squaredNorm(); }

// Mean squared error
double Fit::mse() const { return rss() / nrow_; }

// Mean residual leverage = (1 - trace(H)/n)
double Fit::meanResidualLeverage() const { return 1.0 - (traceH() / nrow_); }

double Fit::traceH() const {
  // trace(H) = trace(X * (X'X + lambda*I)^-1 * X')
  // Using trace(AB) = trace(BA) and X'X = (X'X + lambda * I) - lambda * I
  // trace(H) = p - lambda * trace((X'X + lambda*I)^-1)
  double p{static_cast<double>(ncol_)};

  // If the data was centered in R, we need to add one to capture the dropped
  // intercept column
  if (centered_) {
    p += 1.0;
  }

  return p - lambda_ * xtxLambdaInv_.trace();
}

}  // namespace CV::Ridge
