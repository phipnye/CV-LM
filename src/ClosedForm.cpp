#include "ClosedForm.h"

#include <RcppArmadillo.h>

#include <algorithm>

namespace ClosedForm {

// GCV = MSE / (1 - trace(H)/n)^2
double gcv(const double rss, const double traceHat, const arma::uword nrow) {
  // MSE = RSS / n
  const double nrowDbl{static_cast<double>(nrow)};
  const double mse{std::max(0.0, rss / nrowDbl)};

  // Average leverage = trace(H) / n
  // trace(H) / n should never exceed 1:
  // OLS: trace(H) = rank(X) <= n
  // Ridge: trace(H) = sum_j d_j^2 / (d_j^2 + lambda) [see ESL p.68]
  const double avgLeverage{traceHat / nrowDbl};
  const double denomSqrt{std::max(1.0 - avgLeverage, 0.0)};
  const double denom{denomSqrt * denomSqrt};

  // On IEEE 754 platforms, this division natively handles 0.0 returning NaN if
  // mse is 0.0, or inf if mse > 0.0 for zero division
  return mse / denom;
}

// LOOCV = mean((e_i / (1 - h_ii))^2)
double loocv(const arma::vec& residuals, const arma::vec& diagHat) {
  // Just as we did for gcv, defend against small floating point precision
  // errors (for both OLS and Ridge h_ii should never exceed 1)
  /* OLS: The hat matrix is idempotent and symmetic such that H^2 = H, thus:
   * (H^2)_ii = sum_j h_ij h_ji            = h_ii (idempotent)
   *          = sum_j h_ij^2               = h_ii (symmetric)
   *          = h_ii^2 + sum_{j!=i} h_ij^2 = h_ii
   *          -> h_ii^2 <= h_ii
   *          -> 0 <= h_ii <= 1
   */
  // Ridge: the diagonal entries of (X'X + LI)^-1 increase as L decreases
  // thus the maximum diagonal entry comes from the OLS case shown above

  // In some instances like OLS with a column-rank deficient matrix, we may get
  // leverage values equal to 1 which results in zero divsision (on IEEE 754
  // platforms (required by R), division will correctly result in inf or NaN per
  // element when denom is 0
  return arma::mean(arma::square(
      residuals / arma::clamp(1.0 - diagHat, 0.0, arma::datum::inf)));
}

}  // namespace ClosedForm
