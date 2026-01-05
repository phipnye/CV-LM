#include "include/OLSFit.h"

#include <RcppEigen.h>

namespace CV::OLS {

// Ctor
OLSFit::OLSFit(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
               const bool needHat)
    : qr_{x}, nrow_{x.rows()}, rank_{qr_.rank()}, qty_{y} {
  qty_.applyOnTheLeft(qr_.householderQ().transpose());

  if (needHat) {
    // Leverage values: h_ii = [X(X'X)^-1 X']_ii. Using QR (X = QR),
    // H = QQ' so h_ii = sum_{j=1}^{rank} q_{ij}^2 (rowwise squared norm of thin
    // Q) - instead of evaluating a potentially large Q matrix, we can use
    // backward solving on the triangular matrix R to solve for R^-T X' = Q'
    diagH_ = qr_.matrixR()
                 .topLeftCorner(rank_, rank_)
                 .triangularView<Eigen::Upper>()
                 .transpose()
                 .solve((x * qr_.colsPermutation()).leftCols(rank_).transpose())
                 .colwise()
                 .squaredNorm()
                 .array();
  }
}

// --- Public interface

// GCV = MSE / (1 - trace(H)/n)^2
double OLSFit::gcv() const {
  const double mrl{meanResidualLeverage()};
  return mse() / (mrl * mrl);
}

// LOOCV_error_i = e_i / (1 - h_ii))
double OLSFit::loocv() const {
  return (residuals().array() / (1.0 - diagH_)).square().mean();
}

// --- Internal math

// Sum of squared residuals
double OLSFit::rss() const {
  // Calculate RSS (using the full n x n orthogonal matrix Q, we transform y
  // into Q'y and partition the squared norm of y into two components:
  // ||y||^2 = ||(Q'y).head(rank)||^2 + ||(Q'y).tail(n - rank)||^2
  // where the first term is the ESS the second term is the RSS
  return qty_.tail(nrow_ - rank_).squaredNorm();
}

// Mean squared error
double OLSFit::mse() const { return rss() / nrow_; }

// Mean residual leverage = (1 - trace(H)/n)
double OLSFit::meanResidualLeverage() const {
  return 1.0 - (static_cast<double>(rank_) / nrow_);  // trace(H) = rank(X)
}

Eigen::VectorXd OLSFit::residuals() const {
  // Zero out the components in the column space (the first 'rank' elements,
  // leaving only the components in the orthogonal complement)
  Eigen::VectorXd resid{qty_};
  resid.head(rank_).setZero();

  // Transform back to original space: resid = Q * [0, Q'y.tail(n - rank)]'
  resid.applyOnTheLeft(qr_.householderQ());
  return resid;
}

}  // namespace CV::OLS
