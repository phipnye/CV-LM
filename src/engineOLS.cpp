// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include "include/engineOLS.h"

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>

#include "include/CVWorker.h"
#include "include/utils.h"

namespace CV::OLS {

// Generalized cross-validation for linear regression
// Math shortcut: GCV = MSE / (1 - trace(H)/n)^2 = RSS / (n * (1 -
// trace(H)/n)^2) (for OLS, trace(H) is just the rank of X)
double gcv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x) {
  const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr{x};
  const Eigen::Index rank{qr.rank()};

  // tr(H) = rank(X)
  // leverage is (1 - trace(H)/n)
  const Eigen::Index nrow{x.rows()};
  const double leverage{1.0 - (static_cast<double>(qr.rank()) / nrow)};

  // Calculate RSS (using the full n x n orthogonal matrix Q, we transform y
  // into Q'y we partition the squared norm of y into two components:
  // ||y||^2 = ||(Q'y).head(rank)||^2 + ||(Q'y).tail(n - rank)||^2
  // where the first term is the ESS the second term is the RSS
  Eigen::VectorXd qty{y};
  qty.applyOnTheLeft(qr.householderQ().transpose());
  const double rss{qty.tail(nrow - rank).squaredNorm()};
  return rss / (nrow * leverage * leverage);
}

// Leave-one-out cross-validation for linear regression (leverages shortcut:
// LOOCV_error_i = e_i / (1 - h_ii))
double loocv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x) {
  const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr{x};
  const Eigen::Index rank{qr.rank()};

  // No longer warning about rank-deficiency which differs from the first
  // release

  // Compute OLS coefficients (matching R's lm behavior of zeroing out
  // redundant covariate coefficients in the presence of rank-deficiency)
  const Eigen::Index ncol{x.cols()};
  Eigen::VectorXd beta(ncol);
  const auto rView{
      qr.matrixR().topLeftCorner(rank, rank).triangularView<Eigen::Upper>()};

  // Full-rank
  if (rank == ncol) {
    beta = qr.solve(y);
  } else {  // rank-deficient
    Eigen::VectorXd qty{y};
    qty.applyOnTheLeft(qr.householderQ().transpose());
    beta.setZero();
    beta.head(rank) = rView.solve(qty.head(rank));

    // Permute back to original column order
    beta.applyOnTheLeft(qr.colsPermutation());
  }

  // Leverage values: h_ii = [X(X'X)^-1 X']_ii. Using QR (X = QR),
  // H = QQ' so h_ii = sum_{j=1}^{rank} q_{ij}^2 (rowwise squared norm of thin
  // Q) - instead of evaluating a potentially large Q matrix, we can use
  // backward solving on the triangular matrix R to solve for R^-T X' = Q'
  const Eigen::VectorXd diagH{
      rView.transpose()
          .solve((x * qr.colsPermutation()).leftCols(rank).transpose())
          .colwise()
          .squaredNorm()};

  // LOOCV Formula: mean((res / (1 - h))^2) - NOTE: May be worth adding a max to
  // prevent zero-division in high leverage instances
  return ((y - (x * beta)).array() / (1.0 - diagH.array())).square().mean();
}

// Multi-threaded CV for linear regression
double parCV(const Eigen::VectorXd& y, const Eigen::MatrixXd& x, const int k,
             const int seed, const int nThreads) {
  // Setup folds and reorder data so each fold is a contiguous block, allowing
  // the worker to generate views of the data rather than copying
  const auto [foldIDs, foldSizes]{CV::Utils::cvSetup(seed, x.rows(), k)};

  // Initialize the worker with data and partition info
  CVWorker worker{y, x, foldIDs, foldSizes};
  constexpr std::size_t begin{0};
  const std::size_t end{static_cast<std::size_t>(k)};

  if (nThreads > 1) {
    RcppParallel::parallelReduce(begin, end, worker, 1, nThreads);
  } else {
    // Explicitly call the worker's loop for the full range if single-threaded
    worker(begin, end);
  }

  return worker.mse_;
}

}  // namespace CV::OLS
