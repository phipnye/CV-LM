// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include "include/engineOLS.h"

#include <RcppEigen.h>
#include <RcppParallel.h>

#include "include/Worker.h"
#include "include/utils.h"

namespace CV::OLS {
// Generate OLS coefficients
// Solves for beta using QR decomposition (X = QRP') (handles the math for both
// full-rank and rank-deficient cases)
Eigen::VectorXd coef(const Eigen::VectorXd& y, const Eigen::MatrixXd& x) {
  const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr{x};  // X = QRP'
  const Eigen::Index ncol{x.cols()};
  const Eigen::Index rank{qr.rank()};

  if (rank == ncol) {  // full-rank case: beta = (X'X)^-1 X'y
    return qr.solve(y);
  }

  // Handle rank-deficiency: Solve via the triangular R matrix for the
  // identified rank and then map back to original column space via the
  // permutation matrix P (even though the qr solve method can handle
  // rank-deficient matrices, this mimic's how base R's lm() handles rank
  // deficiency by zeroing out redundant coefficients)
  Eigen::VectorXd w{Eigen::VectorXd::Zero(ncol)};
  const auto sol{qr.householderQ().transpose() * y};
  w.head(rank) = qr.matrixQR()
                     .topLeftCorner(rank, rank)
                     .triangularView<Eigen::Upper>()
                     .solve(sol.head(rank));
  return qr.colsPermutation() * w;
}

// Generalized cross-validation for linear regression
// Math shortcut: GCV = MSE / (1 - trace(H)/n)^2 (for OLS, trace(H) is just the
// rank of X)
double gcv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x) {
  const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr{x};
  const Eigen::VectorXd yHat{x * qr.solve(y)};
  // tr(H) = rank(X). leverage here is (1 - trace(H)/n)
  const double leverage{1.0 - (static_cast<double>(qr.rank()) / x.rows())};
  return ((y - yHat).array() / leverage).square().mean();
}

// Leave-one-out cross-validation for linear regression (leverages shortcut:
// LOO_error_i = e_i / (1 - h_ii))
double loocv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x) {
  const Eigen::Index nrow{x.rows()};
  const Eigen::Index ncol{x.cols()};
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr{x};
  const Eigen::Index rank{qr.rank()};
  Eigen::VectorXd w{Eigen::VectorXd::Zero(ncol)};

  if (rank == ncol) {  // full-rank
    w = qr.solve(y);
  } else {  // rank-deficient
    Rcpp::warning("Rank-deficient design matrix (rank %d < cols %d).", rank,
                  ncol);
    const Eigen::VectorXd sol{qr.householderQ().transpose() * y};
    w.head(rank) = qr.matrixQR()
                       .topLeftCorner(rank, rank)
                       .triangularView<Eigen::Upper>()
                       .solve(sol.head(rank));
    w = qr.colsPermutation() * w;
  }

  // Leverage values: h_ii = [X(X'X)^-1 X']_ii. Using QR (X = QR),
  // H = Q Q' so h_ii = sum_{j=1}^{rank} q_{ij}^2 (rowwise squared norm of thin
  // Q)
  const Eigen::MatrixXd fullQ{qr.householderQ()};
  const Eigen::VectorXd diagH{(fullQ.leftCols(rank)).rowwise().squaredNorm()};

  // LOOCV Formula: mean((res / (1 - h))^2)
  return ((y - x * w).array() / (1.0 - diagH.array())).square().mean();
}

// Multi-threaded CV for linear regression
double parCV(const Eigen::VectorXd& y, const Eigen::MatrixXd& x, const int k,
             const int seed, const int nThreads) {
  const int n{static_cast<int>(x.rows())};
  const auto [s, ns]{CV::Utils::cvSetup(seed, n, k)};

  // Initialize the worker with data and partition info
  Worker cvWorker{y, x, s, ns, n};

  if (nThreads > 1) {
    RcppParallel::parallelReduce(0, k, cvWorker, 1, nThreads);
  } else {
    // Explicitly call the worker's loop for the full range if single-threaded
    cvWorker(0, k);
  }

  return cvWorker.mse;
}

}  // namespace CV::OLS
