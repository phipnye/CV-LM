// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include "include/engineRidge.h"

#include <RcppEigen.h>
#include <RcppParallel.h>

#include "include/Worker.h"
#include "include/utils.h"

namespace CV::Ridge {

// Generate Ridge regression coefficients (solves (X'X + lambda * I)beta = X'y)
Eigen::VectorXd coef(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                     const double lambda) {
  // Efficiently compute X'X using only the lower triangular part (SelfAdjoint)
  const Eigen::Index ncol{x.cols()};
  Eigen::MatrixXd xtx{Eigen::MatrixXd::Zero(ncol, ncol)};
  xtx.selfadjointView<Eigen::Lower>().rankUpdate(x.transpose());

  // Apply the L2 penalty to the diagonal
  xtx.diagonal().array() += lambda;

  // Solve via LDLT decomposition (faster/more stable for positive definite
  // matrices than standard inversion or QR)
  return xtx.ldlt().solve(x.transpose() * y);  // only called when lambda > 0
}

// Generalized cross-validation for ridge regression
double gcv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
           const double lambda) {
  const Eigen::Index nrow{x.rows()};
  const Eigen::Index ncol{x.cols()};

  // For n >= p, it's faster to use the Eigen-decomposition of the covariance
  // matrix X'X
  if (nrow >= ncol) {
    // Use selfadjointView to efficiently compute X'X
    Eigen::MatrixXd xtx{Eigen::MatrixXd::Zero(ncol, ncol)};
    xtx.selfadjointView<Eigen::Lower>().rankUpdate(x.transpose());

    // Spectral Theorem: Symmetry of X'X allows decomposition into VLV' (where
    // V is orthogonal and L is a diagonal matrix of eigenvalues)
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es{xtx};
    const auto evSq{es.eigenvalues().array()};

    // Ridge trace(H) = sum(d_i^2 / (d_i^2 + lambda)) (this represents the
    // effective degrees of freedom used to penalize the GCV denominator)
    const auto df{evSq / (evSq + lambda)};
    const auto& v{es.eigenvectors()};

    // yHat = XV * diag(df / evSq) * V'X'y
    const Eigen::VectorXd yHat{x * (v * ((df / evSq).matrix().asDiagonal() *
                                         (v.transpose() * x.transpose() * y)))};
    const double leverage{1.0 - (df.sum() / nrow)};
    return ((y - yHat).array() / leverage).square().mean();
  }

  // Fallback to SVD of original data matrix
  // For p > n, use SVD on X (avoids building large p x p symmetric matrix which
  // would be O(p^3) to decompose)
  // X = UDV', so X'X = V D^2 V' (SVD singular values squared are eigenvalues
  // of X'X)
  const Eigen::BDCSVD<Eigen::MatrixXd> svd{x, Eigen::ComputeThinU};
  const auto evSq{svd.singularValues().array().square()};
  const auto df{evSq / (evSq + lambda)};

  // yHat shortcut via U matrix: yHat = U * diag(d_i^2 / (d_i^2 + lambda)) * U'y
  // This stays in n-dimensional space (O(n^2)) rather than p-dimensional space
  const Eigen::VectorXd yHat{svd.matrixU() * (df.matrix().asDiagonal() *
                                              (svd.matrixU().transpose() * y))};
  const double leverage{1.0 - (df.sum() / x.rows())};
  return ((y - yHat).array() / leverage).square().mean();
}

// Leave-one-out cross-validation for ridge regression
double loocv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
             const double lambda) {
  const Eigen::Index nrow{x.rows()};
  const Eigen::Index ncol{x.cols()};

  // Prefer eigen-decomposition of X'X when n >= p (relying on the symmetry of
  // X'X for orthogonal diagonalization)
  if (nrow >= ncol) {
    Eigen::MatrixXd xtx{Eigen::MatrixXd::Zero(ncol, ncol)};
    xtx.selfadjointView<Eigen::Lower>().rankUpdate(x.transpose());
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es{xtx};
    const auto evSq{es.eigenvalues().array()};
    const auto df{evSq / (evSq + lambda)};
    const auto s{df.sqrt()};
    const auto& v{es.eigenvectors()};
    // Ridge leverage diagonal (h_ii): diag(X * V * (L + lambda*I)^-1 * V' * X')
    // Computed as rowwise norm of [X * V * diag(sqrt(eigenvalue / (eigenvalue +
    // lambda)) / sqrt(eigenvalue))]
    const Eigen::VectorXd diagH{
        (x * (v * (s / evSq.sqrt()).matrix().asDiagonal()))
            .rowwise()
            .squaredNorm()};

    const Eigen::VectorXd xty{x.transpose() * y};
    const Eigen::VectorXd yHat{
        x * (v * ((df / evSq).matrix().asDiagonal() * (v.transpose() * xty)))};

    return ((y - yHat).array() / (1.0 - diagH.array())).square().mean();
  }

  // Fall back to SVD of original data matrix if n < p
  const Eigen::BDCSVD<Eigen::MatrixXd> svd{x, Eigen::ComputeThinU};
  const auto& u{svd.matrixU()};
  const auto evSq{svd.singularValues().array().square()};
  const auto df{evSq.array() / (evSq.array() + lambda)};
  const auto s{df.sqrt()};

  // leverage values: h_ii = [U * diag(S^2 / (S^2 + lambda)) * U']_ii
  // Computed as rowwise norm of [U * diag(S / sqrt(S^2 + lambda))]
  const Eigen::VectorXd diagH{
      (u * s.matrix().asDiagonal()).rowwise().squaredNorm()};

  // Vectorized yHat = U * diag(df) * U' * y
  const Eigen::VectorXd yHat{u *
                             (df.matrix().asDiagonal() * (u.transpose() * y))};
  return ((y - yHat).array() / (1.0 - diagH.array())).square().mean();
}

// Multi-threaded CV for ridge regression
double parCV(const Eigen::VectorXd& y, const Eigen::MatrixXd& x, const int k,
             const double lambda, const int seed, const int nThreads) {
  const int nrow{static_cast<int>(x.rows())};
  const auto [foldIDs, foldSizes]{CV::Utils::cvSetup(seed, nrow, k)};

  // Initialize the worker
  Worker cvWorker{y, x, lambda, foldIDs, foldSizes, nrow};

  if (nThreads > 1) {
    RcppParallel::parallelReduce(0, k, cvWorker, 1, nThreads);
  } else {
    // Manually execute the range [0, k)
    cvWorker(0, k);
  }

  return cvWorker.mse;
}

}  // namespace CV::Ridge
