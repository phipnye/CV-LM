// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include "include/engineRidge.h"

#include <RcppEigen.h>
#include <RcppParallel.h>

#include "include/Worker.h"
#include "include/utils.h"

namespace CV::Ridge {

// Generalized cross-validation for ridge regression
// Math shortcut: GCV(lambda) = MSE(lambda) / (1 - trace(H(lambda))/n)^2 =
// RSS(lambda) / (n * (1 - trace(H(lambda))/n)^2)
double gcv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
           const double lambda) {
  // Generate X'X + lambda * I
  const Eigen::Index ncol{x.cols()};
  Eigen::MatrixXd xtxLambda{Eigen::MatrixXd::Zero(ncol, ncol)};
  xtxLambda.diagonal().fill(lambda);
  const auto xT{x.transpose()};
  xtxLambda.selfadjointView<Eigen::Lower>().rankUpdate(xT);

  // Use LDLT, which maintains consistency with K-Fold solution and matches
  // Eigen's recommendation for sovling self-adjoint problems (can solve
  // in-place)
  const Eigen::LDLT<Eigen::Ref<Eigen::MatrixXd>> ldlt{xtxLambda};

  // Explicitly evaluate the inverse matrix Z = (X'X + lambda*I)^-1 to be used
  // for both the trace calculation and the ridge coefficients
  Eigen::MatrixXd xtxLambdaInv{Eigen::MatrixXd::Identity(ncol, ncol)};
  xtxLambdaInv = ldlt.solve(xtxLambdaInv);
  const Eigen::Index nrow{x.rows()};

  // trace(H) = trace(X * (X'X + lambda*I)^-1 * X')
  // Using trace(AB) = trace(BA) and X'X = (X'X + lambda * I) - lambda * I
  // trace(H) = p - lambda * trace((X'X + lambda*I)^-1)
  const double leverage{1.0 - ((ncol - lambda * xtxLambdaInv.trace()) / nrow)};

  // Calculate RSS = ||y - X * beta||^2 where beta = (X'X + lambda * I)^-1 * X'y
  const double rss{(y - (x * (xtxLambdaInv * (xT * y)))).squaredNorm()};
  return (rss / (nrow * leverage * leverage));
}

// Leave-one-out cross-validation for ridge regression
double loocv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
             const double lambda) {
  // Generate X'X + lambda * I
  const Eigen::Index ncol{x.cols()};
  Eigen::MatrixXd xtxLambda{Eigen::MatrixXd::Zero(ncol, ncol)};
  xtxLambda.diagonal().fill(lambda);
  const auto xT{x.transpose()};
  xtxLambda.selfadjointView<Eigen::Lower>().rankUpdate(xT);

  // Factorize and compute inverse Z = (X'X + lambda * I)^-1
  const Eigen::LDLT<Eigen::Ref<Eigen::MatrixXd>> ldlt{xtxLambda};
  Eigen::MatrixXd xtxLambdaInv{Eigen::MatrixXd::Identity(ncol, ncol)};
  xtxLambdaInv = ldlt.solve(xtxLambdaInv);

  // Compute ridge coefficients beta = Z * X'y
  const auto beta{xtxLambdaInv * (xT * y)};

  // Compute diagonal of hat matrix H = X * (X'X + lambda * I)^-1 * X'
  // h_ii = x_i' * (X'X + lambda * I)^-1 * x_i, compute this for all i via
  // row-wise dot product of X and (X * (X'X + lambda * I)^-1)
  const auto diagH{(x * xtxLambdaInv).cwiseProduct(x).rowwise().sum()};

  // LOOCV Formula: mean((res / (1 - h))^2) - NOTE: May be worth adding a max to
  // prevent zero-division in high leverage instances
  return ((y - (x * beta)).array() / (1.0 - diagH.array())).square().mean();
}

// Multi-threaded CV for ridge regression
double parCV(const Eigen::VectorXd& y, const Eigen::MatrixXd& x, const int k,
             const double lambda, const int seed, const int nThreads) {
  // Setup folds and reorder data
  const Eigen::Index nrow{x.rows()};
  const auto [foldIDs,
              foldSizes]{CV::Utils::cvSetup(seed, static_cast<int>(nrow), k)};

  // Initialize the worker
  Worker cvWorker{y, x, lambda, foldIDs, foldSizes, nrow, x.cols()};

  if (nThreads > 1) {
    RcppParallel::parallelReduce(0, k, cvWorker, 1, nThreads);
  } else {
    // Manually execute the range [0, k)
    cvWorker(0, k);
  }

  return cvWorker.mse_;
}

}  // namespace CV::Ridge
