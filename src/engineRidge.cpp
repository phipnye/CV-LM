// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include "include/engineRidge.h"

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>

#include "include/CVWorker.h"
#include "include/utils.h"

namespace CV::Ridge {

// Generalized cross-validation for ridge regression
// Math shortcut: GCV(lambda) = MSE(lambda) / (1 - trace(H(lambda))/n)^2 =
// RSS(lambda) / (n * (1 - trace(H(lambda))/n)^2)
double gcv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
           const double lambda, const bool centered) {
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

  // Evaluate the inverse matrix (X'X + lambda*I)^-1 to be used
  // for both the trace calculation and the ridge coefficients
  Eigen::MatrixXd xtxLambdaInv{Eigen::MatrixXd::Identity(ncol, ncol)};
  ldlt.solveInPlace(xtxLambdaInv);  // returns true (no need to check)
  const Eigen::Index nrow{x.rows()};

  // trace(H) = trace(X * (X'X + lambda*I)^-1 * X')
  // Using trace(AB) = trace(BA) and X'X = (X'X + lambda * I) - lambda * I
  // trace(H) = p - lambda * trace((X'X + lambda*I)^-1)
  double traceH{ncol - lambda * xtxLambdaInv.trace()};

  // If the data was centered in R, we need to add one to capture the dropped
  // intercept column
  if (centered) {
    traceH += 1.0;
  }

  // Calculate RSS = ||y - X * beta||^2 where beta = (X'X + lambda * I)^-1 * X'y
  const double rss{(y - (x * (xtxLambdaInv * (xT * y)))).squaredNorm()};
  const double leverage{1.0 - (traceH / nrow)};
  return rss / (nrow * leverage * leverage);
}

// Leave-one-out cross-validation for ridge regression
double loocv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
             const double lambda, const bool centered) {
  // Generate X'X + lambda * I
  const Eigen::Index ncol{x.cols()};
  Eigen::MatrixXd xtxLambda{Eigen::MatrixXd::Zero(ncol, ncol)};
  xtxLambda.diagonal().fill(lambda);
  const auto xT{x.transpose()};
  xtxLambda.selfadjointView<Eigen::Lower>().rankUpdate(xT);

  // Factorize and compute inverse (X'X + lambda * I)^-1
  const Eigen::LDLT<Eigen::Ref<Eigen::MatrixXd>> ldlt{xtxLambda};
  Eigen::MatrixXd xtxLambdaInv{Eigen::MatrixXd::Identity(ncol, ncol)};
  ldlt.solveInPlace(xtxLambdaInv);  // returns true (no need to check)

  // Compute diagonal of hat matrix H = X * (X'X + lambda * I)^-1 * X'
  // h_ii = x_i' * (X'X + lambda * I)^-1 * x_i, compute this for all i via
  // row-wise dot product of X and (X * (X'X + lambda * I)^-1)
  Eigen::ArrayXd diagH{
      (x * xtxLambdaInv).cwiseProduct(x).rowwise().sum().array()};

  // If the data was centered in R, we need to add 1/n to the diagonal entries
  // to capture the dropped intercept column
  if (centered) {
    diagH += (1.0 / x.rows());
  }

  // LOOCV Formula: mean((res / (1 - h))^2) - NOTE: May be worth adding a max to
  // prevent zero-division in high leverage instances
  return ((y - (x * xtxLambdaInv * (xT * y))).array() / (1.0 - diagH))
      .square()
      .mean();
}

// Multi-threaded CV for ridge regression
double parCV(const Eigen::VectorXd& y, const Eigen::MatrixXd& x, const int k,
             const double lambda, const int seed, const int nThreads) {
  // Setup folds and reorder data
  const Eigen::Index nrow{x.rows()};
  const auto [foldIDs,
              foldSizes]{CV::Utils::cvSetup(seed, static_cast<int>(nrow), k)};

  // Pre-calculate fold size bounds
  const auto [minTestSize, maxTestSize]{CV::Utils::testSizeExtrema(foldSizes)};

  // Initialize the worker
  CVWorker worker{y,          x, lambda, foldIDs, foldSizes, nrow - minTestSize,
                  maxTestSize};
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

}  // namespace CV::Ridge
