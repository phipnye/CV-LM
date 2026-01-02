// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include "include/Worker.h"

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <atomic>
#include <cstddef>

namespace CV {

// --- Base implementation

BaseWorker::BaseWorker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                       const Eigen::VectorXi& foldIDs,
                       const Eigen::VectorXi& foldSizes,
                       const Eigen::Index nrow, const Eigen::Index ncol)
    : y_{y},
      x_{x},
      foldIDs_{foldIDs},
      foldSizes_{foldSizes},
      nrow_{nrow},
      ncol_{ncol},
      mse_{0.0} {}

void BaseWorker::operator()(const std::size_t begin, const std::size_t end) {
  // Pre-allocate thread-local buffers to reuse across all folds in this range
  Eigen::MatrixXd xTrain(nrow_, ncol_);
  Eigen::VectorXd yTrain(nrow_);

  // Buffers for holding training and test indices
  Eigen::VectorXi trainIdx(nrow_);
  Eigen::VectorXi testIdx(nrow_);

  for (std::size_t it{begin}; it < end; ++it) {
    const int currentFold{static_cast<int>(it) + 1};
    const Eigen::Index testSize{foldSizes_[static_cast<Eigen::Index>(it)]};
    const Eigen::Index trainSize{nrow_ - testSize};

    // Prepare training and testing containers
    Eigen::Index tr{0};
    Eigen::Index ts{0};

    for (Eigen::Index r{0}; r < nrow_; ++r) {
      if (foldIDs_[r] == currentFold) {
        testIdx[ts++] = r;
      } else {
        trainIdx[tr++] = r;
      }
    }

    // Copy into training buffers
    xTrain.topRows(trainSize) = x_(trainIdx.head(trainSize), Eigen::all);
    yTrain.head(trainSize) = y_(trainIdx.head(trainSize));

    // Fit the model
    const Eigen::VectorXd beta{
        computeCoef(xTrain.topRows(trainSize), yTrain.topRows(trainSize))};

    // Evaluate performance on hold-out fold (MSE)
    const auto xOut{x_(testIdx.head(testSize), Eigen::all)};
    const auto yOut{y_(testIdx.head(testSize))};
    const double cost{(yOut - (xOut * beta)).array().square().mean()};

    // Weighted MSE contribution
    const double alpha{static_cast<double>(testSize) / nrow_};
    mse_ += (alpha * cost);
  }
}

void BaseWorker::join(const BaseWorker& other) { mse_ += other.mse_; }

// --- OLS Implementation

OLS::Worker::Worker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                    const Eigen::VectorXi& foldIDs,
                    const Eigen::VectorXi& foldSizes, const Eigen::Index nrow,
                    const Eigen::Index ncol)
    : BaseWorker{y, x, foldIDs, foldSizes, nrow, ncol} {}

OLS::Worker::Worker(const OLS::Worker& other, RcppParallel::Split)
    : BaseWorker{other.y_,         other.x_,    other.foldIDs_,
                 other.foldSizes_, other.nrow_, other.ncol_} {}

Eigen::VectorXd OLS::Worker::computeCoef(
    const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
    const Eigen::Ref<const Eigen::VectorXd>& yTrain) const {
  const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr{xTrain};

  // if (qr.info() != Eigen::Success) {
  //   Not necessary, Eigen documents this always returns success
  // }

  const Eigen::Index rank{qr.rank()};

  // No rank deficiency
  if (rank == ncol_) {
    return qr.solve(yTrain);
  }

  // Mimic R's behavior of zeroing out coefficients on redundant predictors
  Eigen::VectorXd qty{yTrain};
  qty.applyOnTheLeft(qr.householderQ().transpose());
  Eigen::VectorXd beta{Eigen::VectorXd::Zero(ncol_)};

  // Solve Rz = Q'y for the first 'rank' elements using the top-left rank x rank
  // part of Matrix R
  beta.head(rank).noalias() = qr.matrixR()
                                  .topLeftCorner(rank, rank)
                                  .triangularView<Eigen::Upper>()
                                  .solve(qty.head(rank));

  // Permute back to original column order
  beta.applyOnTheLeft(qr.colsPermutation());
  return beta;
}

// --- Ridge Implementation

Ridge::Worker::Worker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                      double lambda, const Eigen::VectorXi& foldIDs,
                      const Eigen::VectorXi& foldSizes, const Eigen::Index nrow,
                      const Eigen::Index ncol)
    : BaseWorker{y, x, foldIDs, foldSizes, nrow, ncol}, lambda_{lambda} {}

Ridge::Worker::Worker(const Ridge::Worker& other, RcppParallel::Split)
    : BaseWorker{other.y_,         other.x_,    other.foldIDs_,
                 other.foldSizes_, other.nrow_, other.ncol_},
      lambda_{other.lambda_} {}

Eigen::VectorXd Ridge::Worker::computeCoef(
    const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
    const Eigen::Ref<const Eigen::VectorXd>& yTrain) const {
  // Generate cross-products
  Eigen::MatrixXd xtxLambda{Eigen::MatrixXd::Zero(ncol_, ncol_)};
  xtxLambda.diagonal().fill(lambda_);
  const auto xT{xTrain.transpose()};
  xtxLambda.selfadjointView<Eigen::Lower>().rankUpdate(xT);
  Eigen::VectorXd xty{xT * yTrain};

  // Despite positive definiteness, Eigen's documentation states "While the
  // Cholesky decomposition is particularly useful to solve selfadjoint problems
  // like D^*D x = b, for that purpose, we recommend the Cholesky decomposition
  // without square root which is more stable and even faster." We can also
  // perform the decomposition in place here
  const Eigen::LDLT<Eigen::Ref<Eigen::MatrixXd>> ldlt{xtxLambda};

  // if (ldlt.info() != Eigen::Success) {
  // TO DO
  // }

  // xty no longer represents X'y, LDLT.solve supports in-place solves which we
  // use here for efficiency
  xty = ldlt.solve(xty);
  return xty;
}

}  // namespace CV
