// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include "include/Worker.h"

#include <RcppEigen.h>
#include <RcppParallel.h>

#include "include/engineOLS.h"
#include "include/engineRidge.h"
#include "include/utils.h"

namespace CV {

// --- Base implementation

BaseWorker::BaseWorker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                       const Eigen::VectorXi& foldIDs,
                       const Eigen::VectorXi& foldSizes, const int nrow)
    : y{y},
      x{x},
      foldIDs{foldIDs},
      foldSizes{foldSizes},
      nrow{nrow},
      mse{0.0} {}

void BaseWorker::operator()(const std::size_t begin, const std::size_t end) {
  for (std::size_t it{begin}; it < end; ++it) {
    const int foldID{static_cast<int>(it) + 1};
    const Eigen::Index itEigen{static_cast<Eigen::Index>(it)};
    const int nIn{nrow - foldSizes[itEigen]};
    const int nOut{static_cast<int>(foldSizes[itEigen])};

    // Allocate index vectors for this fold
    Eigen::VectorXi inIdxs(nIn);
    Eigen::VectorXi outIdxs(nOut);

    // Fill indices
    Eigen::Index inIdx{0};
    Eigen::Index outIdx{0};

    for (Eigen::Index rowIdx{0}; rowIdx < nrow; ++rowIdx) {
      if (foldIDs[rowIdx] == foldID) {
        outIdxs[outIdx++] = rowIdx;
      } else {
        inIdxs[inIdx++] = rowIdx;
      }
    }

    // Subset using the integer index vectors
    const Eigen::VectorXd beta{computeCoef(x(inIdxs, Eigen::all), y(inIdxs))};
    const Eigen::VectorXd yHat{x(outIdxs, Eigen::all) * beta};
    const double cost{Utils::cost(y(outIdxs), yHat)};

    // Calculate weight: (n_i / n)
    const double alpha{static_cast<double>(foldSizes[itEigen]) / nrow};
    mse += (alpha * cost);
  }
}

void BaseWorker::join(const BaseWorker& other) { mse += other.mse; }

// --- OLS Implementation

OLS::Worker::Worker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                    const Eigen::VectorXi& foldIDs,
                    const Eigen::VectorXi& foldSizes, int nrow)
    : BaseWorker{y, x, foldIDs, foldSizes, nrow} {}

OLS::Worker::Worker(const OLS::Worker& other, RcppParallel::Split split)
    : BaseWorker{other.y, other.x, other.foldIDs, other.foldSizes, other.nrow} {
}

Eigen::VectorXd OLS::Worker::computeCoef(const Eigen::MatrixXd& xTrain,
                                         const Eigen::VectorXd& yTrain) const {
  return OLS::coef(yTrain, xTrain);
}

// --- Ridge Implementation

Ridge::Worker::Worker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                      double lambda, const Eigen::VectorXi& foldIDs,
                      const Eigen::VectorXi& foldSizes, int nrow)
    : BaseWorker{y, x, foldIDs, foldSizes, nrow}, lambda{lambda} {}

Ridge::Worker::Worker(const Ridge::Worker& other, RcppParallel::Split split)
    : BaseWorker{other.y, other.x, other.foldIDs, other.foldSizes, other.nrow},
      lambda{other.lambda} {}

Eigen::VectorXd Ridge::Worker::computeCoef(
    const Eigen::MatrixXd& xTrain, const Eigen::VectorXd& yTrain) const {
  return Ridge::coef(yTrain, xTrain, lambda);
}

}  // namespace CV
