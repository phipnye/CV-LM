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
                       const Eigen::VectorXi& s, const Eigen::VectorXd& ns,
                       const int nrow)
    : y{y}, x{x}, s{s}, ns{ns}, nrow{nrow}, mse{0.0} {}

void BaseWorker::operator()(const std::size_t begin, const std::size_t end) {
  for (std::size_t i{begin}; i < end; ++i) {
    const int fold{static_cast<int>(i) + 1};
    const auto inMask{s.array() != fold};
    const auto outMask{s.array() == fold};
    const Eigen::MatrixXd xTrain{x(inMask, Eigen::all)};
    const Eigen::VectorXd yTrain{y(inMask)};
    const Eigen::VectorXd beta{computeCoef(xTrain, yTrain)};
    const Eigen::VectorXd yHat{x(outMask, Eigen::all) * beta};
    const double cost{Utils::cost(y(outMask), yHat)};
    const double alpha{ns[static_cast<Eigen::Index>(i)] / nrow};
    mse += (alpha * cost);
  }
}

void BaseWorker::join(const BaseWorker& other) { mse += other.mse; }

// --- OLS Implementation

OLS::Worker::Worker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                    const Eigen::VectorXi& s, const Eigen::VectorXd& ns,
                    int nrow)
    : BaseWorker{y, x, s, ns, nrow} {}

OLS::Worker::Worker(const OLS::Worker& other, RcppParallel::Split split)
    : BaseWorker{other.y, other.x, other.s, other.ns, other.nrow} {}

Eigen::VectorXd OLS::Worker::computeCoef(const Eigen::MatrixXd& xTrain,
                                         const Eigen::VectorXd& yTrain) const {
  return OLS::coef(yTrain, xTrain);
}

// --- Ridge Implementation

Ridge::Worker::Worker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                      double lambda, const Eigen::VectorXi& s,
                      const Eigen::VectorXd& ns, int nrow)
    : BaseWorker{y, x, s, ns, nrow}, lambda{lambda} {}

Ridge::Worker::Worker(const Ridge::Worker& other, RcppParallel::Split split)
    : BaseWorker{other.y, other.x, other.s, other.ns, other.nrow},
      lambda{other.lambda} {}

Eigen::VectorXd Ridge::Worker::computeCoef(
    const Eigen::MatrixXd& xTrain, const Eigen::VectorXd& yTrain) const {
  return Ridge::coef(yTrain, xTrain, lambda);
}

}  // namespace CV
