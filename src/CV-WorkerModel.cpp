#include "CV-WorkerModel.h"

#include <RcppEigen.h>

namespace CV {

namespace OLS {

WorkerModel::WorkerModel(const Eigen::Index ncol,
                         const Eigen::Index maxTrainSize,
                         const double threshold)
    : cod_(maxTrainSize, ncol) {
  // Threshold at which to consider singular values zero "A pivot will be
  // considered nonzero if its absolute value is strictly greater than
  // |pivot|⩽threshold×|maxpivot| "
  cod_.setThreshold(threshold);
}

WorkerModel::WorkerModel(const WorkerModel& other)
    : info_{other.info_}, cod_(other.cod_.rows(), other.cod_.cols()) {
  cod_.setThreshold(other.cod_.threshold());
}

void WorkerModel::computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                              const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                              Eigen::VectorXd& beta) {
  // Decompose training set into the for XP = QTZ
  cod_.compute(xTrain);

  // if (cod_.info() != Eigen::Success) {
  //   Not necessary, documentation states this always returns success for COD
  // }

  // This behavior strays from R's lm behavior when the design matrix is not of
  // full-column rank, R (as of 2026) uses Dqrdc2/Linpack which zeros out the
  // last ncol - rank coefficients on the "redundnant" columns of the design
  // matrix while COD gives the unique minimum norm solution via a second
  // orthogonal transform XP = QR = QTZ, which allows solving through a
  // truncated upper triangular matrix T* [rank x rank]
  beta = cod_.solve(yTrain);
}

}  // namespace OLS

namespace Ridge {

WorkerModel::WorkerModel(const Eigen::Index ncol, const double lambda)
    : info_{Eigen::Success},
      lambda_{lambda},
      xtxLambda_(ncol, ncol),
      ldlt_(ncol) {}

WorkerModel::WorkerModel(const WorkerModel& other)
    : info_{other.info_},
      lambda_{other.lambda_},
      xtxLambda_(other.xtxLambda_.rows(), other.xtxLambda_.cols()),
      ldlt_(other.ldlt_.cols()) {}

void WorkerModel::computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                              const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                              Eigen::VectorXd& beta) {
  // Generate cross-products (re-use pre-allocated buffers)
  xtxLambda_.setZero();
  xtxLambda_.diagonal().fill(lambda_);
  const auto xT{xTrain.transpose()};
  xtxLambda_.selfadjointView<Eigen::Lower>().rankUpdate(xT);

  // beta_ is a misnomer at this point LDLT supports in-place solving so we fill
  // beta_ with X'y (the RHS of solve)
  beta.noalias() = xT * yTrain;

  // Despite positive definiteness, Eigen's documentation states "While the
  // Cholesky decomposition is particularly useful to solve selfadjoint problems
  // like D^*D x = b, for that purpose, we recommend the Cholesky decomposition
  // without square root which is more stable and even faster." We can also
  // perform the decomposition in place here
  ldlt_.compute(xtxLambda_);

  if (const Eigen::ComputationInfo info{ldlt_.info()}; info != Eigen::Success) {
    info_ = info;
    return;
  }

  // LDLT::solve supports in-place solves which we use here for efficiency
  ldlt_.solveInPlace(beta);  // just returns true (no need to check)
}

}  // namespace Ridge

}  // namespace CV
