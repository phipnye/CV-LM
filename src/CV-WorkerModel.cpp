#include "CV-WorkerModel.h"

#include <RcppEigen.h>

namespace CV {

namespace OLS {

// Main ctor
WorkerModel::WorkerModel(const Eigen::Index ncol,
                         const Eigen::Index maxTrainSize,
                         const double threshold)
    : cod_(maxTrainSize, ncol) {
  // Threshold at which to consider singular values zero "A pivot will be
  // considered nonzero if its absolute value is strictly greater than
  // |pivot|⩽threshold×|maxpivot| "
  cod_.setThreshold(threshold);
}

// Copy ctor - required for RcppParallel split
WorkerModel::WorkerModel(const WorkerModel& other)
    : cod_(other.cod_.rows(), other.cod_.cols()) {
  cod_.setThreshold(other.cod_.threshold());
}

// Compute OLS coefficients using Complete Orthogonal Decomposition
void WorkerModel::computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                              const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                              Eigen::VectorXd& beta) {
  // Decompose training set into the form XP = QTZ
  cod_.compute(xTrain);

  // if (cod_.info() != Eigen::Success) {
  //   Not necessary, documentation states this always returns success for COD
  // }

  // This behavior strays from R's lm behavior when the design matrix is not of
  // full-column rank, R (as of 2026) uses Dqrdc2/Linpack which zeros out the
  // last ncol - rank coefficients on the "redundant" columns of the design
  // matrix while COD gives the unique minimum norm solution via a second
  // orthogonal transform XP = QR = QTZ, which allows solving through a
  // truncated upper triangular matrix T* [rank x rank]
  // Note: While this results in different coefficients for any rank-deficient
  // matrix, out-of-sample predictions will only diverge from R's when
  // the system is underdetermined
  beta = cod_.solve(yTrain);
}

}  // namespace OLS

namespace Ridge {

// Use primal form X'X + lambda * I
namespace Narrow {

// Main ctor
WorkerModel::WorkerModel(const Eigen::Index ncol, const double lambda)
    : ldlt_(ncol),
      xtxLambda_(ncol, ncol),
      lambda_{lambda},
      info_{Eigen::Success} {}

// Copy ctor - required for RcppParallel split
WorkerModel::WorkerModel(const WorkerModel& other)
    : ldlt_(other.ldlt_.cols()),
      xtxLambda_(other.xtxLambda_.rows(), other.xtxLambda_.cols()),
      lambda_{other.lambda_},
      info_{other.info_} {}

// Estimate ridge coefficients using LDLT of regularized covariance matrix
void WorkerModel::computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                              const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                              Eigen::VectorXd& beta) {
  // Generate regularized covariance matrix
  xtxLambda_.setZero();
  xtxLambda_.diagonal().fill(lambda_);
  const auto xTranspose{xTrain.transpose()};
  xtxLambda_.selfadjointView<Eigen::Lower>().rankUpdate(xTranspose);

  // beta_ is a misnomer at this point, LDLT supports in-place solving so we
  // fill beta_ with X'y (the RHS of solve)
  beta.noalias() = xTranspose * yTrain;

  // Despite positive definiteness, Eigen's documentation states "While the
  // Cholesky decomposition is particularly useful to solve selfadjoint problems
  // like D^*D x = b, for that purpose, we recommend the Cholesky decomposition
  // without square root which is more stable and even faster."
  ldlt_.compute(xtxLambda_);

  // Make sure decomposition was successful
  if (const Eigen::ComputationInfo info{ldlt_.info()}; info != Eigen::Success) {
    info_ = info;
    return;
  }

  // LDLT::solve supports in-place solves which we use here for efficiency
  ldlt_.solveInPlace(beta);  // just returns true (no need to check)
}

// Get decomposition success information
Eigen::ComputationInfo WorkerModel::getInfo() const noexcept { return info_; }

}  // namespace Narrow

// Use dual form XX' + lambda * I
namespace Wide {

// Main ctor
WorkerModel::WorkerModel(const Eigen::Index maxTrainSize, const double lambda)
    : ldlt_(maxTrainSize),
      xxtLambda_(maxTrainSize, maxTrainSize),
      alpha_(maxTrainSize),
      lambda_{lambda},
      info_{Eigen::Success} {}

// Copy ctor - required for RcppParallel split
WorkerModel::WorkerModel(const WorkerModel& other)
    : ldlt_(other.ldlt_.cols()),
      xxtLambda_(other.xxtLambda_.rows(), other.xxtLambda_.cols()),
      alpha_(other.alpha_.size()),
      lambda_{other.lambda_},
      info_{other.info_} {}

void WorkerModel::computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                              const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                              Eigen::VectorXd& beta) {
  // Current fold's training size (we are not guaranteed consistent training
  // sizes across folds)
  const Eigen::Index trainSize{xTrain.rows()};

  // Generate regularized outer product
  auto xxtLambdaBlock{xxtLambda_.topLeftCorner(trainSize, trainSize)};
  xxtLambdaBlock.setZero();
  xxtLambdaBlock.diagonal().fill(lambda_);
  xxtLambdaBlock.selfadjointView<Eigen::Lower>().rankUpdate(xTrain);

  // Decompose regularized outer product (see above in narrow case for why we
  // use LDLT over LLT)
  ldlt_.compute(xxtLambdaBlock);

  // Make sure decomposition was successful
  if (const Eigen::ComputationInfo info{ldlt_.info()}; info != Eigen::Success) {
    info_ = info;
    return;
  }

  // Solve for dual coefficients alpha: (XX' + lambda * I) * alpha = y
  alpha_.head(trainSize) = ldlt_.solve(yTrain);

  // Map back to primal space: beta = X' * alpha
  beta.noalias() = xTrain.transpose() * alpha_.head(trainSize);
}

// Get decomposition success information
Eigen::ComputationInfo WorkerModel::getInfo() const noexcept { return info_; }

}  // namespace Wide

}  // namespace Ridge

}  // namespace CV
