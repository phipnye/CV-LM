#include "CV-WorkerModel.h"

#include <RcppEigen.h>

#include <algorithm>

#include "Utils-Decompositions-utils.h"

namespace CV {

namespace OLS {

// Main ctor
WorkerModel::WorkerModel(const Eigen::Index ncol,
                         const Eigen::Index maxTrainSize,
                         const double threshold)
    : qtz_{maxTrainSize, ncol} {
  // Threshold at which to consider pivots zero "A pivot will be
  // considered nonzero if its absolute value is strictly greater than
  // |pivot|⩽threshold×|maxpivot| "
  qtz_.setThreshold(threshold);
}

// Copy ctor - required for RcppParallel split
WorkerModel::WorkerModel(const WorkerModel& other)
    : qtz_{other.qtz_.rows(), other.qtz_.cols()} {
  // Threshold at which to consider pivots zero
  qtz_.setThreshold(other.qtz_.threshold());
}

// Compute OLS coefficients using Complete Orthogonal Decomposition
void WorkerModel::computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                              const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                              Eigen::VectorXd& beta) {
  // Decompose training set into the form XP = QTZ
  qtz_.compute(xTrain);

  // if (qtz_.info() != Eigen::Success) {
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
  beta = qtz_.solve(yTrain);
}

}  // namespace OLS

namespace Ridge {

// Main ctor
WorkerModel::WorkerModel(const Eigen::Index ncol,
                         const Eigen::Index maxTrainSize,
                         const double threshold, const double lambda)
    : udvT_{maxTrainSize, ncol, Eigen::ComputeThinU | Eigen::ComputeThinV},
      uTy_{std::min(maxTrainSize, ncol)},
      singularVals_{std::min(maxTrainSize, ncol)},
      singularShrinkFactors_{std::min(maxTrainSize, ncol)},
      lambda_{lambda},
      info_{Eigen::Success} {
  // Prescribe threshold to SVD decomposition where singular values are
  // considered zero "A singular value will be considered nonzero if its value
  // is strictly greater than |singularvalue|⩽threshold×|maxsingularvalue|."
  udvT_.setThreshold(threshold);
}

// Copy ctor - required for RcppParallel split
WorkerModel::WorkerModel(const WorkerModel& other)
    : udvT_{other.udvT_.rows(), other.udvT_.cols(),
            Eigen::ComputeThinU | Eigen::ComputeThinV},
      uTy_{other.uTy_.size()},
      singularVals_{other.singularVals_.size()},
      singularShrinkFactors_{other.singularShrinkFactors_.size()},
      lambda_{other.lambda_},
      info_{other.info_} {
  // Prescribe threshold to SVD decomposition where singular values are
  // considered zero
  udvT_.setThreshold(other.udvT_.threshold());
}

// Estimate ridge coefficients using singular value decomposition
void WorkerModel::computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                              const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                              Eigen::VectorXd& beta) {
  // Obtain singular value decomposition of the training set
  udvT_.compute(xTrain);

  // Make sure decomposition was successful
  if (const Eigen::ComputationInfo info{udvT_.info()}; info != Eigen::Success) {
    info_ = info;
    return;
  }

  // The number of singular values may change across folds
  const Eigen::Index singularValsSize{udvT_.singularValues().size()};
  // ReSharper disable once CppDFAUnusedValue
  auto singularVals{singularVals_.head(singularValsSize)};
  singularVals = Utils::Decompositions::getSingularVals(udvT_);

  // Compute the projection of y
  auto uTy{uTy_.head(singularValsSize)};
  uTy.noalias() = udvT_.matrixU().transpose() * yTrain;

  // Apply the shrinkage to the singular values
  // This function should only be called in the case where lambda > 0 so this is
  // safe regardless of rank (these shrinkage factors are related to the
  // coordinate shrinkage factors (d_j^2 / (d_j^2 + lambda)) for solving fitted
  // values X * beta but are reduced by d_j since we're solving explicitly for
  // beta)
  // ReSharper disable once CppDFAUnusedValue
  auto singularShrinkFactors{singularShrinkFactors_.head(singularValsSize)};
  singularShrinkFactors =
      singularVals.array() / (singularVals.array().square() + lambda_);

  // beta_ridge = V * diag(Sigma^2 + lambda * I)^-1 Sigma * U'y
  beta.noalias() =
      udvT_.matrixV() * (singularShrinkFactors.array() * uTy.array()).matrix();
}

// Get decomposition success information
Eigen::ComputationInfo WorkerModel::getInfo() const noexcept { return info_; }

}  // namespace Ridge

}  // namespace CV
