#include "CV-Stochastic-WorkerModel.h"

#include <RcppEigen.h>

namespace CV::Stochastic {

namespace OLS {

WorkerModel::WorkerModel(const Eigen::Index ncol,
                         const Eigen::Index maxTrainSize,
                         const Eigen::Index maxTestSize, const double threshold)
    : qtz_{maxTrainSize, ncol}, testResid_{maxTestSize} {
  // Threshold at which to consider pivots zero "A pivot will be considered
  // nonzero if its absolute value is strictly greater than
  // |pivot|⩽threshold×|maxpivot| "
  qtz_.setThreshold(threshold);
}

WorkerModel::WorkerModel(const WorkerModel& other)
    : qtz_{other.qtz_.rows(), other.qtz_.cols()},
      testResid_{other.testResid_.size()} {
  // Threshold at which to consider pivots zero
  qtz_.setThreshold(other.qtz_.threshold());
}

}  // namespace OLS

namespace Ridge {

WorkerModel::WorkerModel(const Eigen::Index ncol,
                         const Eigen::Index maxTrainSize,
                         const Eigen::Index maxTestSize, const double threshold,
                         const double lambda)
    : udvT_{maxTrainSize, ncol, Eigen::ComputeThinU | Eigen::ComputeThinV},
      beta_{ncol},
      testResid_{maxTestSize},
      uTy_{std::min(maxTrainSize, ncol)},
      singularVals_{std::min(maxTrainSize, ncol)},
      singularShrinkFactors_{std::min(maxTrainSize, ncol)},
      lambda_{lambda},
      info_{Eigen::Success} {
  // Prescribe threshold to SVD where singular values are considered zero "A
  // singular value will be considered nonzero if its value is strictly greater
  // than |singularvalue|⩽threshold×|maxsingularvalue|."
  udvT_.setThreshold(threshold);
}

WorkerModel::WorkerModel(const WorkerModel& other)
    : udvT_{other.udvT_.rows(), other.udvT_.cols(),
            Eigen::ComputeThinU | Eigen::ComputeThinV},
      beta_{other.beta_.size()},
      testResid_{other.testResid_.size()},
      uTy_{other.uTy_.size()},
      singularVals_{other.singularVals_.size()},
      singularShrinkFactors_{other.singularShrinkFactors_.size()},
      lambda_{other.lambda_},
      info_{other.info_} {
  // Prescribe threshold to SVD where singular values are considered zero
  udvT_.setThreshold(other.udvT_.threshold());
}

Eigen::ComputationInfo WorkerModel::getInfo() const noexcept { return info_; }

}  // namespace Ridge

}  // namespace CV::Stochastic
