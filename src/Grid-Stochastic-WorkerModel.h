#ifndef CV_LM_GRID_STOCHASTIC_WORKERMODEL_H
#define CV_LM_GRID_STOCHASTIC_WORKERMODEL_H

#include <RcppEigen.h>

#include <algorithm>

#include "Utils-Data.h"
#include "Utils-Decompositions.h"

namespace Grid::Stochastic {

class WorkerModel {
  // Pre-allocate for SVD
  Eigen::BDCSVD<Eigen::MatrixXd> udvT_;

  // Thread-specific data buffers
  Eigen::VectorXd uTy_;
  Eigen::VectorXd beta_;
  Eigen::VectorXd testResid_;
  Eigen::VectorXd singularVals_;
  Eigen::VectorXd singularValsSq_;
  Eigen::VectorXd singularShrinkFactors_;

  // Sizes
  Eigen::Index singularValsSize_;  // m = min(n, p) may change across folds

 public:
  // Main ctor
  WorkerModel(const Eigen::Index ncol, const Eigen::Index maxTrainSize,
              const Eigen::Index maxTestSize, const double threshold)
      : udvT_{maxTrainSize, ncol, Eigen::ComputeThinU | Eigen::ComputeThinV},
        uTy_{std::min(maxTrainSize, ncol)},  // m = min(n,p) where U'y exists
        beta_{ncol},
        testResid_{maxTestSize},
        singularVals_{std::min(maxTrainSize, ncol)},
        singularValsSq_{std::min(maxTrainSize, ncol)},
        singularShrinkFactors_{std::min(maxTrainSize, ncol)},
        singularValsSize_{-1}  // gets set in the fit method
  {
    // Prescribe threshold to SVD where singular values are considered zero "A
    // singular value will be considered nonzero if its value is strictly
    // greater than |singularvalue|⩽threshold×|maxsingularvalue|."
    udvT_.setThreshold(threshold);
  }

  // Copy ctor
  WorkerModel(const WorkerModel& other)
      : udvT_{other.udvT_.rows(), other.udvT_.cols(),
              Eigen::ComputeThinU | Eigen::ComputeThinV},
        uTy_{other.uTy_.size()},
        beta_{other.beta_.size()},
        testResid_{other.testResid_.size()},
        singularVals_{other.singularVals_.size()},
        singularValsSq_{other.singularValsSq_.size()},
        singularShrinkFactors_{other.singularShrinkFactors_.size()},
        singularValsSize_{-1}  // gets set in the fit method
  {
    // Prescribe threshold to SVD where singular values are considered zero
    udvT_.setThreshold(other.udvT_.threshold());
  }

  // Fit the model to the training set (this method takes place in the outer
  // loop that we parallelize across folds before we iterate over shrinkage
  // values since it's only specific to the training set data
  template <typename DerivedX, typename DerivedY>
  [[nodiscard]] Eigen::ComputationInfo fit(
      const Eigen::MatrixBase<DerivedX>& xTrain,
      const Eigen::MatrixBase<DerivedY>& yTrain) {
    // Make sure the data is what we expect
    Utils::Data::assertDataStructure(xTrain, yTrain);

    // Obtain singular value decomposition of the training set
    udvT_.compute(xTrain);

    // Make sure decomposition was successful
    if (const Eigen::ComputationInfo info{udvT_.info()};
        info != Eigen::Success) {
      return info;
    }

    // Across folds (particularly with wide data), m = min(n, p) may change
    singularValsSize_ = udvT_.singularValues().size();

    // Extract the singular values
    auto singularVals{singularVals_.head(singularValsSize_)};
    Utils::Decompositions::getSingularVals(udvT_, singularVals);
    singularValsSq_.head(singularValsSize_) = singularVals.array().square();

    // Compute projection of y onto the left singular vectors of X
    uTy_.head(singularValsSize_).noalias() =
        udvT_.matrixU().transpose() * yTrain;
    return Eigen::Success;
  }

  // Compute the unique minimum-norm solution via the MP pseudoinverse (note
  // that this produces different coefficients than R's lm but will only produce
  // different out-of-sample predictions when the system is underdetermined
  // [this may also differ slightly from cvLM because of small numerical
  // differences between COD and SVD])
  template <typename DerivedX, typename DerivedY>
  [[nodiscard]] double olsEvalTestMSE(
      const Eigen::MatrixBase<DerivedX>& xTest,
      const Eigen::MatrixBase<DerivedY>& yTest) {
    // Make sure the data is what we expect
    Utils::Data::assertDataStructure(xTest, yTest);

    // Shrinkage factors simplify to 1 / diag(D) when lambda == 0.0
    singularShrinkFactors_.setZero();

    // See "Matrix Analysis and Applied Linear Algebra" [Meyer p.423]
    const Eigen::Index rank{udvT_.rank()};
    singularShrinkFactors_.head(rank) =
        singularVals_.head(rank).array().inverse();
    return evalTestMSE(xTest, yTest);
  }

  template <typename DerivedX, typename DerivedY>
  [[nodiscard]] double ridgeEvalTestMSE(
      const Eigen::MatrixBase<DerivedX>& xTest,
      const Eigen::MatrixBase<DerivedY>& yTest, const double lambda) {
    // Make sure the data is what we expect
    Utils::Data::assertDataStructure(xTest, yTest);

    // At this point, we should have lambda > 0 (these shrinkage factors are
    // related to the coordinate shrinkage factors (d_j^2 / (d_j^2 + lambda))
    // for solving fitted values X * beta but are reduced by d_j since we're
    // solving explicitly for beta)
    singularShrinkFactors_.head(singularValsSize_) =
        singularVals_.head(singularValsSize_).array() /
        (singularValsSq_.head(singularValsSize_).array() + lambda);
    return evalTestMSE(xTest, yTest);
  }

 private:
  template <typename DerivedX, typename DerivedY>
  [[nodiscard]] double evalTestMSE(const Eigen::MatrixBase<DerivedX>& xTest,
                                   const Eigen::MatrixBase<DerivedY>& yTest) {
    // Make sure the data is what we expect
    Utils::Data::assertDataStructure(xTest, yTest);

    /*
     * beta = (X'X + LI)^-1 X'y
     * (V D^2 V' + LI) * beta = VDU'y
     * beta = V * alpha
     * (V D^2 V' + LI) V * alpha = VDU'y
     * V D^2 V'V * alpha + lambda * V * alpha = VDU'y
     * V (D^2 + LI) alpha = VDU'y
     * (D^2 + LI) alpha = DU'y
     * alpha = (D^2 + LI)^-1 DU'y
     * beta = V * alpha = V * [(D^2 + LI)^-1 D] * U'y
     */
    const auto alpha{(singularShrinkFactors_.head(singularValsSize_).array() *
                      uTy_.head(singularValsSize_).array())};
    beta_.noalias() = udvT_.matrixV() * alpha.matrix();

    // Compute test set residuals
    const Eigen::Index testSize{xTest.rows()};
    auto testResid{testResid_.head(testSize)};
    testResid.noalias() = yTest - (xTest * beta_);

    // Compute mse = rss / n
    return testResid.squaredNorm() / static_cast<double>(testSize);
  }
};

}  // namespace Grid::Stochastic

#endif  // CV_LM_GRID_STOCHASTIC_WORKERMODEL_H
