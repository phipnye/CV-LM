#ifndef CV_LM_CV_STOCHASTIC_WORKERMODEL_H
#define CV_LM_CV_STOCHASTIC_WORKERMODEL_H

#include <RcppEigen.h>

#include "Utils-Data.h"
#include "Utils-Decompositions.h"

namespace CV::Stochastic {

namespace OLS {

class WorkerModel {
 public:
  static constexpr bool canFail{false};  // COD is always succesful
  static constexpr bool requiresLambda{false};

 private:
  // Members
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> qtz_;
  Eigen::VectorXd testResid_;

 public:
  // Main ctor
  explicit WorkerModel(Eigen::Index ncol, Eigen::Index maxTrainSize,
                       Eigen::Index maxTestSize, double threshold);

  // Copy ctor - required for RcppParallel split
  WorkerModel(const WorkerModel& other);

  // Evaluate out-of-sample model performance
  template <typename DerivedX, typename DerivedY>
  [[nodiscard]] double evalTestMSE(const Eigen::MatrixBase<DerivedX>& xTrain,
                                   const Eigen::MatrixBase<DerivedY>& yTrain,
                                   const Eigen::MatrixBase<DerivedX>& xTest,
                                   const Eigen::MatrixBase<DerivedY>& yTest) {
    // Make sure the data is what we expect (test and train use same derivation)
    Utils::Data::assertDataStructure(xTrain, yTrain);

    // Decompose training set into the form XP = QTZ (we do not need to check
    // for success of this decomposition as documentation states info method
    // always returns success for COD)
    qtz_.compute(xTrain);

    // Compute test set residuals:
    // This behavior strays from R's lm behavior when the design matrix is not
    // of full-column rank, R (as of 2026) uses Dqrdc2/Linpack which zeros out
    // the last (ncol - rank) coefficients on the "redundant" columns of the
    // design matrix while COD gives the unique minimum norm solution (while
    // this results in different coefficients for any rank-deficient matrix,
    // out-of-sample predictions should only diverge from R's when the system is
    // underdetermined)
    const Eigen::Index testSize{xTest.rows()};
    auto testResid{testResid_.head(testSize)};
    testResid.noalias() = yTest - (xTest * qtz_.solve(yTrain));

    // Return mse = rss / n
    return testResid.squaredNorm() / static_cast<double>(testSize);
  }
};

}  // namespace OLS

namespace Ridge {

class WorkerModel {
 public:
  static constexpr bool canFail{true};  // BDCSVD can fail
  static constexpr bool requiresLambda{true};

 private:
  // Members
  Eigen::BDCSVD<Eigen::MatrixXd> udvT_;
  Eigen::VectorXd beta_;
  Eigen::VectorXd testResid_;
  Eigen::VectorXd uTy_;
  Eigen::VectorXd singularVals_;
  Eigen::VectorXd singularShrinkFactors_;
  const double lambda_;
  Eigen::ComputationInfo info_;

 public:
  // Main ctor
  explicit WorkerModel(Eigen::Index ncol, Eigen::Index maxTrainSize,
                       Eigen::Index maxTestSize, double threshold,
                       double lambda);

  // Copy ctor - required for RcppParallel split
  WorkerModel(const WorkerModel& other);

  // Evaluate out-of-sample model performance
  template <typename DerivedX, typename DerivedY>
  [[nodiscard]] double evalTestMSE(const Eigen::MatrixBase<DerivedX>& xTrain,
                                   const Eigen::MatrixBase<DerivedY>& yTrain,
                                   const Eigen::MatrixBase<DerivedX>& xTest,
                                   const Eigen::MatrixBase<DerivedY>& yTest) {
    // Make sure the data is what we expect (test and train use same derivation)
    Utils::Data::assertDataStructure(xTrain, yTrain);

    // Obtain singular value decomposition of the training set
    udvT_.compute(xTrain);

    // Make sure decomposition was successful
    if (const Eigen::ComputationInfo info{udvT_.info()};
        info != Eigen::Success) {
      info_ = info;
      return 0.0;
    }

    // The number of singular values may change across folds
    const Eigen::Index singularValsSize{udvT_.singularValues().size()};
    auto uTy{uTy_.head(singularValsSize)};
    uTy.noalias() = udvT_.matrixU().transpose() * yTrain;

    // Apply the shrinkage to the singular values (these shrinkage factors are
    // related to the coordinate shrinkage factors (d_j^2 / (d_j^2 + lambda))
    // for solving fitted values X * beta but are reduced by d_j since we're
    // solving explicitly for beta)
    Utils::Decompositions::getSingularVals(udvT_, singularVals_);
    const auto singularVals{singularVals_.head(singularValsSize)};
    auto singularShrinkFactors{singularShrinkFactors_.head(singularValsSize)};
    singularShrinkFactors =
        singularVals.array() / (singularVals.array().square() + lambda_);

    // beta_ridge = V * diag(D^2 + LI)^-1 D * U'y (this function should only be
    // called for lambda > 0 or else we would need to modify this to provide
    // minimum norm solution for rank-deficient matrices)
    beta_.noalias() = udvT_.matrixV() *
                      (singularShrinkFactors.array() * uTy.array()).matrix();

    // Compute test set residuals
    const Eigen::Index testSize{xTest.rows()};
    auto testResid{testResid_.head(testSize)};
    testResid.noalias() = yTest - (xTest * beta_);

    // Return mse = rss / n
    return testResid.squaredNorm() / static_cast<double>(testSize);
  }

  // Get decomposition success information
  [[nodiscard]] Eigen::ComputationInfo getInfo() const noexcept;
};

}  // namespace Ridge

}  // namespace CV::Stochastic

#endif  // CV_LM_CV_STOCHASTIC_WORKERMODEL_H
