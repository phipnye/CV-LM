#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>
#include <utility>

#include "Utils-Decompositions-utils.h"
#include "Utils-Folds-utils.h"

namespace CV {

template <typename WorkerModel>
class Worker : public RcppParallel::Worker {
  // Thread-local buffers
  Eigen::VectorXd beta_;
  Eigen::VectorXd resid_;
  Eigen::VectorXi trainIdxs_;
  Eigen::VectorXi testIdxs_;

  // Should be one of the WorkerModel objects from OLS or Ridge namespaces
  WorkerModel model_;  // fit-specific data (e.g., lambda for Ridge)

  // References
  const Eigen::Map<Eigen::VectorXd>& y_;
  const Eigen::Map<Eigen::MatrixXd>& x_;
  const Utils::Folds::FoldInfo& foldInfo_;

  // Accumulator
  double cvRes_;

  // Sizes
  const Eigen::Index nrow_;

  // Enum for checking success of decompositions (only relevant to svd cases)
  Eigen::ComputationInfo info_;

 public:
  // Main Ctor
  template <typename... Lambda>
  explicit Worker(const Eigen::Map<Eigen::VectorXd>& y,
                  const Eigen::Map<Eigen::MatrixXd>& x,
                  const Utils::Folds::FoldInfo& foldInfo,
                  const double threshold, Lambda&&... lambda)
      : beta_{x.cols()},
        resid_{foldInfo.maxTestSize_},
        trainIdxs_{foldInfo.maxTrainSize_},
        testIdxs_{foldInfo.maxTestSize_},
        model_{x.cols(), foldInfo.maxTrainSize_, threshold,
               std::forward<Lambda>(lambda)...},
        y_{y},
        x_{x},
        foldInfo_{foldInfo},
        cvRes_{0.0},
        nrow_{x.rows()},
        info_{Eigen::Success} {}

  // Split Ctor
  Worker(const Worker& other, const RcppParallel::Split)
      : beta_{other.beta_.size()},
        resid_{other.resid_.size()},
        trainIdxs_{other.trainIdxs_.size()},
        testIdxs_{other.testIdxs_.size()},
        model_{other.model_},
        y_{other.y_},
        x_{other.x_},
        foldInfo_{other.foldInfo_},
        cvRes_{0.0},
        nrow_{other.nrow_},
        info_{other.info_} {}

  // Worker should only be copied via split ctor
  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  // parallelReduce requires an operator() to perform the work
  void operator()(const std::size_t begin, const std::size_t end) override {
    // Casting from std::size_t to int is safe here (end is the number of folds
    // which is a signed 32-bit integer from R)
    for (int testID{static_cast<int>(begin)}, endID{static_cast<int>(end)};
         testID < endID; ++testID) {
      // Extract test and training set sizes
      const Eigen::Index testSize{foldInfo_.testFoldSizes_[testID]};
      const Eigen::Index trainSize{nrow_ - testSize};

      // Split the test and training indices
      foldInfo_.testTrainSplit(testID, testIdxs_, trainIdxs_);

      // Fit the model on the training set
      model_.computeBeta(x_(trainIdxs_.head(trainSize), Eigen::all),
                         y_(trainIdxs_.head(trainSize)), beta_);

      // Check whether computation was successful (we only need to check this in
      // the ridge case since it uses singular value decomposition whereas OLS
      // uses complete orthogonal decomposition which is documented to always be
      // successful)
      if constexpr (WorkerModel::canFail) {
        if (const Eigen::ComputationInfo modelInfo{model_.getInfo()};
            modelInfo != Eigen::Success) {
          info_ = modelInfo;
          return;
        }
      }

      // Evaluate performance on hold-out set (MSE)
      auto testResid{resid_.head(testSize)};
      testResid.noalias() = y_(testIdxs_.head(testSize)) -
                            (x_(testIdxs_.head(testSize), Eigen::all) * beta_);
      const double testMSE{testResid.squaredNorm() / testSize};

      // Weighted MSE contribution
      const double wt{static_cast<double>(testSize) / nrow_};
      cvRes_ += (wt * testMSE);
    }
  }

  // parallelReduce uses join method to compose the operations of two worker
  // instances
  void join(const Worker& other) {
    // Record unsuccessful decompositions for ridge instances
    if constexpr (WorkerModel::canFail) {
      if (info_ != Eigen::Success) {
        return;
      }

      if (other.info_ != Eigen::Success) {
        info_ = other.info_;
        return;
      }
    }

    cvRes_ += other.cvRes_;
  }

  // Member access
  [[nodiscard]] double getCV() const {
    // Make sure singular value decomposition was successful before we return a
    // result
    if constexpr (WorkerModel::canFail) {
      Utils::Decompositions::checkSvdInfo(
          info_);  // Important we don't call this from a multithreaded context
                   // since Rcpp::stop may be called
    }

    return cvRes_;
  }
};

}  // namespace CV
