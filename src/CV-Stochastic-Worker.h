#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>
#include <utility>

#include "CenteringBuffers.h"
#include "Enums.h"
#include "Utils-Decompositions.h"
#include "Utils-Folds.h"

namespace CV::Stochastic {

template <typename WorkerModel, Enums::CenteringMethod Centering>
class Worker : public RcppParallel::Worker {
  // Static boolean to check if we mean center the data
  static constexpr bool meanCenter{Centering == Enums::CenteringMethod::Mean};

  // Thread-local buffers
  Eigen::VectorXi trainIdxs_;
  Eigen::VectorXi testIdxs_;

  // Should be one of the WorkerModel objects from OLS or Ridge namespaces
  WorkerModel model_;

  // References
  const Eigen::Map<Eigen::VectorXd>& y_;
  const Eigen::Map<Eigen::MatrixXd>& x_;
  const Utils::Folds::FoldInfo& foldInfo_;

  // Accumulator
  double cvRes_;

  // Sizes
  const Eigen::Index nrow_;

  // Optional buffers for when centering is needed
  CenteringBuffers<meanCenter> centeringBuffers_;

  // Enum indicating success of singular value decompositions of training sets
  Eigen::ComputationInfo info_;

 public:
  // Main Ctor
  template <typename... Lambda>
  explicit Worker(const Eigen::Map<Eigen::VectorXd>& y,
                  const Eigen::Map<Eigen::MatrixXd>& x,
                  const Utils::Folds::FoldInfo& foldInfo,
                  const double threshold, Lambda&&... lambda)
      : trainIdxs_{foldInfo.maxTrainSize_},
        testIdxs_{foldInfo.maxTestSize_},
        model_{x.cols(), foldInfo.maxTrainSize_, foldInfo.maxTestSize_,
               threshold, std::forward<Lambda>(lambda)...},
        y_{y},
        x_{x},
        foldInfo_{foldInfo},
        cvRes_{0.0},
        nrow_{x.rows()},
        centeringBuffers_{x.cols(), foldInfo.maxTrainSize_,
                          foldInfo.maxTestSize_},
        info_{Eigen::Success} {}

  // Split Ctor
  Worker(const Worker& other, const RcppParallel::Split)
      : trainIdxs_{other.trainIdxs_.size()},
        testIdxs_{other.testIdxs_.size()},
        model_{other.model_},
        y_{other.y_},
        x_{other.x_},
        foldInfo_{other.foldInfo_},
        cvRes_{0.0},
        nrow_{other.nrow_},
        centeringBuffers_{other.centeringBuffers_},
        info_{other.info_} {}

  // Worker should only be copied via split ctor
  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  // parallelReduce requires an operator() to perform the work
  void operator()(const std::size_t begin, const std::size_t end) override {
    // Casting from std::size_t to int is safe here (end is the number of folds
    // which is a signed 32-bit integer from R)
    const int endID{static_cast<int>(end)};

    for (int testID{static_cast<int>(begin)}; testID < endID; ++testID) {
      // Extract test and training set sizes
      const Eigen::Index testSize{foldInfo_.testFoldSizes_[testID]};
      const Eigen::Index trainSize{nrow_ - testSize};

      // Split the test and training indices
      foldInfo_.testTrainSplit(testID, testIdxs_, trainIdxs_);

      // Some folds may have fewer observations for the training set (these
      // subsets are views)
      const auto trainIdxs{trainIdxs_.head(trainSize)};
      const auto testIdxs{testIdxs_.head(testSize)};
      const auto xTrain{x_(trainIdxs, Eigen::all)};
      const auto yTrain{y_(trainIdxs)};
      const auto xTest{x_(testIdxs, Eigen::all)};
      const auto yTest{y_(testIdxs)};
      double testMSE{0.0};
      
      // Fit the model on the training set
      if constexpr (meanCenter) {
        // Center the training data and the testing data using the training set
        // column means
        centeringBuffers_.centerData(xTrain, yTrain, xTest, yTest);
        testMSE +=
            model_.evalTestMSE(centeringBuffers_.getXTrain().topRows(trainSize),
                               centeringBuffers_.getYTrain().head(trainSize),
                               centeringBuffers_.getXTest().topRows(testSize),
                               centeringBuffers_.getYTest().head(testSize));
      } else {
        Enums::assertExpected<Centering, Enums::CenteringMethod::None>();
        testMSE += model_.evalTestMSE(xTrain, yTrain, xTest, yTest);
      }

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

      // Weighted MSE contribution
      const double wt{static_cast<double>(testSize) / nrow_};
      cvRes_ += (testMSE * wt);
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
      // Important we don't call this from a multithreaded context since
      // Rcpp::stop will be called if any decomposition was unsuccessful
      Utils::Decompositions::checkSvdInfo(info_);
    }

    return cvRes_;
  }
};

}  // namespace CV::Stochastic
