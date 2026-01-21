#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>

#include "CenteringBuffers.h"
#include "Enums.h"
#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"
#include "Grid-Stochastic-WorkerModel.h"
#include "Utils-Decompositions.h"
#include "Utils-Folds.h"

namespace Grid::Stochastic {

template <Enums::CenteringMethod Centering>
class Worker : public RcppParallel::Worker {
  // Static boolean to check if we mean center the data
  static constexpr bool meanCenter{Centering == Enums::CenteringMethod::Mean};

  // Accumulator (vector of cv results - one per lambda)
  Eigen::VectorXd cvs_;

  // Thread-specific data buffers
  Eigen::VectorXi trainIdxs_;
  Eigen::VectorXi testIdxs_;

  // Worker model object for computing coefficients and test MSEs
  WorkerModel model_;

  // References
  const Utils::Folds::FoldInfo& foldInfo_;
  const Eigen::Map<Eigen::VectorXd>& y_;
  const Eigen::Map<Eigen::MatrixXd>& x_;
  const Generator& lambdasGrid_;

  // Sizes
  const Eigen::Index nrow_;

  // Optional buffers for when centering is needed
  CenteringBuffers<meanCenter> centeringBuffers_;

  // Enum indicating success of singular value decompositions of training sets
  Eigen::ComputationInfo info_;

 public:
  // Main ctor
  explicit Worker(const Eigen::Map<Eigen::VectorXd>& y,
                  const Eigen::Map<Eigen::MatrixXd>& x,
                  const Utils::Folds::FoldInfo& foldInfo,
                  const Generator& lambdasGrid, const double threshold)
      : cvs_{Eigen::VectorXd::Zero(lambdasGrid.size())},
        trainIdxs_{foldInfo.maxTrainSize_},
        testIdxs_{foldInfo.maxTestSize_},
        model_{x.cols(), foldInfo.maxTrainSize_, foldInfo.maxTestSize_,
               threshold},
        foldInfo_{foldInfo},
        y_{y},
        x_{x},
        lambdasGrid_{lambdasGrid},
        nrow_{x.rows()},
        centeringBuffers_{x.cols(), foldInfo.maxTrainSize_,
                          foldInfo.maxTestSize_},
        info_{Eigen::Success} {}

  // Split ctor
  Worker(const Worker& other, const RcppParallel::Split)
      : cvs_{Eigen::VectorXd::Zero(other.cvs_.size())},
        trainIdxs_{other.trainIdxs_.size()},
        testIdxs_{other.testIdxs_.size()},
        model_{other.model_},
        foldInfo_{other.foldInfo_},
        y_{other.y_},
        x_{other.x_},
        lambdasGrid_{other.lambdasGrid_},
        nrow_{other.nrow_},
        centeringBuffers_{other.centeringBuffers_},
        info_{other.info_} {}

  // Worker should only be copied via split ctor
  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  // RcppParallel requires an operator() to perform the work
  void operator()(const std::size_t begin, const std::size_t end) override {
    // Casting from std::size_t to int is safe here (end is the number of folds
    // which is a 32-bit integer from R)
    const int endID{static_cast<int>(end)};

    for (int testID{static_cast<int>(begin)}; testID < endID; ++testID) {
      // Retrieve size of test and training sets
      const Eigen::Index testSize{foldInfo_.testFoldSizes_[testID]};
      const Eigen::Index trainSize{nrow_ - testSize};

      // MSE contribution weight
      const double wt{static_cast<double>(testSize) /
                      static_cast<double>(nrow_)};

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

      // Perform SVD on training set (uses computation options given in ctor)
      if constexpr (meanCenter) {
        // Center the training data and the testing data using the testing set
        // column means
        centeringBuffers_.centerData(xTrain, yTrain, xTest, yTest);
        info_ = model_.fit(centeringBuffers_.getXTrain().topRows(trainSize),
                           centeringBuffers_.getYTrain().head(trainSize));
        // info_ = Fit the model
      } else {
        Enums::assertExpected<Centering, Enums::CenteringMethod::None>();
        info_ = model_.fit(xTrain, yTrain);
      }

      // Terminate early if SVD wasn't successful
      if (info_ != Eigen::Success) {
        return;
      }

      Eigen::Index lambdaIdx{0};

      // Handle the OLS case first in case of rank-deficiency
      if (lambdasGrid_[lambdaIdx] <= 0.0) {
        double testMSE{0.0};

        if constexpr (meanCenter) {
          testMSE += model_.olsEvalTestMSE(
              centeringBuffers_.getXTest().topRows(testSize),
              centeringBuffers_.getYTest().head(testSize));
        } else {
          testMSE += model_.olsEvalTestMSE(xTest, yTest);
        }

        cvs_[lambdaIdx++] += (wt * testMSE);
      }

      const Eigen::Index nLambda{lambdasGrid_.size()};

      // Now solve for the ridge regression (lambda > 0) case where we don't
      // have to worry about zero division for the cooridate shrinkage factors
      while (lambdaIdx < nLambda) {
        const double lambda{lambdasGrid_[lambdaIdx]};
        double testMSE{0.0};

        if constexpr (meanCenter) {
          testMSE += model_.ridgeEvalTestMSE(
              centeringBuffers_.getXTest().topRows(testSize),
              centeringBuffers_.getYTest().head(testSize), lambda);
        } else {
          testMSE += model_.ridgeEvalTestMSE(xTest, yTest, lambda);
        }

        cvs_[lambdaIdx++] += (wt * testMSE);
      }
    }
  }

  // parallelReduce uses join to compose the operations of two worker instances
  // that were previously split
  void join(const Worker& other) {
    // Record unsuccessful decompositions
    if (info_ != Eigen::Success) {
      return;
    }

    if (other.info_ != Eigen::Success) {
      info_ = other.info_;
      return;
    }

    // Add up cross-validation results across folds of the data
    cvs_ += other.cvs_;
  }

  // Retrive optimal CV-lambda pairing
  [[nodiscard]] LambdaCV getOptimalPair() const {
    // Make sure SVD worked consistently before returning results (important we
    // don't call this from a multithreaded context since Rcpp::stop will be
    // called if any decomposition was unsuccessful)
    Utils::Decompositions::checkSvdInfo(info_);

    // Find the smallest cv result
    Eigen::Index bestIdx;
    const double minCV{cvs_.minCoeff(&bestIdx)};

    // Designated initializers not supported until C++20
    // return LambdaCV{.lambda{lambdasGrid[bestIdx]}, .cv{minMSE}};
    return LambdaCV{lambdasGrid_[bestIdx], minCV};
  }
};

}  // namespace Grid::Stochastic
