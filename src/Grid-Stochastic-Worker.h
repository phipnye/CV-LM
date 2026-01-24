#ifndef CV_LM_GRID_STOCHASTIC_WORKER_H
#define CV_LM_GRID_STOCHASTIC_WORKER_H

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>

#include "Enums.h"
#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"
#include "Grid-Stochastic-WorkerModel.h"
#include "Utils-Decompositions.h"
#include "Utils-Folds.h"

namespace Grid::Stochastic {

template <Enums::CenteringMethod Centering>
class Worker : public RcppParallel::Worker {
  // --- Data members

  // Worker model object for computing coefficients and test MSEs
  WorkerModel model_;

  // Container in charge of retrieving (and centering) test and training data
  Utils::Folds::DataLoader<Centering> loader_;

  // Reference to fold paritioning information
  const Utils::Folds::DataSplitter& splitter_;

  // Generator for retrieving shrinkage parameter values to test
  const Generator& lambdasGrid_;

  // Accumulator (vector of cv results - one per lambda)
  Eigen::VectorXd cvs_;

  // Number of rows in the design matrix (observations)
  const Eigen::Index nrow_;

  // Enum indicating success of singular value decompositions of training sets
  Eigen::ComputationInfo info_;

 public:
  // Main ctor
  explicit Worker(const Eigen::VectorXd& ySorted,
                  const Eigen::MatrixXd& xSorted,
                  const Utils::Folds::DataSplitter& splitter,
                  const Generator& lambdasGrid, const double threshold)
      : model_{xSorted.cols(), splitter.maxTrain(), splitter.maxTest(),
               threshold},
        loader_{ySorted, xSorted, splitter.maxTrain(), splitter.maxTest()},
        splitter_{splitter},
        lambdasGrid_{lambdasGrid},
        cvs_{Eigen::VectorXd::Zero(lambdasGrid.size())},
        nrow_{xSorted.rows()},
        info_{Eigen::Success} {}

  // Split ctor
  Worker(const Worker& other, const RcppParallel::Split)
      : model_{other.model_},
        loader_{other.loader_},
        splitter_{other.splitter_},
        lambdasGrid_{other.lambdasGrid_},
        cvs_{Eigen::VectorXd::Zero(other.cvs_.size())},
        nrow_{other.nrow_},
        info_{other.info_} {}

  // Worker should only be copied via split ctor
  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  // Work operator for parallel reduction - each thread gets its own exclusive
  // range
  void operator()(const std::size_t begin, const std::size_t end) override {
    // Casting from std::size_t to Index is safe here (end is the number of
    // folds which is a signed 32-bit integer from R)
    const Eigen::Index endID{static_cast<Eigen::Index>(end)};

    for (Eigen::Index testID{static_cast<Eigen::Index>(begin)}; testID < endID;
         ++testID) {
      // Extract where the test data set starts from (which rows) and how many
      // observations are used in the test set
      const auto [testStart, testSize]{splitter_[testID]};

      // Get the (potentially centered) test and training data sets
      const auto [xTrain, yTrain, xTest,
                  yTest]{loader_.prepData(testStart, testSize)};

      // Fit the model to the training dataset
      info_ = model_.fit(xTrain, yTrain);

      // Terminate early if SVD wasn't successful
      if (info_ != Eigen::Success) {
        return;
      }

      // MSE contribution weight
      const double wt{static_cast<double>(testSize) /
                      static_cast<double>(nrow_)};
      Eigen::Index lambdaIdx{0};

      // Handle the OLS case first in case of rank-deficiency
      if (lambdasGrid_[lambdaIdx] <= 0.0) {
        const double testMSE{model_.olsEvalTestMSE(xTest, yTest)};
        cvs_[lambdaIdx++] += (wt * testMSE);
      }

      const Eigen::Index nLambda{lambdasGrid_.size()};

      // Now solve for the ridge regression (lambda > 0) case where we don't
      // have to worry about zero division for the cooridate shrinkage factors
      while (lambdaIdx < nLambda) {
        const double lambda{lambdasGrid_[lambdaIdx]};
        const double testMSE{model_.ridgeEvalTestMSE(xTest, yTest, lambda)};
        cvs_[lambdaIdx++] += (wt * testMSE);
      }
    }
  }

  // Reduce results across multiple threads
  void join(const Worker& other) {
    // Record unsuccessful decompositions
    if (info_ != Eigen::Success) {
      return;
    }

    if (other.info_ != Eigen::Success) {
      info_ = other.info_;
      return;
    }

    // Add up cross-validation results across folds of the data (vector addition
    // across shrinkage parameter values)
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

#endif  // CV_LM_GRID_STOCHASTIC_WORKER_H
