#ifndef CV_LM_GRID_STOCHASTIC_WORKER_H
#define CV_LM_GRID_STOCHASTIC_WORKER_H

#include <RcppArmadillo.h>
#include <RcppParallel.h>

#include <cstddef>
#include <utility>

#include "DataLoader.h"
#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"
#include "Utils-Decompositions.h"

namespace Grid::Stochastic {

template <typename Decomp>
class Worker : public RcppParallel::Worker {
  // --- Data members

  // Decomposition object for doing the math
  Decomp decomp_;

  // Data buffers to write training data into
  arma::mat XtrainBuf_;
  arma::vec yTrainBuf_;

  // Container in charge of retrieving test and training data
  const DataLoader& loader_;

  // Generator for retrieving shrinkage parameter values to test
  const Generator& lambdasGrid_;

  // Accumulator (vector of cv results - one per lambda)
  arma::vec cvs_;

  // Boolean indicating success of decompositions
  bool success_;

 public:
  // Main ctor
  explicit Worker(Decomp decomp, const DataLoader& loader,
                  const Generator& lambdasGrid)
      : decomp_{std::move(decomp)},
        XtrainBuf_(loader.maxTrain(), loader.ncol()),
        yTrainBuf_(loader.maxTrain()),
        loader_{loader},
        lambdasGrid_{lambdasGrid},
        cvs_{arma::zeros(lambdasGrid.size())},
        success_{true} {}

  // Split ctor
  Worker(const Worker& other, const RcppParallel::Split)
      : decomp_{other.decomp_.clone()},  // just copies tolerance
        XtrainBuf_(other.XtrainBuf_.n_rows, other.XtrainBuf_.n_cols),
        yTrainBuf_(other.yTrainBuf_.n_elem),
        loader_{other.loader_},
        lambdasGrid_{other.lambdasGrid_},
        cvs_{arma::zeros(other.cvs_.n_elem)},
        success_{other.success_} {}

  // Worker should only be copied via split ctor
  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  // Work operator for parallel reduction - each thread gets its own exclusive
  // range
  void operator()(const std::size_t foldStart,
                  const std::size_t foldEnd) override {
    // This is safe, foldEnd is bound by signed 32-bit integer values
    const arma::uword endID{static_cast<arma::uword>(foldEnd)};

    for (arma::uword testID{static_cast<arma::uword>(foldStart)};
         testID < endID; ++testID) {
      // Load the test and training data sets
      const auto [Xtest, yTest, testSize,
                  trainSize]{loader_.load(testID, XtrainBuf_, yTrainBuf_)};
      const arma::subview Xtrain{XtrainBuf_.head_rows(trainSize)};
      const arma::subview_col yTrain{yTrainBuf_.head(trainSize)};

      // MSE contribution weight
      const double wt{static_cast<double>(testSize) /
                      static_cast<double>(loader_.nrow())};

      static_assert(Decomp::requiresLambda,
                    "Attempting to instantiate grid search with an object that "
                    "doesn't support shrinkage.");
      arma::uword lambdaIdx{0};

      // Fit the model to the training dataset
      success_ = Utils::Decompositions::setParams(decomp_, Xtrain, yTrain,
                                                  lambdasGrid_[lambdaIdx]);

      // Terminate early if a decomposition was unsuccessful
      if (!success_) {
        return;
      }

      // Evaluate the OLS case
      cvs_[lambdaIdx++] += (wt * decomp_.testMSE(Xtest, yTest));
      const arma::uword nLambda{lambdasGrid_.size()};

      // Now loop over the remaining lambdas and compute the corresponding CV
      // values
      while (lambdaIdx < nLambda) {
        decomp_.setLambda(lambdasGrid_[lambdaIdx]);
        cvs_[lambdaIdx++] += (wt * decomp_.testMSE(Xtest, yTest));
      }
    }
  }

  // Reduce results across multiple threads
  void join(const Worker& other) {
    // Make sure all decompositions were successful
    if (!success_ || !other.success_) {
      success_ = false;
      return;
    }

    // Add up cross-validation results across folds of the data (vector addition
    // across shrinkage parameter values)
    cvs_ += other.cvs_;
  }

  // Retrive optimal CV-lambda pairing
  [[nodiscard]] LambdaCV getOptimalPair() const {
    // Make sure all decompositions were successful before returning a result
    if (!success_) {
      // getCV won't be called from a multithreaded context
      Rcpp::stop(
          "One or more decompositions were unsuccessul in evaluation of K-Fold "
          "CV.");
    }

    // Find the smallest cv result
    const arma::uword bestIdx{cvs_.index_min()};

    // Designated initializers not supported until C++20
    return LambdaCV{lambdasGrid_[bestIdx], cvs_[bestIdx]};
  }
};

}  // namespace Grid::Stochastic

#endif  // CV_LM_GRID_STOCHASTIC_WORKER_H
