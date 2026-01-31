#ifndef CV_LM_CV_STOCHASTIC_WORKER_H
#define CV_LM_CV_STOCHASTIC_WORKER_H

#include <RcppArmadillo.h>
#include <RcppParallel.h>

#include <cstddef>
#include <utility>

#include "ConstexprOptional.h"
#include "DataLoader.h"
#include "Utils-Decompositions.h"

namespace CV::Stochastic {

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

  // Accumulator
  double cvRes_;

  // Conditional data
  ConstexprOptional<Decomp::requiresLambda, double> lambda_;

  // Boolean indicating success of decompositions
  bool success_;

 public:
  // Main ctor
  explicit Worker(Decomp decomp, const DataLoader& loader,
                  const double lambda = 0.0)
      : decomp_{std::move(decomp)},
        XtrainBuf_(loader.maxTrain(), loader.ncol()),
        yTrainBuf_(loader.maxTrain()),
        loader_{loader},
        cvRes_{0.0},
        lambda_{lambda},
        success_{true} {}

  // Split ctor
  Worker(const Worker& other, const RcppParallel::Split)
      : decomp_{other.decomp_.clone()},  // just copies tolerance
        XtrainBuf_(other.XtrainBuf_.n_rows, other.XtrainBuf_.n_cols),
        yTrainBuf_(other.yTrainBuf_.n_elem),
        loader_{other.loader_},
        cvRes_{0.0},
        lambda_{other.lambda_},
        success_{other.success_} {}

  // Worker should only be copied via split ctor
  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  // Work operator for parallel reduction - each thread gets its own exclusive
  // range
  void operator()(const std::size_t foldStart,
                  const std::size_t foldEnd) override {
    // This is safe, foldEnd is bound by signed 32-bit integer value
    const arma::uword endID{static_cast<arma::uword>(foldEnd)};

    for (arma::uword testID{static_cast<arma::uword>(foldStart)};
         testID < endID; ++testID) {
      // Load the test and training data sets
      const auto [Xtest, yTest, testSize,
                  trainSize]{loader_.load(testID, XtrainBuf_, yTrainBuf_)};
      const arma::subview Xtrain{XtrainBuf_.head_rows(trainSize)};
      const arma::subview_col yTrain{yTrainBuf_.head(trainSize)};

      // Set the design matrix, response vector, and lambda
      if constexpr (Decomp::requiresLambda) {
        success_ = Utils::Decompositions::setParams(decomp_, Xtrain, yTrain,
                                                    lambda_.value());
      } else {
        success_ = Utils::Decompositions::setParams(decomp_, Xtrain, yTrain);
      }

      // Terminate early if a decomposition was unsuccessful
      if (!success_) {
        return;
      }

      // Evaluate out-of-sample performance
      const double testMSE{decomp_.testMSE(Xtest, yTest)};

      // Weighted MSE contribution
      const double wt{static_cast<double>(testSize) /
                      static_cast<double>(loader_.nrow())};
      cvRes_ += (testMSE * wt);
    }
  }

  // Reduce results across multiple threads
  void join(const Worker& other) {
    // Make sure all decompositions were successful
    if (!success_ || !other.success_) {
      success_ = false;
      return;
    }

    cvRes_ += other.cvRes_;
  }

  // Retrieve final results
  [[nodiscard]] double getCV() const {
    // Make sure all decompositions were successful before returning a result
    if (!success_) {
      // getCV won't be called from a multithreaded context
      Rcpp::stop(
          "One or more decompositions were unsuccessul in evaluation of K-Fold "
          "CV.");
    }

    return cvRes_;
  }
};

}  // namespace CV::Stochastic

#endif  // CV_LM_CV_STOCHASTIC_WORKER_H
