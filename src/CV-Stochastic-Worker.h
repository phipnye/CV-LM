#ifndef CV_LM_CV_STOCHASTIC_WORKER_H
#define CV_LM_CV_STOCHASTIC_WORKER_H

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>
#include <type_traits>

#include "Enums.h"
#include "Utils-Decompositions.h"
#include "Utils-Folds.h"

namespace CV::Stochastic {

template <typename WorkerModel, Enums::CenteringMethod Centering>
class Worker : public RcppParallel::Worker {
  // --- Data members

  // One of the WorkerModel objects from OLS or Ridge namespaces in charge of
  // fitting the model and evaluating out-of-sample performance
  WorkerModel model_;

  // Container in charge of retrieving (and centering) test and training data
  Utils::Folds::DataLoader<Centering> loader_;

  // Reference to fold paritioning information
  const Utils::Folds::DataSplitter& splitter_;

  // Accumulator
  double cvRes_;

  // Number of rows in the design matrix (observations)
  const Eigen::Index nrow_;

  // Enum indicating success of singular value decompositions of training sets
  Eigen::ComputationInfo info_;

 public:
  // --- Ctors

  // OLS ctor
  template <typename WM = WorkerModel,
            typename = std::enable_if_t<!WM::requiresLambda>>
  explicit Worker(const Eigen::VectorXd& ySorted,
                  const Eigen::MatrixXd& xSorted,
                  const Utils::Folds::DataSplitter& splitter,
                  const double threshold)
      : model_{xSorted.cols(), splitter.maxTrain(), splitter.maxTest(),
               threshold},
        loader_{ySorted, xSorted, splitter.maxTrain(), splitter.maxTest()},
        splitter_{splitter},
        cvRes_{0.0},
        nrow_{xSorted.rows()},
        info_{Eigen::Success} {}

  // Ridge ctor
  template <typename WM = WorkerModel,
            typename = std::enable_if_t<WM::requiresLambda>>
  explicit Worker(const Eigen::VectorXd& ySorted,
                  const Eigen::MatrixXd& xSorted,
                  const Utils::Folds::DataSplitter& splitter,
                  const double threshold, const double lambda)
      : model_{xSorted.cols(), splitter.maxTrain(), splitter.maxTest(),
               threshold, lambda},
        loader_{ySorted, xSorted, splitter.maxTrain(), splitter.maxTest()},
        splitter_{splitter},
        cvRes_{0.0},
        nrow_{xSorted.rows()},
        info_{Eigen::Success} {}

  // Split ctor
  Worker(const Worker& other, const RcppParallel::Split)
      : model_{other.model_},
        loader_{other.loader_},
        splitter_{other.splitter_},
        cvRes_{0.0},
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

      // Evaluate out-of-sample performance
      const double testMSE{model_.evalTestMSE(xTrain, yTrain, xTest, yTest)};

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

  // Reduce results across multiple threads
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

  // Retrieve final results
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

#endif  // CV_LM_CV_STOCHASTIC_WORKER_H
