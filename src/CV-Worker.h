#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>

#include "CV-Utils-utils.h"

namespace CV {

template <typename WorkerModelType, typename ModelFactory>
class Worker : public RcppParallel::Worker {
  // Thread-local buffers
  Eigen::MatrixXd xTrain_;
  Eigen::VectorXd yTrain_;
  Eigen::VectorXd beta_;
  Eigen::VectorXd resid_;
  Eigen::VectorXi trainIdxs_;
  Eigen::VectorXi testIdxs_;

  // Should be one of the WorkerModel objects from OLS, Ridge::Narrow, or
  // Ridge::Wide
  WorkerModelType model_;  // fit-specific data (e.g., lambda for Ridge)

  // References
  const Eigen::Map<Eigen::VectorXd>& y_;
  const Eigen::Map<Eigen::MatrixXd>& x_;
  const Eigen::VectorXi& testFoldIDs_;
  const Eigen::VectorXi& testFoldSizes_;

  // Accumulator
  double cvRes_;

  // Sizes
  const Eigen::Index nrow_;
  const Eigen::Index ncol_;

  // Enum for checking success of decompositions (only relevant to cholesky)
  Eigen::ComputationInfo info_;

 public:
  // Main Ctor
  explicit Worker(const Eigen::Map<Eigen::VectorXd>& y,
                  const Eigen::Map<Eigen::MatrixXd>& x,
                  const Eigen::VectorXi& testFoldIDs,
                  const Eigen::VectorXi& testFoldSizes,
                  const Eigen::Index maxTrainSize,
                  const Eigen::Index maxTestSize, const ModelFactory& factory)
      : xTrain_(maxTrainSize, x.cols()),
        yTrain_(maxTrainSize),
        beta_(x.cols()),
        resid_(maxTestSize),
        trainIdxs_(maxTrainSize),
        testIdxs_(maxTestSize),
        model_{factory()},
        y_{y},
        x_{x},
        testFoldIDs_{testFoldIDs},
        testFoldSizes_{testFoldSizes},
        cvRes_{0.0},
        nrow_{x.rows()},
        ncol_{x.cols()},
        info_{Eigen::Success} {}

  // Split Ctor
  Worker(const Worker& other, const RcppParallel::Split)
      : xTrain_(other.xTrain_.rows(), other.xTrain_.cols()),
        yTrain_(other.yTrain_.size()),
        beta_(other.beta_.size()),
        resid_(other.resid_.size()),
        trainIdxs_(other.trainIdxs_.size()),
        testIdxs_(other.testIdxs_.size()),
        model_{other.model_},
        y_{other.y_},
        x_{other.x_},
        testFoldIDs_{other.testFoldIDs_},
        testFoldSizes_{other.testFoldSizes_},
        cvRes_{0.0},
        nrow_{other.nrow_},
        ncol_{other.ncol_},
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
      const Eigen::Index testSize{testFoldSizes_[testID]};
      const Eigen::Index trainSize{nrow_ - testSize};

      // Split the test and training indices
      Utils::testTrainSplit(testID, testFoldIDs_, testIdxs_, trainIdxs_);

      // Copy training data using pre-allocated buffers
      xTrain_.topRows(trainSize) = x_(trainIdxs_.head(trainSize), Eigen::all);
      yTrain_.head(trainSize) = y_(trainIdxs_.head(trainSize));

      // Fit the model on the training set
      model_.computeBeta(xTrain_.topRows(trainSize), yTrain_.head(trainSize),
                         beta_);

      // Check whether computation was successful (we only need to check this in
      // the ridge case since it uses cholesky decomposition whereas OLS uses
      // complete orthogonal decomposition which is documented to always be
      // successful)
      if constexpr (WorkerModelType::canFail) {
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
    if constexpr (WorkerModelType::canFail) {
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
    // Make sure LDLT decomposition was successful before we return a result
    // NOTE: Imporant we don't call this from a multithreaded context since
    // Rcpp::stop may be called
    if constexpr (WorkerModelType::canFail) {
      Utils::checkLdltStatus(info_);
    }

    return cvRes_;
  }
};

}  // namespace CV
