#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>
#include <utility>

namespace CV {

template <typename Model>
struct Worker : public RcppParallel::Worker {
  // References
  const Eigen::VectorXd& y_;
  const Eigen::MatrixXd& x_;
  const Eigen::VectorXi& foldIDs_;
  const Eigen::VectorXi& foldSizes_;

  // Sizes
  const Eigen::Index nrow_;
  const Eigen::Index ncol_;

  // Accumulator
  double mse_;

  // Thread-local buffers
  Eigen::MatrixXd xTrain_;
  Eigen::VectorXd yTrain_;
  Eigen::VectorXd beta_;
  Eigen::VectorXd resid_;
  Eigen::VectorXi trainIdxs_;
  Eigen::VectorXi testIdxs_;

  // Fit-specific data (e.g., lambda for Ridge)
  Model model_;

  // Main Ctor
  template <typename... Args>
  explicit Worker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                  const Eigen::VectorXi& foldIDs,
                  const Eigen::VectorXi& foldSizes,
                  const Eigen::Index maxTrainSize,
                  const Eigen::Index maxTestSize, Args&&... modelArgs)
      : y_{y},
        x_{x},
        foldIDs_{foldIDs},
        foldSizes_{foldSizes},
        nrow_{x.rows()},
        ncol_{x.cols()},
        mse_{0.0},
        xTrain_(maxTrainSize, ncol_),
        yTrain_(maxTrainSize),
        trainIdxs_(maxTrainSize),
        testIdxs_(maxTestSize),
        beta_(ncol_),
        resid_(maxTestSize),
        model_{std::forward<Args>(modelArgs)...} {}

  // Split Ctor
  explicit Worker(const Worker& other, const RcppParallel::Split)
      : y_{other.y_},
        x_{other.x_},
        foldIDs_{other.foldIDs_},
        foldSizes_{other.foldSizes_},
        nrow_{other.nrow_},
        ncol_{other.ncol_},
        mse_{0.0},
        xTrain_(other.xTrain_.rows(), other.xTrain_.cols()),
        yTrain_(other.yTrain_.size()),
        trainIdxs_(other.trainIdxs_.size()),
        testIdxs_(other.testIdxs_.size()),
        beta_(other.beta_.size()),
        resid_(other.resid_.size()),
        model_{other.model_} {}

  // parallelReduce requires an operator() to perform the work
  void operator()(const std::size_t begin, const std::size_t end) {
    // Casting from std::size_t to int is safe here (end is the number of folds
    // which is a signed 32-bit integer from R)
    for (int foldID{static_cast<int>(begin)}, endID{static_cast<int>(end)};
         foldID < endID; ++foldID) {
      const Eigen::Index testSize{foldSizes_[foldID]};
      const Eigen::Index trainSize{nrow_ - testSize};

      // Prepare training and testing containers
      Eigen::Index tr{0};
      Eigen::Index ts{0};

      for (int row{0}; row < nrow_; ++row) {
        if (foldIDs_[row] == foldID) {
          testIdxs_[ts++] = row;
        } else {
          trainIdxs_[tr++] = row;
        }
      }

      // Copy training data using pre-allocated buffers
      xTrain_.topRows(trainSize) = x_(trainIdxs_.head(trainSize), Eigen::all);
      yTrain_.head(trainSize) = y_(trainIdxs_.head(trainSize));

      // Fit the model
      model_.computeBeta(xTrain_.topRows(trainSize), yTrain_.head(trainSize),
                         beta_);

      // Evaluate performance on hold-out fold (MSE)
      resid_.head(testSize) = y_(testIdxs_.head(testSize));
      resid_.head(testSize).noalias() -=
          (x_(testIdxs_.head(testSize), Eigen::all) * beta_);
      const double foldMSE{resid_.head(testSize).squaredNorm() / testSize};

      // Weighted MSE contribution
      const double alpha{static_cast<double>(testSize) / nrow_};
      mse_ += (alpha * foldMSE);
    }
  }

  // parallelReduce uses join method to compose the operations of two worker
  // instances
  void join(const Worker& other) { mse_ += other.mse_; }
};

}  // namespace CV
