#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>

namespace CV {

template <typename FitType>
struct CVWorker : public RcppParallel::Worker {
  // Data members
  const Eigen::VectorXd& y_;
  const Eigen::MatrixXd& x_;
  const Eigen::VectorXi& foldIDs_;
  const Eigen::VectorXi& foldSizes_;
  const Eigen::Index nrow_;
  const Eigen::Index ncol_;
  const Eigen::Index maxTrainSize_;
  const Eigen::Index maxTestSize_;
  double mse_;

  // Thread-local buffers
  Eigen::MatrixXd xTrain_;
  Eigen::VectorXd yTrain_;
  Eigen::VectorXi trainIdxs_;
  Eigen::VectorXi testIdxs_;
  Eigen::VectorXd beta_;
  Eigen::VectorXd resid_;

  // Fit-specific data (e.g., lambda for Ridge)
  FitType model_;

  // Main Ctor
  template <typename... Args>
  explicit CVWorker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                    const Eigen::VectorXi& foldIDs,
                    const Eigen::VectorXi& foldSizes,
                    const Eigen::Index maxTrainSize,
                    const Eigen::Index maxTestSize, Args&&... args)
      : y_{y},
        x_{x},
        foldIDs_{foldIDs},
        foldSizes_{foldSizes},
        nrow_{x.rows()},
        ncol_{x.cols()},
        maxTrainSize_{maxTrainSize},
        maxTestSize_{maxTestSize},
        mse_{0.0},
        xTrain_(maxTrainSize_, ncol_),
        yTrain_(maxTrainSize_),
        trainIdxs_(maxTrainSize_),
        testIdxs_(maxTestSize_),
        beta_(ncol_),
        resid_(maxTestSize_),
        model_{std::forward<Args>(args)...} {}

  // Split Ctor
  explicit CVWorker(const CVWorker& other, const RcppParallel::Split)
      : y_{other.y_},
        x_{other.x_},
        foldIDs_{other.foldIDs_},
        foldSizes_{other.foldSizes_},
        nrow_{other.nrow_},
        ncol_{other.ncol_},
        maxTrainSize_{other.maxTrainSize_},
        maxTestSize_{other.maxTestSize_},
        mse_{0.0},
        xTrain_(maxTrainSize_, ncol_),
        yTrain_(maxTrainSize_),
        trainIdxs_(maxTrainSize_),
        testIdxs_(maxTestSize_),
        beta_(ncol_),
        resid_(maxTestSize_),
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
  void join(const CVWorker& other) { mse_ += other.mse_; }
};

}  // namespace CV
