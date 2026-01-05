#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>

namespace Grid::Stochastic {

struct Worker : RcppParallel::Worker {
  // Data members
  const Eigen::VectorXd& y_;
  const Eigen::MatrixXd& x_;
  const Eigen::VectorXi& foldIDs_;
  const Eigen::VectorXi& foldSizes_;
  const Eigen::VectorXd& lambdas_;
  const Eigen::Index nrow_;
  const Eigen::Index maxTrainSize_;
  const Eigen::Index maxTestSize_;

  // Accumulator (vector of MSEs (one per lambda))
  Eigen::VectorXd mses_;

  // Thread-specific data buffers
  Eigen::VectorXi trainIdxs_;
  Eigen::VectorXi testIdxs_;
  Eigen::VectorXd uty_;
  Eigen::ArrayXd eigenVals_;
  Eigen::ArrayXd eigenValsSq_;
  Eigen::ArrayXd diagD_;
  Eigen::VectorXd beta_;
  Eigen::VectorXd resid_;

  // Ctor
  explicit Worker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                  const Eigen::VectorXi& foldIDs,
                  const Eigen::VectorXi& foldSizes,
                  const Eigen::VectorXd& lambdas, const Eigen::Index nrow,
                  const Eigen::Index maxTrainSize,
                  const Eigen::Index maxTestSize);

  // Split ctor
  Worker(const Worker& other, const RcppParallel::Split);

  // RcppParallel requires an operator() to perform the work
  void operator()(const std::size_t begin, const std::size_t end) override;

  // parallelReduce uses join to compose the operations of two worker instances
  // that were previously split
  void join(const Worker& other);
};

};  // namespace Grid::Stochastic
