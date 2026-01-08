#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

// ReSharper disable once CppUnusedIncludeDirective
#include <cstddef>

#include "Grid-Generator.h"

namespace Grid::Stochastic {

struct Worker : RcppParallel::Worker {
  // References
  const Eigen::VectorXd& y_;
  const Eigen::MatrixXd& x_;
  const Eigen::VectorXi& foldIDs_;
  const Eigen::VectorXi& foldSizes_;
  const Generator& lambdasGrid_;

  // Sizes
  const Eigen::Index nrow_;

  // Accumulator (vector of MSEs - one per lambda)
  Eigen::VectorXd mses_;

  // Thread-specific data buffers
  Eigen::VectorXd uty_;
  Eigen::VectorXd beta_;
  Eigen::VectorXd resid_;
  Eigen::ArrayXd eigenVals_;
  Eigen::ArrayXd eigenValsSq_;
  Eigen::ArrayXd diagD_;
  Eigen::VectorXi trainIdxs_;
  Eigen::VectorXi testIdxs_;

  // Pre-allocate for SVD
  Eigen::BDCSVD<Eigen::MatrixXd> svd_;

  // Flag indicating success of decompositions
  Eigen::ComputationInfo info_;

  // Ctor
  explicit Worker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                  const Eigen::VectorXi& foldIDs,
                  const Eigen::VectorXi& foldSizes,
                  const Generator& lambdasGrid, Eigen::Index maxTrainSize,
                  Eigen::Index maxTestSize, double threshold);

  // Split ctor
  Worker(const Worker& other, RcppParallel::Split);

  // RcppParallel requires an operator() to perform the work
  void operator()(std::size_t begin, std::size_t end) override;

  // parallelReduce uses join to compose the operations of two worker instances
  // that were previously split
  void join(const Worker& other);
};

};  // namespace Grid::Stochastic
