#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

// ReSharper disable once CppUnusedIncludeDirective
#include <cstddef>

#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"

namespace Grid::Stochastic {

class Worker : public RcppParallel::Worker {
  // Pre-allocate for SVD
  Eigen::BDCSVD<Eigen::MatrixXd> svd_;

  // Accumulator (vector of MSEs - one per lambda)
  Eigen::VectorXd cvs_;

  // Thread-specific data buffers
  Eigen::VectorXd uty_;
  Eigen::VectorXd beta_;
  Eigen::VectorXd resid_;
  Eigen::ArrayXd eigenVals_;
  Eigen::ArrayXd eigenValsSq_;
  Eigen::ArrayXd diagW_;
  Eigen::VectorXi trainIdxs_;
  Eigen::VectorXi testIdxs_;

  // References
  const Eigen::VectorXi& testFoldIDs_;
  const Eigen::VectorXi& testFoldSizes_;
  const Eigen::Map<Eigen::VectorXd>& y_;
  const Eigen::Map<Eigen::MatrixXd>& x_;
  const Generator& lambdasGrid_;

  // Sizes
  const Eigen::Index nrow_;

  // Enum indicating success of singular value decompositions of training sets
  Eigen::ComputationInfo info_;

 public:
  // Main tor
  explicit Worker(const Eigen::Map<Eigen::VectorXd>& y,
                  const Eigen::Map<Eigen::MatrixXd>& x,
                  const Eigen::VectorXi& testFoldIDs,
                  const Eigen::VectorXi& testFoldSizes,
                  const Generator& lambdasGrid, Eigen::Index maxTrainSize,
                  Eigen::Index maxTestSize, double threshold);

  // Split ctor
  Worker(const Worker& other, RcppParallel::Split);

  // Worker should only be copied via split ctor
  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  // RcppParallel requires an operator() to perform the work
  void operator()(std::size_t begin, std::size_t end) override;

  // parallelReduce uses join to compose the operations of two worker instances
  // that were previously split
  void join(const Worker& other);

  // Member access
  LambdaCV getOptimalPair() const;

 private:
  // Evalutate out-of-sample performance
  void evalTestMSE(Eigen::Index lambdaIdx, Eigen::Index testSize,
                   const Eigen::MatrixXd& v, double wt);
};

};  // namespace Grid::Stochastic
