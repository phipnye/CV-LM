#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

// ReSharper disable once CppUnusedIncludeDirective
#include <cstddef>

#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"
#include "Utils-Folds-utils.h"

namespace Grid::Stochastic {

class Worker : public RcppParallel::Worker {
  // Pre-allocate for SVD
  Eigen::BDCSVD<Eigen::MatrixXd> udvT_;

  // Accumulator (vector of MSEs - one per lambda)
  Eigen::VectorXd cvs_;

  // Thread-specific data buffers
  Eigen::VectorXd uTy_;
  Eigen::VectorXd beta_;
  Eigen::VectorXd resid_;
  Eigen::VectorXd singularVals_;
  Eigen::VectorXd singularValsSq_;
  Eigen::VectorXd singularShrinkFactors_;
  Eigen::VectorXi trainIdxs_;
  Eigen::VectorXi testIdxs_;

  // References
  const Utils::Folds::FoldInfo& foldInfo_;
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
                  const Utils::Folds::FoldInfo& foldInfo,
                  const Generator& lambdasGrid, double threshold);

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
  [[nodiscard]] LambdaCV getOptimalPair() const;

 private:
  // Evalutate out-of-sample performance
  void evalTestMSE(
      const Eigen::Ref<const Eigen::VectorXd>& uTy,
      const Eigen::Ref<const Eigen::VectorXd>& singularShrinkFactors,
      Eigen::Index lambdaIdx, Eigen::Index testSize, double wt);
};

}  // namespace Grid::Stochastic
