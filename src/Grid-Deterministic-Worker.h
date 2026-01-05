#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>
#include <utility>

namespace Grid::Deterministic {

// Base class for searching grid of deterministic CV (LOOCV and GCV) results
struct Worker : public RcppParallel::Worker {
  // Common data members
  const Eigen::VectorXd& lambdas_;
  const Eigen::ArrayXd& eigenValsSq_;
  const Eigen::VectorXd& uty_;
  const Eigen::Index nrow_;
  const bool centered_;

  // Accumulator (pair of doubles for the [lambda, CV])
  std::pair<double, double> results_;

  // Thread-local buffer for repeated denominator computations
  Eigen::ArrayXd denom_;

  // Ctor
  explicit Worker(const Eigen::VectorXd& lambdas,
                  const Eigen::ArrayXd& eigenValsSq, const Eigen::VectorXd& uty,
                  const Eigen::Index nrow, const bool centered);

  virtual ~Worker() override = default;

  // Join logic for parallel reduction
  void join(const Worker& other);
};

struct GCVGridWorker : public Worker {
  // Unique data members
  const double rssNull_;
  const Eigen::ArrayXd& utySq_;

  // Ctor
  explicit GCVGridWorker(const Eigen::VectorXd& lambdas,
                         const Eigen::ArrayXd& eigenValsSq,
                         const Eigen::VectorXd& uty, const Eigen::Index nrow,
                         const bool centered, const Eigen::ArrayXd& utySq,
                         const double rssNull);

  // Split ctor
  explicit GCVGridWorker(const GCVGridWorker& other,
                         const RcppParallel::Split split);

  // RcppParallel requires an operator() to perform the work
  void operator()(const std::size_t begin, const std::size_t end) override;
};

struct LOOCVGridWorker : public Worker {
  // Unique data members
  const Eigen::VectorXd& yNull_;
  const Eigen::MatrixXd& u_;
  const Eigen::MatrixXd& uSq_;

  // Thread-specific data buffers
  Eigen::ArrayXd diagD_;
  Eigen::ArrayXd diagH_;
  Eigen::VectorXd resid_;

  // Ctor
  explicit LOOCVGridWorker(const Eigen::VectorXd& lambdas,
                           const Eigen::ArrayXd& eigenValsSq,
                           const Eigen::VectorXd& uty, const Eigen::Index nrow,
                           const bool centered, const Eigen::VectorXd& yNull,
                           const Eigen::MatrixXd& u,
                           const Eigen::MatrixXd& uSq);

  // Split ctor
  explicit LOOCVGridWorker(const LOOCVGridWorker& other,
                           const RcppParallel::Split split);

  // RcppParallel requires an operator() to perform the work
  void operator()(const std::size_t begin, const std::size_t end) override;
};

};  // namespace Grid::Deterministic
