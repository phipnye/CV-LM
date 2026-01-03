#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>
#include <utility>

namespace Grid {

// Base class for searching grid of deterministic CV (LOOCV and GCV) results
struct DeterministicGridWorker : public RcppParallel::Worker {
  // Common data members
  const double maxLambda_;
  const double precision_;
  const bool centered_;
  const Eigen::ArrayXd& eigenValsSq_;
  const Eigen::VectorXd& uty_;
  const Eigen::Index nrow_;
  std::pair<double, double> results_;

  // Thread-local buffer for repeated denominator computations
  Eigen::ArrayXd denom_;

  // Ctor
  explicit DeterministicGridWorker(double maxLambda, double precision,
                                   bool centered,
                                   const Eigen::ArrayXd& eigenValsSq,
                                   const Eigen::VectorXd& uty,
                                   Eigen::Index nrow);

  virtual ~DeterministicGridWorker() override = default;

  // Join logic for parallel reduction
  void join(const DeterministicGridWorker& other);
};

struct GCVGridWorker : public DeterministicGridWorker {
  // Unique data members
  const double rssNull_;
  const Eigen::ArrayXd& utySq_;

  // Ctor
  explicit GCVGridWorker(double maxLambda, double precision, bool centered,
                         const Eigen::ArrayXd& eigenValsSq,
                         const Eigen::VectorXd& uty, Eigen::Index nrow,
                         double rssNull, const Eigen::ArrayXd& utySq);

  // Split ctor
  explicit GCVGridWorker(const GCVGridWorker& other, RcppParallel::Split split);

  // RcppParallel requires an operator() to perform the work
  void operator()(std::size_t begin, std::size_t end) override;
};

struct LOOCVGridWorker : public DeterministicGridWorker {
  // Unique data members
  const Eigen::VectorXd& yNull_;
  const Eigen::MatrixXd& u_;
  const Eigen::MatrixXd& uSq_;

  // Thread-specific data buffers
  Eigen::ArrayXd diagD_;
  Eigen::ArrayXd diagH_;
  Eigen::VectorXd resid_;

  // Ctor
  explicit LOOCVGridWorker(double maxLambda, double precision, bool centered,
                           const Eigen::ArrayXd& eigenValsSq,
                           const Eigen::VectorXd& uty, Eigen::Index nrow,
                           const Eigen::VectorXd& yNull,
                           const Eigen::MatrixXd& u,
                           const Eigen::MatrixXd& uSq);

  // Split ctor
  explicit LOOCVGridWorker(const LOOCVGridWorker& other,
                           RcppParallel::Split split);

  // RcppParallel requires an operator() to perform the work
  void operator()(std::size_t begin, std::size_t end) override;
};

};  // namespace Grid
