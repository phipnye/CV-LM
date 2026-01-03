#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>

namespace CV {

struct BaseCVWorker : public RcppParallel::Worker {
  // Data members
  const Eigen::VectorXd& y_;
  const Eigen::MatrixXd& x_;
  const Eigen::VectorXi& foldIDs_;
  const Eigen::VectorXi& foldSizes_;
  const Eigen::Index nrow_;
  const Eigen::Index ncol_;
  double mse_;

  // Thread-local buffers
  Eigen::MatrixXd xTrain_;
  Eigen::VectorXd yTrain_;
  Eigen::VectorXi trainIdxs_;
  Eigen::VectorXi testIdxs_;
  Eigen::VectorXd beta_;

  // Ctor
  explicit BaseCVWorker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                        const Eigen::VectorXi& foldIDs,
                        const Eigen::VectorXi& foldSizes, Eigen::Index nrow,
                        Eigen::Index ncol);

  // Virtaul dtor
  virtual ~BaseCVWorker() override = default;

  // RcppParallel's parallel reduce requires:
  // 1) An operator() which performs the work
  // 2) A join method which composes the operations of two worker instances that
  // were previously split Here we simply combine the accumulated value of the
  // instance we are being joined with to our own
  void operator()(std::size_t begin, std::size_t end) override;
  void join(const BaseCVWorker& other);

  // Derived classes implement the specific math engine (use Eigen::Ref to
  // accept expressions without forcing evaluation to MatrixXd)
  virtual void computeCoef(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                           const Eigen::Ref<const Eigen::VectorXd>& yTrain) = 0;
};

namespace OLS {

struct CVWorker : public BaseCVWorker {
  // Ctor
  explicit CVWorker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                    const Eigen::VectorXi& foldIDs,
                    const Eigen::VectorXi& foldSizes, Eigen::Index nrow,
                    Eigen::Index ncol);

  // Split ctor
  explicit CVWorker(const CVWorker& other, RcppParallel::Split split);

  // Compute OLS coefficients
  void computeCoef(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                   const Eigen::Ref<const Eigen::VectorXd>& yTrain) override;
};

}  // namespace OLS

namespace Ridge {

struct CVWorker : public BaseCVWorker {
  // (Additional) data member
  const double lambda_;

  // Pre-allocated buffers for X'X + lambda * I
  Eigen::MatrixXd xtxLambda_;

  // Ctor
  explicit CVWorker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                    double lambda, const Eigen::VectorXi& foldIDs,
                    const Eigen::VectorXi& foldSizes, Eigen::Index nrow,
                    Eigen::Index ncol);

  // Split ctor
  explicit CVWorker(const CVWorker& other, RcppParallel::Split split);

  // Compute ridge coefficients
  void computeCoef(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                   const Eigen::Ref<const Eigen::VectorXd>& yTrain) override;
};
}  // namespace Ridge

}  // namespace CV
