#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>

namespace CV {

struct BaseWorker : public RcppParallel::Worker {
  // Data members
  const Eigen::VectorXd& y_;
  const Eigen::MatrixXd& x_;
  const Eigen::VectorXi& foldIDs_;
  const Eigen::VectorXi& foldSizes_;
  const Eigen::Index nrow_;
  const Eigen::Index ncol_;
  double mse_;

  // Ctor
  BaseWorker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
             const Eigen::VectorXi& foldIDs, const Eigen::VectorXi& foldSizes,
             Eigen::Index nrow, Eigen::Index ncol);

  // Virtaul dtor
  virtual ~BaseWorker() override = default;

  void operator()(std::size_t begin, std::size_t end) override;
  void join(const BaseWorker& other);

  // Derived classes implement the specific math engine (use Eigen::Ref to
  // accept expressions without forcing evaluation to MatrixXd)
  virtual Eigen::VectorXd computeCoef(
      const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
      const Eigen::Ref<const Eigen::VectorXd>& yTrain) const = 0;
};

namespace OLS {

struct Worker : public BaseWorker {
  // Ctor
  Worker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
         const Eigen::VectorXi& foldIDs, const Eigen::VectorXi& foldSizes,
         Eigen::Index nrow, Eigen::Index ncol);

  // Split ctor
  Worker(const Worker& other, RcppParallel::Split split);

  // Compute OLS coefficients
  Eigen::VectorXd computeCoef(
      const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
      const Eigen::Ref<const Eigen::VectorXd>& yTrain) const override;
};

}  // namespace OLS

namespace Ridge {

struct Worker : public BaseWorker {
  // (Additional) data member
  const double lambda_;

  // Ctor
  Worker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x, double lambda,
         const Eigen::VectorXi& foldIDs, const Eigen::VectorXi& foldSizes,
         Eigen::Index nrow, Eigen::Index ncol);

  // Split ctor
  Worker(const Worker& other, RcppParallel::Split split);

  // Compute ridge coefficients
  Eigen::VectorXd computeCoef(
      const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
      const Eigen::Ref<const Eigen::VectorXd>& yTrain) const override;
};
}  // namespace Ridge

}  // namespace CV
