#pragma once

#include <RcppEigen.h>

#include "CV-WorkerModel.h"

namespace CV {

namespace OLS {

struct WorkerModelFactory {
  const Eigen::Index ncol_;
  const Eigen::Index maxTrainSize_;

  [[nodiscard]] WorkerModel operator()() const;
};

}  // namespace OLS

namespace Ridge {

struct WorkerModelFactory {
  const Eigen::Index ncol_;
  const double lambda_;

  [[nodiscard]] WorkerModel operator()() const;
};

}  // namespace Ridge

}  // namespace CV
