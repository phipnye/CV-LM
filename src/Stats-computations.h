#pragma once

#include <RcppEigen.h>

namespace Stats {

// Closed-form solution for generalized cross-validation
[[nodiscard]] double gcv(double rss, double traceHat, Eigen::Index nrow);

// Closed-form solution for leave-one-out cross-validation
[[nodiscard]] double loocv(const Eigen::VectorXd& residuals,
                           const Eigen::VectorXd& diagHat);

}  // namespace Stats
