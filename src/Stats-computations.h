#pragma once

#include <RcppEigen.h>

namespace Stats {

[[nodiscard]] double gcv(double rss, double traceHat, Eigen::Index nrow);
[[nodiscard]] double loocv(const Eigen::VectorXd& residuals,
                           const Eigen::VectorXd& diagHat);

}  // namespace Stats
