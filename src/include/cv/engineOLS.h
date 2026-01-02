#pragma once

#include <RcppEigen.h>

namespace CV::OLS {

// Generalized cross-validation for linear regression
double gcv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x);

// Leave-one-out cross-validation for linear regression
double loocv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x);

// Multi-threaded CV for linear regression
double parCV(const Eigen::VectorXd& y, const Eigen::MatrixXd& x, int k,
             int seed, int nThreads);

}  // namespace CV::OLS
