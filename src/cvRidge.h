#ifndef CV_RIDGE_H
#define CV_RIDGE_H

#include <Rcpp.h>
#include <RcppEigen.h>

using namespace Rcpp;

DataFrame cvRidge(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const int &K0, const double &lambda,
                  const bool &generalized, const int &seed, const int &nthreads);

#endif
