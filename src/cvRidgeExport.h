#ifndef CV_RIDGE_EXPORT_H
#define CV_RIDGE_EXPORT_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
using namespace Rcpp;

List cvRidge(const Eigen::VectorXd& y, const Eigen::MatrixXd& X, const double& lambda, int K, const int& seed, const bool& pivot = true);
List parcvRidge(const Eigen::VectorXd& y, const Eigen::MatrixXd& X, const double& lambda, int K, const int& seed, const bool& pivot = true)

#endif