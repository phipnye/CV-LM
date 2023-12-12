#ifndef CV_LM_EXPORT_H
#define CV_LM_EXPORT_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
using namespace Rcpp;

List cvLM(const Eigen::VectorXd& y, const Eigen::MatrixXd& X, int K, const int& seed, const String& pivot = "full", const bool& rankCheck = true);
List parcvLM(const Eigen::VectorXd& y, const Eigen::MatrixXd& X, int K, const int& seed, const String& pivot = "full", const bool& rankCheck = true);

#endif