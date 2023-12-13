#ifndef CV_LM_H
#define CV_LM_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>

using namespace Rcpp;

// Calculate MSE
double cost(const Eigen::VectorXd& y, const Eigen::VectorXd& yhat);

// Generate OLS coefficients
Eigen::VectorXd OLScoef(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const String& pivot, const bool& check);

// Generate Ridge regression coefficients
Eigen::VectorXd Ridgecoef(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const bool& pivot, const double& lambda);

// Extract elements of our features that are in-sample
Eigen::MatrixXd XinSample(const Eigen::MatrixXd& X, const Eigen::VectorXi& s, const int& i);

// Extract elements of our target that are in-sample
Eigen::VectorXd yinSample(const Eigen::VectorXd& y, const Eigen::VectorXi& s, const int& i);

// Extract elements of our features that are out-of-sample
Eigen::MatrixXd XoutSample(const Eigen::MatrixXd& X, const Eigen::VectorXi& s, const int& i);

// Extract elements of our target that are out-of-sample
Eigen::VectorXd youtSample(const Eigen::VectorXd& y, const Eigen::VectorXi& s, const int& i);

// Sampling assignment for CV
IntegerVector sampleCV(const IntegerVector& x, const int& size);

// Setup partitions for CV
List cvSetup(const int& seed, const int& n, const int& K);

#endif
