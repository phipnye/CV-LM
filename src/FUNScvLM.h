#ifndef FUNS_CV_LM_H
#define FUNS_CV_LM_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>

using namespace Rcpp;

// Confirm valid value for K
int Kcheck(const int &n, const int &K0);

// Calculate MSE
double cost(const Eigen::VectorXd &y, const Eigen::VectorXd &yhat);

// Generate OLS coefficients
Eigen::VectorXd OLScoef(const Eigen::VectorXd &y, const Eigen::MatrixXd &X);

// Generate Ridge regression coefficients
Eigen::VectorXd Ridgecoef(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const double &lambda);

// Extract elements of our features that are in-sample
Eigen::MatrixXd XinSample(const Eigen::MatrixXd &X, const Eigen::VectorXi &s, const int &i);

// Extract elements of our target that are in-sample
Eigen::VectorXd yinSample(const Eigen::VectorXd &y, const Eigen::VectorXi &s, const int &i);

// Extract elements of our features that are out-of-sample
Eigen::MatrixXd XoutSample(const Eigen::MatrixXd &X, const Eigen::VectorXi &s, const int &i);

// Extract elements of our target that are out-of-sample
Eigen::VectorXd youtSample(const Eigen::VectorXd &y, const Eigen::VectorXi &s, const int &i);

// Sampling assignment for CV
IntegerVector sampleCV(const IntegerVector &x, const int &size);

// Setup partitions for CV
List cvSetup(const int &seed, const int &n, const int &K);

// Generalized cross-validation for linear regression
double gcvOLS(const Eigen::VectorXd &y, const Eigen::MatrixXd &X);

// Generalized cross-validation for ridge regression
double gcvRidge(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const double &lambda);

// Leave-one-out cross-validation for linear regression
double loocvOLS(const Eigen::VectorXd &y, const Eigen::MatrixXd &X);

// Leave-one-out cross-validation for ridge regression
double loocvRidge(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const double &lambda);

// Multi-threaded CV for linear regression
double parcvOLS(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const int &K, const int &seed, const int &nthreads);

// Multi-threaded CV for ridge regression
double parcvRidge(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const int &K, const double &lambda,
                  const int &seed, const int &nthreads);

// CV for linear regression
double cvOLS(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const int &K, const int &seed);

// CV for ridge regression
double cvRidge(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const int &K, const double &lambda, const int &seed);

#endif
