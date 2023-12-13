#ifndef CV_RIDGE_WORKER_H
#define CV_RIDGE_WORKER_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>

using namespace Rcpp;

// Struct for threaded execution of CV
struct cvRidgeWorker : public RcppParallel::Worker {
  const Eigen::VectorXd& y;
  const Eigen::MatrixXd& X;
  const double& lambda;
  const Eigen::VectorXi& s;
  const NumericVector& ns;
  const int& n;
  const bool& pivot;
  double MSE;

  cvRidgeWorker(const Eigen::VectorXd& y_, const Eigen::MatrixXd& X_, const double& lamda_, const Eigen::VectorXi& s_, const NumericVector& ns_, const int& n_, const bool& pivot_);
  cvRidgeWorker(const cvRidgeWorker& CVRW, RcppParallel::Split);

  void operator()(std::size_t begin, std::size_t end);
  void join(const cvRidgeWorker& CVRW);
};

#endif
