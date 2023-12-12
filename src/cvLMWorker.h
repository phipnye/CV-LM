#ifndef CV_LM_WORKER_H
#define CV_LM_WORKER_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>

using namespace Rcpp;

// Struct for threaded execution of K-fold CV
struct cvLMWorker : public RcppParallel::Worker {
  const Eigen::VectorXd& y;
  const Eigen::MatrixXd& X;
  const Eigen::VectorXi& s;
  const IntegerVector& ns;
  const int& n;
  const String& pivot;
  const bool& rankCheck;
  double MSE;

  cvLMWorker(const Eigen::VectorXd& y_, const Eigen::MatrixXd& X_, const Eigen::VectorXi& s_, const IntegerVector& ns_, const int& n_, const String& pivot_, const bool& rankCheck_);
  cvLMWorker(const cvLMWorker& CVLMW, RcppParallel::Split);

  void operator()(std::size_t begin, std::size_t end);
  void join(const cvLMWorker& CVLMW);
};

#endif
