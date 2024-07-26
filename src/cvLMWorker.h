#ifndef CV_LM_WORKER_H
#define CV_LM_WORKER_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>

using namespace Rcpp;

// Struct for threaded execution of K-fold CV
struct cvLMWorker : public RcppParallel::Worker {
  const Eigen::VectorXd &y;
  const Eigen::MatrixXd &X;
  const Eigen::VectorXi &s;
  const Eigen::VectorXd &ns;
  const int &n;
  double MSE;

  cvLMWorker(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const Eigen::VectorXi &s, const Eigen::VectorXd &ns,
             const int &n);
  cvLMWorker(const cvLMWorker &CVLMW, RcppParallel::Split);
  ~cvLMWorker(){};

  void operator()(std::size_t begin, std::size_t end);
  void join(const cvLMWorker &CVLMW);
};

#endif
