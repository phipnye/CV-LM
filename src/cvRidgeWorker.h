#ifndef CV_RIDGE_WORKER_H
#define CV_RIDGE_WORKER_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>

using namespace Rcpp;

// Struct for threaded execution of CV
struct cvRidgeWorker : public RcppParallel::Worker {
  const Eigen::VectorXd &y;
  const Eigen::MatrixXd &X;
  const double &lambda;
  const Eigen::VectorXi &s;
  const Eigen::VectorXd &ns;
  const int &n;
  double MSE;

  cvRidgeWorker(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const double &lamda, const Eigen::VectorXi &s,
                const Eigen::VectorXd &ns, const int &n);
  cvRidgeWorker(const cvRidgeWorker &CVRW, RcppParallel::Split);
  ~cvRidgeWorker(){};

  void operator()(std::size_t begin, std::size_t end);
  void join(const cvRidgeWorker &CVRW);
};

#endif
