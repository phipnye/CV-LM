// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
#include "FUNScvLM.h"
#include "cvRidgeWorker.h"

cvRidgeWorker::cvRidgeWorker(const Eigen::VectorXd& y_, const Eigen::MatrixXd& X_, const double& lambda_, const Eigen::VectorXi& s_, const IntegerVector& ns_, const int& n_, const bool& pivot_)
  : y(y_), X(X_), lambda(lambda_), s(s_), ns(ns_), n(n_), pivot(pivot_), MSE(0.0) {}
cvRidgeWorker::cvRidgeWorker(const cvRidgeWorker& CVRW, RcppParallel::Split)
  : y(CVRW.y), X(CVRW.X), lambda(CVRW.lambda), s(CVRW.s), ns(CVRW.ns), n(CVRW.n), pivot(CVRW.pivot), MSE(0.0) {}

void cvRidgeWorker::operator()(std::size_t begin, std::size_t end) {
  for (std::size_t i = begin; i < end; ++i) {
    Eigen::MatrixXd XinS = XinSample(X, s, i);
    Eigen::VectorXd yinS = yinSample(y, s, i);
    Eigen::MatrixXd XoutS = XoutSample(X, s, i);
    Eigen::VectorXd youtS = youtSample(y, s, i);
    Eigen::VectorXd W = Ridgecoef(XinS, yinS, pivot, lambda);
    Eigen::VectorXd yhat = XoutS * W;
    double costI = cost(youtS, yhat);
    double alphaI = static_cast<double>(ns[i]) / n;
    MSE += alphaI * costI;
  }
}

void cvRidgeWorker::join(const cvRidgeWorker& CVRW) {
  MSE += CVRW.MSE;
}
