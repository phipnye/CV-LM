// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include "cvRidgeWorker.h"

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>

#include "FUNScvLM.h"

cvRidgeWorker::cvRidgeWorker(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const double &lambda,
                             const Eigen::VectorXi &s, const Eigen::VectorXd &ns, const int &n)
    : y(y), X(X), lambda(lambda), s(s), ns(ns), n(n), MSE(0) {}
cvRidgeWorker::cvRidgeWorker(const cvRidgeWorker &CVRW, RcppParallel::Split)
    : y(CVRW.y), X(CVRW.X), lambda(CVRW.lambda), s(CVRW.s), ns(CVRW.ns), n(CVRW.n), MSE(0) {}

void cvRidgeWorker::operator()(std::size_t begin, std::size_t end) {
  for (std::size_t i = begin; i < end; ++i) {
    Eigen::MatrixXd XinS = XinSample(X, s, i);
    Eigen::VectorXd yinS = yinSample(y, s, i);
    Eigen::MatrixXd XoutS = XoutSample(X, s, i);
    Eigen::VectorXd youtS = youtSample(y, s, i);
    Eigen::VectorXd winS = Ridgecoef(yinS, XinS, lambda);
    Eigen::VectorXd yhat = XoutS * winS;
    double costI = cost(youtS, yhat);
    double alphaI = ns[i] / n;
    MSE += (alphaI * costI);
  }
}

void cvRidgeWorker::join(const cvRidgeWorker &CVRW) { MSE += CVRW.MSE; }
