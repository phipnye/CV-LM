// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include "cvLMWorker.h"

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>

#include "FUNScvLM.h"

cvLMWorker::cvLMWorker(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const Eigen::VectorXi &s,
                       const Eigen::VectorXd &ns, const int &n)
    : y(y), X(X), s(s), ns(ns), n(n), MSE(0) {}
cvLMWorker::cvLMWorker(const cvLMWorker &CVLMW, RcppParallel::Split)
    : y(CVLMW.y), X(CVLMW.X), s(CVLMW.s), ns(CVLMW.ns), n(CVLMW.n), MSE(0) {}

void cvLMWorker::operator()(std::size_t begin, std::size_t end) {
  for (std::size_t i = begin; i < end; ++i) {
    Eigen::MatrixXd XinS = XinSample(X, s, i);
    Eigen::VectorXd yinS = yinSample(y, s, i);
    Eigen::MatrixXd XoutS = XoutSample(X, s, i);
    Eigen::VectorXd youtS = youtSample(y, s, i);
    Eigen::VectorXd winS = OLScoef(yinS, XinS);
    Eigen::VectorXd yhat = XoutS * winS;
    double costI = cost(youtS, yhat);
    double alphaI = ns[i] / n;
    MSE += (alphaI * costI);
  }
}

void cvLMWorker::join(const cvLMWorker &CVLMW) { MSE += CVLMW.MSE; }
