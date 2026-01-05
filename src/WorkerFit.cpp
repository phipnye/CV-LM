#include "include/WorkerFit.h"

#include <RcppEigen.h>

namespace CV {

namespace OLS {

void WorkerFit::computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                            const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                            Eigen::VectorXd& beta) const {
  // Decompose training set into the for XP = QR
  const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr{xTrain};

  // if (qr.info() != Eigen::Success) {
  //   Not necessary, Eigen documents this always returns success
  // }

  const Eigen::Index rank{qr.rank()};

  // No rank deficiency
  if (rank == xTrain.cols()) {
    beta = qr.solve(yTrain);
    return;
  }

  // Mimic R's behavior of zeroing out coefficients on redundant predictors
  Eigen::VectorXd qty{
      yTrain};  // explicitly generate new data in this instance because
                // ColPivHouseholderQR::solve does not support in-place solving
                // like LDLT and this is a special case of rank-deficiency
                // (hopefully rare)
  qty.applyOnTheLeft(qr.householderQ().transpose());
  beta.setZero();

  // Solve Rz = Q'y for the first 'rank' elements using the top-left rank x rank
  // part of Matrix R
  beta.head(rank) = qr.matrixR()
                        .topLeftCorner(rank, rank)
                        .triangularView<Eigen::Upper>()
                        .solve(qty.head(rank));

  // Permute back to original column order
  beta.applyOnTheLeft(qr.colsPermutation());
}

}  // namespace OLS

namespace Ridge {

WorkerFit::WorkerFit(const double lambda, const Eigen::Index ncol)
    : lambda_{lambda}, xtxLambda_(ncol, ncol) {}

// Explicitly handle copying/splitting to ensure buffer allocation
WorkerFit::WorkerFit(const WorkerFit& other)
    : lambda_{other.lambda_},
      xtxLambda_(other.xtxLambda_.rows(), other.xtxLambda_.cols()) {}

void WorkerFit::computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                            const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                            Eigen::VectorXd& beta) const {
  // Generate cross-products (re-use pre-allocated buffers)
  xtxLambda_.setZero();
  xtxLambda_.diagonal().fill(lambda_);
  const auto xT{xTrain.transpose()};
  xtxLambda_.selfadjointView<Eigen::Lower>().rankUpdate(xT);

  // beta_ is a misnomer at this point LDLT supports in-place solving so we fill
  // beta_ with X'y (the RHS of solve)
  beta.noalias() = xT * yTrain;

  // Despite positive definiteness, Eigen's documentation states "While the
  // Cholesky decomposition is particularly useful to solve selfadjoint problems
  // like D^*D x = b, for that purpose, we recommend the Cholesky decomposition
  // without square root which is more stable and even faster." We can also
  // perform the decomposition in place here
  const Eigen::LDLT<Eigen::Ref<Eigen::MatrixXd>> ldlt{xtxLambda_};

  // if (ldlt.info() != Eigen::Success) {
  // TO DO
  // }

  // LDLT::solve supports in-place solves which we use here for efficiency
  ldlt.solveInPlace(beta);  // just returns true (no need to check)
}

}  // namespace Ridge

}  // namespace CV
