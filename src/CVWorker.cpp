// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include "include/CVWorker.h"

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>

namespace CV {

// --- Base implementation

BaseCVWorker::BaseCVWorker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                           const Eigen::VectorXi& foldIDs,
                           const Eigen::VectorXi& foldSizes)
    : y_{y},
      x_{x},
      foldIDs_{foldIDs},
      foldSizes_{foldSizes},
      nrow_{x.rows()},
      ncol_{x.cols()},
      mse_{0.0},
      xTrain_(nrow_, ncol_),
      yTrain_(nrow_),
      trainIdxs_(nrow_),
      testIdxs_(nrow_),
      beta_(ncol_),
      resid_(nrow_) {}

void BaseCVWorker::operator()(const std::size_t begin, const std::size_t end) {
  // Casting from std::size_t to int is safe here (end is the number of folds
  // which is a 32-bit integer from R)
  for (int foldID{static_cast<int>(begin)}, endID{static_cast<int>(end)};
       foldID < endID; ++foldID) {
    const Eigen::Index testSize{foldSizes_[foldID]};
    const Eigen::Index trainSize{nrow_ - testSize};

    // Prepare training and testing containers
    Eigen::Index tr{0};
    Eigen::Index ts{0};

    for (int row{0}; row < nrow_; ++row) {
      if (foldIDs_[row] == foldID) {
        testIdxs_[ts++] = row;
      } else {
        trainIdxs_[tr++] = row;
      }
    }

    // Copy training data using pre-allocated buffers
    xTrain_.topRows(trainSize) = x_(trainIdxs_.head(trainSize), Eigen::all);
    yTrain_.head(trainSize) = y_(trainIdxs_.head(trainSize));

    // Fit the model
    computeBeta(xTrain_.topRows(trainSize), yTrain_.topRows(trainSize));

    // Evaluate performance on hold-out fold (MSE)
    resid_.head(testSize) = y_(testIdxs_.head(testSize));
    resid_.head(testSize).noalias() -=
        (x_(testIdxs_.head(testSize), Eigen::all) * beta_);
    const double foldMSE{resid_.head(testSize).squaredNorm() / testSize};

    // Weighted MSE contribution
    const double alpha{static_cast<double>(testSize) / nrow_};
    mse_ += (alpha * foldMSE);
  }
}

void BaseCVWorker::join(const BaseCVWorker& other) { mse_ += other.mse_; }

// --- OLS Implementation

OLS::CVWorker::CVWorker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                        const Eigen::VectorXi& foldIDs,
                        const Eigen::VectorXi& foldSizes)
    : BaseCVWorker{y, x, foldIDs, foldSizes} {}

OLS::CVWorker::CVWorker(const OLS::CVWorker& other, const RcppParallel::Split)
    : BaseCVWorker{other.y_, other.x_, other.foldIDs_, other.foldSizes_} {}

void OLS::CVWorker::computeBeta(
    const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
    const Eigen::Ref<const Eigen::VectorXd>& yTrain) {
  const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr{xTrain};

  // if (qr.info() != Eigen::Success) {
  //   Not necessary, Eigen documents this always returns success
  // }

  const Eigen::Index rank{qr.rank()};

  // No rank deficiency
  if (rank == ncol_) {
    beta_ = qr.solve(yTrain);
    return;
  }

  // Mimic R's behavior of zeroing out coefficients on redundant predictors
  Eigen::VectorXd qty{
      yTrain};  // explicitly generate new data in this instance because
                // ColPivHouseholderQR::solve does not support in-place solving
                // like LDLT and this is a special case of rank-deficiency
                // (hopefully rare)
  qty.applyOnTheLeft(qr.householderQ().transpose());
  beta_.setZero();

  // Solve Rz = Q'y for the first 'rank' elements using the top-left rank x rank
  // part of Matrix R
  beta_.head(rank) = qr.matrixR()
                         .topLeftCorner(rank, rank)
                         .triangularView<Eigen::Upper>()
                         .solve(qty.head(rank));

  // Permute back to original column order
  beta_.applyOnTheLeft(qr.colsPermutation());
}

// --- Ridge Implementation

Ridge::CVWorker::CVWorker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                          double lambda, const Eigen::VectorXi& foldIDs,
                          const Eigen::VectorXi& foldSizes)
    : BaseCVWorker{y, x, foldIDs, foldSizes},
      lambda_{lambda},
      xtxLambda_(BaseCVWorker::ncol_, BaseCVWorker::ncol_) {}

Ridge::CVWorker::CVWorker(const Ridge::CVWorker& other,
                          RcppParallel::Split split)
    : BaseCVWorker(other.y_, other.x_, other.foldIDs_, other.foldSizes_),
      lambda_{other.lambda_},
      xtxLambda_(other.ncol_, other.ncol_) {}

void Ridge::CVWorker::computeBeta(
    const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
    const Eigen::Ref<const Eigen::VectorXd>& yTrain) {
  // Generate cross-products (re-use pre-allocated buffers)
  xtxLambda_.setZero();
  xtxLambda_.diagonal().fill(lambda_);
  const auto xT{xTrain.transpose()};
  xtxLambda_.selfadjointView<Eigen::Lower>().rankUpdate(xT);

  // beta_ is a misnomer at this point LDLT supports in-place solving so we fill
  // beta_ with X'y (the RHS of solve)
  beta_.noalias() = xT * yTrain;

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
  ldlt.solveInPlace(beta_);  // just returns true (no need to check)
}

}  // namespace CV
