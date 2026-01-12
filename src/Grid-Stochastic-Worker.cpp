#include "Grid-Stochastic-Worker.h"

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <algorithm>

#include "CV-Utils-utils.h"
#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"
#include "Grid-Utils-utils.h"

namespace Grid::Stochastic {

// Main ctor
Worker::Worker(const Eigen::Map<Eigen::VectorXd>& y,
               const Eigen::Map<Eigen::MatrixXd>& x,
               const Eigen::VectorXi& testFoldIDs,
               const Eigen::VectorXi& testFoldSizes,
               const Generator& lambdasGrid, const Eigen::Index maxTrainSize,
               const Eigen::Index maxTestSize, const double threshold)
    : svd_(maxTrainSize, x.cols(), Eigen::ComputeThinU | Eigen::ComputeThinV),
      cvs_{Eigen::VectorXd::Zero(lambdasGrid.size())},
      uty_(std::min(x.rows(), x.cols())),
      beta_(x.cols()),
      resid_(maxTestSize),
      eigenVals_(std::min(x.rows(), x.cols())),
      eigenValsSq_(std::min(x.rows(), x.cols())),
      diagW_(std::min(x.rows(), x.cols())),
      trainIdxs_(maxTrainSize),
      testIdxs_(maxTestSize),
      testFoldIDs_{testFoldIDs},
      testFoldSizes_{testFoldSizes},
      y_{y},
      x_{x},
      lambdasGrid_{lambdasGrid},
      nrow_{x.rows()},
      info_{Eigen::Success} {
  // Prescribe threshold to SVD decomposition where singular values are
  // considered zero "A singular value will be considered nonzero if its value
  // is strictly greater than |singularvalue|⩽threshold×|maxsingularvalue|."
  svd_.setThreshold(threshold);
}

// Split ctor
Worker::Worker(const Worker& other, const RcppParallel::Split)
    : svd_(other.svd_.rows(), other.svd_.cols(),
           Eigen::ComputeThinU | Eigen::ComputeThinV),
      cvs_{Eigen::VectorXd::Zero(other.cvs_.size())},
      uty_(other.uty_.size()),
      beta_(other.beta_.size()),
      resid_(other.resid_.size()),
      eigenVals_(other.eigenVals_.size()),
      eigenValsSq_(other.eigenValsSq_.size()),
      diagW_(other.diagW_.size()),
      trainIdxs_(other.trainIdxs_.size()),
      testIdxs_(other.testIdxs_.size()),
      testFoldIDs_{other.testFoldIDs_},
      testFoldSizes_{other.testFoldSizes_},
      y_{other.y_},
      x_{other.x_},
      lambdasGrid_{other.lambdasGrid_},
      nrow_{other.nrow_},
      info_{other.info_} {
  // Prescribe threshold to SVD decomposition where singular values are
  // considered zero
  svd_.setThreshold(other.svd_.threshold());
}

// Work operator
void Worker::operator()(const std::size_t begin, const std::size_t end) {
  // Casting from std::size_t to int is safe here (end is the number of folds
  // which is a 32-bit integer from R)
  for (int testID{static_cast<int>(begin)}, endID{static_cast<int>(end)};
       testID < endID; ++testID) {
    // Retrieve size of test and training sets
    const Eigen::Index testSize{testFoldSizes_[testID]};
    const Eigen::Index trainSize{nrow_ - testSize};

    // MSE contribution weight
    const double wt{static_cast<double>(testSize) / static_cast<double>(nrow_)};

    // Split the test and training indices
    CV::Utils::testTrainSplit(testID, testFoldIDs_, testIdxs_, trainIdxs_);

    // Perform SVD on training set (uses computation options given in ctor)
    svd_.compute(x_(trainIdxs_.head(trainSize), Eigen::all));

    // Make sure SVD is successful
    if (const Eigen::ComputationInfo info{svd_.info()};
        info != Eigen::Success) {
      info_ = info;
      return;
    }

    // Extract relevant SVD computations
    const Eigen::MatrixXd& v{svd_.matrixV()};
    uty_.noalias() =
        svd_.matrixU().transpose() * y_(trainIdxs_.head(trainSize));
    eigenVals_ = svd_.singularValues();
    eigenValsSq_ = eigenVals_.square();

    // Handle unique case of rank-deficient design matrix with OLS (<= threshold
    // should always be true since grid start at zero)
    Eigen::Index lambdaIdx{0};

    if (const Eigen::Index rank{svd_.rank()};
        lambdasGrid_[0] <= svd_.threshold() && rank < x_.cols()) {
      // Manually compute Moore-penrose (minimum-norm solution - note that this
      // strays from R's lm behavior)
      diagW_.setZero();

      // diag(W)_i simplifies to diag(1 / eigenVal_i) when lambda == 0.0
      diagW_.head(rank) = eigenVals_.head(rank).array().inverse();

      // Evaluate performance on hold-out set (MSE)
      evalTestMSE(lambdaIdx++, testSize, v, wt);
    }

    for (const Eigen::Index nLambda{lambdasGrid_.size()}; lambdaIdx < nLambda;
         ++lambdaIdx) {
      // diag(W)_i = diag(eigenVal_i / eigenVal_i^2 + lambda)
      diagW_ = eigenVals_ / (eigenValsSq_ + lambdasGrid_[lambdaIdx]);

      // Evaluate performance on hold-out fold (MSE)
      evalTestMSE(lambdaIdx, testSize, v, wt);
    }
  }
}

// Reduce results
void Worker::join(const Worker& other) {
  // Record unsuccessful decompositions
  if (info_ != Eigen::Success) {
    return;
  }

  if (other.info_ != Eigen::Success) {
    info_ = other.info_;
    return;
  }

  // Add up cross-validation results across folds of the data
  cvs_ += other.cvs_;
}

// Retrive optimal CV-lambda pairing
LambdaCV Worker::getOptimalPair() const {
  // Make sure SVD worked consistently before returning results
  // NOTE: Imporant we don't call this from a multithreaded context since this
  // can call Rcpp::stop
  Utils::checkSvdStatus(info_);

  // Find the smallest cv result
  Eigen::Index bestIdx;
  const double minCV{cvs_.minCoeff(&bestIdx)};

  // Designated initializers not supported until C++20
  // return LambdaCV{.lambda{lambdasGrid[bestIdx]}, .cv{minMSE}};
  return LambdaCV{lambdasGrid_[bestIdx], minCV};
}

// Evaluate out-of-sample performance
void Worker::evalTestMSE(const Eigen::Index lambdaIdx,
                         const Eigen::Index testSize, const Eigen::MatrixXd& v,
                         const double wt) {
  /*
   * beta = (X'X + lambda * I)^-1 X'y
   * (V D^2 V' + lambda * I) * beta = VDU'y
   * beta = V * alpha
   * (V D^2 V' + lambda * I) V * alpha = VDU'y
   * V D^2 V'V * alpha + lambda * V * alpha = VDU'y
   * V (D^2 + lambda * I) alpha = VDU'y
   * (D^2 + lambda * I) alpha = DU'y
   * alpha = (D^2 + lambda * I)^-1 DU'y
   * beta = V * alpha = V * [(D^2 + lambda * I)^-1 D] * U'y
   */
  // Note that alpha_i = W_ii * (U'y)_i
  beta_.noalias() = v * (diagW_ * uty_.array()).matrix();

  // Compute residuals
  auto testResid{resid_.head(testSize)};
  testResid.noalias() = y_(testIdxs_.head(testSize)) -
                        (x_(testIdxs_.head(testSize), Eigen::all) * beta_);

  // Accumulate weighted MSE
  const double testMSE{testResid.squaredNorm() / static_cast<double>(testSize)};
  cvs_[lambdaIdx] += (wt * testMSE);
}

}  // namespace Grid::Stochastic
