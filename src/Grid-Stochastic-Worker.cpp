#include "Grid-Stochastic-Worker.h"

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <algorithm>

#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"
#include "Utils-Decompositions-utils.h"
#include "Utils-Folds-utils.h"

namespace Grid::Stochastic {

// Main ctor
Worker::Worker(const Eigen::Map<Eigen::VectorXd>& y,
               const Eigen::Map<Eigen::MatrixXd>& x,
               const Utils::Folds::FoldInfo& foldInfo,
               const Generator& lambdasGrid, const double threshold)
    : udvT_{foldInfo.maxTrainSize_, x.cols(),
            Eigen::ComputeThinU | Eigen::ComputeThinV},
      cvs_{Eigen::VectorXd::Zero(lambdasGrid.size())},
      uTy_{std::min(foldInfo.maxTrainSize_,
                    x.cols())},  // m = min(n,p) where U'y exists
      beta_{x.cols()},
      resid_{foldInfo.maxTestSize_},
      singularVals_{std::min(foldInfo.maxTrainSize_, x.cols())},
      singularValsSq_{std::min(foldInfo.maxTrainSize_, x.cols())},
      singularShrinkFactors_{std::min(foldInfo.maxTrainSize_, x.cols())},
      trainIdxs_{foldInfo.maxTrainSize_},
      testIdxs_{foldInfo.maxTestSize_},
      foldInfo_{foldInfo},
      y_{y},
      x_{x},
      lambdasGrid_{lambdasGrid},
      nrow_{x.rows()},
      info_{Eigen::Success} {
  // Prescribe threshold to SVD where singular values are considered zero "A
  // singular value will be considered nonzero if its value is strictly greater
  // than |singularvalue|⩽threshold×|maxsingularvalue|."
  udvT_.setThreshold(threshold);
}

// Split ctor
Worker::Worker(const Worker& other, const RcppParallel::Split)
    : udvT_{other.udvT_.rows(), other.udvT_.cols(),
            Eigen::ComputeThinU | Eigen::ComputeThinV},
      cvs_{Eigen::VectorXd::Zero(other.cvs_.size())},
      uTy_{other.uTy_.size()},
      beta_{other.beta_.size()},
      resid_{other.resid_.size()},
      singularVals_{other.singularVals_.size()},
      singularValsSq_{other.singularValsSq_.size()},
      singularShrinkFactors_{other.singularShrinkFactors_.size()},
      trainIdxs_{other.trainIdxs_.size()},
      testIdxs_{other.testIdxs_.size()},
      foldInfo_{other.foldInfo_},
      y_{other.y_},
      x_{other.x_},
      lambdasGrid_{other.lambdasGrid_},
      nrow_{other.nrow_},
      info_{other.info_} {
  // Prescribe threshold to SVD where singular values are
  // considered zero
  udvT_.setThreshold(other.udvT_.threshold());
}

// Work operator
void Worker::operator()(const std::size_t begin, const std::size_t end) {
  // Casting from std::size_t to int is safe here (end is the number of folds
  // which is a 32-bit integer from R)
  for (int testID{static_cast<int>(begin)}, endID{static_cast<int>(end)};
       testID < endID; ++testID) {
    // Retrieve size of test and training sets
    const Eigen::Index testSize{foldInfo_.testFoldSizes_[testID]};
    const Eigen::Index trainSize{nrow_ - testSize};

    // MSE contribution weight
    const double wt{static_cast<double>(testSize) / static_cast<double>(nrow_)};

    // Split the test and training indices
    foldInfo_.testTrainSplit(testID, testIdxs_, trainIdxs_);

    // Perform SVD on training set (uses computation options given in ctor)
    udvT_.compute(x_(trainIdxs_.head(trainSize), Eigen::all));

    // Make sure SVD is successful
    if (const Eigen::ComputationInfo info{udvT_.info()};
        info != Eigen::Success) {
      info_ = info;
      return;
    }

    // Across folds (particularly with wide data), m = min(n,p) may change
    const Eigen::Index singularValsSize{udvT_.singularValues().size()};
    // ReSharper disable once CppDFAUnusedValue
    auto singularVals{singularVals_.head(singularValsSize)};
    // ReSharper disable once CppDFAUnusedValue
    auto singularValsSq{singularValsSq_.head(singularValsSize)};
    auto uTy{uTy_.head(singularValsSize)};
    auto singularShrinkFactors{singularShrinkFactors_.head(singularValsSize)};

    // Extract the projection of y and the singular values
    singularVals = Utils::Decompositions::getSingularVals(udvT_);
    singularValsSq = singularVals.array().square();
    uTy.noalias() =
        udvT_.matrixU().transpose() * y_(trainIdxs_.head(trainSize));

    // Handle unique case of a design matrix that is not of full column rank
    // with OLS - this is one area where cvLM can differ from grid.search by a
    // small amount although both the complete orthoogonal decomposition used
    // for the OLS should match the following svd approach, small numerical
    // imprecisions could lead to different results
    Eigen::Index lambdaIdx{0};

    if (const Eigen::Index rank{udvT_.rank()};
        lambdasGrid_[0] <= 0.0 && rank < x_.cols()) {
      // Compute the unique minimum-norm solution via the MP pseudoinverse (note
      // that this produces different coefficients than R's lm but will only
      // produce different out-of-sample predictions when the system is
      // underdetermined)
      singularShrinkFactors.setZero();

      // Shrinkage_j simplifies to diag(1 / d_j) when lambda == 0.0
      singularShrinkFactors.head(rank) =
          singularVals.head(rank).array().inverse();

      // Evaluate performance on hold-out set (MSE)
      evalTestMSE(uTy, singularShrinkFactors, lambdaIdx++, testSize, wt);
    }

    for (const Eigen::Index nLambda{lambdasGrid_.size()}; lambdaIdx < nLambda;
         ++lambdaIdx) {
      // At this point, we should have lambda > 0 or all singular vals > 0
      // (these shrinkage factors are related to the coordinate shrinkage
      // factors (d_j^2 / (d_j^2 + lambda)) for solving fitted values X * beta
      // but are reduced by d_j since we're solving explicitly for beta)
      singularShrinkFactors = singularVals.array() / (singularValsSq.array() +
                                                      lambdasGrid_[lambdaIdx]);

      // Evaluate performance on hold-out fold (MSE)
      evalTestMSE(uTy, singularShrinkFactors, lambdaIdx, testSize, wt);
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
  Utils::Decompositions::checkSvdInfo(info_);

  // Find the smallest cv result
  Eigen::Index bestIdx;
  const double minCV{cvs_.minCoeff(&bestIdx)};

  // Designated initializers not supported until C++20
  // return LambdaCV{.lambda{lambdasGrid[bestIdx]}, .cv{minMSE}};
  return LambdaCV{lambdasGrid_[bestIdx], minCV};
}

// Evaluate out-of-sample performance
void Worker::evalTestMSE(
    const Eigen::Ref<const Eigen::VectorXd>& uTy,
    const Eigen::Ref<const Eigen::VectorXd>& singularShrinkFactors,
    const Eigen::Index lambdaIdx, const Eigen::Index testSize,
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
  beta_.noalias() =
      udvT_.matrixV() * (singularShrinkFactors.array() * uTy.array()).matrix();

  // Compute residuals
  auto testResid{resid_.head(testSize)};
  testResid.noalias() = y_(testIdxs_.head(testSize)) -
                        (x_(testIdxs_.head(testSize), Eigen::all) * beta_);

  // Accumulate weighted MSE
  const double testMSE{testResid.squaredNorm() / static_cast<double>(testSize)};
  cvs_[lambdaIdx] += (wt * testMSE);
}

}  // namespace Grid::Stochastic
