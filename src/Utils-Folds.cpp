#include <Rcpp.h>
#include <RcppEigen.h>

#include <cfenv>
#include <cmath>
#include <tuple>

#include "Utils-Folds.h"

namespace Utils::Folds {

// RAII guard for the rounding mode
ScopedRoundingMode::ScopedRoundingMode(const int mode)
    : oldMode_{std::fegetround()} {
  std::fesetround(mode);
}

ScopedRoundingMode::~ScopedRoundingMode() { std::fesetround(oldMode_); }

// Confirm valid value for K
// Mimics boot::cv.glm logic to find a K that evenly (or nearly evenly) divides
// n (uses FE_TONEAREST to match R's banker rounding)
int kCheck(const int nrow, const int k0, const bool generalized) {
  // GCV or LOOCV
  if (generalized || nrow == k0) {
    return k0;
  }

  /*
   * We're essentially trying to mimic (from boot::cv.glm):
   *   kvals <- unique(round(n/(1L:floor(n/2))))
   *   temp <- abs(kvals - K)
   *   if (!any(temp == 0))
   *     K <- kvals[temp == min(temp)][1L]
   *   if (K != K.o)
   *     warning(gettextf("'K' has been set to %f", K), domain = NA)
   */

  [[maybe_unused]] const ScopedRoundingMode roundGuard{
      FE_TONEAREST};  // round to nearest, ties to even
  const double nDbl{static_cast<double>(nrow)};
  const int floorNHalf{nrow / 2};

  // Start with den = 1 (we already checked for no difference at beginning of
  // function)
  int closestK{nrow};
  int minDiff{closestK - k0};

  // Consider k values between n and 2 (iterates through possible denominators
  // to find a K value that fits n
  for (int den{2}; den <= floorNHalf; ++den) {
    const int kVal{
        static_cast<int>(std::nearbyint(nDbl / den))};  // use banker's rounding
    const int absDiff{std::abs(k0 - kVal)};

    if (absDiff == 0) {
      return k0;
    }

    if (absDiff < minDiff) {
      minDiff = absDiff;
      closestK = kVal;
    }
  }

  Rcpp::warning("K has been changed from %d to %d.", k0, closestK);
  return closestK;
}

FoldInfo::FoldInfo(const int seed, const int nrow, const int k)
    : testFoldIDs_{[=]() -> Eigen::VectorXi {
        // Call back into R for sample and set.seed to guarantee the exact same
        // random partitions as boot::cv.glm (using C++ RNG would break
        // reproducibility)
        const Rcpp::Function setSeed{"set.seed"};
        const Rcpp::Function sampleR{"sample"};
        std::ignore = setSeed(seed);  // returns NULL that we can ignore

        /*
         * Replicate fold assignment from boot::cv.glm:
         *   f <- ceiling(n/K)
         *   s <- sample0(rep(1L:K, f), n)
         */
        const int repeats{
            static_cast<int>(std::ceil(static_cast<double>(nrow) / k))};
        const Rcpp::IntegerVector seqVec{Rcpp::rep(Rcpp::seq(1, k), repeats)};
        const Rcpp::IntegerVector sampled{sampleR(seqVec, nrow)};
        Eigen::VectorXi testFoldIDs{Rcpp::as<Eigen::VectorXi>(sampled)};

        // Convert to zero indexing
        testFoldIDs.array() -= 1;
        return testFoldIDs;
      }()},

      testFoldSizes_{[&]() -> Eigen::VectorXi {
        // R's internal documentation states the number of rows for a matrix are
        // limited to 32-bit values so VectorXi is safe here
        Eigen::VectorXi testFoldSizes{Eigen::VectorXi::Zero(k)};

        // Store test fold sizes to calculate the weighted CV estimate:
        // sum((n_i / n) * cost_i)
        const Eigen::Index size{testFoldIDs_.size()};

        for (Eigen::Index idx{0}; idx < size; ++idx) {
          ++testFoldSizes[testFoldIDs_[idx]];
        }

        return testFoldSizes;
      }()},

      // Pre-compute the max test and train sizes so we can allocate appropriate
      // buffer sizes in worker instances
      maxTestSize_{static_cast<Eigen::Index>(testFoldSizes_.maxCoeff())},
      maxTrainSize_{
          static_cast<Eigen::Index>(nrow - testFoldSizes_.minCoeff())},
      nrow_{nrow} {}

// Split the test and training indices
void FoldInfo::testTrainSplit(const int testID, Eigen::VectorXi& testIdxs,
                              Eigen::VectorXi& trainIdxs) const {
  Eigen::Index trIdx{0};
  Eigen::Index tsIdx{0};

  // Fill the Idxs buffers with the corresponding test/train row indices
  for (int rowIdx{0}; rowIdx < nrow_; ++rowIdx) {
    if (testFoldIDs_[rowIdx] == testID) {
      testIdxs[tsIdx++] = rowIdx;
    } else {
      trainIdxs[trIdx++] = rowIdx;
    }
  }
}

}  // namespace Utils::Folds
