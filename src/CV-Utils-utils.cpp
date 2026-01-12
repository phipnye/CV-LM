#include "CV-Utils-utils.h"

#include <Rcpp.h>
#include <RcppEigen.h>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <utility>

namespace CV::Utils {

// RAII guard for the rounding mode
ScopedRoundingMode::ScopedRoundingMode(const int mode)
    : oldMode_{std::fegetround()} {
  std::fesetround(mode);
}

ScopedRoundingMode::~ScopedRoundingMode() { std::fesetround(oldMode_); }

// Confirm valid value for K
// Mimics boot::cv.glm logic to find a K that evenly (or nearly evenly) divides
// n (uses FE_TONEAREST to match R's banker rounding)
int kCheck(const int nrow, const int k0) {
  // LOOCV
  if (nrow == k0) {
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

// Generates test fold assignments
std::pair<Eigen::VectorXi, Eigen::VectorXi> setupFolds(const int seed,
                                                       const int nrow,
                                                       const int k) {
  // Call back into R for sample and set.seed to guarantee the exact same
  // random partitions as boot::cv.glm (using C++ RNG would break
  // reproducibility)
  const Rcpp::Function setSeed{"set.seed"};
  const Rcpp::Function sampleR{"sample"};
  // ReSharper disable once CppExpressionWithoutSideEffects
  setSeed(seed);

  /*
   * Replicate fold assignment from boot::cv.glm:
   *   f <- ceiling(n/K)
   *   s <- sample0(rep(1L:K, f), n)
   */
  const int repeats{static_cast<int>(std::ceil(static_cast<double>(nrow) / k))};
  const Rcpp::IntegerVector seqVec{Rcpp::rep(Rcpp::seq(1, k), repeats)};
  const Rcpp::IntegerVector sampled{sampleR(seqVec, nrow)};

  // R's internal documentation states the number of rows for a matrix are
  // limited to 32-bit values so VectorXi is safe here
  Eigen::VectorXi testFoldIDs{Rcpp::as<Eigen::VectorXi>(sampled)};
  Eigen::VectorXi testFoldSizes{Eigen::VectorXi::Zero(k)};

  // Convert testFoldIDs to be zero-indexed
  testFoldIDs.array() -= 1;

  // Store test fold sizes to calculate the weighted CV estimate: sum((n_i / n)
  // * cost_i)
  for (Eigen::Index idx{0}, size{testFoldIDs.size()}; idx < size; ++idx) {
    ++testFoldSizes[testFoldIDs[idx]];
  }

  return {testFoldIDs, testFoldSizes};
}

// Determine the min and max test fold sizes
std::pair<Eigen::Index, Eigen::Index> testSizeExtrema(
    const Eigen::VectorXi& testFoldSizes) {
  const auto [minIter, maxIter]{std::minmax_element(
      testFoldSizes.data(), testFoldSizes.data() + testFoldSizes.size())};
  return {static_cast<Eigen::Index>(*minIter),
          static_cast<Eigen::Index>(*maxIter)};
}

// Split the test and training indices
void testTrainSplit(const int testID, const Eigen::VectorXi& testFoldIDs,
                    Eigen::VectorXi& testIdxs, Eigen::VectorXi& trainIdxs) {
  const int nrow{static_cast<int>(testFoldIDs.rows())};
  Eigen::Index trIdx{0};
  Eigen::Index tsIdx{0};

  // Fill the Idxs buffers with the corresponding test/train row indices
  for (int rowIdx{0}; rowIdx < nrow; ++rowIdx) {
    if (testFoldIDs[rowIdx] == testID) {
      testIdxs[tsIdx++] = rowIdx;
    } else {
      trainIdxs[trIdx++] = rowIdx;
    }
  }
}

// Check for success of LDLT decomposition
void checkLdltStatus(const Eigen::ComputationInfo info) {
  if (info == Eigen::Success) {
    return;
  }

  // Documentation states NumericalIssue is the only non-success message for
  // LDLT "Returns: Success if computation was successful, NumericalIssue if the
  // factorization failed because of a zero pivot."
  switch (info) {
    case Eigen::NumericalIssue:
      Rcpp::stop(
          "LDLT failed: Numerical issue (zero pivot). This suggests the matrix "
          "X'X + lambda * I is nearly singular. Try increasing the "
          "regularization parameter `lambda` or checking for columns with "
          "constant values/low variance.");
      // break; - not reachable

    default:
      Rcpp::stop("LDLT failed: An unknown error occurred.");
      // break; - not reachable
  }
}

}  // namespace CV::Utils
