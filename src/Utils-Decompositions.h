#ifndef CV_LM_UTILS_DECOMPOSITIONS_H
#define CV_LM_UTILS_DECOMPOSITIONS_H

#include "Utils-Data.h"

namespace Utils::Decompositions {

// Helper to set the parameters for a decomposition in order
template <typename Decomp, typename TX, typename TY>
[[nodiscard]] bool setParams(Decomp& decomp, const TX& X, const TY& y,
                             const double lambda = 0.0) {
  Data::assertMat<TX>();
  Data::assertVec<TY>();

  // Set the design matrix
  if (!decomp.setDesign(X)) {
    return false;
  }

  // Set the response vector
  if (!decomp.setResponse(y)) {
    return false;
  }

  // Set lambda (only if the class requires it, e.g., Ridge/SVD)
  if constexpr (Decomp::requiresLambda) {
    decomp.setLambda(lambda);
  }

  return true;
}

}  // namespace Utils::Decompositions

#endif  // CV_LM_UTILS_DECOMPOSITIONS_H
