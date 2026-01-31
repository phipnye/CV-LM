#ifndef CV_LM_ENUMS_H
#define CV_LM_ENUMS_H

#include <cstdint>

namespace Enums {

enum class CrossValidationMethod : std::int8_t { KCV, GCV, LOOCV };
enum class CenteringMethod : std::int8_t { None, Mean };

// Make sure the AnalyticMethod is what we expect (compile-time check)
template <CrossValidationMethod Actual, CrossValidationMethod Expected>
constexpr void assertExpected() {
  static_assert(Actual == Expected, "Unexpected deterministic cv method");
}

// Make sure the CenteringMode is what we expect (compile-time check)
template <CenteringMethod Actual, CenteringMethod Expected>
constexpr void assertExpected() {
  static_assert(Actual == Expected, "Unexpected centering method");
}

}  // namespace Enums

#endif  // CV_LM_ENUMS_H
