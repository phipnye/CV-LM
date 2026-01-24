#ifndef CV_LM_ENUMS_H
#define CV_LM_ENUMS_H

#include <cstdint>

namespace Enums {

enum class FitMethod : std::int8_t { OLS, Ridge };
enum class AnalyticMethod : std::int8_t { GCV, LOOCV };
enum class CenteringMethod : std::int8_t { None, Mean };

// Make sure the FitMethod is what we expect (compile-time check)
// consteval not supported until c++20
template <FitMethod Actual, FitMethod Expected>
constexpr void assertExpected() {
  static_assert(Actual == Expected, "Unexpected fit method");
}

// Make sure the AnalyticMethod is what we expect (compile-time check)
template <AnalyticMethod Actual, AnalyticMethod Expected>
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
