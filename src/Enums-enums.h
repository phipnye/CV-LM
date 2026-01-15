#pragma once

// ReSharper disable once CppUnusedIncludeDirective
#include <cstdint>

namespace Enums {

enum class FitMethod : std::int8_t { OLS, Ridge };
enum class AnalyticMethod : std::int8_t { GCV, LOOCV };

// consteval not supported until c++20

// Make sure the FitMethod is what we expect (complile-time check)
template <FitMethod Actual, FitMethod Expected>
constexpr void assertExpected() {
  static_assert(Actual == Expected, "Unexpected fit method");
}

// Make sure the AnalyticMethod is what we expect (complile-time check)
template <AnalyticMethod Actual, AnalyticMethod Expected>
constexpr void assertExpected() {
  static_assert(Actual == Expected, "Unexpected deterministic cv method");
}

}  // namespace Enums
