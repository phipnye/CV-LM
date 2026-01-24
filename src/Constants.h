#ifndef CV_LM_CONSTANTS_H
#define CV_LM_CONSTANTS_H

#include <cstddef>
#include <limits>

namespace Constants {

// First iteration of worker operator()
inline constexpr std::size_t begin{0};

// Grain size for RcppParallel (minimum chunk size for parallelization -
// preventing the creation of threads that process less than the grain size)
inline constexpr std::size_t grainSize{1};

// These static asserts should be safe since IEEE 754 supports NaN and Inf
// "All R platforms are required to work with values conforming to the IEC 60559
// (also known as IEEE 754) standard" -
// https://stat.ethz.ch/R-manual/R-devel/library/base/html/double.html

// Not a number
static_assert(std::numeric_limits<double>::has_quiet_NaN,
              "System does not support NaN values potentially required for "
              "closed-form solutions.");
inline constexpr double NaN{std::numeric_limits<double>::quiet_NaN()};

// Infinity
static_assert(std::numeric_limits<double>::has_infinity,
              "System does not support infinite values required for "
              "cross-validation calculations and grid searching.");
inline constexpr double Inf{std::numeric_limits<double>::infinity()};

}  // namespace Constants

#endif  // CV_LM_CONSTANTS_H
