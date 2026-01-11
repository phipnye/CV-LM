#pragma once

#include <cstddef>

namespace Constants {

// First iterator of worker operator()
inline constexpr std::size_t begin{0};

// Grain size for RcppParallel splitting
inline constexpr std::size_t grainSize{1};

}  // namespace Constants
