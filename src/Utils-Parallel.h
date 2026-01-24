#ifndef CV_LM_UTILS_PARALLEL_H
#define CV_LM_UTILS_PARALLEL_H

#include <RcppParallel.h>

#include <cstddef>

#include "Constants.h"

namespace Utils::Parallel {

template <typename Worker>
void reduce(Worker& worker, std::size_t end, int nThreads) {
  RcppParallel::parallelReduce(Constants::begin, end, worker,
                               Constants::grainSize, nThreads);
}

}  // namespace Utils::Parallel

#endif  // CV_LM_UTILS_PARALLEL_H
