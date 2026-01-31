#ifndef CV_LM_UTILS_PARALLEL_H
#define CV_LM_UTILS_PARALLEL_H

#include <RcppParallel.h>

#include <cstddef>

namespace Utils::Parallel {

template <typename Worker>
void reduce(Worker& worker, std::size_t end, int nThreads) {
  constexpr std::size_t begin{0};

  if (nThreads > 1) {
    constexpr std::size_t grainSize{1};
    RcppParallel::parallelReduce(begin, end, worker, grainSize, nThreads);
    return;
  }

  // Directly call operator() if not multithreaded
  worker(begin, end);
}

}  // namespace Utils::Parallel

#endif  // CV_LM_UTILS_PARALLEL_H
