#pragma once

namespace Grid {

// Simple container for holding [lambda, CV] pairs for grid searches
struct LambdaCV {
  double lambda;
  double cv;
};

}  // namespace Grid
