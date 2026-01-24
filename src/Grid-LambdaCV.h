#ifndef CV_LM_GRID_LAMBDACV_H
#define CV_LM_GRID_LAMBDACV_H

namespace Grid {

// Simple container for holding [lambda, CV] pairs for grid searches
struct LambdaCV {
  double lambda_;
  double cv_;
};

}  // namespace Grid

#endif  // CV_LM_GRID_LAMBDACV_H
