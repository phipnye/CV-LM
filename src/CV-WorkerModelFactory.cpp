#include "CV-WorkerModelFactory.h"

#include "CV-WorkerModel.h"

namespace CV {

namespace OLS {

WorkerModel WorkerModelFactory::operator()() const {
  return WorkerModel{ncol_, maxTrainSize_, threshold_};
}

}  // namespace OLS

namespace Ridge {

namespace Narrow {

WorkerModel WorkerModelFactory::operator()() const {
  return WorkerModel{ncol_, lambda_};
}

}

namespace Wide {

WorkerModel WorkerModelFactory::operator()() const {
  return WorkerModel{maxTrainSize_, lambda_};
}

}

}  // namespace Ridge

}  // namespace CV
