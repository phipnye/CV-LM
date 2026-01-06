## cvLM.R: Fast cross-validation for linear and ridge regression models using RcppEigen
##
## This file is part of the cvLM package.

.eval_cvLM <- function(
  y,
  X,
  K.vals,
  lambda,
  generalized,
  seed,
  n.threads,
  centered
) {
  cvs <- vapply(
    K.vals,
    function(K) {
      cv.lm.rcpp(
        y,
        X,
        K,
        lambda,
        generalized,
        seed,
        min(K, n.threads),
        centered
      )
    },
    numeric(1L),
    USE.NAMES = FALSE
  )

  data.frame(K = K.vals, CV = cvs, seed = seed)
}

cvLM <- function(object, ...) UseMethod("cvLM")

cvLM.formula <- function(
  object,
  data,
  subset,
  na.action,
  K.vals = 10L,
  lambda = 0,
  generalized = FALSE,
  seed = 1L,
  n.threads = 1L,
  penalize.intercept = FALSE,
  ...
) {
  # Extract data
  dat <- .prepare_lm_data(object, data, subset, na.action)
  y <- dat$y
  X <- dat$X
  mt <- dat$mt

  # --- Confirm validity of arguments

  # Number of folds
  K.vals <- .assert_valid_kvals(K.vals, nrow(X))

  # Shrinkage parameter
  lambda <- .assert_double_scalar(lambda, "lambda", nonneg = TRUE)

  # Generalized boolean
  .assert_logical_scalar(generalized, "generalized")

  # Seed
  seed <- .assert_integer_scalar(seed, "seed", nonneg = TRUE)

  # Number of threads (-1 -> defaultNumThreads)
  n.threads <- .assert_valid_threads(n.threads)

  # Whether to penalize the intercept coefficient in the case of ridge regression
  if (lambda > 0) {
    .assert_logical_scalar(penalize.intercept, "penalize.intercept")
  }

  # We only center if it's a ridge regression model with an intercept
  centered <- !penalize.intercept && lambda > 0 && attr(mt, "intercept") == 1L

  # Center the data and drop the intercept column
  if (centered) {
    tmp <- .center_data(y, X, mt)
    y <- tmp$y
    X <- tmp$X
  }

  # Check for valid regression data before passing to C++
  .assert_valid_data(y, X)
  .eval_cvLM(y, X, K.vals, lambda, generalized, seed, n.threads, centered)
}

cvLM.lm <- function(
  object,
  data,
  K.vals = 10L,
  lambda = 0,
  generalized = FALSE,
  seed = 1L,
  n.threads = 1L,
  penalize.intercept = FALSE,
  ...
) {
  # Raise warning for unsupported lm features (weights and offset)
  if (!is.null(object$weights)) {
    warning(
      "cvLM does not currently support weighted least squares. Weights will be ignored.",
      call. = FALSE
    )
  }

  if (!is.null(object$offset)) {
    warning(
      "cvLM does not currently support offsets. Offset will be ignored.",
      call. = FALSE
    )
  }

  cvLM.formula(
    formula(object),
    data = data,
    K.vals = K.vals,
    lambda = lambda,
    generalized = generalized,
    seed = seed,
    n.threads = n.threads,
    penalize.intercept = penalize.intercept
  )
}

cvLM.glm <- function(
  object,
  data,
  K.vals = 10L,
  lambda = 0,
  generalized = FALSE,
  seed = 1L,
  n.threads = 1L,
  penalize.intercept = FALSE,
  ...
) {
  if (!.is_lm(object)) {
    stop(
      "cvLM only performs cross-validation for linear and ridge regression models.",
      call. = FALSE
    )
  }

  cvLM.lm(
    object,
    data = data,
    K.vals = K.vals,
    lambda = lambda,
    generalized = generalized,
    seed = seed,
    n.threads = n.threads,
    penalize.intercept = penalize.intercept
  )
}
