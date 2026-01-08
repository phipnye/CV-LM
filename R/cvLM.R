## cvLM.R: Fast cross-validation for linear and ridge regression models using RcppEigen
##
## This file is part of the cvLM package.

# Internal function that accepts prepared data and paramters
.cvLM_fit <- function(
  y,
  X,
  K.vals,
  lambda,
  generalized,
  seed,
  n.threads,
  tol,
  penalize.intercept,
  has.intercept,
  mt
) {
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
  
  # Threshold for QR decomposition (and at which to consider lambda 0 for OLS)
  tol <- .assert_double_scalar(tol, "tol", nonneg = TRUE)

  # Whether to penalize the intercept coefficient in the case of ridge regression
  centered <- FALSE
  
  if (lambda > tol) {
    .assert_logical_scalar(penalize.intercept, "penalize.intercept")
  
    # We only center if it's a ridge regression model with an intercept
    centered <- !penalize.intercept && has.intercept
  
    # Center the data and drop the intercept column
    if (centered) {
      tmp <- .center_data(y, X, mt)
      y <- tmp$y
      X <- tmp$X
    }
  }

  # Check for valid regression data before passing to C++
  .assert_valid_data(y, X)

  # Pass off to C++
  cvs <- vapply(
    K.vals,
    function(K) {
      cv.lm.rcpp(
        y = y,
        x = X,
        k0 = K,
        lambda = lambda,
        generalized = generalized,
        seed = seed,
        nThreads = min(K, n.threads),
        threshold = tol,
        centered = centered
      )
    },
    numeric(1L),
    USE.NAMES = FALSE
  )
  
  data.frame(K = K.vals, CV = cvs, seed = seed)
}

cvLM <- function(object, ...) UseMethod("cvLM")

# Formula method
cvLM.formula <- function(
  object,
  data,
  subset = NULL,
  na.action = NULL,
  K.vals = 10L,
  lambda = 0,
  generalized = FALSE,
  seed = 1L,
  n.threads = 1L,
  tol = 1e-7,
  penalize.intercept = FALSE,
  ...
) {
  # --- Extract data (mimic lm behavior)

  mf <- match.call(expand.dots = FALSE)
  m <- match(c("object", "data", "subset", "na.action"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  names(mf)[names(mf) == "object"] <- "formula"
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")

  if (is.empty.model(mt)) {
    stop("Empty model specified.", call. = FALSE)
  }

  X <- model.matrix(mt, mf)
  y <- model.response(mf, "double")

  .cvLM_fit(
    y = y,
    X = X,
    K.vals = K.vals,
    lambda = lambda,
    generalized = generalized,
    seed = seed,
    n.threads = n.threads,
    tol = tol,
    penalize.intercept = penalize.intercept,
    has.intercept = (attr(mt, "intercept") == 1L),
    mt = mt
  )
}

# lm method
cvLM.lm <- function(
  object,
  data,
  K.vals = 10L,
  lambda = 0,
  generalized = FALSE,
  seed = 1L,
  n.threads = 1L,
  tol = 1e-7,
  penalize.intercept = FALSE,
  ...
) {
  # Raise warning for unsupported lm features (weights and offset)
  if (!is.null(object$weights) && length(unique(object$weights)) > 1L) {
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

  # --- Reconstruct model frame

  # We cannot simply use object$model because the user may have supplied a new 'subset' or 'na.action' in
  # this call (we must merge the original call with the current arguments)

  # Start with a generic model.frame call
  mf.call <- quote(stats::model.frame())

  # Use the formula/terms from the object (preserves environment)
  mf.call$formula <- terms(object)

  # If data provided in current call, use it (otherwise, fall back to original)
  if (!missing(data)) {
    mf.call$data <- data 
  } else {
    mf.call$data <- object$call$data
  }
  
  cl.curr <- match.call(expand.dots = FALSE)

  # Handle subset
  if ("subset" %in% names(cl.curr)) {
    mf.call$subset <- cl.curr$subset
  } else {
    mf.call$subset <- object$call$subset
  }

  # Handle na.action
  if ("na.action" %in% names(cl.curr)) {
    mf.call$na.action <- cl.curr$na.action
  } else {
    mf.call$na.action <- object$call$na.action
  }

  mf.call$drop.unused.levels <- TRUE

  # Evaluate in parent.frame() to ensure any 'subset' symbols (like a vector of indices) defined in current
  # scope are found
  mf <- eval(mf.call, parent.frame())

  # --- Extract data

  mt <- attr(mf, "terms")
  X <- model.matrix(mt, mf, contrasts.arg = object$contrasts)
  y <- model.response(mf, "double")

  .cvLM_fit(
    y = y,
    X = X,
    K.vals = K.vals,
    lambda = lambda,
    generalized = generalized,
    seed = seed,
    n.threads = n.threads,
    tol = tol,
    penalize.intercept = penalize.intercept,
    has.intercept = (attr(mt, "intercept") == 1L),
    mt = mt
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
  tol = 1e-7,
  penalize.intercept = FALSE,
  ...
) {
  if (!.is_lm(object)) {
    stop(
      "cvLM only performs cross-validation for linear and ridge regression models.",
      call. = FALSE
    )
  }

  # Use NextMethod to dispatch to cvLM.lm
  NextMethod("cvLM")
}
