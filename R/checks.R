## checks.R: Validation of arguments
##
## This file is part of the cvLM package.

is.wholenumber <- function(x, tol = .Machine$double.eps^0.5) {
  abs(x - round(x)) < tol
}

.is_lm <- function(model) {
  inherits(model, "lm")
}

.assert_scalar <- function(x, type, name) {
  if (length(x) != 1L || !type(x) || is.na(x)) {
    stop(
      sprintf(
        "Argument '%s' must be a single, non-missing %s value.",
        name,
        sub("^is\\.", "", deparse(substitute(type)))
      ),
      call. = FALSE
    )
  }
}

.assert_logical_scalar <- function(x, name) {
  .assert_scalar(x, is.logical, name)
}

.assert_integer_scalar <- function(x, name, nonneg = FALSE) {
  .assert_scalar(x, is.wholenumber, name)
  x <- as.integer(x)

  if (nonneg && x < 0L) {
    stop(sprintf("Argument '%s' must be non-negative.", name), call. = FALSE)
  }

  if (!is.finite(x)) {
    stop(
      sprintf("Argument '%s' must be finite and fit in an integer.", name),
      call. = FALSE
    )
  }

  x
}

.assert_double_scalar <- function(x, name, nonneg = FALSE) {
  .assert_scalar(x, is.numeric, name)

  if (nonneg && x < 0) {
    stop(sprintf("Argument '%s' must be non-negative.", name), call. = FALSE)
  }

  if (!is.finite(x)) {
    stop(sprintf("Argument '%s' must be finite.", name), call. = FALSE)
  }

  # Return as double
  as.double(x)
}

.assert_valid_data <- function(y, X) {
  # X must be a matrix
  if (!is.matrix(X) || !is.double(X)) {
    stop("The design matrix must be a numeric matrix.", call. = FALSE)
  }

  # y must be a vector
  if (!is.atomic(y) || !is.double(y)) {
    stop("The response vector must be a numeric vector", call. = FALSE)
  }

  pred.nrow <- nrow(X)
  resp.nrow <- length(y)

  # Ensure dimensions align
  if (pred.nrow != resp.nrow) {
    stop(
      sprintf(
        "Dimension mismatch: Response has %d observations, but design matrix has %d rows.",
        resp.nrow,
        pred.nrow
      ),
      call. = FALSE
    )
  }

  # Check numerical integrity (no NA, NaN, or Inf)
  if (!all(is.finite(y))) {
    stop(
      "The response variable contains invalid values (NA, NaN, or Inf).",
      call. = FALSE
    )
  }

  if (!all(is.finite(X))) {
    stop(
      "The design matrix contains invalid values (NA, NaN, or Inf).",
      call. = FALSE
    )
  }

  # Make sure data isn't empty
  if (pred.nrow < 2L) {
    stop("Insufficient dataset size.", call. = FALSE)
  }

  if (ncol(X) == 0L) {
    stop("The design matrix has no columns.", call. = FALSE)
  }
}

.assert_valid_kvals <- function(K.vals, n) {
  # Confirm integer values
  K.vals <- vapply(
    as.vector(K.vals),
    .assert_integer_scalar,
    integer(1L),
    name = "K.val",
    USE.NAMES = FALSE
  )

  # Make sure number of folds between 2 and n
  if (any(K.vals < 2L)) {
    stop("All values of K must be >= 2.", call. = FALSE)
  }

  if (any(K.vals > n)) {
    stop(
      "All values of K must be <= number of observations.",
      call. = FALSE
    )
  }

  # Return unique and as integer
  unique(K.vals)
}

.assert_valid_threads <- function(n.threads) {
  n.threads <- .assert_integer_scalar(n.threads, "n.threads", nonneg = FALSE)
  def.threads <- RcppParallel::defaultNumThreads()

  if (identical(n.threads, -1L)) {
    return(def.threads)
  }

  max(1L, min(def.threads, n.threads))
}
