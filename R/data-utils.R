## data-utils.R: Data preparation for C++ tasks
##
## This file is part of the cvLM package.

.center_data <- function(y, X, mt) {
  # Identify the intercept column
  intercept.col <- which(colnames(X) == "(Intercept)")

  if (length(intercept.col) > 0L) {
    # Center y
    y <- y - mean(y)

    # Drop intercept and center X columns (when you center y and X, the regression line is forced to pass
    # through the origin of the new coordinate system, where the intercept is mathematically guaranteed to be
    # zero, hence a column of ones is not needed anymore
    X <- X[, -intercept.col, drop = FALSE]
    X <- sweep(X, 2L, colMeans(X), FUN = "-", check.margin = FALSE)
  }

  list(y = y, X = X)
}

.prepare_lm_data <- function(object, data, subset = NULL, na.action = NULL, env = parent.frame()) {
  if (!inherits(object, "formula")) stop("`object` must be a formula.", call. = FALSE)
  
  mf_args <- list(
    formula = object,
    data = data,
    drop.unused.levels = TRUE
  )
  if (!is.null(subset)) mf_args$subset <- subset
  if (!is.null(na.action)) mf_args$na.action <- na.action
  
  mf <- do.call(stats::model.frame, mf_args, envir = env)
  
  mt <- attr(mf, "terms")
  if (is.null(mt)) stop("No terms object found.", call. = FALSE)
  
  y <- model.response(mf, type = "numeric")
  if (is.null(y)) stop("Response must be numeric.", call. = FALSE)
  
  X <- model.matrix(mt, mf)
  if (!is.numeric(y)) stop("Response variable must be numeric.")
  if (!is.matrix(X) || !is.numeric(X)) stop("Design matrix must be a numeric matrix.")
  
  attr(X, "xlevels") <- .getXlevels(mt, mf)
  attr(y, "na.action") <- attr(mf, "na.action")
  
  list(y = y, X = X, mt = mt)
}
