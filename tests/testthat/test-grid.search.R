# --- Global helpers and shared data

gen.data <- function(n, p, sparse.p, noise.sd, rank.def = FALSE) {
  # Raw scrambled features
  X.raw <- (sample.int(1e7, n * p, replace = TRUE) %% 2000) / 100 - 10
  X <- matrix(X.raw, n, p)

  # Coefficients with controlled signal
  beta <- (sample.int(500, p) / 100)
  beta[sample.int(p, sparse.p)] <- 0

  # Rank-deficiency option
  if (rank.def && p >= 3) {
    X[, p] <- X[, 1] + X[, 2]
  }

  # Response
  signal <- X %*% beta
  noise <- rnorm(n, sd = noise.sd)
  y <- signal + noise

  cbind(data.frame(y = as.numeric(y)), X)
}

muffle <- function(expr) {
  withCallingHandlers(
    expr,
    warning = function(w) {
      if (grepl("K has been changed", w$message, fixed = TRUE)) {
        invokeRestart("muffleWarning")
      }
    }
  )
}

# --- Scenario data sets

### Expect min cases to select near zero shrinkage (if not zero)
### Expect mid cases to select a mid-range shrinkage value
### Expect max cases to select max.lambda

set.seed(213)

# Narrow (n > p)
df.narrow.min <- gen.data(n = 183, p = 5, sparse.p = 0, noise.sd = 1e-4)
df.narrow.mid <- gen.data(n = 126, p = 20, sparse.p = 10, noise.sd = 10)
df.narrow.max <- gen.data(n = 181, p = 32, sparse.p = 30, noise.sd = 1e4)

# Wide (p > n)
df.wide.min <- gen.data(n = 51, p = 103, sparse.p = 1, noise.sd = 1e-3)
df.wide.mid <- gen.data(n = 75, p = 85, sparse.p = 35, noise.sd = 25)
df.wide.max <- gen.data(n = 56, p = 181, sparse.p = 176, noise.sd = 1e4)

# Rank-deficient
df.rd.mid <- gen.data(
  n = 132,
  p = 4,
  sparse.p = 2,
  noise.sd = 3,
  rank.def = TRUE
)

# --- Test parameters

K.vals <- list(2, 5, 10, NULL)
seed <- 73568569
max.lambda <- 100
precision <- 0.5
scenarios <- list(
  df.narrow.min,
  df.narrow.mid,
  df.narrow.max,
  df.wide.min,
  df.wide.mid,
  df.wide.max,
  df.rd.mid
)

# --- Run tests

test_that("grid.search matches brute-force cvLM sweep", {
  # Can take a long time to run
  skip_on_cran()

  # Simulate grid generator
  lambdas <- seq(0, max.lambda, by = precision)

  if (max.lambda != tail(lambdas, 1)) {
    lambdas <- c(lambdas, max.lambda)
  }

  for (data.set in scenarios) {
    for (K in K.vals) {
      K <- K %||% nrow(data.set)
      is.loocv <- K == nrow(data.set)
      generalized.opts <- if (is.loocv) c(FALSE, TRUE) else FALSE

      for (generalized in generalized.opts) {
        for (center in c(FALSE, TRUE)) {
          common.args <- list(
            y ~ .,
            data = data.set,
            generalized = generalized,
            seed = seed,
            center = center
          )

          grid.res <- muffle(do.call(
            grid.search,
            c(
              common.args,
              list(K = K, max.lambda = max.lambda, precision = precision)
            )
          ))

          tm <- system.time({
            manual.cvs <- vapply(
              lambdas,
              function(lambda) {
                muffle(do.call(
                  cvLM,
                  c(common.args, list(K.vals = K, lambda = lambda))
                ))$CV
              },
              numeric(1)
            )
          })

          best.idx <- which.min(manual.cvs)
          expect_equal(grid.res$CV, manual.cvs[best.idx])
          expect_equal(grid.res$lambda, lambdas[best.idx])
        }
      }
    }
  }
})

test_that("grid.search results are agnostic to the number of threads", {
  # Skip multithreaded tests on CRAN
  skip_on_cran()
  multi.threads <- max(RcppParallel::defaultNumThreads(), 2L)

  for (data.set in scenarios) {
    for (K in K.vals) {
      K <- K %||% nrow(data.set)
      is.loocv <- K == nrow(data.set)
      generalized.opts <- if (is.loocv) c(FALSE, TRUE) else FALSE

      for (generalized in generalized.opts) {
        for (center in c(FALSE, TRUE)) {
          common.args <- list(
            y ~ .,
            data = data.set,
            generalized = generalized,
            seed = seed,
            center = center,
            K = K,
            max.lambda = max.lambda,
            precision = precision
          )

          res.single <- muffle(do.call(
            grid.search,
            c(common.args, list(n.threads = 1L))
          ))

          res.multiple <- muffle(do.call(
            grid.search,
            c(common.args, list(n.threads = multi.threads))
          ))

          # Results may not be exactly identical because of the lack of associativity for floating point
          # addition
          expect_equal(res.single$CV, res.multiple$CV)
          expect_identical(res.single$lambda, res.multiple$lambda)
        }
      }
    }
  }
})
