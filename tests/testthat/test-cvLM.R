# --- Global helpers and shared data

gen.data <- function(n, p, scale = 10) {
  X <- matrix(runif(n * p, -scale, scale), n, p)
  y <- runif(n, -scale, scale)
  cbind(data.frame(y = y), X)
}

muffle <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    # Expected warning patterns from boot::cv.glm and cvLM
    pats <- "K has been changed|'K' has been set to|prediction from rank-deficient fit"

    if (grepl(pats, w$message)) {
      invokeRestart("muffleWarning")
    }
  })
}

set.seed(1)

# Typical narrow regression data (using a prime n to ensure the CV handles uneven folds)
df.narrow <- gen.data(101, 11)

# Ill-conditioned data
df.ill <- gen.data(83, 4)
df.ill[, 2] <- df.ill[, 2] * 1e-3
df.ill[, 3] <- df.ill[, 3] * 1e3

# Rank-deficient (but still overdetermined)
df.rd <- gen.data(113, 4)
df.rd <- cbind(df.rd, `5` = df.rd[, 2] + df.rd[, 3])

# Wide data - here the results should differ for OLS w/ boot::cv.glm
df.wide <- gen.data(63, 117)

# Test parameters
seed <- 73568569
K.vals <- list(2, 3, 4, 5, 7, 8, 10, NULL) # null corresponds to loocv
lambdas <- c(0.6148868, 48.08172, 7230.901)

test_that("cvLM matches boot::cv.glm for OLS", {
  # Note: cvLM will not match boot::cv.glm for matrices that are wide, R's lm uses dqrc2/linpack,
  # which gives a least-squares solution on overdetermined systems by placing 0 coefficients on redundant
  # covariates, cvLM instead estimates using the unique minimum norm solution; however, this will only
  # affect out-of-sample predictions on wide data sets

  for (data.set in list(df.narrow, df.ill, df.rd, df.wide)) {
    fit.glm <- glm(y ~ ., data = data.set)
    is.wide <- nrow(data.set) < ncol(data.set)

    for (K in K.vals) {
      K <- K %||% nrow(data.set) # LOOCV if null

      # Skip computing boot for wide LOOCV because diag(H) = 1 and we get zero division
      if (is.wide && K == nrow(data.set)) {
        res <- cvLM(fit.glm, data = data.set, K.vals = K, center = FALSE)$CV
        expect_true(is.nan(res))
        next
      }

      set.seed(seed)

      boot.res <- muffle(boot::cv.glm(
        data = data.set,
        glmfit = fit.glm,
        K = K
      ))$delta[1]

      cvLM.res <- muffle(cvLM(
        fit.glm,
        data = data.set,
        K.vals = K,
        seed = seed,
        n.threads = 1,
        tol = min(1e-07, fit.glm$control$epsilon / 1000), # match glm.fit
        center = FALSE
      ))$CV

      if (is.wide) {
        expect_true(abs(boot.res - cvLM.res) > testthat_tolerance())
      } else {
        expect_equal(boot.res, cvLM.res)
      }
    }
  }
})

test_that("OLS CV is invariant to centering on narrow data", {
  # Only narrow / overdetermined datasets (could differ on underdetermined systems since we use minimum norm
  # solutions)
  for (data.set in list(df.narrow, df.ill, df.rd)) {
    for (K in K.vals) {
      K <- K %||% nrow(data.set)
      common.args <- list(
        y ~ .,
        data = data.set,
        K.vals = K,
        lambda = 0,
        seed = seed
      )

      res.center <- muffle(do.call(
        cvLM,
        c(common.args, list(center = TRUE))
      ))$CV

      res.nocenter <- muffle(do.call(
        cvLM,
        c(common.args, list(center = FALSE))
      ))$CV

      expect_equal(res.center, res.nocenter)
    }
  }
})

test_that("cvLM matches manual matrix algebra for Ridge (K-fold and GCV)", {
  ridge.cv.ref <- function(y, X, K, lambda, seed, center) {
    set.seed(seed)
    n <- nrow(X)

    # Mirror boot's logic of changing the number of folds
    if ((K > n) || (K <= 1)) {
      stop("'K' outside allowable range")
    }

    K <- round(K)
    kvals <- unique(round(n / (1L:floor(n / 2))))
    temp <- abs(kvals - K)

    if (!any(temp == 0)) {
      K <- kvals[temp == min(temp)][1L]
    }

    s <- sample(rep(1L:K, ceiling(n / K)), n)
    cv.errors <- numeric(K)
    weights <- table(s) / n

    for (i in seq_len(max(s))) {
      j.in <- which(s != i)
      j.out <- which(s == i)

      if (center) {
        X.in.colMeans <- colMeans(X[j.in, , drop = FALSE])
        y.in.mean <- mean(y[j.in])
        X.in <- scale(
          X[j.in, , drop = FALSE],
          center = X.in.colMeans,
          scale = FALSE
        )
        y.in <- y[j.in] - y.in.mean
        X.out <- scale(
          X[j.out, , drop = FALSE],
          center = X.in.colMeans,
          scale = FALSE
        )
        beta <- solve(
          crossprod(X.in) + lambda * diag(ncol(X.in)),
          crossprod(X.in, y.in)
        )
        preds <- (X.out %*% beta) + y.in.mean
      } else {
        X.in <- cbind(1, X[j.in, , drop = FALSE])
        y.in <- y[j.in]
        X.out <- cbind(1, X[j.out, , drop = FALSE])
        beta <- solve(
          crossprod(X.in) + lambda * diag(ncol(X.in)),
          crossprod(X.in, y.in)
        )
        preds <- X.out %*% beta
      }

      cv.errors[i] <- mean((y[j.out] - preds)^2)
    }

    sum(as.numeric(weights) * cv.errors)
  }

  ridge.gcv.ref <- function(y, X, lambda, center) {
    n <- nrow(X)

    if (center) {
      y.mean <- mean(y)
      X.mat <- scale(X, center = colMeans(X), scale = FALSE)
      y.target <- y - y.mean
      XtX.lambda.inv <- solve(crossprod(X.mat) + (lambda * diag(ncol(X.mat))))
      beta <- XtX.lambda.inv %*% crossprod(X.mat, y.target)
      H <- X.mat %*% XtX.lambda.inv %*% t(X.mat) + matrix(1 / n, n, n)
      preds <- (X.mat %*% beta) + y.mean
    } else {
      X.mat <- cbind(1, X)
      XtX.lambda.inv <- solve(crossprod(X.mat) + (lambda * diag(ncol(X.mat))))
      beta <- XtX.lambda.inv %*% crossprod(X.mat, y)
      H <- X.mat %*% XtX.lambda.inv %*% t(X.mat)
      preds <- X.mat %*% beta
    }

    trH <- sum(diag(H))
    mean(((y - preds) / (1 - trH / n))^2)
  }

  for (data.set in list(df.narrow, df.wide, df.rd)) {
    X.mat <- as.matrix(data.set[, -1])
    y.vec <- data.set[, 1]

    for (lambda in lambdas) {
      for (center in c(FALSE, TRUE)) {
        # GCV test
        expect_equal(
          cvLM(
            y ~ .,
            data = data.set,
            lambda = lambda,
            generalized = TRUE,
            center = center
          )$CV,
          ridge.gcv.ref(y.vec, X.mat, lambda, center)
        )

        # K-fold and LOOCV tests
        for (K in K.vals) {
          K <- K %||% nrow(data.set)
          expect_equal(
            muffle(cvLM(
              y ~ .,
              data = data.set,
              K.vals = K,
              lambda = lambda,
              seed = seed,
              center = center
            ))$CV,
            ridge.cv.ref(y.vec, X.mat, K, lambda, seed, center)
          )
        }
      }
    }
  }
})

test_that("cvLM S3 methods return identical results", {
  for (data.set in list(df.narrow, df.wide, df.rd, df.ill)) {
    fit.lm <- lm(y ~ ., data = data.set)
    fit.glm <- glm(y ~ ., data = data.set)

    for (K in K.vals) {
      K <- K %||% nrow(data.set)

      for (lambda in lambdas) {
        common.args <- list(K.vals = K, lambda = lambda, seed = seed)
        res.formula <- muffle(do.call(
          cvLM,
          c(list(y ~ ., data = data.set), common.args)
        ))
        res.lm <- muffle(do.call(cvLM, c(list(fit.lm), common.args)))
        expect_identical(res.formula, res.lm)
        res.glm <- muffle(do.call(cvLM, c(list(fit.glm), common.args)))
        expect_identical(res.formula, res.glm)
      }
    }
  }
})

test_that("cvLM results are agnostic to the number of threads", {
  # Skip multithreaded tests on CRAN
  skip_on_cran()
  multi.threads <- max(RcppParallel::defaultNumThreads(), 2L)

  for (data.set in list(df.narrow, df.wide, df.rd, df.ill)) {
    for (K in K.vals) {
      if (is.null(K)) {
        next # LOOCV isn't multithreaded
      }

      for (lambda in lambdas) {
        common.args <- list(
          y ~ .,
          data = data.set,
          K.vals = K,
          lambda = lambda,
          seed = seed
        )
        res.single <- muffle(do.call(
          cvLM,
          c(common.args, list(n.threads = 1L))
        ))
        res.multiple <- muffle(do.call(
          cvLM,
          c(common.args, list(n.threads = multi.threads))
        ))

        # Results may not be exactly identical because of the lack of associativity for floating point
        # addition
        expect_equal(res.single, res.multiple)
      }
    }
  }
})
