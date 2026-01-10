source("tests/helpers.R")

test_that("cvLM matches boot::cv.glm for OLS under varied conditions, including rank-deficient and ill-conditioned data", {
  sample.sizes <- c(20, 50, 100, 1000)
  predictors <- c(2, 4, 6)
  Ks.list <- list(
    few = c(2, 3),
    typical = c(5, 10),
    full = NULL # will use nrow as K
  )
  seeds <- c(1L, 42L, 123L)

  for (n in sample.sizes) {
    for (p in predictors) {
      # In the case of wide data, R zeroes out coefficients (a general solution) whereas we use complete
      # orthogonal decomposition which gives the unique minimum norm solution, hence results won't match
      if (p >= n) {
        next
      }

      dat <- make.data(n = n, p = p)
      nrow.df <- nrow(dat$df)

      # --- Setup varied data scenarios

      # Categorical data
      dat.fact <- dat$df
      dat.fact$group <- factor(rep(c("A", "B"), length.out = n))

      # Interaction and polynomial
      formulas <- list(
        standard = dat$formula,
        interaction = as.formula(paste(
          names(dat$df)[1],
          "~",
          names(dat$df)[2],
          "*",
          names(dat$df)[3]
        )),
        polynomial = as.formula(paste(
          names(dat$df)[1],
          "~ poly(",
          names(dat$df)[2],
          ", 2)"
        ))
      )

      # Main testing loop
      for (seed in seeds) {
        for (Ks.name in names(Ks.list)) {
          Ks <- Ks.list[[Ks.name]]

          # LOOCV
          if (is.null(Ks)) {
            Ks <- nrow.df
          }

          for (K in Ks) {
            # Loop through the varied formula types
            for (f_type in names(formulas)) {
              target_formula <- formulas[[f_type]]
              cv.boot <- boot.cv.ols(target_formula, dat$df, K, seed)
              cv.pkg <- cvLM(
                target_formula,
                data = dat$df,
                K.vals = K,
                lambda = 0,
                generalized = FALSE,
                seed = seed,
                n.threads = 1
              )$CV

              diff <- abs(cv.pkg - cv.boot)
              msg <- paste0(
                "Mismatch detected!\n",
                "Scenario: ",
                f_type,
                "\n",
                "n = ",
                n,
                ", p = ",
                p,
                ", K = ",
                K,
                ", seed = ",
                seed,
                "\n",
                "cvLM = ",
                cv.pkg,
                "\n",
                "boot::cv.glm = ",
                cv.boot,
                "\n",
                "abs(diff) = ",
                diff
              )

              expect_true(diff < 1e-10, info = msg)
            }

            # Test categorical data
            cat_formula <- as.formula(paste(names(dat.fact)[1], "~ ."))
            cv.boot_cat <- boot.cv.ols(cat_formula, dat.fact, K, seed)
            cv.pkg_cat <- cvLM(
              cat_formula,
              data = dat.fact,
              K.vals = K,
              lambda = 0,
              seed = seed
            )$CV

            expect_true(
              abs(cv.pkg_cat - cv.boot_cat) < 1e-10,
              info = paste("Categorical mismatch at n =", n, "K =", K)
            )
          }
        }
      }

      # --- Rank-deficient case (duplicate last column)
      if (p > 1) {
        dat.df.rd <- dat$df
        last.col <- names(dat.df.rd)[p + 1]
        second.last.col <- names(dat.df.rd)[p]
        dat.df.rd[[last.col]] <- dat.df.rd[[second.last.col]]

        for (K in c(2, 5, 10)) {
          cv.boot <- boot.cv.ols(dat$formula, dat.df.rd, K, seed = 123L)
          cv.pkg <- cvLM(
            dat$formula,
            data = dat.df.rd,
            K.vals = K,
            lambda = 0,
            seed = 123L
          )$CV

          diff <- abs(cv.pkg - cv.boot)
          msg <- paste0(
            "Rank-deficient mismatch!\n",
            "n = ",
            n,
            ", p = ",
            p,
            ", K = ",
            K,
            "\n",
            "cvLM = ",
            cv.pkg,
            "\n",
            "boot::cv.glm = ",
            cv.boot,
            "\n",
            "abs(diff) = ",
            diff
          )
          expect_true(diff < 1e-10, info = msg)
        }
      }

      # --- Ill-conditioned case (make last column almost linear combination)
      if (p > 1) {
        dat.df.ic <- dat$df
        last.col <- names(dat.df.ic)[p + 1]
        second.last.col <- names(dat.df.ic)[p]
        dat.df.ic[[last.col]] <- dat.df.ic[[second.last.col]] *
          0.999 +
          rnorm(n) * 1e-6

        for (K in c(2, 5, 10)) {
          cv.boot <- boot.cv.ols(dat$formula, dat.df.ic, K, seed = 123L)
          cv.pkg <- cvLM(
            dat$formula,
            data = dat.df.ic,
            K.vals = K,
            lambda = 0,
            seed = 123L
          )$CV

          diff <- abs(cv.pkg - cv.boot)
          msg <- paste0(
            "Ill-conditioned mismatch!\n",
            "n = ",
            n,
            ", p = ",
            p,
            ", K = ",
            K,
            "\n",
            "cvLM = ",
            cv.pkg,
            "\n",
            "boot::cv.glm = ",
            cv.boot,
            "\n",
            "abs(diff) = ",
            diff
          )
          expect_true(diff < 1e-10, info = msg)
        }
      }
    }
  }
})
