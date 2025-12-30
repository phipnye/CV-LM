## reg.table.R: Linear regression latex and html tables
##
## This file is part of the cvLM package.

get.fun <- function(FUN) get(FUN, envir = asNamespace("cvLM"), mode = "function")

p.a <- function(...) paste0(..., collapse = " & ")
p.n <- function(...) paste0(..., collapse = "")

fmt <- function(x, n.digits, big.mark) {
  format(round(x, n.digits), nsmall = n.digits, big.mark = big.mark, trim = TRUE)
}

r.squared <- function(model) summary(model)[["r.squared"]]
adj.r.squared <- function(model) summary(model)[["adj.r.squared"]]
fstatistic <- function(model) summary(model)[[c("fstatistic", "value")]]

coeff.df <- function(model.summary, n.digits, big.mark, type = c("latex", "html")) {
  type <- match.arg(type)
  coefs <- coef(model.summary)

  predictors <- rownames(coefs)
  estimates <- fmt(coefs[, "Estimate"], n.digits, big.mark)
  p.vals <- coefs[, "Pr(>|t|)"]
  p.stars <- rep_len("", length(p.vals))

  if (type == "latex") {
    p.stars[p.vals <= 0.1] <- "$^{*}$"
    p.stars[p.vals <= 0.05] <- "$^{**}$"
    p.stars[p.vals <= 0.01] <- "$^{***}$"
  } else if (type == "html") {
    p.stars[p.vals <= 0.1] <- "<sup>*</sup>"
    p.stars[p.vals <= 0.05] <- "<sup>**</sup>"
    p.stars[p.vals <= 0.01] <- "<sup>***</sup>"
  }

  estimates <- paste0(estimates, p.stars)
  std.err <- paste0("(", fmt(coefs[, "Std. Error"], n.digits, big.mark), ")")

  DF <- data.frame(
    Predictor = c(rbind(predictors, sprintf("%s.std.err", predictors))),
    Value = c(rbind(estimates, std.err))
  )

  return(DF)
}


latex <- function(models, n.digits, big.mark, caption, spacing, stats, cv.args) {
  stats.ncv <- setdiff(stats, "CV")

  n.models <- length(models)
  if (is.null(model.names <- names(models))) {
    model.names <- paste0("(", seq_len(n.models), ")")
  }

  summs <- lapply(models, summary)

  reg.df <- coeff.df(summs[[1L]], n.digits, big.mark, "latex")
  colnames(reg.df) <- c("Predictor", model.names[1L])
  for (i in seq_along(summs)[-1L]) {
    temp.df <- coeff.df(summs[[i]], n.digits, big.mark, "latex")
    colnames(temp.df) <- c("Predictor", model.names[i])
    reg.df <- merge(reg.df, temp.df, by = "Predictor", all = TRUE, sort = FALSE)
  }

  reg.df[is.na(reg.df)] <- ""
  reg.df[["Predictor"]][grepl("\\.std\\.err$", reg.df[["Predictor"]], perl = TRUE)] <- ""

  reg.mat <- as.matrix(reg.df)
  ncr <- ncol(reg.mat)
  reg.mat[, ncr] <- paste0(reg.mat[, ncr], " \\\\\n")

  regression.table <- paste0(
    "\\begin{table}[!htbp]\n\\centering\n",
    sprintf("\\caption{%s}\n", caption),
    sprintf("\\begin{tabular}{@{\\extracolsep{%gpt}}l", spacing),
    p.n(rep("c", n.models)),
    "}\n",
    "\\hline\n\\hline\n",
    " & ", p.a(model.names), " \\\\\n\\hline\n",
    gsub("\\\\\n & ", "\\\\\n", p.a(t(reg.mat)), fixed = TRUE)
  )

  stats.mat <-
    if ((n.stats <- length(stats.ncv)) > 1L) {
      sapply(models, function(model) {
        vapply(stats.ncv, function(stat) {
          if (stat != "nobs") {
            fmt(do.call(get.fun(stat), list(model)), n.digits, big.mark)
          } else {
            as.character(do.call(get.fun(stat), list(model)))
          }
        }, character(1L))
      }, simplify = TRUE)
    } else if (n.stats == 1L) {
      matrix(
        vapply(models, function(model) {
          if (stats.ncv != "nobs") {
            fmt(do.call(get.fun(stats.ncv), list(model)), n.digits, big.mark)
          } else {
            as.character(do.call(get.fun(stats.ncv), list(model)))
          }
        }, character(1L)),
        nrow = 1L, ncol = n.models, dimnames = list(stats.ncv, NULL)
      )
    } else {
      matrix(nrow = 0L, ncol = n.models)
    }

  if ("CV" %in% stats) {
    CV <- vapply(models, function(model) do.call(cvLM, c(list(model), cv.args))[["CV"]], numeric(1L))
    CV <- fmt(CV, n.digits, big.mark)
    stats.mat <- rbind(stats.mat, CV)
  }

  stats.mat <- stats.mat[stats, , drop = FALSE]

  stat.labels <- stats
  stat.labels[stat.labels == "r.squared"] <- "$R^2$"
  stat.labels[stat.labels == "adj.r.squared"] <- "$\\bar{R}^2$"
  stat.labels[stat.labels == "fstatistic"] <- "F statistic"
  stat.labels[stat.labels == "nobs"] <- "Observations"
  stats.mat <- cbind(stat.labels, stats.mat)

  ncs <- ncol(stats.mat)
  stats.mat[, ncs] <- paste0(stats.mat[, ncs], " \\\\\n")

  regression.table <- paste0(
    regression.table,
    "\\hline\n",
    gsub("\\\\\n & ", "\\\\\n", p.a(t(stats.mat)), fixed = TRUE),
    "\\hline\n\\hline\n",
    sprintf("\\multicolumn{%d}{c}{\\textit{Note:} \\hfill $^{*}$p$<$0.1; $^{**}$p$<$0.05; $^{***}$p$<$0.01}\n", n.models + 1L),
    "\\end{tabular}\n\\end{table}\n"
  )

  return(regression.table)
}

html <- function(models, n.digits, big.mark, caption, spacing, stats, cv.args) {
  stats.ncv <- setdiff(stats, "CV")

  n.models <- length(models)
  if (is.null(model.names <- names(models))) {
    model.names <- paste0("(", seq_len(n.models), ")")
  }

  summs <- lapply(models, summary)

  reg.df <- coeff.df(summs[[1L]], n.digits, big.mark, "html")
  colnames(reg.df) <- c("Predictor", model.names[1L])
  for (i in seq_along(summs)[-1L]) {
    temp.df <- coeff.df(summs[[i]], n.digits, big.mark, "html")
    colnames(temp.df) <- c("Predictor", model.names[i])
    reg.df <- merge(reg.df, temp.df, by = "Predictor", all = TRUE, sort = FALSE)
  }

  reg.df[is.na(reg.df)] <- ""
  reg.df[["Predictor"]][grepl("\\.std\\.err$", reg.df[["Predictor"]], perl = TRUE)] <- ""

  reg.mat <- as.matrix(reg.df)
  reg.mat <- matrix(sprintf("<td style='text-align: center;'>%s</td>\n", reg.mat), dim(reg.mat))
  ncr <- ncol(reg.mat)
  reg.mat[, 1L] <- paste0("<tr>\n", sub("<td style='text-align: center;'>", "<td>", reg.mat[, 1L], fixed = TRUE))
  reg.mat[, ncr] <- paste0(reg.mat[, ncr], "</tr>\n")

  single.hline <- sprintf("<tr>\n<td colspan='%d'>\n<hr style='margin: 0.5px'>\n</td>\n</tr>\n", n.models + 1L)
  double.hline <- sprintf("<tr>\n<td colspan='%d'>\n<hr style='margin: 0.5px'>\n<hr style='margin: 0.5px;'>\n</td>\n</tr>\n", n.models + 1L)

  regression.table <- paste0(
    sprintf("<table style='border-spacing: %gpx 0;'>\n", spacing),
    sprintf("<caption>%s</caption>\n", caption),
    "<tbody>\n",
    double.hline,
    "<tr>\n",
    p.n(sprintf("<td>%s</td>\n", c("", model.names))),
    "</tr>\n",
    single.hline,
    p.n(t(reg.mat))
  )

  stats.mat <-
    if ((n.stats <- length(stats.ncv)) > 1L) {
      sapply(models, function(model) {
        vapply(stats.ncv, function(stat) {
          if (stat != "nobs") {
            fmt(do.call(get.fun(stat), list(model)), n.digits, big.mark)
          } else {
            as.character(do.call(get.fun(stat), list(model)))
          }
        }, character(1L))
      }, simplify = TRUE)
    } else if (n.stats == 1L) {
      matrix(
        vapply(models, function(model) {
          if (stats.ncv != "nobs") {
            fmt(do.call(get.fun(stats.ncv), list(model)), n.digits, big.mark)
          } else {
            as.character(do.call(get.fun(stats.ncv), list(model)))
          }
        }, character(1L)),
        nrow = 1L, ncol = n.models, dimnames = list(stats.ncv, NULL)
      )
    } else {
      matrix(nrow = 0L, ncol = n.models)
    }

  if ("CV" %in% stats) {
    CV <- vapply(models, function(model) do.call(cvLM, c(list(model), cv.args))[["CV"]], numeric(1L))
    CV <- fmt(CV, n.digits, big.mark)
    stats.mat <- rbind(stats.mat, CV)
  }

  stats.mat <- stats.mat[stats, , drop = FALSE]

  stat.labels <- stats
  stat.labels[stat.labels == "r.squared"] <- "<i>R</i><sup>2</sup>"
  stat.labels[stat.labels == "adj.r.squared"] <- "<span style='text-decoration: overline;'><i>R</i></span><sup>2</sup>"
  stat.labels[stat.labels == "fstatistic"] <- "F statistic"
  stat.labels[stat.labels == "nobs"] <- "Observations"
  stats.mat <- cbind(stat.labels, stats.mat)

  stats.mat <- matrix(sprintf("<td style='text-align: center;'>%s</td>\n", stats.mat), dim(stats.mat))
  ncs <- ncol(stats.mat)
  stats.mat[, 1L] <- paste0("<tr>\n", sub("<td style='text-align: center;'>", "<td>", stats.mat[, 1L], fixed = TRUE))
  stats.mat[, ncs] <- paste0(stats.mat[, ncs], "</tr>\n")

  regression.table <- paste0(
    regression.table,
    single.hline,
    p.n(t(stats.mat)),
    double.hline,
    sprintf("<tr>\n<td colspan='%d' style='text-align: left;'>\n<i>Note: </i>\n<span style='float: right;'>\n<sup>*</sup>p &lt; 0.1; \n<sup>**</sup>p &lt; 0.05; \n<sup>***</sup>p &lt; 0.01\n</span>\n</td>\n</tr>\n", n.models + 1),
    "</tbody>\n</table>"
  )

  return(regression.table)
}

reg.table <- function(models, type = c("latex", "html"), split.size = 4L, n.digits = 3L, big.mark = "",
                      caption = "Regression Results", spacing = 5,
                      stats = c("AIC", "BIC", "CV", "r.squared", "adj.r.squared", "fstatistic", "nobs"),
                      ...) {
  if (!all(vapply(models, is.linear.reg.model, logical(1L)))) {
    stop("All models should be a linear regression model.", call. = FALSE)
  }
  models <- lapply(models, function(model) {
    if (inherits(model, "glm")) {
      class(model) <- c("lm", setdiff(class(model), "lm"))
    }
    return(model)
  })

  stats <- match.arg(stats, c("CV", "AIC", "BIC", "r.squared", "adj.r.squared", "fstatistic", "nobs"), TRUE)

  if (length(split.size) != 1L || !is.integer(split.size)) {
    stop("Argument 'split.size' must be a single integer value.", call. = FALSE)
  }
  model.chunks <- split(models, ceiling(seq_along(models) / split.size))

  type <- match.arg(type, c("latex", "html"))
  TBL.FUN <- get.fun(type)

  if (length(n.digits) != 1L || !is.integer(n.digits)) {
    stop("Argument 'n.digits' must be a single integer value.", call. = FALSE)
  }
  if (length(big.mark) != 1L || !is.character(big.mark)) {
    stop("Argument 'big.mark' must be a single string value.", call. = FALSE)
  }
  if (length(caption) != 1L || !is.character(caption)) {
    stop("Argument 'caption' must be a single string value.", call. = FALSE)
  }
  if (length(spacing) != 1L || !is.numeric(spacing)) {
    stop("Argument 'spacing' must be a single numeric value.", call. = FALSE)
  }

  cv.args <- list(...)

  if ("K.vals" %in% (cv.argnames <- names(cv.args)) && length(cv.args[["K.vals"]]) > 1L) {
    stop("Argument 'K.vals' must be a single integer value.", call. = FALSE)
  } else if (is.null(cv.argnames) && length(cv.args[[2L]]) > 1L) {
    stop("Argument 'K.vals' must be a single integer value.", call. = FALSE)
  }

  regression.tables <- vapply(model.chunks, TBL.FUN, character(1L),
    n.digits = n.digits, big.mark = big.mark,
    caption = caption, spacing = spacing, stats = stats, cv.args = cv.args,
    USE.NAMES = FALSE
  )

  return(regression.tables)
}
