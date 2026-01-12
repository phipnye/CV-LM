## reg.table.R: Linear regression latex and html tables
##
## This file is part of the cvLM package.

.get_model_names <- function(models) {
  model.names <- names(models)

  if (is.null(model.names)) {
    model.names <- paste0("(", seq_along(models), ")")
  }

  model.names
}

.build_coef_df <- function(models, n.digits, big.mark, type) {
  summs <- lapply(models, summary)
  model.names <- .get_model_names(models)
  reg.df <- .coef_df(summs[[1L]], n.digits, big.mark, type)
  colnames(reg.df) <- c("Predictor", model.names[1L])

  for (i in seq_along(summs)[-1L]) {
    temp.df <- .coef_df(summs[[i]], n.digits, big.mark, type)
    colnames(temp.df) <- c("Predictor", model.names[i])
    reg.df <- merge(reg.df, temp.df, by = "Predictor", all = TRUE, sort = FALSE)
  }

  reg.df[is.na(reg.df)] <- ""
  reg.df$Predictor[grepl("\\.std\\.err$", reg.df$Predictor, perl = TRUE)] <- ""
  reg.df
}

.get_stat_value <- function(model, stat, n.digits, big.mark) {
  val <- do.call(.get_fun(stat), list(model))

  if (stat != "nobs") {
    .fmt(val, n.digits, big.mark)
  } else {
    as.character(val)
  }
}

.build_stats_mat <- function(models, stats, n.digits, big.mark, cv.args) {
  stats.ncv <- setdiff(stats, "CV")
  n.models <- length(models)

  stats.mat <-
    if ((n.stats <- length(stats.ncv)) > 1L) {
      sapply(
        models,
        function(model) {
          vapply(
            stats.ncv,
            .get_stat_value,
            character(1L),
            model = model,
            n.digits = n.digits,
            big.mark = big.mark
          )
        },
        simplify = TRUE
      )
    } else if (n.stats == 1L) {
      matrix(
        vapply(
          models,
          .get_stat_value,
          character(1L),
          stat = stats.ncv,
          n.digits = n.digits,
          big.mark = big.mark
        ),
        nrow = 1L,
        ncol = n.models,
        dimnames = list(stats.ncv, NULL)
      )
    } else {
      matrix(nrow = 0L, ncol = n.models)
    }

  if ("CV" %in% stats) {
    CV <- vapply(
      models,
      function(model) do.call(cvLM, c(list(model), cv.args))$CV,
      numeric(1L)
    )
    stats.mat <- rbind(stats.mat, .fmt(CV, n.digits, big.mark))
  }

  stats.mat[stats, , drop = FALSE]
}

latex <- function(
  models,
  n.digits,
  big.mark,
  caption,
  spacing,
  stats,
  cv.args
) {
  n.models <- length(models)
  model.names <- .get_model_names(models)
  reg.df <- .build_coef_df(models, n.digits, big.mark, "latex")
  reg.mat <- as.matrix(reg.df)
  ncr <- ncol(reg.mat)
  reg.mat[, ncr] <- paste0(reg.mat[, ncr], " \\\\\n")
  regression.table <- paste0(
    "\\begin{table}[!htbp]\n\\centering\n",
    sprintf("\\caption{%s}\n", caption),
    sprintf("\\begin{tabular}{@{\\extracolsep{%gpt}}l", spacing),
    .p_n(rep("c", n.models)),
    "}\n",
    "\\hline\n\\hline\n",
    " & ",
    .p_a(model.names),
    " \\\\\n\\hline\n",
    gsub("\\\\\n & ", "\\\\\n", .p_a(t(reg.mat)), fixed = TRUE)
  )
  stats.mat <- .build_stats_mat(models, stats, n.digits, big.mark, cv.args)
  stat.labels <- stats
  stat.labels[stat.labels == "r.squared"] <- "$R^2$"
  stat.labels[stat.labels == "adj.r.squared"] <- "$\\bar{R}^2$"
  stat.labels[stat.labels == "fstatistic"] <- "F statistic"
  stat.labels[stat.labels == "nobs"] <- "Observations"
  stats.mat <- cbind(stat.labels, stats.mat)
  ncs <- ncol(stats.mat)
  stats.mat[, ncs] <- paste0(stats.mat[, ncs], " \\\\\n")
  paste0(
    regression.table,
    "\\hline\n",
    gsub("\\\\\n & ", "\\\\\n", .p_a(t(stats.mat)), fixed = TRUE),
    "\\hline\n\\hline\n",
    sprintf(
      "\\multicolumn{%d}{c}{\\textit{Note:} \\hfill $^{*}$p$<$0.1; $^{**}$p$<$0.05; $^{***}$p$<$0.01}\n",
      n.models + 1L
    ),
    "\\end{tabular}\n\\end{table}\n"
  )
}

html <- function(models, n.digits, big.mark, caption, spacing, stats, cv.args) {
  n.models <- length(models)
  model.names <- .get_model_names(models)
  reg.df <- .build_coef_df(models, n.digits, big.mark, "html")
  reg.mat <- matrix(
    sprintf("<td style='text-align: center;'>%s</td>\n", as.matrix(reg.df)),
    dim(reg.df)
  )
  ncr <- ncol(reg.mat)
  reg.mat[, 1L] <- paste0(
    "<tr>\n",
    sub("<td style='text-align: center;'>", "<td>", reg.mat[, 1L], fixed = TRUE)
  )
  reg.mat[, ncr] <- paste0(reg.mat[, ncr], "</tr>\n")
  single.hline <- sprintf(
    "<tr>\n<td colspan='%d'>\n<hr style='margin: 0.5px'>\n</td>\n</tr>\n",
    n.models + 1L
  )
  double.hline <- sprintf(
    "<tr>\n<td colspan='%d'>\n<hr style='margin: 0.5px'>\n<hr style='margin: 0.5px;'>\n</td>\n</tr>\n",
    n.models + 1L
  )
  regression.table <- paste0(
    sprintf("<table style='border-spacing: %gpx 0;'>\n", spacing),
    sprintf("<caption>%s</caption>\n", caption),
    "<tbody>\n",
    double.hline,
    "<tr>\n",
    .p_n(sprintf("<td>%s</td>\n", c("", model.names))),
    "</tr>\n",
    single.hline,
    .p_n(t(reg.mat))
  )
  stats.mat <- .build_stats_mat(models, stats, n.digits, big.mark, cv.args)
  stat.labels <- stats
  stat.labels[stat.labels == "r.squared"] <- "<i>R</i><sup>2</sup>"
  stat.labels[stat.labels == "adj.r.squared"] <-
    "<span style='text-decoration: overline;'><i>R</i></span><sup>2</sup>"
  stat.labels[stat.labels == "fstatistic"] <- "F statistic"
  stat.labels[stat.labels == "nobs"] <- "Observations"
  stats.mat <- cbind(stat.labels, stats.mat)
  stats.mat <- matrix(
    sprintf("<td style='text-align: center;'>%s</td>\n", stats.mat),
    dim(stats.mat)
  )
  ncs <- ncol(stats.mat)
  stats.mat[, 1L] <- paste0(
    "<tr>\n",
    sub(
      "<td style='text-align: center;'>",
      "<td>",
      stats.mat[, 1L],
      fixed = TRUE
    )
  )
  stats.mat[, ncs] <- paste0(stats.mat[, ncs], "</tr>\n")
  paste0(
    regression.table,
    single.hline,
    .p_n(t(stats.mat)),
    double.hline,
    sprintf(
      "<tr>\n<td colspan='%d' style='text-align: left;'>\n<i>Note: </i>\n<span style='float: right;'>\n<sup>*</sup>p &lt; 0.1; \n<sup>**</sup>p &lt; 0.05; \n<sup>***</sup>p &lt; 0.01\n</span>\n</td>\n</tr>\n",
      n.models + 1L
    ),
    "</tbody>\n</table>"
  )
}

reg.table <- function(
  models,
  type = c("latex", "html"),
  split.size = 4L,
  n.digits = 3L,
  big.mark = "",
  caption = "Regression Results",
  spacing = 5,
  stats = c(
    "AIC",
    "BIC",
    "CV",
    "r.squared",
    "adj.r.squared",
    "fstatistic",
    "nobs"
  ),
  ...
) {
  if (!all(vapply(models, .is_lm, logical(1L)))) {
    stop("All models should be a linear regression model.", call. = FALSE)
  }

  stats <- match.arg(
    stats,
    c("CV", "AIC", "BIC", "r.squared", "adj.r.squared", "fstatistic", "nobs"),
    TRUE
  )

  if (length(split.size) != 1L || !is.integer(split.size)) {
    stop("Argument 'split.size' must be a single integer value.", call. = FALSE)
  }

  model.chunks <- split(models, ceiling(seq_along(models) / split.size))
  type <- match.arg(type, c("latex", "html"))
  TBL.FUN <- .get_fun(type)
  n.digits <- .assert_integer_scalar(n.digits, "n.digits", nonneg = TRUE)
  .assert_scalar(big.mark, is.character, "big.mark")
  .assert_scalar(caption, is.character, "caption")
  spacing <- .assert_double_scalar(spacing, "spacing", nonneg = TRUE)
  cv.args <- list(...)
  cv.argnames <- names(cv.args)

  if (length(cv.args$K.vals) > 1L) {
    stop("Argument 'K.vals' must be a single integer value.", call. = FALSE)
  }
  
  # TO DO: Add warnings for when model statistics won't align with CV results (like introducing a lambda > 0
  # on a linear model)
  
  vapply(
    model.chunks,
    TBL.FUN,
    character(1L),
    n.digits = n.digits,
    big.mark = big.mark,
    caption = caption,
    spacing = spacing,
    stats = stats,
    cv.args = cv.args,
    USE.NAMES = FALSE
  )
}
